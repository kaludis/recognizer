/*
Copyright (c) 2014 Alexander Bezsilko <demonsboots@gmail.com>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @file recogizer.cpp
 * @author Alexander Bezsilko
 * @date 4 Nov 2014
 * @brief File containing recognizer static methods and types definitions.
 */

#include <cstddef>
#include <memory>
#include <algorithm>
#include <iostream>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include "recognizer.h"

namespace recognizer
{

/**
 * Recognizer public interface.
 */
std::string
Recognizer::get_text(const std::string& file) throw (RecException)
{
    if (file.empty()) {
        throw RecException("bad file name");
    }

    cv::Mat image = cv::imread(file);

    return get_text(image);
}

/**
 * Recognizer public interface.
 */
std::string
Recognizer::get_text(const cv::Mat& image) throw (RecException)
{
    if (image.empty()) {
        throw RecException("failed to load image");
    }

    BoxesGroups boxes_groups = find_text_rects(image);
    if (!boxes_groups.size()) {
        return std::string();
    }

    remove_dup(boxes_groups);
    TextAreas text_areas = create_text_areas(image, boxes_groups);

    return alphabet_analisis(text_areas);
}

/**
 * Find rectangles containing characters.
 */    
Recognizer::BoxesGroups
Recognizer::find_text_rects(const cv::Mat& image)
{
    Channels channels;
    cv::text::computeNMChannels(image, channels);

    std::size_t cn = channels.size();
    for (std::size_t c = 0; c < cn - 1; ++c) {
        channels.push_back(max_channel_ - channels[c]);
    }

    BoxesGroups boxes_groups;

    try {
        // Creating external filtrs for 1nd stage classifier of N&M algorithm
        ERFilterPtr er_filter1 =
                cv::text::createERFilterNM1(
                    cv::text::loadClassifierNM1(classifierNM1_),
                    16,
                    0.00015f,
                    0.13f,
                    0.2f,
                    true,
                    0.1f);
        // Creating external filtrs for 2nd stage classifier of N&M algorithm
        ERFilterPtr er_filter2 =
                cv::text::createERFilterNM2(
                    cv::text::loadClassifierNM2(classifierNM2_),
                    0.5);

        if (!er_filter1 || !er_filter2) {
            throw RecException("could not create external region filters");
        }

        // Apply the default cascade classifier to each
        // independent channel (could be done in parallel)    
        cn = channels.size();
        Regions regions(cn);
        for (std::size_t c = 0; c < cn; ++c) {
            er_filter1->run(channels[c], regions[c]);
            er_filter2->run(channels[c], regions[c]);
        }

        // Detect character groups    
        RegionGroups region_groups;


        cv::text::erGrouping(image,
                             channels,
                             regions,
                             region_groups,
                             boxes_groups,
                             cv::text::ERGROUPING_ORIENTATION_ANY,
                             classifierGrouping_,
                             0.5);

    return boxes_groups;
    
    } catch (const cv::Exception& ex) {
        throw RecException(ex.what());
    }
}

/**
 * Removing duplicate rectangles.
 */        
void
Recognizer::remove_dup(BoxesGroups& boxes)
{
    BoxesGroups::iterator outer = boxes.begin();

    // Remove equals and contains rectangles
    for (; outer != boxes.end(); ++outer) {
        BoxesGroups::iterator inner = outer;
        ++inner;

        for (; inner != boxes.end(); ) {
            if (outer->contains(inner->tl()) &&
                (outer->contains(inner->br()))) {
                inner = boxes.erase(inner);
                continue;                
            }

            if (inner->contains(outer->tl()) &&
                inner->contains(outer->br())) {
                outer = boxes.erase(outer);
                break;                
            }
            
            ++inner;
        }
    }

    BoxesGroups::iterator outer_iter = boxes.begin();
    BoxesGroups::iterator outer_end = boxes.end();

    for (; outer_iter != boxes.end(); ) {
        BoxesGroups::iterator inner_iter = outer_iter;
        ++inner_iter;

        for (; inner_iter != boxes.end(); ) {
            if (((*inner_iter) & (*outer_iter)).area() != 0) {
                if (outer_iter->area() > inner_iter->area()) {
                    inner_iter = boxes.erase(inner_iter);
                    continue;
                } else {
                    outer_iter = boxes.erase(outer_iter);
                    inner_iter = outer_iter;
                    ++inner_iter;
                    continue;
                }
            } else {
                ++inner_iter;
            }
        }

        ++outer_iter;
    }
}

/**
 * Create images pieces with characters.
 */        
Recognizer::TextAreas
Recognizer::create_text_areas(const cv::Mat& image, const BoxesGroups& boxes)
{
    int sum_area = 0;
    for (const cv::Rect& r : boxes) {
        sum_area += r.area();
    }

    TextAreas text_areas;

    if (sum_area >= (image.size().area() / 2)) {
        cv::Mat gray;
        cv::cvtColor(image, gray, CV_RGB2GRAY, 0);
        text_areas.push_back(gray);
    } else {
        for (const cv::Rect& r : boxes) {
            cv::Mat gray, bw;
            cv::cvtColor(cv::Mat(image, r), gray, CV_RGB2GRAY, 0);
            cv::threshold(gray, bw, 127.5f, max_channel_, cv::THRESH_OTSU);
            text_areas.push_back(bw);
        }
    }

    return text_areas;
}

/**
 * Recognize preprocessed images for characters using tesseract ocr.
 */        
std::string
Recognizer::alphabet_analisis(const Recognizer::TextAreas& areas)
{
    Text rec_text;
    std::unique_ptr<tesseract::TessBaseAPI>
            ocr(new tesseract::TessBaseAPI());

    // Init tesseract dictionary for English language
    if (ocr->Init(nullptr, "eng", tesseract::OEM_DEFAULT)) {
        throw RecException("could not initialize tesseract ocr");
    }

    // Step by step recognize text areas
    for (const cv::Mat& area : areas) {
        ocr->SetImage(static_cast<uchar*>(static_cast<void*>(area.data)),
                      area.size().width,
                      area.size().height,
                      area.channels(),
                      area.step1());
        
        if (!ocr->Recognize(0)) {

            // Processing recognize result for trash characters elimination
            std::string res = string_processing(ocr->GetUTF8Text());
            if (!res.empty()) {
                rec_text.push_back(res);
            }
        }

        ocr->Clear();
    }

    ocr->End();

    std::string result;
    for (const std::string& s : rec_text) {
        result.append(s).append(1, ' ');
    }

    return result;
        //    return normalize_result(rec_text);
}

/**
 * Parse the string for the presence of unnecessary characters.
 */            
std::string Recognizer::string_processing(const std::string& str)
{
    std::size_t len = str.length();
    std::string norm1;
    for (std::size_t idx = 0; idx < len; ++idx) {
        if (((str[idx] >= 'a') && (str[idx] <= 'z')) ||
            ((str[idx] >= 'A') && (str[idx] <= 'Z')) ||
            ((str[idx] >= '0') && (str[idx] <= '9')) ||
            (str[idx] == ' ') || (str[idx] == ',') || (str[idx] == '.') ||
            (str[idx] == '!') || (str[idx] == '?')) {
            norm1.push_back(str[idx]);
        } else if (str[idx] == '\n') {
            if ((idx != 0) && (idx != len - 1) &&
                (str[idx - 1] != '\n' ) && (str[idx + 1] != '\n')) {
                    norm1.push_back(str[idx]);                
            }
        }
    }

    // Remove extra spaces
    len = norm1.length();
    std::string norm2;
    for (std::size_t idx = 0; idx < len; ++idx) {
        if (norm1[idx] == ' ') {
            if ((idx != 0) && (idx != len - 1) && (norm1[idx - 1] != ' ')) {
                norm2.push_back(' ');
            }
        } else {
            norm2.push_back(norm1[idx]);
        }
    }

    return norm2;
}

/**
 * Remove same character groups from string.
 */
std::string Recognizer::normalize_result(const Text& text)
{
    std::string str;

    for (const std::string s : text) {
        str.append(s).append(1, ' ');
    }

    std::vector<std::string> words;
    std::string::size_type start = 0, end;
    do {
        end = str.find(' ', start);
        if (end == std::string::npos) {
            end = str.length();
        }

        std::string tmp = str.substr(start, end - start);
        
        if (std::find(words.begin(), words.end(), tmp) == words.end()) {
            words.push_back(tmp);
        }
        
        start = end + 1;
    } while (start < str.length());

    str.clear();
    for (const std::string& s : words) {
        str.append(s).append(1, ' ');
    }

    str = str.substr(0, str.length() - 2);

    return str;
}

/**
 * Set full paths to algorithm classifiers.
 */
void Recognizer::set_classifiers(const std::string& classifierNM1,
                            const std::string& classifierNM2,
                            const std::string& classifierGrouping)
{
    classifierNM1_ = classifierNM1;
    classifierNM2_ = classifierNM2;
    classifierGrouping_ = classifierGrouping;
}

const uchar Recognizer::max_channel_ = 255;

std::string Recognizer::classifierNM1_ =
        "trained_classifierNM1.xml";

std::string Recognizer::classifierNM2_ =
        "trained_classifierNM2.xml";

std::string Recognizer::classifierGrouping_ =
        "trained_classifier_erGrouping.xml";
}









