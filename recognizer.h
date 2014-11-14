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
 * @file recogizer.h
 * @author Alexander Bezsilko
 * @date 4 Nov 2014
 * @brief File containing recognizer static methods and types declaration.
 */

#ifndef RECOGNIZER_H_
#define RECOGNIZER_H_

#include <opencv2/core.hpp>
#include <opencv2/text.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

#include "recexcept.h"

namespace recognizer
{

/**
 * @brief Class Recognizer provide performs for searching and
 * recognizing English text and numbers on the noisy and low
 * quality images.
 *
 * @detailed Recognizer using OpenCV library for detecting
 * image areas that contained characters and pass then to
 * tesseract ocr library for recognition to machine data
 * representation.
 */

class Recognizer {
public:

    /**
     * Internal typedefs.
     */

    typedef std::vector<std::string> Text;    
    typedef std::vector<cv::Mat> Channels;
    typedef Channels TextAreas;
    typedef cv::Ptr<cv::text::ERFilter> ERFilterPtr;
    typedef std::vector<std::vector<cv::text::ERStat>> Regions;
    typedef std::vector<std::vector<cv::Vec2i>> RegionGroups;
    typedef std::vector<cv::Rect> BoxesGroups;    

    /**
     * @brief Recognizer public interface.
     * @detailed Static method get_text is a general method for
     * image processing and text recognition.
     *
     * @param[in] file Name of image file.
     * @return Not empty string with recognized text,
     * empty string on failure.
     * @throw RecException if occured critical error.
     */
    static std::string get_text(const std::string& file) throw (RecException);

    /**
     * @brief Recognizer public interface.
     * @detailed Static method get_text is a general method for
     * image processing and text recognition.
     *
     * @param[in] file OpenCV matrix image representation.
     * @return Not empty string with recognized text,
     * empty string on failure.
     * @throw RecException if occured critical error.
     */    
    static std::string get_text(const cv::Mat& image) throw (RecException);

    /**
     * @brief Set full paths to algorithm classifiers.
     * @detailed Set paths to algorithm classifiers. By default searching
     * path is application path.
     *     
     * @param[in] classifierNM1 Classifier for stage 1 of Neumann algorithm.
     * @param[in] classifierNM2 Classifier for stage 2 of Neumann algorithm.
     * @param[in] classifierGrouping for grouping.
     */
    static void set_classifiers(const std::string& classifierNM1,
                                const std::string& classifierNM2,
                                const std::string& classifierGrouping);

private:

    /**
     * @brief Find rectangles containing characters.
     * @detailed Find rectangles containing characters.
     *
     * @param[in] image OpenCV matrix image representation.
     * @return All founded rectangles containing with characters.
     */    
    static BoxesGroups find_text_rects(const cv::Mat& image);

    /**
     * @brief Removing duplicate rectangles.
     * @detailed Removing duplicate rectangles.
     *
     * @param[out] boxes Rectangles containing characters.
     */        
    static void remove_dup(BoxesGroups& boxes);

    /**
     * @brief Create images pieces with characters.
     * @detailed Find Create images pieces with characters
     * via overlay rectangles on the source image.
     *
     * @param[in] image OpenCV matrix image representation.
     * @param[in] boxes Rectangles containing characters. 
     * @return All founded text areas.
     */        
    static TextAreas create_text_areas(const cv::Mat& image,
                                     const BoxesGroups& boxes);

    /**
     * @brief Recognize preprocessed images for characters using tesseract ocr.
     * @detailed Recognize preprocessed images for characters using
     * tesseract ocr.
     *
     * @param[in] areas Preprocessed images.
     * @return Not empty string with recognized text.
     */        
    static std::string alphabet_analisis(const TextAreas& areas);

    /**
     * @brief Parse the string for the presence of unnecessary characters.
     * @detailed Parse the string for the presence of unnecessary characters.
     *
     * @param[in] s Row string with trash.
     * @return string without unwanted characters and multiple spaces.
     */            
    static std::string string_processing(const std::string& s);

    /**
     * @brief Remove same characters pieces from string.
     * @detailed Remove same characters pieces from string.
     *
     * @param[in] text Set text pieces.
     * @return Normalize string.
     */
    static std::string normalize_result(const Text& text);

private:
    static const uchar max_channel_;
    static std::string classifierNM1_;
    static std::string classifierNM2_;
    static std::string classifierGrouping_;
};

}

#endif // RECOGNIZER_H_
