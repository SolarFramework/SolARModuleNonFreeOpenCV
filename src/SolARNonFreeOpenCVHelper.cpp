/**
 * @copyright Copyright (c) 2017 B-com http://www.b-com.com/
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SolARNonFreeOpenCVHelper.h"
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "datastructure/DescriptorBuffer.h"

using namespace org::bcom::xpcf;

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

static std::map<DescriptorDataType,uint32_t> solarDescriptor2cvType =
{{ DescriptorDataType::TYPE_8U,CV_8U},{DescriptorDataType::TYPE_32F,CV_32F}};



static std::map<std::tuple<uint32_t,std::size_t,uint32_t>,int> solar2cvTypeConvertMap = {{std::make_tuple(8,1,3),CV_8UC3},{std::make_tuple(8,1,1),CV_8UC1}};

static std::map<int,std::pair<Image::ImageLayout,Image::DataType>> cv2solarTypeConvertMap = {{CV_8UC3,{Image::ImageLayout::LAYOUT_BGR,Image::DataType::TYPE_8U}},
                                                                                                      {CV_8UC1,{Image::ImageLayout::LAYOUT_GREY,Image::DataType::TYPE_8U}}};

uint32_t SolARNonFreeOpenCVHelper::deduceOpenDescriptorCVType(DescriptorDataType querytype){
    return solarDescriptor2cvType.at(querytype);
}


int SolARNonFreeOpenCVHelper::deduceOpenCVType(SRef<Image> img)
{
    // TODO : handle safe mode if missing map entry
    // is it ok when destLayout != img->ImageLayout ?
    return solar2cvTypeConvertMap.at(std::forward_as_tuple(img->getNbBitsPerComponent(),1,img->getNbChannels()));
}

void SolARNonFreeOpenCVHelper::mapToOpenCV (SRef<Image> imgSrc, cv::Mat& imgDest)
{
    cv::Mat imgCV(imgSrc->getHeight(),imgSrc->getWidth(),deduceOpenCVType(imgSrc), imgSrc->data()); 
    imgDest = imgCV;
}


cv::Mat SolARNonFreeOpenCVHelper::mapToOpenCV (SRef<Image> imgSrc)
{
    cv::Mat imgCV(imgSrc->getHeight(),imgSrc->getWidth(),deduceOpenCVType(imgSrc), imgSrc->data());
    return imgCV;
}

FrameworkReturnCode SolARNonFreeOpenCVHelper::convertToSolar (cv::Mat&  imgSrc, SRef<Image>& imgDest)
{
    if (cv2solarTypeConvertMap.find(imgSrc.type()) == cv2solarTypeConvertMap.end() || imgSrc.empty()) {
        return FrameworkReturnCode::_ERROR_LOAD_IMAGE;
    }
    std::pair<Image::ImageLayout,Image::DataType> type = cv2solarTypeConvertMap.at(imgSrc.type());
    imgDest = utils::make_shared<Image>(imgSrc.ptr(), imgSrc.cols, imgSrc.rows, type.first, Image::PixelOrder::INTERLEAVED, type.second);

    return FrameworkReturnCode::_SUCCESS;
}

std::vector<cv::Point2i> SolARNonFreeOpenCVHelper::convertToOpenCV (const Contour2Di &contour)
{
    std::vector<cv::Point2i> output;
    for (int i = 0; i < contour.size(); i++)
    {
        output.push_back(cv::Point2i(contour[i]->getX(), contour[i]->getY()));
    }
    return output;
}

std::vector<cv::Point2f> SolARNonFreeOpenCVHelper::convertToOpenCV (const Contour2Df &contour)
{
    std::vector<cv::Point2f> output;
    for (int i = 0; i < contour.size(); i++)
    {
        output.push_back(cv::Point2f(contour[i].getX(), contour[i].getY()));
    }
    return output;
}

}
}
}

