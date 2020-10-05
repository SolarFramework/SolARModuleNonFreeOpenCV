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

#include "SolARDescriptorsExtractorSURF128Opencv.h"
#include "SolARNonFreeOpenCVHelper.h"
#include <core/Log.h>

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF128Opencv);

namespace xpcf = org::bcom::xpcf;

using namespace cv;
using namespace cv::xfeatures2d;

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

SolARDescriptorsExtractorSURF128Opencv::SolARDescriptorsExtractorSURF128Opencv():ConfigurableBase(xpcf::toUUID<SolARDescriptorsExtractorSURF128Opencv>())
{
    declareInterface<api::features::IDescriptorsExtractor>(this);
    LOG_DEBUG(" SolARDescriptorsExtractorSURF128Opencv constructor")
    declareProperty("hessianThreshold", m_hessianThreshold);
    declareProperty("nbOctaves", m_nbOctaves);
    declareProperty("nbOctaveLayers", m_nbOctaveLayers);
    declareProperty("extended", m_extended);
    declareProperty("upright", m_upright);
}

xpcf::XPCFErrorCode SolARDescriptorsExtractorSURF128Opencv::onConfigured()
{
    // m_extractor must have a default implementation : initialize default extractor type
    m_extractor=SURF::create(m_hessianThreshold,m_nbOctaves,m_nbOctaveLayers,(bool)m_extended, (bool)m_upright);
    return xpcf::_SUCCESS;
}

SolARDescriptorsExtractorSURF128Opencv::~SolARDescriptorsExtractorSURF128Opencv()
{
    LOG_DEBUG(" SolARDescriptorExtractorSURF128Opencv destructor")
}

void SolARDescriptorsExtractorSURF128Opencv::extract(const SRef<Image> image, const std::vector<Keypoint> & keypoints, SRef<DescriptorBuffer>& descriptors){

    //transform all SolAR data to openCv data
/*    SRef<Image> convertedImage = image;

    if (image->getImageLayout() != Image::ImageLayout::LAYOUT_GREY) {
        // input Image not in grey levels : convert it !
        convertedImage = xpcf::utils::make_shared<Image>(Image::ImageLayout::LAYOUT_GREY,Image::PixelOrder::INTERLEAVED,image->getDataType());

        convertedImage->setSize(image->getWidth(),image->getHeight());

        cv::Mat imgSource, imgConverted;
        SolARNonFreeOpenCVHelper::mapToOpenCV(image,imgSource);

        SolARNonFreeOpenCVHelper::mapToOpenCV(convertedImage,imgConverted);

        if (image->getImageLayout() == Image::ImageLayout::LAYOUT_RGB)
            cv::cvtColor(imgSource, imgConverted, cv::COLOR_RGB2GRAY);
        else
            cv::cvtColor(imgSource, imgConverted, cv::COLOR_BGR2GRAY);
    }

    cv::Mat opencvImage;
    SolARNonFreeOpenCVHelper::mapToOpenCV(convertedImage,opencvImage);
*/
    // Convert image in greyscale is not already done
    cv::Mat opencvImage;
    if (image->getImageLayout() != Image::ImageLayout::LAYOUT_GREY)
    {
        cv::Mat opencvColorImage;
        SolARNonFreeOpenCVHelper::mapToOpenCV(image,opencvColorImage);
        if (image->getImageLayout() == Image::ImageLayout::LAYOUT_RGB)
            cv::cvtColor(opencvColorImage, opencvImage, cv::COLOR_RGB2GRAY);
        else
            cv::cvtColor(opencvColorImage, opencvImage, cv::COLOR_BGR2GRAY);

    }
    else
    {
        SolARNonFreeOpenCVHelper::mapToOpenCV(image,opencvImage);
    }


    cv::Mat out_mat_descps;

    std::vector<cv::KeyPoint> transform_to_data;

    for(unsigned int k =0; k < keypoints.size(); ++k)
    {
        transform_to_data.push_back(
                //instantiate keypoint
                 cv::KeyPoint(keypoints[k].getX(),
                              keypoints[k].getY(),
                              keypoints[k].getSize(),
                              keypoints[k].getAngle(),
                              keypoints[k].getResponse(),
                              keypoints[k].getOctave(),
                              keypoints[k].getClassId())
        );
    }

   m_extractor->compute(opencvImage, transform_to_data, out_mat_descps);

  // m_ex
  // enum DESCRIPTOR::TYPE desc_type = descriptors->getDescriptorType();

    descriptors.reset( new DescriptorBuffer(out_mat_descps.data,DescriptorType::SURF_128, DescriptorDataType::TYPE_32F, 128, out_mat_descps.rows)) ;

}

}
}
}  // end of namespace SolAR
