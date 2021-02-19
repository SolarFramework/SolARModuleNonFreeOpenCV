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

#include "SolARKeypointDetectorNonFreeOpencv.h"
#include "SolARNonFreeOpenCVHelper.h"
#include "core/Log.h"

#include "xpcf/api/IComponentManager.h"

#include <string>

#include <iostream>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv)

namespace xpcf = org::bcom::xpcf;

using namespace cv;
using namespace cv::xfeatures2d;

namespace SolAR {
using namespace datastructure;
using namespace api::features;
namespace MODULES {
namespace NONFREEOPENCV {




static std::map<std::string,IKeypointDetector::KeypointDetectorType> stringToType = {{"SURF",IKeypointDetector::KeypointDetectorType::SURF}};

static std::map<IKeypointDetector::KeypointDetectorType,std::string> typeToString = {{IKeypointDetector::KeypointDetectorType::SURF, "SURF"}};


SolARKeypointDetectorNonFreeOpencv::SolARKeypointDetectorNonFreeOpencv():ConfigurableBase(xpcf::toUUID<SolARKeypointDetectorNonFreeOpencv>())
{
    addInterface<api::features::IKeypointDetector>(this);
    declareProperty("imageRatio", m_imageRatio);
    declareProperty("nbDescriptors", m_nbDescriptors);
    declareProperty("type", m_type);
    LOG_DEBUG("SolARKeypointDetectorOpencv constructor");
}



SolARKeypointDetectorNonFreeOpencv::~SolARKeypointDetectorNonFreeOpencv()
{
    LOG_DEBUG("SolARKeypointDetectorNonFreeOpencv destructor");
}

xpcf::XPCFErrorCode SolARKeypointDetectorNonFreeOpencv::onConfigured()
{
    LOG_DEBUG(" SolARKeypointDetectorOpencv onConfigured");
    setType(stringToType.at(m_type));
    return xpcf::XPCFErrorCode::_SUCCESS;
}



void SolARKeypointDetectorNonFreeOpencv::setType(KeypointDetectorType type)
{

    /*
     * 	SURF,
     */
    m_type=typeToString.at(type);
    switch (type) {
    case (KeypointDetectorType::SURF):
        LOG_DEBUG("KeypointDetectorImp::setType(SURF)");
        m_detector = SURF::create();
        break;
    default :
        LOG_DEBUG("KeypointDetectorImp::setType(SURF DEFAULT)");
        m_detector=SURF::create();
        break;
    }
}

IKeypointDetector::KeypointDetectorType SolARKeypointDetectorNonFreeOpencv::getType()
{
    return stringToType.at(m_type);
}

void SolARKeypointDetectorNonFreeOpencv::detect(const SRef<Image> image, std::vector<Keypoint> & keypoints)
{
    std::vector<cv::KeyPoint> kpts;

    // the input image is down-scaled to accelerate the keypoints extraction

    float ratioInv=1.f/m_imageRatio;

    keypoints.clear();

    // instantiation of an opencv image from an input IImage
    cv::Mat opencvImage = SolARNonFreeOpenCVHelper::mapToOpenCV(image);

    cv::Mat img_1;
    cvtColor( opencvImage, img_1, cv::COLOR_BGR2GRAY );
    cv::resize(img_1, img_1, Size(img_1.cols*m_imageRatio,img_1.rows*m_imageRatio), 0, 0);



    try
    {
        if(!m_detector){
            LOG_DEBUG(" detector is initialized with default value : {}", this->m_type)
            setType(stringToType.at(this->m_type));
        }
        m_detector->detect(img_1, kpts, Mat());
    }
    catch (Exception& e)
    {
        LOG_ERROR("Feature : {}", m_detector->getDefaultName())
        LOG_ERROR("{}",e.msg)
        return;
    }


    kptsFilter.retainBest(kpts,m_nbDescriptors);

    int kpID=0;
    for(const auto& itr : kpts){
        Keypoint kpa = Keypoint();
        float px = itr.pt.x*ratioInv;
        float py = itr.pt.y*ratioInv;
        cv::Vec3b bgr{ 0, 0, 0 };
        if (opencvImage.channels() == 3)
            bgr = opencvImage.at<cv::Vec3b>((int)py, (int)px);
        kpa.init(kpID++, px, py, bgr[2], bgr[1], bgr[0], itr.size, itr.angle, itr.response, itr.octave, itr.class_id) ;
        keypoints.push_back(kpa);
    }

}
}
}
}  // end of namespace SolAR
