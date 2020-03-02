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
#include "SolAROpenCVHelper.h"
#include "core/Log.h"

#include "xpcf/api/IComponentManager.h"


#include "SolAROpencvAPI.h"
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
using namespace SolAR::MODULES::OPENCV;

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {




static std::map<std::string,IKeypointDetector::KeypointDetectorType> stringToType = {{"SURF",IKeypointDetector::KeypointDetectorType::SURF},
                                                                  {"SIFT",IKeypointDetector::KeypointDetectorType::SIFT}};

static std::map<IKeypointDetector::KeypointDetectorType,std::string> typeToString = {{IKeypointDetector::KeypointDetectorType::SURF, "SURF"},
                                                                  {IKeypointDetector::KeypointDetectorType::SIFT, "SIFT"}};


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
    return xpcf::_SUCCESS;
}



void SolARKeypointDetectorNonFreeOpencv::setType(KeypointDetectorType type)
{

    /*
     * 	SURF,
        SIFT
        */
    m_type=typeToString.at(type);
    switch (type) {
    case (KeypointDetectorType::SURF):
        LOG_DEBUG("KeypointDetectorImp::setType(SURF)");
        m_detector = SURF::create();
        break;
    case (KeypointDetectorType::SIFT):
        LOG_DEBUG("KeypointDetectorImp::setType(SIFT)");
        m_detector = SIFT::create();
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
    cv::Mat opencvImage = SolAROpenCVHelper::mapToOpenCV(image);

    cv::Mat img_1;
    cvtColor( opencvImage, img_1, CV_BGR2GRAY );
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

    for(std::vector<cv::KeyPoint>::iterator itr=kpts.begin();itr!=kpts.end();++itr){
        Keypoint kpa = Keypoint();

        kpa.init((*itr).pt.x*ratioInv,(*itr).pt.y*ratioInv,(*itr).size,(*itr).angle,(*itr).response,(*itr).octave,(*itr).class_id) ;
        keypoints.push_back(kpa);
    }

}
}
}
}  // end of namespace SolAR
