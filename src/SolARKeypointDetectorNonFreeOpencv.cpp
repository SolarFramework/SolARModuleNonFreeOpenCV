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
#include <iostream>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "IComponentManager.h"

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv);

namespace xpcf = org::bcom::xpcf;

using namespace cv;
using namespace cv::xfeatures2d;
using namespace SolAR::MODULES::OPENCV;

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

SolARKeypointDetectorNonFreeOpencv::SolARKeypointDetectorNonFreeOpencv()
{
    setUUID(SolARKeypointDetectorNonFreeOpencv::UUID);
    addInterface<api::features::IKeypointDetector>(this,api::features::IKeypointDetector::UUID, "interface api::features::IKeypointDetector");

    LOG_DEBUG("SolARKeypointDetectorNonFreeOpencv constructor");
    m_type=KeypointDetectorType::AKAZE;
}


SolARKeypointDetectorNonFreeOpencv::~SolARKeypointDetectorNonFreeOpencv()
{
    LOG_DEBUG("SolARKeypointDetectorNonFreeOpencv destructor");
}


void SolARKeypointDetectorNonFreeOpencv::setType(KeypointDetectorType type)
{

    /*
     * 	SURF,
        ORB,
        SIFT,
        DAISY,
        LATCH,
        AKAZE,
        AKAZEUP,
        BRISK,
        BRIEF,
        */
    m_type=type;

    switch (m_type) {

    case (KeypointDetectorType::SIFT):

        LOG_DEBUG("KeypointDetectorImp::setType(SIFT)");

        m_detector=SIFT::create();

        break;

    case (KeypointDetectorType::SURF):

        LOG_DEBUG("KeypointDetectorImp::setType(SURF)");

        m_detector=SURF::create();

        break;

    case (KeypointDetectorType::DAISY):

        LOG_DEBUG("KeypointDetectorImp::setType(DAISY)");

        m_detector=DAISY::create();

        break;

    case (KeypointDetectorType::AKAZE):

        LOG_DEBUG("KeypointDetectorImp::setType(AKAZE)");

        m_detector=AKAZE::create();

        break;

    case (KeypointDetectorType::ORB):

        LOG_DEBUG("KeypointDetectorImp::setType(ORB)");

        m_detector=ORB::create();

        break;

    case (KeypointDetectorType::BRISK):

        LOG_DEBUG("KeypointDetectorImp::setType(BRISK)");

        m_detector=BRISK::create();

        break;



    default :

        LOG_DEBUG("KeypointDetectorImp::setType(AKAZE)");
        m_detector=AKAZE::create();
        break;
    }
}

KeypointDetectorType SolARKeypointDetectorNonFreeOpencv::getType()
{
    return m_type;
}

void SolARKeypointDetectorNonFreeOpencv::detect(const SRef<Image> &image, std::vector<SRef<Keypoint>> &keypoints)
{
    std::vector<cv::KeyPoint> kpts;

    // the input image is down-scaled to accelerate the keypoints extraction

    float ratioInv=1.f/m_ratio;

    keypoints.clear();

    // instantiation of an opencv image from an input IImage
    cv::Mat opencvImage = SolAROpenCVHelper::mapToOpenCV(image);

    cv::Mat img_1;
    cvtColor( opencvImage, img_1, CV_BGR2GRAY );
    cv::resize(img_1, img_1, Size(img_1.cols*m_ratio,img_1.rows*m_ratio), 0, 0);

    try
    {
        if(!m_detector){
            LOG_DEBUG(" detector is initialized with default value : {}", (int)this->m_type)
            setType(this->m_type);
        }
        m_detector->detect(img_1, kpts, Mat());
    }
    catch (Exception& e)
    {
        LOG_ERROR("Feature : {}", m_detector->getDefaultName())
        LOG_ERROR("{}",e.msg)
        return;
    }


    kptsFilter.retainBest(kpts,m_select_best_N_features);

    for(std::vector<cv::KeyPoint>::iterator itr=kpts.begin();itr!=kpts.end();++itr){
        sptrnms::shared_ptr<Keypoint> kpa = sptrnms::make_shared<Keypoint>();

        kpa->init((*itr).pt.x*ratioInv,(*itr).pt.y*ratioInv,(*itr).size,(*itr).angle,(*itr).response,(*itr).octave,(*itr).class_id) ;
        keypoints.push_back(kpa);
    }
}

}
}
}  // end of namespace SolAR