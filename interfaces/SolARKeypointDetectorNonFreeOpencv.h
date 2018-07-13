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

#ifndef SOLARKEYPOINTDETECTORNONFREEOPENCV_H
#define SOLARKEYPOINTDETECTORNONFREEOPENCV_H

#include "api/features/IKeypointDetector.h"
// Definition of SolARKeypointDetectorOpencv Class //
// part of SolAR namespace //

#include "xpcf/component/ComponentBase.h"
#include "SolAROpencvNonFreeAPI.h"
#include <string>
#include "opencv2/opencv.hpp"


namespace SolAR {
using namespace datastructure;
using namespace api::features;
namespace MODULES {
namespace NONFREEOPENCV {

class SOLAROPENCVNONFREE_EXPORT_API SolARKeypointDetectorNonFreeOpencv : public org::bcom::xpcf::ComponentBase,
        public IKeypointDetector {
public:
    SolARKeypointDetectorNonFreeOpencv();
    ~SolARKeypointDetectorNonFreeOpencv();
    void unloadComponent () override final;
    void setType(KeypointDetectorType type);
    KeypointDetectorType  getType();
 
    void detect (const SRef<Image> &image, std::vector<SRef<Keypoint>> &keypoints);

private:
	int m_id;
    KeypointDetectorType m_type;
    cv::Ptr<cv::Feature2D> m_detector;
    cv::KeyPointsFilter kptsFilter;

    //TODO: user parameters to expose
    unsigned int m_select_best_N_features = 1000; //select the first 1000 best features
    float m_ratio=0.5f;//resize image to speedup computation.
};

extern int deduceOpenCVType(SRef<Image> img);

}
}
}  // end of namespace SolAR



#endif // SOLARKEYPOINTDETECTORNONFREEOPENCV_H
