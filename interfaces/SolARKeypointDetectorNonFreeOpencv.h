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

#include "xpcf/component/ConfigurableBase.h"
#include "SolAROpencvNonFreeAPI.h"
#include <string>
#include "opencv2/opencv.hpp"


namespace SolAR {
namespace MODULES {
namespace NONFREEOPENCV {

/** @class SolARKeypointDetectorNonFreeOpencv
  * @brief <B>Detects keypoints in an image (based on SIFT or SURF algorithm).</B>
  * <TT>UUID: d1f9317c-9519-4671-8ff5-4629773544f2</TT>
  *
  */

class SOLAROPENCVNONFREE_EXPORT_API SolARKeypointDetectorNonFreeOpencv : public org::bcom::xpcf::ConfigurableBase,
    public api::features::IKeypointDetector {
public:
    SolARKeypointDetectorNonFreeOpencv();
    ~SolARKeypointDetectorNonFreeOpencv();
    void unloadComponent () override final;

    org::bcom::xpcf::XPCFErrorCode onConfigured() override final;

    /// @brief Set the type of method used to detect keypoints in the image
    /// @param[in] type The type of method used to detect keypoints.
    void setType(KeypointDetectorType type) override;

    /// @brief Get the type of method used to detect keypoints in the image
    /// @return The type of method used to detect keypoints.
    KeypointDetectorType  getType() override;
 
    /// @brief This method detects keypoints in an input Image
    /// @param[in] image input image on which we are extracting keypoints.
    /// @param[out] keypoints The keypoints detected from the image passed as first argument.
    void detect (const SRef<datastructure::Image> image, std::vector<datastructure::Keypoint> & keypoints) override;

private:
	int m_id;
    std::string m_type = "SURF";
    cv::Ptr<cv::Feature2D> m_detector;
    cv::KeyPointsFilter kptsFilter;

    //TODO: user parameters to expose
    int m_nbDescriptors = 1000; //select the first 1000 best features
    float m_imageRatio=1.0f;//resize image to speedup computation.
};

extern int deduceOpenCVType(SRef<datastructure::Image> img);

}
}
}  // end of namespace SolAR



#endif // SOLARKEYPOINTDETECTORNONFREEOPENCV_H
