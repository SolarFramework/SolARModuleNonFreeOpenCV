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

#ifndef SOLARDESCRIPTORSEXTRACTORSURF64OPENCV_H
#define SOLARDESCRIPTORSEXTRACTORSURF64OPENCV_H

#include "api/features/IDescriptorsExtractor.h"
// Definition of SolARDescriptorsExtractorSURF64Opencv Class //
// part of SolAR namespace //

#include "xpcf/component/ConfigurableBase.h"
#include "SolAROpencvNonFreeAPI.h"
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

/**
 * @class SolARDescriptorsExtractorSURF64Opencv
 * @brief <B>Extracts the SURF descriptors (size 64) for a set of keypoints.</B>
 * <TT>UUID: 1a437804-d0a3-11e7-8fab-cec278b6b50a</TT>
 *
 */

class SOLAROPENCVNONFREE_EXPORT_API SolARDescriptorsExtractorSURF64Opencv : public org::bcom::xpcf::ConfigurableBase,
        public api::features::IDescriptorsExtractor {
public:
    SolARDescriptorsExtractorSURF64Opencv();
    ~SolARDescriptorsExtractorSURF64Opencv();
    org::bcom::xpcf::XPCFErrorCode onConfigured() override final;
    void unloadComponent () override final;
    inline std::string getTypeString() override { return std::string("DescriptorExtractorType::SURF64") ;};

    /// @brief Extracts a set of descriptors (size 64) from a given image around a set of keypoints based on SURF algorithm
    /// [in] image: source image.
    /// [in] keypoints: set of keypoints.
    /// [out] decsriptors: set of computed descriptors.
    void extract(const SRef<Image> image, const std::vector<Keypoint> & keypoints, SRef<DescriptorBuffer>& descriptors) override;

private:
    cv::Ptr<cv::Feature2D> m_extractor;
    double m_hessianThreshold = 100.0f; // Threshold for hessian keypoint detector used in SURF.
    int m_nbOctaves = 4;                // Number of pyramid octaves the keypoint detector will use.
    int m_nbOctaveLayers = 3;           // Number of octave layers within each octave.
    int m_extended = 0;              // Extended descriptor flag (1 - use extended 128-element descriptors; 0 - use 64-element descriptors).
    int m_upright = 0;               // Up-right or rotated features flag (1 - do not compute orientation of features; 0 - compute orientation).
};

}
}
}  // end of namespace SolAR



#endif // SOLARDESCRIPTORSEXTRACTORSURF64OPENCV_H
