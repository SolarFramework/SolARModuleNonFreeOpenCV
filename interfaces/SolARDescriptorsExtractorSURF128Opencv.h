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

#ifndef SOLARDESCRIPTORSEXTRACTORSURF128OPENCV_H
#define SOLARDESCRIPTORSEXTRACTORSURF128OPENCV_H

#include "api/features/IDescriptorsExtractor.h"
// Definition of SolARDescriptorExtractorOpencv Class //
// part of SolAR namespace //

#include "xpcf/component/ComponentBase.h"
#include "SolAROpencvNonFreeAPI.h"
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

/**
 * @class SolARDescriptorsExtractorSURF128Opencv
 * @brief <B>Extracts the SURF descriptors (size 128) for a set of keypoints.</B>
 * <TT>UUID: fe14a310-d0a2-11e7-8fab-cec278b6b50a</TT>
 *
 */

class SOLAROPENCVNONFREE_EXPORT_API SolARDescriptorsExtractorSURF128Opencv : public org::bcom::xpcf::ComponentBase,
        public api::features::IDescriptorsExtractor {
public:
    SolARDescriptorsExtractorSURF128Opencv();
    ~SolARDescriptorsExtractorSURF128Opencv();
    void unloadComponent () override final;
    inline std::string getTypeString() override { return std::string("DescriptorExtractorType::SURF128") ;};

    /// @brief Extracts a set of descriptors (size 128) from a given image around a set of keypoints based on SURF algorithm
    /// [in] image: source image.
    /// [in] keypoints: set of keypoints.
    /// [out] decsriptors: set of computed descriptors.
    void extract (const SRef<Image> image, const std::vector<SRef<Keypoint>> &keypoints, SRef<DescriptorBuffer>& descriptors) override;

private:
    cv::Ptr<cv::Feature2D> m_extractor;
};

}
}
}  // end of namespace SolAR



#endif // SOLARDESCRIPTORSEXTRACTORSURF128OPENCV_H
