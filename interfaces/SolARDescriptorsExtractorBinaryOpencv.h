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

#ifndef SOLARDESCRIPTORSEXTRACTORBINARYOPENCV_H
#define SOLARDESCRIPTORSEXTRACTORBINARYOPENCV_H

#include "api/features/IDescriptorsExtractorBinary.h"

#include "xpcf/component/ConfigurableBase.h"

#include "SolAROpencvNonFreeAPI.h"
#include "opencv2/opencv.hpp"
#include "opencv2/line_descriptor.hpp"

#include "datastructure/DescriptorBuffer.h"
#include "datastructure/Keyline.h"

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

/** 
 * @class SolARDescriptorsExtractorBinaryOpencv
 * @brief <B>Extracts the Binary descriptors for a set of keylines.</B>
 * <TT>UUID: ccf87cfe-446e-4bc2-8e79-c2eee71dbf4d</TT>
 *
 */

class SOLAROPENCVNONFREE_EXPORT_API SolARDescriptorsExtractorBinaryOpencv : public org::bcom::xpcf::ConfigurableBase,
    public api::features::IDescriptorsExtractorBinary
{
public:
	SolARDescriptorsExtractorBinaryOpencv();
	~SolARDescriptorsExtractorBinaryOpencv();

	org::bcom::xpcf::XPCFErrorCode onConfigured() override final;
	void unloadComponent() override final;

    /// @brief Set keyline detector component.
    /// @param[in] a keyline detector instance
    /// @return FrameworkReturnCode::_SUCCESS if successful, else FrameworkReturnCode::_ERROR
    FrameworkReturnCode setDetector(const SRef<api::features::IKeylineDetector> detector) override;

    /// @brief Get keyline detector component.
    /// @param[in] a keyline detector instance
    /// @return FrameworkReturnCode::_SUCCESS if successful, else FrameworkReturnCode::_ERROR
    FrameworkReturnCode getDetector(SRef<api::features::IKeylineDetector> & detector) const override;
	
	/// @brief Extracts the descriptors for a set of keylines
	/// @param[in] image The image on which the keylines have been detected
	/// @param[int] keylines The set of keylines on which the descriptors are extracted
	/// @param[out] descriptors The extracted descriptors. The nth descriptor corresponds to the nth keyline of the second argument.
	void extract(const SRef<Image> image,
				 const std::vector<Keyline> & keylines,
				 SRef<DescriptorBuffer> & descriptors) override;

	/// @brief Detects keylines and extracts the corresponding descriptors
	/// @param[in] image The image on which the keylines have been detected
	/// @param[out] keylines The set of detected keylines on which the descriptors are extracted
	/// @param[out] descriptors The extracted descriptors. The nth descriptor corresponds to the nth keyline of the second argument.
	void compute(const SRef<Image> image,
				 std::vector<Keyline> & keylines,
				 SRef<DescriptorBuffer> & descriptors) override;

private:
	SRef<api::features::IKeylineDetector> m_detector;

	cv::Ptr<cv::line_descriptor::BinaryDescriptor> m_extractor;

	float m_imageRatio{ 1.f };
	int m_reductionRatio{ 2 };
	int m_widthOfBand{ 7 };
};
}
}
}


#endif // SOLARDESCRIPTORSEXTRACTORBINARYOPENCV_H
