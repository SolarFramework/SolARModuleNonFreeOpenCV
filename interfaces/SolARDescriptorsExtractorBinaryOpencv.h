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
	cv::Ptr<cv::line_descriptor::LSDDetector> m_detector;
	cv::Ptr<cv::line_descriptor::BinaryDescriptor> m_extractor;

	std::string m_type = "BINARY";
	float m_imageRatio = 1.0;
	int m_scale = 2;
	int m_numOctave = 1;
	int m_widthOfBand = 7;
	int m_minLineLength = 0;
};
}
}
}


#endif // SOLARDESCRIPTORSEXTRACTORBINARYOPENCV_H
