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

#ifndef SOLARDESCRIPTORMATCHERBINARYOPENCV_H
#define SOLARDESCRIPTORMATCHERBINARYOPENCV_H

#include "api/features/IDescriptorMatcher.h"

#include "xpcf/component/ConfigurableBase.h"
#include <opencv2/line_descriptor.hpp>
#include "SolAROpencvNonFreeAPI.h"

#include "datastructure/DescriptorBuffer.h"
#include "datastructure/DescriptorMatch.h"

namespace SolAR {
using namespace datastructure;
using namespace api::features;
namespace MODULES {
namespace NONFREEOPENCV {

/**
 * @class SolARDescriptorMatcherBinaryOpencv
 * @brief <B>Matches two sets of binary descriptors.</B>
 * <TT>UUID: 5b2a6059-e704-4196-aa6d-b7066243c308</TT>
 *
 */

class SOLAROPENCVNONFREE_EXPORT_API SolARDescriptorMatcherBinaryOpencv : public org::bcom::xpcf::ConfigurableBase,
	public api::features::IDescriptorMatcher
{
public:
	SolARDescriptorMatcherBinaryOpencv();
	~SolARDescriptorMatcherBinaryOpencv();

	org::bcom::xpcf::XPCFErrorCode onConfigured() override final;
	void unloadComponent() override final;

	/// @brief Matches two descriptors desc1 and desc2 respectively based on KNN search strategy.
	/// [in] desc1: source descriptor.
	/// [in] desc2: target descriptor.
	/// [out] matches: ensemble of detected matches, a pair of source/target indices.
	/// @return IDescriptorMatcher::RetCode::DESCRIPTORS_MATCHER_OK if succeed.
	IDescriptorMatcher::RetCode match(	const SRef<DescriptorBuffer> descriptors1,
										const SRef<DescriptorBuffer> descriptors2,
										std::vector<DescriptorMatch> & matches) override;

	IDescriptorMatcher::RetCode match(
		const SRef<DescriptorBuffer> descriptors1,
		const std::vector<SRef<DescriptorBuffer>> & descriptors2,
		std::vector<DescriptorMatch> & matches
	)
	{ return RetCode::DESCRIPTORS_MATCHER_OK; };

	IDescriptorMatcher::RetCode matchInRegion(	const std::vector<Point2Df> & points2D,
												const std::vector<SRef<DescriptorBuffer>> & descriptors,
												const SRef<Frame> frame,
												std::vector<DescriptorMatch> &matches,
												const float radius)
	{ return RetCode::DESCRIPTORS_MATCHER_OK; };

private:
	unsigned getBestMatchIndex(const std::vector<cv::DMatch> & candidates);

	cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> m_matcher;

	/// @brief k parameter for knn Matcher
	/// The number of closest descriptors to be returned
	int m_numClosestDescriptors = 2;

	/// @brief distance ratio used to keep good matches.
	/// Several matches can correspond to a given keyline on the first image. The first match with the best score is always retained.
	/// But here, we can also retain the second match if its distance or score is greater than the score of the best match * m_distanceRatio.
	float m_distanceRatio = 0.75f;
};

}
}
}

#endif // SOLARDESCRIPTORMATCHERBINARYOPENCV_H
