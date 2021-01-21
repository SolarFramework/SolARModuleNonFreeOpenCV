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

#include "SolARDescriptorMatcherBinaryOpencv.h"
#include "SolARNonFreeOpenCVHelper.h"
#include "SolAROpencvNonFreeAPI.h"
#include "core/Log.h"

namespace xpcf = org::bcom::xpcf;

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorMatcherBinaryOpencv)

namespace SolAR {
using namespace datastructure;
using namespace api::features;
namespace MODULES {
namespace NONFREEOPENCV {

SolARDescriptorMatcherBinaryOpencv::SolARDescriptorMatcherBinaryOpencv() : ConfigurableBase(xpcf::toUUID<SolARDescriptorMatcherBinaryOpencv>())
{
	declareInterface<IDescriptorMatcher>(this);
	declareProperty<float>("distanceRatio", m_distanceRatio);
	declareProperty<int>("numClosestDescriptors", m_numClosestDescriptors);
}

SolARDescriptorMatcherBinaryOpencv::~SolARDescriptorMatcherBinaryOpencv() { }

xpcf::XPCFErrorCode SolARDescriptorMatcherBinaryOpencv::onConfigured()
{
	m_matcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

	if (m_matcher->empty())
		return xpcf::_ERROR_NULL_POINTER;

	return xpcf::_SUCCESS;
}

IDescriptorMatcher::RetCode SolARDescriptorMatcherBinaryOpencv::match(const SRef<DescriptorBuffer> descriptors1,
																	  const SRef<DescriptorBuffer> descriptors2,
																	  std::vector<DescriptorMatch>& matches)
{
	matches.clear();

	// Check if the descriptors can be matched
	if (descriptors1->getDescriptorType() != descriptors2->getDescriptorType())
	{
		LOG_WARNING(" Descriptors type don't match!");
		return IDescriptorMatcher::RetCode::DESCRIPTORS_DONT_MATCH;
	}
	if (descriptors1->getNbDescriptors() == 0 || descriptors2->getNbDescriptors() == 0)
	{
		LOG_WARNING(" Descriptors are empty.");
		return IDescriptorMatcher::RetCode::DESCRIPTOR_EMPTY;
	}
	if (descriptors1->getNbDescriptors() < 2 || descriptors2->getNbDescriptors() < 2)
	{
		LOG_WARNING(" Not enough descriptors to perform matching.");
		return IDescriptorMatcher::RetCode::DESCRIPTORS_MATCHER_OK;
	}

	// Mapping to OpenCV
	uint32_t type_conversion = SolARNonFreeOpenCVHelper::deduceOpenDescriptorCVType(descriptors1->getDescriptorDataType());

	cv::Mat cvDescriptors1(descriptors1->getNbDescriptors(), descriptors1->getNbElements(), type_conversion);
	cvDescriptors1.data = (uchar*)descriptors1->data();

	cv::Mat cvDescriptors2(descriptors2->getNbDescriptors(), descriptors2->getNbElements(), type_conversion);
	cvDescriptors2.data = (uchar*)descriptors2->data();

	// Matching
	std::vector<std::vector<cv::DMatch>> cvMatches;
	m_matcher->knnMatch(cvDescriptors1, cvDescriptors2, cvMatches, m_numClosestDescriptors, cv::Mat(), true);

	// Keep only best matches
	for (const auto& match : cvMatches)
	{
		cv::DMatch best;
		if (getBestMatch(match, best))
			matches.push_back(DescriptorMatch(best.queryIdx, best.trainIdx, best.distance));
	}

	return IDescriptorMatcher::RetCode::DESCRIPTORS_MATCHER_OK;
}

bool SolARDescriptorMatcherBinaryOpencv::getBestMatch(const std::vector<cv::DMatch> & candidates, cv::DMatch & bestMatch)
{
	if (candidates.empty())
		return false;

	bestMatch = *std::min_element(candidates.begin(), candidates.end());
	return true;
}

}
}
}