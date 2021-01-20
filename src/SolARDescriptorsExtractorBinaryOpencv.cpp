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

#include "SolARDescriptorsExtractorBinaryOpencv.h"
#include "SolARNonFreeOpenCVHelper.h"
#include "core/Log.h"

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorBinaryOpencv)

namespace xpcf = org::bcom::xpcf;

namespace SolAR {
using namespace datastructure;
using namespace api::features;
namespace MODULES {
namespace NONFREEOPENCV {

SolARDescriptorsExtractorBinaryOpencv::SolARDescriptorsExtractorBinaryOpencv() : ConfigurableBase(xpcf::toUUID<SolARDescriptorsExtractorBinaryOpencv>())
{
    declareInterface<api::features::IDescriptorsExtractorBinary>(this);
	declareProperty("imageRatio", m_imageRatio);
	declareProperty("reductionRatio", m_reductionRatio);
	declareProperty("widthOfBand", m_widthOfBand);
}

SolARDescriptorsExtractorBinaryOpencv::~SolARDescriptorsExtractorBinaryOpencv() { }

org::bcom::xpcf::XPCFErrorCode SolARDescriptorsExtractorBinaryOpencv::onConfigured()
{
	m_extractor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

	if (m_extractor->empty())
		return xpcf::_ERROR_NULL_POINTER;
	m_extractor->setReductionRatio(m_reductionRatio);
	m_extractor->setWidthOfBand(m_widthOfBand);
	return xpcf::_SUCCESS;
}

FrameworkReturnCode SolARDescriptorsExtractorBinaryOpencv::setDetector(const SRef<IKeylineDetector> detector)
{
	m_detector = detector;
	return FrameworkReturnCode::_SUCCESS;
}

FrameworkReturnCode SolARDescriptorsExtractorBinaryOpencv::getDetector(SRef<IKeylineDetector> & detector) const
{
	detector = m_detector;
	return FrameworkReturnCode::_SUCCESS;
}

void SolARDescriptorsExtractorBinaryOpencv::extract(const SRef<Image> image, const std::vector<Keyline>& keylines, SRef<DescriptorBuffer>& descriptors)
{
	cv::Mat opencvImage = SolARNonFreeOpenCVHelper::mapToOpenCV(image);
	cv::resize(opencvImage, opencvImage, cv::Size(), m_imageRatio, m_imageRatio);

	// Convert SolAR keylines to OpenCV KeyLines
	std::vector<cv::line_descriptor::KeyLine> cvKeylines(keylines.size());
	std::transform(
		keylines.begin(), keylines.end(), cvKeylines.begin(),
		[&](const auto& kl)
		{
			cv::line_descriptor::KeyLine cvKl;
			cvKl.pt = cv::Point2f(kl.getX() * m_imageRatio, kl.getY() * m_imageRatio);
			cvKl.startPointX = kl.getStartPointX() * m_imageRatio;
			cvKl.startPointY = kl.getStartPointY() * m_imageRatio;
			cvKl.sPointInOctaveX = kl.getSPointInOctaveX() * m_imageRatio;
			cvKl.sPointInOctaveY = kl.getSPointInOctaveY() * m_imageRatio;
			cvKl.endPointX = kl.getEndPointX() * m_imageRatio;
			cvKl.endPointY = kl.getEndPointY() * m_imageRatio;
			cvKl.ePointInOctaveX = kl.getEPointInOctaveX() * m_imageRatio;
			cvKl.ePointInOctaveY = kl.getEPointInOctaveY() * m_imageRatio;
			cvKl.lineLength = kl.getLineLength() * m_imageRatio;
			cvKl.numOfPixels = kl.getNumOfPixels() * m_imageRatio;
			cvKl.angle = kl.getAngle();
			cvKl.size = kl.getSize() * m_imageRatio;
			cvKl.response = kl.getResponse();
			cvKl.octave = kl.getOctave();
			cvKl.class_id = kl.getClassId();	
			return cvKl;
		}
	);
	// Descriptors extraction
	cv::Mat cvDescriptors;
	m_extractor->compute(opencvImage, cvKeylines, cvDescriptors);
	// Fill out extracted descriptors
    descriptors.reset(new DescriptorBuffer(cvDescriptors.data, DescriptorType::BINARY, DescriptorDataType::TYPE_8U, 32, cvDescriptors.rows) );
}

void SolARDescriptorsExtractorBinaryOpencv::compute(const SRef<Image> image, std::vector<Keyline>& keylines, SRef<DescriptorBuffer>& descriptors)
{
	// Keylines detection
	m_detector->detect(image, keylines);
	// Descriptors extraction
	this->extract(image, keylines, descriptors);
}


}
}
}
