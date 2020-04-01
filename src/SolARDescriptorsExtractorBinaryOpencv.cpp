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
#include "SolAROpenCVHelper.h"
#include "core/Log.h"

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorBinaryOpencv)

namespace xpcf = org::bcom::xpcf;
using namespace SolAR::MODULES::OPENCV;

namespace SolAR {
using namespace datastructure;
using namespace api::features;
namespace MODULES {
namespace NONFREEOPENCV {

SolARDescriptorsExtractorBinaryOpencv::SolARDescriptorsExtractorBinaryOpencv() : ConfigurableBase(xpcf::toUUID<SolARDescriptorsExtractorBinaryOpencv>())
{
    declareInterface<api::features::IDescriptorsExtractorBinary>(this);
	declareProperty("scale", m_scale);
	declareProperty("numOctave", m_numOctave);
	declareProperty("widthOfBand", m_widthOfBand);

	m_extractor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
}

SolARDescriptorsExtractorBinaryOpencv::~SolARDescriptorsExtractorBinaryOpencv() { }

org::bcom::xpcf::XPCFErrorCode SolARDescriptorsExtractorBinaryOpencv::onConfigured()
{
	if (m_extractor->empty())
		return xpcf::_FAIL;
	m_extractor->setReductionRatio(m_scale);
	m_extractor->setNumOfOctaves(m_numOctave);
	m_extractor->setWidthOfBand(m_widthOfBand);
	return xpcf::_SUCCESS;
}

void SolARDescriptorsExtractorBinaryOpencv::extract(const SRef<Image> image, const std::vector<Keyline>& keylines, SRef<DescriptorBuffer>& descriptors)
{
	cv::Mat opencvImage;
	SolAROpenCVHelper::mapToOpenCV(image, opencvImage);
	
	cv::Mat cvDescriptors;
	std::vector<cv::line_descriptor::KeyLine> cvKeylines;
	for (int i = 0; i < keylines.size(); i++)
	{
		cv::line_descriptor::KeyLine kli;
		kli.pt = cv::Point2f(keylines[i].getX(), keylines[i].getY());
		kli.startPointX = keylines[i].getStartPointX();
		kli.startPointY = keylines[i].getStartPointY();
		kli.sPointInOctaveX = keylines[i].getSPointInOctaveX();
		kli.sPointInOctaveY = keylines[i].getSPointInOctaveY();
		kli.endPointX = keylines[i].getEndPointX();
		kli.endPointY = keylines[i].getEndPointY();
		kli.sPointInOctaveX = keylines[i].getEPointInOctaveX();
		kli.sPointInOctaveY = keylines[i].getEPointInOctaveY();
		kli.lineLength = keylines[i].getLineLength();
		kli.numOfPixels = keylines[i].getNumOfPixels();
		kli.angle = keylines[i].getAngle();
		kli.size = keylines[i].getSize();
		kli.response = keylines[i].getResponse();
		kli.octave = keylines[i].getOctave();
		kli.class_id = keylines[i].getClassId();
		cvKeylines.push_back(kli);
	}

	m_extractor->compute(opencvImage, cvKeylines, cvDescriptors);
	LOG_DEBUG("descriptor size: {}", m_extractor->descriptorSize());
	LOG_DEBUG("descriptor type: {}", m_extractor->descriptorType());
	LOG_DEBUG("cvDescriptors size: {}", cvDescriptors.size());
	LOG_DEBUG("cvDescriptors rows: {}", cvDescriptors.rows);
	LOG_DEBUG("cvDescriptors cols: {}", cvDescriptors.cols);
	LOG_DEBUG("cvDescriptors at (0,0): {}", cvDescriptors.at<uint8_t>(0,0) );

	descriptors.reset(new DescriptorBuffer(cvDescriptors.data, DescriptorType::BINARY, DescriptorDataType::TYPE_8U, 32, cvDescriptors.rows) );
}

}
}
}
