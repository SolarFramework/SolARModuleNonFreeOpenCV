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
	declareProperty("type", m_type);
	declareProperty("imageRatio", m_imageRatio);
	declareProperty("scale", m_scale);
	declareProperty("numOctave", m_numOctave);
	declareProperty("widthOfBand", m_widthOfBand);
}

SolARDescriptorsExtractorBinaryOpencv::~SolARDescriptorsExtractorBinaryOpencv() { }

org::bcom::xpcf::XPCFErrorCode SolARDescriptorsExtractorBinaryOpencv::onConfigured()
{
	if (m_type == "LSD")
		m_detector = cv::line_descriptor::LSDDetector::createLSDDetector();
	m_extractor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

	if (m_extractor->empty())
		return xpcf::_ERROR_NULL_POINTER;
	m_extractor->setReductionRatio(m_scale);
	m_extractor->setNumOfOctaves(m_numOctave);
	m_extractor->setWidthOfBand(m_widthOfBand);
	return xpcf::_SUCCESS;
}

void SolARDescriptorsExtractorBinaryOpencv::extract(const SRef<Image> image, const std::vector<Keyline>& keylines, SRef<DescriptorBuffer>& descriptors)
{
	cv::Mat opencvImage;
	SolAROpenCVHelper::mapToOpenCV(image, opencvImage);

	cv::Mat img_1;
	cv::resize(opencvImage, img_1, cv::Size(opencvImage.cols * m_imageRatio, opencvImage.rows * m_imageRatio), 0, 0);
	float ratioInv = 1.f / m_imageRatio;
	
	cv::Mat cvDescriptors;
	std::vector<cv::line_descriptor::KeyLine> cvKeylines;
	for (int i = 0; i < keylines.size(); i++)
	{
		cv::line_descriptor::KeyLine kli;
		kli.pt = cv::Point2f(keylines[i].getX() * ratioInv, keylines[i].getY() * ratioInv);
		kli.startPointX = keylines[i].getStartPointX() * ratioInv;
		kli.startPointY = keylines[i].getStartPointY() * ratioInv;
		kli.sPointInOctaveX = keylines[i].getSPointInOctaveX() * ratioInv;
		kli.sPointInOctaveY = keylines[i].getSPointInOctaveY() * ratioInv;
		kli.endPointX = keylines[i].getEndPointX() * ratioInv;
		kli.endPointY = keylines[i].getEndPointY() * ratioInv;
		kli.sPointInOctaveX = keylines[i].getEPointInOctaveX() * ratioInv;
		kli.sPointInOctaveY = keylines[i].getEPointInOctaveY() * ratioInv;
		kli.lineLength = keylines[i].getLineLength() * ratioInv;
		kli.numOfPixels = keylines[i].getNumOfPixels() * ratioInv;
		kli.angle = keylines[i].getAngle();
		kli.size = keylines[i].getSize() * ratioInv;
		kli.response = keylines[i].getResponse();
		kli.octave = keylines[i].getOctave();
		kli.class_id = keylines[i].getClassId();
		cvKeylines.push_back(kli);
	}
	// Descriptors extraction
	m_extractor->compute(img_1, cvKeylines, cvDescriptors);
	// Use ORB as DescriptorType as it is equivalent to the binary descriptor format
    descriptors.reset(new DescriptorBuffer(cvDescriptors.data, DescriptorType::ORB, DescriptorDataType::TYPE_8U, 32, cvDescriptors.rows) );
}

void SolARDescriptorsExtractorBinaryOpencv::compute(const SRef<Image> image, std::vector<Keyline>& keylines, SRef<DescriptorBuffer>& descriptors)
{
	cv::Mat opencvImage;
	SolAROpenCVHelper::mapToOpenCV(image, opencvImage);

	cv::Mat img_1;
	cv::resize(opencvImage, img_1, cv::Size(opencvImage.cols * m_imageRatio, opencvImage.rows * m_imageRatio), 0, 0);
	float ratioInv = 1.f / m_imageRatio;

	cv::Mat cvDescriptors;
	std::vector<cv::line_descriptor::KeyLine> cvKeylines;
	// Keyline detection
	if (m_type == "LSD")
		m_detector->detect(img_1, cvKeylines, m_scale, m_numOctave);
	else
		m_extractor->detect(img_1, cvKeylines);
	// Descriptors extraction
	m_extractor->compute(img_1, cvKeylines, cvDescriptors);

	keylines.clear();
	for (int i = 0; i < cvKeylines.size(); i++)
	{
		Keyline kli;
		kli.init(
			cvKeylines[i].pt.x * ratioInv,
			cvKeylines[i].pt.y * ratioInv,
			cvKeylines[i].getStartPoint().x * ratioInv,
			cvKeylines[i].getStartPoint().y * ratioInv,
			cvKeylines[i].getStartPointInOctave().x * ratioInv,
			cvKeylines[i].getStartPointInOctave().y * ratioInv,
			cvKeylines[i].getEndPoint().x * ratioInv,
			cvKeylines[i].getEndPoint().y * ratioInv,
			cvKeylines[i].getEndPointInOctave().x * ratioInv,
			cvKeylines[i].getEndPointInOctave().y * ratioInv,
			cvKeylines[i].lineLength * ratioInv,
			cvKeylines[i].size * ratioInv,
			cvKeylines[i].angle,
			cvKeylines[i].response,
			cvKeylines[i].numOfPixels * ratioInv,
			cvKeylines[i].octave,
			cvKeylines[i].class_id
		);
		keylines.push_back(kli);
	}
	// Use ORB as DescriptorType as it is equivalent to the binary descriptor format
    descriptors.reset(new DescriptorBuffer(cvDescriptors.data, DescriptorType::ORB, DescriptorDataType::TYPE_8U, 32, cvDescriptors.rows));
}


}
}
}
