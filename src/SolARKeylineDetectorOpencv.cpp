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

#include "SolARKeylineDetectorOpencv.h"
#include "SolAROpenCVHelper.h"
#include "core/Log.h"

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARKeylineDetectorOpencv)

namespace xpcf = org::bcom::xpcf;
using namespace SolAR::MODULES::OPENCV;

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

static std::map<std::string, IKeylineDetector::KeylineDetectorType> stringToType = {
	{ "LSD", IKeylineDetector::KeylineDetectorType::LSD },
	{ "MSLD", IKeylineDetector::KeylineDetectorType::MSLD }
};

static std::map<IKeylineDetector::KeylineDetectorType, std::string> typeToString = {
	{ IKeylineDetector::KeylineDetectorType::LSD, "LSD" },
	{ IKeylineDetector::KeylineDetectorType::MSLD, "MSLD" }
};

SolARKeylineDetectorOpencv::SolARKeylineDetectorOpencv() : ConfigurableBase(xpcf::toUUID<SolARKeylineDetectorOpencv>())
{
	declareInterface<api::features::IKeylineDetector>(this);

	declareProperty("imageRatio", m_imageRatio);
	declareProperty("nbDescriptors", m_nbDescriptors);
	declareProperty("scale", m_scale);
	declareProperty("numOctave", m_numOctave);
	declareProperty("type", m_type);
	LOG_DEBUG("SolARKeylineDetectorOpencv constructor");
}

SolARKeylineDetectorOpencv::~SolARKeylineDetectorOpencv()
{
	LOG_DEBUG("SolARKeylineDetectorOpencv destructor");
}

xpcf::XPCFErrorCode SolARKeylineDetectorOpencv::onConfigured()
{
	LOG_DEBUG("SolARKeylineDetectorOpencv onConfigured");
	if (stringToType.find(m_type) != stringToType.end())
	{
		setType(stringToType.at(m_type));
		return xpcf::_SUCCESS;
	}
	else
	{
		LOG_WARNING("Keyline detector of type {} defined in your configuration file does not exist", m_type);
		return xpcf::_ERROR_NOT_IMPLEMENTED;
	}
}

void SolARKeylineDetectorOpencv::setType(KeylineDetectorType type)
{
	switch (type)
	{
	case (KeylineDetectorType::LSD):
		LOG_DEBUG("KeylineDetectorType::setType(LSD)");
		m_detector = cv::line_descriptor::LSDDetector::createLSDDetector();
		// /!\ LineSegmentDetector is no longer implemented in OpenCV >= 4 /!\ 
		break;
	case (KeylineDetectorType::MSLD):
		LOG_DEBUG("KeylineDetectorType::setType(MSLD) not implemented");
		break;
	}
}

IKeylineDetector::KeylineDetectorType SolARKeylineDetectorOpencv::getType()
{
	return stringToType.at(m_type);
}

void SolARKeylineDetectorOpencv::detect(const SRef<Image> image, std::vector<Keyline>& keylines)
{
	float ratioInv = 1.f / m_imageRatio;

	keylines.clear();

	cv::Mat opencvImage = SolAROpenCVHelper::mapToOpenCV(image);

	cv::Mat img_1;
    //cv::cvtColor(opencvImage, img_1, cv::COLOR_BGR2GRAY);
	cv::resize(opencvImage, img_1, cv::Size(opencvImage.cols * m_imageRatio, opencvImage.rows * m_imageRatio), 0, 0);

	std::vector<cv::line_descriptor::KeyLine> cvKeylines;
	try
	{
		if (!m_detector)
		{
			LOG_DEBUG(" detector is initialiazed with default value : {}", m_type);
			setType(stringToType.at(m_type));
		}
		m_detector->detect(img_1, cvKeylines, m_scale, m_numOctave, cv::Mat());
	}
	catch (cv::Exception & e)
	{
		LOG_ERROR("Feature: {}", m_detector->getDefaultName().c_str());
		LOG_ERROR("{}", e.msg);
		return;
	}

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
			cvKeylines[i].numOfPixels * ratioInv,
			cvKeylines[i].size * ratioInv,
			cvKeylines[i].angle,
			cvKeylines[i].response,
			cvKeylines[i].octave,
			cvKeylines[i].class_id
		);
		keylines.push_back(kli);
	}
}

}
}
}
