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
#include "SolARNonFreeOpenCVHelper.h"
#include "core/Log.h"

XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARKeylineDetectorOpencv)

namespace xpcf = org::bcom::xpcf;
using namespace cv::line_descriptor;
using namespace cv::ximgproc;

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

static std::map<std::string, IKeylineDetector::KeylineDetectorType> stringToType = {
	{ "FLD", IKeylineDetector::KeylineDetectorType::FLD },
	{ "LSD", IKeylineDetector::KeylineDetectorType::LSD },
	{ "MSLD", IKeylineDetector::KeylineDetectorType::MSLD }
};

static std::map<IKeylineDetector::KeylineDetectorType, std::string> typeToString = {
	{ IKeylineDetector::KeylineDetectorType::FLD, "FLD" },
	{ IKeylineDetector::KeylineDetectorType::LSD, "LSD" },
	{ IKeylineDetector::KeylineDetectorType::MSLD, "MSLD" }
};

SolARKeylineDetectorOpencv::SolARKeylineDetectorOpencv() : ConfigurableBase(xpcf::toUUID<SolARKeylineDetectorOpencv>())
{
	declareInterface<api::features::IKeylineDetector>(this);

	declareProperty("imageRatio", m_imageRatio);
	declareProperty("scale", m_scale);
	declareProperty("numOctave", m_numOctave);
	declareProperty("type", m_type);
	declareProperty("minLineLength", m_minLineLength);
	LOG_DEBUG("SolARKeylineDetectorOpencv constructor");
}

SolARKeylineDetectorOpencv::~SolARKeylineDetectorOpencv()
{
	LOG_DEBUG("SolARKeylineDetectorOpencv destructor");
}

xpcf::XPCFErrorCode SolARKeylineDetectorOpencv::onConfigured()
{
	LOG_DEBUG("SolARKeylineDetectorOpencv onConfigured");
	if (stringToType.find(m_type) != stringToType.end()) // TODO better detector type handling
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

void SolARKeylineDetectorOpencv::setType(KeylineDetectorType type) // TODO RENAME init detector
{
	switch (type)
	{
	case (KeylineDetectorType::FLD):
		LOG_DEBUG("KeylineDetectorType::setType(FLD) - Fast Line Detector");
		m_detector = createFastLineDetector(m_minLineLength);
		break;
	case (KeylineDetectorType::LSD): /* /!\ LSD is no longer implemented in OpenCV >= 4 /!\ */
		LOG_DEBUG("KeylineDetectorType::setType(LSD) - Line Segment Detector");
		m_detector = LSDDetector::createLSDDetector();
		break;
	case (KeylineDetectorType::MSLD):
		LOG_DEBUG("KeylineDetectorType::setType(MSLD) - Multi Scale Line Detector not implemented");
		break;
	default:
		LOG_WARNING("Failed to initialize detector - unknown type");
	}
}

IKeylineDetector::KeylineDetectorType SolARKeylineDetectorOpencv::getType()
{
	return stringToType.at(m_type);
}

void SolARKeylineDetectorOpencv::detect(const SRef<Image> image, std::vector<Keyline>& keylines)
{
	float ratioInv = 1.f / m_imageRatio;

	keylines.clear(); // TODO: needed ?

	cv::Mat opencvImage = SolARNonFreeOpenCVHelper::mapToOpenCV(image);
	cv::cvtColor(opencvImage, opencvImage, cv::COLOR_BGR2GRAY);
	cv::resize(opencvImage, opencvImage, cv::Size(), m_imageRatio, m_imageRatio);

	try
	{
		if (!m_detector)
		{
			LOG_ERROR(" detector is not initialized!");
			return;
		}
		switch (stringToType.at(m_type))
		{
		case KeylineDetectorType::FLD:
		{
			// Perform keyline detection
			std::vector<cv::Vec4f> cvKeylines;
			m_detector.dynamicCast<FastLineDetector>()
				->detect(opencvImage, cvKeylines);
			// Convert to SolAR Keylines
			keylines.resize(cvKeylines.size());
			std::transform(cvKeylines.begin(), cvKeylines.end(), keylines.begin(),
				[ratioInv](const auto& kl)
			{
				return Keyline( // TODO compute other keyline params (length, midpoint, angle,... ?)
					kl[0] * ratioInv, kl[1] * ratioInv,		// startPoint
					kl[2] * ratioInv, kl[3] * ratioInv		// endPoint
				);
			});
			break;
		}
		case KeylineDetectorType::LSD:
		{
			// Perform keyline detection
			std::vector<KeyLine> cvKeylines;
			m_detector.dynamicCast<LSDDetector>()
				->detect(opencvImage, cvKeylines, m_scale, m_numOctave, cv::Mat());
			// Convert to SolAR Keylines
			for (const auto& kl : cvKeylines)
			{
				float length = kl.lineLength * ratioInv;
				if (length < m_minLineLength) continue;

				Keyline kli;
				kli.init(
					kl.pt.x * ratioInv,
					kl.pt.y * ratioInv,
					kl.getStartPoint().x * ratioInv,
					kl.getStartPoint().y * ratioInv,
					kl.getStartPointInOctave().x * ratioInv,
					kl.getStartPointInOctave().y * ratioInv,
					kl.getEndPoint().x * ratioInv,
					kl.getEndPoint().y * ratioInv,
					kl.getEndPointInOctave().x * ratioInv,
					kl.getEndPointInOctave().y * ratioInv,
					length,
					kl.size * ratioInv,
					kl.angle,
					kl.response,
					kl.numOfPixels * ratioInv,
					kl.octave,
					kl.class_id
				);
				keylines.push_back(kli);
			}
			break;
		}
		}
	}
	catch (cv::Exception & e)
	{
		LOG_ERROR("Feature: {}", m_detector->getDefaultName().c_str());
		LOG_ERROR("{}", e.msg);
		return;
	}
}

}
}
}
