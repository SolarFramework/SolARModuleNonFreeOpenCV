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
	declareProperty("numOctaves", m_numOctaves);
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
	case (KeylineDetectorType::LSD): /* /!\ LSD implementation has been removed in OpenCV 4+ /!\ */
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
	cv::resize(opencvImage, opencvImage, cv::Size(), m_imageRatio, m_imageRatio);
	// Conversion to grayscale
	if (opencvImage.channels() != 1)
		cv::cvtColor(opencvImage, opencvImage, cv::COLOR_BGR2GRAY);

	std::vector<KeyLine> cvKeylines;
	try
	{
		if (!m_detector)
		{
			LOG_ERROR(" detector is not initialized!");
			return;
		}
		switch (stringToType.at(m_type))
		{
		case KeylineDetectorType::FLD: // TODO: Check->This implementation should be license free
		{
			// Prepare different scale/octave
			std::vector<std::vector<cv::Vec4f>> lines;
			lines.resize(m_numOctaves);
			std::vector<cv::Mat> gaussianPyrs = computeGaussianPyramid(opencvImage, m_numOctaves, m_scale);
			// Perform line detection on each octave
			int class_counter = -1;
			for (int i = 0; i < m_numOctaves; ++i)
			{
				m_detector.dynamicCast<FastLineDetector>()
					->detect(gaussianPyrs[i], lines[i]);
			}
			// Create keylines
			float octaveScale = 1.f;
			for (int i = 0; i < m_numOctaves; ++i)
			{
				for (const auto& l : lines[i])
				{
					KeyLine kl;
					/* fill KeyLine's fields */
					kl.startPointX = l[0] * octaveScale;
					kl.startPointY = l[1] * octaveScale;
					kl.endPointX = l[2] * octaveScale;
					kl.endPointY = l[3] * octaveScale;
					kl.sPointInOctaveX = l[0];
					kl.sPointInOctaveY = l[1];
					kl.ePointInOctaveX = l[2];
					kl.ePointInOctaveY = l[3];
					kl.lineLength = std::sqrtf(std::powf(l[0] - l[2], 2.f) + std::powf(l[1] - l[3], 2.f));

					/* compute number of pixels covered by line */
					cv::LineIterator li(gaussianPyrs[i], cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]));
					kl.numOfPixels = li.count;

					kl.angle = std::atan2f((kl.endPointY - kl.startPointY), (kl.endPointX - kl.startPointX));
					kl.class_id = ++class_counter;
					kl.octave = i;
					kl.size = (kl.endPointX - kl.startPointX) * (kl.endPointY - kl.startPointY);
					kl.response = kl.lineLength / std::max(gaussianPyrs[i].cols, gaussianPyrs[i].rows);
					kl.pt = cv::Point2f((kl.endPointX + kl.startPointX) / 2.f, (kl.endPointY + kl.startPointY) / 2.f);

					cvKeylines.push_back(kl);
				}
				// Prepare next octave scale factor
				if (i < m_numOctaves - 1)
					octaveScale *= m_scale;
			}
			break;
		}
		case KeylineDetectorType::LSD:
		{
			// Perform keyline detection
			m_detector.dynamicCast<LSDDetector>()
				->detect(opencvImage, cvKeylines, m_scale, m_numOctaves, cv::Mat());
			break;
		}
		}
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
	}
	catch (cv::Exception & e)
	{
		LOG_ERROR("Feature: {}", m_detector->getDefaultName().c_str());
		LOG_ERROR("{}", e.msg);
		return;
	}
}

std::vector<cv::Mat> SolARKeylineDetectorOpencv::computeGaussianPyramid(const cv::Mat & opencvImage, int numOctaves, int scale)
{
	std::vector<cv::Mat> gaussianPyrs;
	gaussianPyrs.resize(numOctaves);
	cv::Mat currentImg = opencvImage.clone();
	gaussianPyrs[0] = currentImg;
	for (int i = 1; i < numOctaves; ++i)
	{
		cv::pyrDown( currentImg, currentImg, cv::Size(currentImg.cols / scale, currentImg.rows / scale) );
		gaussianPyrs[i] = currentImg;
	}
	return gaussianPyrs;
}

}
}
}
