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

#include "SolARFiducialMarkerPoseEstimatorNonFreeOpencv.h"
#include "SolARNonFreeOpenCVHelper.h"
#include "core/Log.h"

//#include <boost/thread/thread.hpp>
XPCF_DEFINE_FACTORY_CREATE_INSTANCE(SolAR::MODULES::NONFREEOPENCV::SolARFiducialMarkerPoseEstimatorNonFreeOpencv);

namespace xpcf = org::bcom::xpcf;

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

SolARFiducialMarkerPoseEstimatorNonFreeOpencv::SolARFiducialMarkerPoseEstimatorNonFreeOpencv():ConfigurableBase(xpcf::toUUID<SolARDescriptorsExtractorSURF64Opencv>())
{
    declareInterface<api::solver::pose::ITrackablePose>(this);
	declareInjectable<api::solver::pose::I3DTransformFinderFrom2D3D>(m_pnp);
	declareInjectable<api::geom::IProject>(m_projector);
	declareProperty("nbThreshold", m_nbThreshold);
	declareProperty("minThreshold", m_minThreshold);
	declareProperty("maxThreshold", m_maxThreshold);
	declareProperty("maxReprojError", m_maxReprojError);
    LOG_DEBUG(" SolARFiducialMarkerPoseEstimatorNonFreeOpencv constructor")
}

void SolARFiducialMarkerPoseEstimatorNonFreeOpencv::setDictionary(const datastructure::SquaredBinaryPattern &pattern)
{
	int patternSize = pattern.getSize();
	const SquaredBinaryPatternMatrix &patternMatrix = pattern.getPatternMatrix();
	cv::Mat markerBits(patternSize, patternSize, CV_8UC1);
	for (int i = 0; i < patternSize; i++)
		for (int j = 0; j < patternSize; j++)
			markerBits.at<uchar>(i, j) = (uchar)patternMatrix(i, j);
	cv::Mat markerCompressed = cv::aruco::Dictionary::getByteListFromBits(markerBits);
	// create dictionary from binary marker
	m_dictionary = cv::aruco::Dictionary::create(1, patternSize);
	m_dictionary->maxCorrectionBits = 0;
	m_dictionary->bytesList.push_back(markerCompressed);
}

xpcf::XPCFErrorCode SolARFiducialMarkerPoseEstimatorNonFreeOpencv::onConfigured()
{	
	m_detectorParams = cv::aruco::DetectorParameters::create();
	m_detectorParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    LOG_DEBUG(" SolARFiducialMarkerPoseEstimatorNonFreeOpencv configured");
    return xpcf::XPCFErrorCode::_SUCCESS;
}

FrameworkReturnCode SolARFiducialMarkerPoseEstimatorNonFreeOpencv::setTrackable(const SRef<datastructure::Trackable> trackable)
{
    // components initialisation for marker detection
    if (trackable->getType() == TrackableType::FIDUCIAL_MARKER)
    {
        m_fiducialMarker = xpcf::utils::dynamic_pointer_cast<FiducialMarker>(trackable);
        setDictionary(m_fiducialMarker->getPattern());
        LOG_DEBUG("Marker pattern:\n {}", m_fiducialMarker->getPattern().getPatternMatrix());
    }
    else {
        LOG_ERROR("The SolARFiducialMarkerPoseEstimator should only use a trackable of type FIDUCIAL_MARKER")
        return FrameworkReturnCode::_ERROR_;
    }
    return FrameworkReturnCode::_SUCCESS;


//    m_binaryMarker = marker;

//	LOG_DEBUG("Marker pattern:\n {}", m_binaryMarker->getPattern().getPatternMatrix());
}

FrameworkReturnCode SolARFiducialMarkerPoseEstimatorNonFreeOpencv::estimate(const SRef<Image> image, const SolAR::datastructure::CameraParameters & camParams, Transform3Df & pose)
{	
	cv::Mat opencvImage;
	SolARNonFreeOpenCVHelper::mapToOpenCV(image, opencvImage);

	std::vector<int> ids;
	std::vector<std::vector<cv::Point2f>> corners, rejected;
	std::vector<cv::Mat> rvecs, tvecs;

	// detect markers 
	cv::aruco::detectMarkers(opencvImage, m_dictionary, corners, ids, m_detectorParams, rejected);
	if (ids.size() == 0)
		return FrameworkReturnCode::_ERROR_;

	// find 2D-3D correspondences
	std::vector<Point2Df> img2DPoints;
	std::vector<Point3Df> pattern3DPoints;
	for (const auto &it : corners[0])
		img2DPoints.push_back(Point2Df(it.x, it.y));
    m_fiducialMarker->getWorldCorners(pattern3DPoints);
	// Compute the pose of the camera using a Perspective n Points algorithm using only the 4 corners of the marker
	if (m_pnp->estimate(img2DPoints, pattern3DPoints, camParams, pose) == FrameworkReturnCode::_SUCCESS){
		std::vector<Point2Df> projected2DPts;
		m_projector->project(pattern3DPoints, pose, camParams, projected2DPts);
		float errorReproj(0.f);
		for (int j = 0; j < projected2DPts.size(); ++j)
			errorReproj += (projected2DPts[j] - img2DPoints[j]).norm();
		errorReproj /= projected2DPts.size();
		LOG_DEBUG("Mean reprojection error: {}", errorReproj);
		if (errorReproj < m_maxReprojError)
			return FrameworkReturnCode::_SUCCESS;
		pose = Transform3Df::Identity();
		return FrameworkReturnCode::_ERROR_;
	}
}

}
}
}  // end of namespace SolAR
