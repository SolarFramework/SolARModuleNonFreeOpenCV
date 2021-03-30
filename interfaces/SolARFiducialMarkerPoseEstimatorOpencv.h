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

#ifndef SOLARFIDUCIALMARKERPOSEESTIMATOROPENCV_H
#define SOLARFIDUCIALMARKERPOSEESTIMATOROPENCV_H

#include "api/solver/pose/IFiducialMarkerPose.h"
#include "api/geom/IProject.h"
#include "api/solver/pose/I3DTransformFinderFrom2D3D.h"
#include "xpcf/component/ConfigurableBase.h"
#include "SolAROpencvNonFreeAPI.h"
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/aruco.hpp"

namespace SolAR {
namespace MODULES {
namespace NONFREEOPENCV {

/**
* @class SolARFiducialMarkerPoseEstimatorOpencv
* @brief <B>Estimate camera pose based on a fiducial marker using Aruco library.</B>
* <TT>UUID: 2b952e6c-ddd4-4316-ac9a-d3fad0b33b32</TT>
*
* @SolARComponentInjectablesBegin
* @SolARComponentInjectable{SolAR::api::input::files::IMarker2DSquaredBinary}
* @SolARComponentInjectable{SolAR::api::solver::pose::I3DTransformFinderFrom2D3D}
* @SolARComponentInjectable{SolAR::api::geom::IProject}
* @SolARComponentInjectablesEnd
*
* @SolARComponentPropertiesBegin
* @SolARComponentProperty{ nbThreshold,
*                         ,
*                         @SolARComponentPropertyDescNum{ int, [0..MAX INT], 3 }}
* @SolARComponentProperty{ minThreshold,
*                          ,
*                          @SolARComponentPropertyDescNum{ int, [-1..MAX INT], -1 }}
* @SolARComponentProperty{ maxThreshold,
*                          ,
*                          @SolARComponentPropertyDescNum{ int, [0..MAX INT], 220 }}
* @SolARComponentPropertiesEnd
*
*
*/

class SOLAROPENCVNONFREE_EXPORT_API SolARFiducialMarkerPoseEstimatorOpencv : public org::bcom::xpcf::ConfigurableBase,
        public api::solver::pose::IFiducialMarkerPose {
public:
	///@brief SolAR3DTransformEstimationFrom3D3D constructor;
	SolARFiducialMarkerPoseEstimatorOpencv();
	///@brief SolAR3DTransformEstimationFrom3D3D destructor;
	~SolARFiducialMarkerPoseEstimatorOpencv() = default;
    org::bcom::xpcf::XPCFErrorCode onConfigured() override final;
	/// @brief this method is used to set intrinsic parameters and distorsion of the camera
	/// @param[in] Camera calibration matrix parameters.
	/// @param[in] Camera distorsion parameters.
	void setCameraParameters(const datastructure::CamCalibration & intrinsicParams, const datastructure::CamDistortion & distorsionParams) override;

	/// @brief this method is used to set the fiducial marker
	/// @param[in] Fiducial marker.
	void setMarker(const SRef<api::input::files::IMarker2DSquaredBinary> marker) override;

	/// @brief this method is used to set the fiducial marker
	/// @param[in] Fiducial marker.
	void setMarker(const SRef<datastructure::FiducialMarker> marker) override;

	/// @brief Estimates camera pose based on a fiducial marker.
	/// @param[in] image: input image.
	/// @param[out] pose: camera pose.
	/// @return FrameworkReturnCode::_SUCCESS if the estimation succeed, else FrameworkReturnCode::_ERROR_
	FrameworkReturnCode estimate(const SRef<datastructure::Image> image, datastructure::Transform3Df & pose) override;

	void unloadComponent() override final;

private:

	void setDictionary(const datastructure::SquaredBinaryPattern &pattern);

private:
	cv::Mat												m_camMatrix;
	cv::Mat												m_camDistortion;
	cv::Ptr<cv::aruco::Dictionary>						m_dictionary;
	cv::Ptr<cv::aruco::DetectorParameters>				m_detectorParams;
	SRef<api::input::files::IMarker2DSquaredBinary>		m_binaryMarker;
	SRef<api::solver::pose::I3DTransformFinderFrom2D3D>	m_pnp;
	SRef<api::geom::IProject>							m_projector;
	int													m_nbThreshold = 3;
	int													m_minThreshold = -1;
	int													m_maxThreshold = 220;
	float												m_maxReprojError = 0.5f;
};

}
}
}  // end of namespace SolAR



#endif // SOLARFIDUCIALMARKERPOSEESTIMATOROPENCV_H
