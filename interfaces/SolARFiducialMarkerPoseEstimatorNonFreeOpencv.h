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

#ifndef SOLARFIDUCIALMARKERPOSEESTIMATORNONFREEOPENCV_H
#define SOLARFIDUCIALMARKERPOSEESTIMATORNONFREEOPENCV_H

#include "xpcf/component/ConfigurableBase.h"
#include "api/solver/pose/ITrackablePose.h"
#include "api/geom/IProject.h"
#include "api/solver/pose/I3DTransformFinderFrom2D3D.h"
#include "datastructure/FiducialMarker.h"
#include "SolAROpencvNonFreeAPI.h"
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/aruco.hpp"

namespace SolAR {
namespace MODULES {
namespace NONFREEOPENCV {

/**
* @class SolARFiducialMarkerPoseEstimatorNonFreeOpencv
* @brief <B>Estimate camera pose based on a fiducial marker using Aruco library.</B>
* <TT>UUID: 2b952e6c-ddd4-4316-ac9a-d3fad0b33b32</TT>
*
* @SolARComponentInjectablesBegin
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

class SOLAROPENCVNONFREE_EXPORT_API SolARFiducialMarkerPoseEstimatorNonFreeOpencv : public org::bcom::xpcf::ConfigurableBase,
        public api::solver::pose::ITrackablePose {
public:
	///@brief SolAR3DTransformEstimationFrom3D3D constructor;
    SolARFiducialMarkerPoseEstimatorNonFreeOpencv();

	///@brief SolAR3DTransformEstimationFrom3D3D destructor;
    ~SolARFiducialMarkerPoseEstimatorNonFreeOpencv() = default;

    org::bcom::xpcf::XPCFErrorCode onConfigured() override final;

	/// @brief this method is used to set the fiducial marker
	/// @param[in] Fiducial marker.
    FrameworkReturnCode setTrackable(const SRef<SolAR::datastructure::Trackable> trackable) override;

    /// @brief Estimates camera pose based on a fiducial marker.
    /// @param[in] image input image.
    /// @param[in] camParams the camera parameters.
    /// @param[out] pose camera pose.
    /// @return FrameworkReturnCode::_SUCCESS if the estimation succeed, else FrameworkReturnCode::_ERROR_
    FrameworkReturnCode estimate(const SRef<SolAR::datastructure::Image> image,
                                 const SolAR::datastructure::CameraParameters & camParams,
                                 SolAR::datastructure::Transform3Df & pose) override;

	void unloadComponent() override final;

private:

    void setDictionary(const datastructure::SquaredBinaryPattern &pattern);

private:
	cv::Ptr<cv::aruco::Dictionary>						m_dictionary;
	cv::Ptr<cv::aruco::DetectorParameters>				m_detectorParams;
    SRef<datastructure::FiducialMarker>                 m_fiducialMarker;
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



#endif // SOLARFIDUCIALMARKERPOSEESTIMATORNONFREEOPENCV_H
