
#include <xpcf/xpcf.h>
#include <xpcf/threading/BaseTask.h>
#include <xpcf/threading/DropBuffer.h>
#include <boost/log/core.hpp>
#include <core/Log.h>

#include "SolARModuleOpencv_traits.h"
#include "SolARModuleNonFreeOpencv_traits.h"
#include "SolARModuleTools_traits.h"

#include "api/display/I2DOverlay.h"
#include "api/display/I3DOverlay.h"
#include "api/display/I3DPointsViewer.h"
#include "api/display/IImageViewer.h"
#include "api/features/IContoursExtractor.h"
#include "api/features/IContoursFilter.h"
#include "api/features/IDescriptorMatcher.h"
#include "api/features/IDescriptorsExtractorBinary.h"
#include "api/features/IDescriptorsExtractorSBPattern.h"
#include "api/features/IMatchesFilter.h"
#include "api/features/ISBPatternReIndexer.h"
#include "api/geom/IImage2WorldMapper.h"
#include "api/image/IImageConvertor.h"
#include "api/image/IImageFilter.h"
#include "api/image/IPerspectiveController.h"
#include "api/input/devices/ICamera.h"
#include "api/input/files/IMarker2DSquaredBinary.h"
#include "api/solver/map/ITriangulator.h"
#include "api/solver/pose/I3DTransformFinderFrom2D3D.h"

#include "SolAROpenCVHelper.h"

#include <opencv2/core.hpp>
#include "opencv2/video/video.hpp"


#define MIN_THRESHOLD -1
#define MAX_THRESHOLD 220
#define NB_THRESHOLD 8
#define EPSILON 0.0001

namespace xpcf = org::bcom::xpcf;

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;
using namespace SolAR::MODULES::NONFREEOPENCV;
using namespace SolAR::MODULES::TOOLS;

cv::Mat fundamentalMatrix(	const cv::Mat & pose1, const cv::Mat & pose2,
							const cv::Mat & pose1Inv, const cv::Mat & pose2Inv,
							const cv::Mat & m_cameraMatrix)
{
	cv::Mat pose12 = pose1Inv * pose2;
	cv::Mat R12 = (cv::Mat_<double>(3, 3) << pose12.at<double>(0, 0), pose12.at<double>(0, 1), pose12.at<double>(0, 2),
											 pose12.at<double>(1, 0), pose12.at<double>(1, 1), pose12.at<double>(1, 2),
											 pose12.at<double>(2, 0), pose12.at<double>(2, 1), pose12.at<double>(2, 2));
	cv::Mat T12 = (cv::Mat_<double>(3, 1) << pose12.at<double>(0, 3), pose12.at<double>(1, 3), pose12.at<double>(2, 3));

	cv::Mat T12x = (cv::Mat_<double>(3, 3) <<                 0, -T12.at<double>(2),  T12.at<double>(1),
											  T12.at<double>(2),				  0, -T12.at<double>(0),
											 -T12.at<double>(1),  T12.at<double>(0),				 0);

	cv::Mat F12 = m_cameraMatrix.t().inv() * T12x * R12 * m_cameraMatrix.inv();
	return F12;
}

cv::Mat_<double> linearTriangulation(const cv::Point3d & u,
	const cv::Matx34d & P,
	const cv::Point3d & u1,
	const cv::Matx34d & P1) {

	cv::Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
	);
	cv::Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3)));

	cv::Mat_<double> X;
	cv::solve(A, B, X, cv::DECOMP_SVD);
	return X;
}

cv::Mat_<double> iterativeLinearTriangulation(const cv::Point3d & u,
	const cv::Matx34d & P,
	const cv::Point3d & u1,
	const cv::Matx34d & P1) {

	double wi = 1, wi1 = 1;
	cv::Mat_<double> X(4, 1);

	cv::Mat_<double> X_ = linearTriangulation(u, P, u1, P1);
	X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

	//std::cout<<" output of linear triangulation: "<<X<<std::endl;
	for (int i = 0; i < 10; i++) {
		//recalculate weights
		double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2)*X)(0);
		double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		cv::Matx43d A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
			(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
			(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
			(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
		);
		cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
			-(u.y*P(2, 3) - P(1, 3)) / wi,
			-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
			-(u1.y*P1(2, 3) - P1(1, 3)) / wi1
			);

		cv::solve(A, B, X_, cv::DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
		//std::cout<<"Output Non Linear Triangulation iteration "<<i<<" X: "<<X<<std::endl;
	}
	return X;
}

double triangulate(	const std::vector<Keyline> & keylines1,
					const std::vector<Keyline> & keylines2,
					const std::vector<DescriptorMatch> & matches,
					const Transform3Df & pose1,
					const Transform3Df & pose2,
					const CamCalibration & intrinsicParams,
					std::vector<Edge3Df> & lines3D,
					std::vector<int> & indices,
					const SRef<Image> & img1, const SRef<Image> & img2,
					SRef<Image> & imgDebug)
{
	if (imgDebug == nullptr)
	{
		imgDebug = xpcf::utils::make_shared<Image>(img1->getWidth() + img2->getWidth(), std::max(img1->getHeight(), img2->getHeight()), img1->getImageLayout(), img1->getPixelOrder(), img1->getDataType());
	}
	else if ((imgDebug->getWidth() != img1->getWidth() + img2->getWidth()) || (imgDebug->getHeight() != std::max(img1->getHeight(), img2->getHeight())))
	{
		imgDebug->setSize(img1->getWidth() + img2->getWidth(), std::max(img1->getHeight(), img2->getHeight()));
	}
	cv::Mat cv_img1, cv_img2, cv_imgDebug;
	cv_img1 = SolAROpenCVHelper::mapToOpenCV(img1);
	cv_img2 = SolAROpenCVHelper::mapToOpenCV(img2);
	cv_imgDebug = SolAROpenCVHelper::mapToOpenCV(imgDebug);
	cv_imgDebug.setTo(0);
	int img1_width = img1->getWidth();
	cv_img1.copyTo(cv_imgDebug(cv::Rect(0, 0, img1_width, img1->getHeight())));
	cv_img2.copyTo(cv_imgDebug(cv::Rect(img1_width, 0, img2->getWidth(), img2->getHeight())));

	LOG_INFO("Performing triangulation...");
	cv::Mat_<double> m_camMatrix;
	m_camMatrix.create(3, 3);
	m_camMatrix.at<double>(0, 0) = (double)intrinsicParams(0, 0);
	m_camMatrix.at<double>(0, 1) = (double)intrinsicParams(0, 1);
	m_camMatrix.at<double>(0, 2) = (double)intrinsicParams(0, 2);
	m_camMatrix.at<double>(1, 0) = (double)intrinsicParams(1, 0);
	m_camMatrix.at<double>(1, 1) = (double)intrinsicParams(1, 1);
	m_camMatrix.at<double>(1, 2) = (double)intrinsicParams(1, 2);
	m_camMatrix.at<double>(2, 0) = (double)intrinsicParams(2, 0);
	m_camMatrix.at<double>(2, 1) = (double)intrinsicParams(2, 1);
	m_camMatrix.at<double>(2, 2) = (double)intrinsicParams(2, 2);

	cv::Mat Kinv = m_camMatrix.inv();


	cv::Matx44d Pose1(	pose1(0, 0), pose1(0, 1), pose1(0, 2), pose1(0, 3),
						pose1(1, 0), pose1(1, 1), pose1(1, 2), pose1(1, 3),
						pose1(2, 0), pose1(2, 1), pose1(2, 2), pose1(2, 3),
						pose1(3, 0), pose1(3, 1), pose1(3, 2), pose1(3, 3));

	cv::Matx44d Pose2(	pose2(0, 0), pose2(0, 1), pose2(0, 2), pose2(0, 3),
						pose2(1, 0), pose2(1, 1), pose2(1, 2), pose2(1, 3),
						pose2(2, 0), pose2(2, 1), pose2(2, 2), pose2(2, 3),
						pose2(3, 0), pose2(3, 1), pose2(3, 2), pose2(3, 3));

	cv::Mat P1 = cv::Mat(Pose1);
	cv::Mat P2 = cv::Mat(Pose2);

	cv::Mat pose1Inv = P1.inv();
	cv::Mat pose2Inv = P2.inv();

	cv::Mat KPose1 = m_camMatrix * pose1Inv.rowRange(0, 3);
	cv::Mat KPose2 = m_camMatrix * pose2Inv.rowRange(0, 3);

	cv::Mat F12 = fundamentalMatrix(P1, P2, pose1Inv, pose2Inv, m_camMatrix);
	LOG_INFO("F12: {}", F12);
	cv::Mat um1, um2;
	cv::Point3f u1, u2;
	// For each match
	for (unsigned i = 0; i < matches.size(); i++)
	//unsigned i = 0;
	{

		Keyline kl1 = keylines1[matches[i].getIndexInDescriptorA()];
		Keyline kl2 = keylines2[matches[i].getIndexInDescriptorB()];

		// Start & End points in homogeneous space
		cv::Mat start1 = (cv::Mat_<double>(3, 1) << kl1.getStartPointX(), kl1.getStartPointY(), 1);
		cv::Mat end1 = (cv::Mat_<double>(3, 1) << kl1.getEndPointX(), kl1.getEndPointY(), 1);
		cv::Mat start2 = (cv::Mat_<double>(3, 1) << kl2.getStartPointX(), kl2.getStartPointY(), 1);
		cv::Mat end2 = (cv::Mat_<double>(3, 1) << kl2.getEndPointX(), kl2.getEndPointY(), 1);

		// Draw l1 and l2 on their respective image
		cv::line(cv_imgDebug, cv::Point2f(start1.at<double>(0), start1.at<double>(1)), cv::Point2f(end1.at<double>(0), end1.at<double>(1)), cv::Scalar(0, 255, 0), 3);
		cv::line(cv_imgDebug, cv::Point2f(start2.at<double>(0) + img1_width, start2.at<double>(1)), cv::Point2f(end2.at<double>(0) + img1_width, end2.at<double>(1)), cv::Scalar(0, 255, 0), 3);

		// Plücker coord of l2
		cv::Mat l2 = start2.cross(end2);

		/////////////////////////////
		// 1. Compute epipolar line from start/end point of line l1 in img2
		cv::Mat epipolarLine_start1 = F12 * start1;
		cv::Mat epipolarLine_end1 = F12 * end1;


		// Draw start1/end1 on img1
		cv::circle(cv_imgDebug, cv::Point2f(start1.at<double>(0), start1.at<double>(1)), 10, cv::Scalar(255, 0, 0), 5);
		cv::circle(cv_imgDebug, cv::Point2f(end1.at<double>(0), end1.at<double>(1)), 10, cv::Scalar(0, 0, 255), 5);

		// 2. Compute intersection of l1_se* and line l2 -> estimate start point of l2 in img2 s2*
		cv::Mat intersect_start = epipolarLine_start1.cross(l2);
		cv::Mat intersect_end = epipolarLine_end1.cross(l2);
		// Draw intersecting points on img2
		cv::circle(cv_imgDebug, cv::Point2f(intersect_start.at<double>(0), intersect_start.at<double>(1)), 10, cv::Scalar(255, 0, 0), 5);
		cv::circle(cv_imgDebug, cv::Point2f(intersect_end.at<double>(0), intersect_end.at<double>(1)), 10, cv::Scalar(0, 0, 255), 5);

		// 3. Solve for 3D point S using s1 and s2* (same as point triangulation)
		um1 = Kinv * start1;
		um2 = Kinv * intersect_start;
		u1 = cv::Point3f((float) um1.at<double>(0), (float) um1.at<double>(1), (float) um1.at<double>(2));
		u2 = cv::Point3f((float) um2.at<double>(0), (float) um2.at<double>(1), (float) um2.at<double>(2));
		LOG_INFO("um1: {}", um1);

		cv::Mat S = iterativeLinearTriangulation(u1, KPose1, u2, KPose2);

		LOG_INFO("S: {}", S);
		um1 = Kinv * end1;
		um2 = Kinv * intersect_end;
		u1 = cv::Point3f((float)um1.at<double>(0), (float)um1.at<double>(1), (float)um1.at<double>(2));
		u2 = cv::Point3f((float)um2.at<double>(0), (float)um2.at<double>(1), (float)um2.at<double>(2));

		cv::Mat E = iterativeLinearTriangulation(u1, KPose1, u2, KPose2);

		Edge3Df line3D;
		line3D.p1.setX(S.at<double>(0));
		line3D.p1.setY(S.at<double>(1));
		line3D.p1.setZ(S.at<double>(2));
		line3D.p2.setX(E.at<double>(0));
		line3D.p2.setY(E.at<double>(1));
		line3D.p2.setZ(E.at<double>(2));

		LOG_INFO("line3D.p1: {}", line3D.p1);

		lines3D.push_back(line3D);
		indices.push_back(i);

		// reproject S in img1 and img2
		cv::Mat s1r_m = KPose1 * S;
		LOG_INFO("s1r_m: {}", s1r_m);
		cv::Point3f s1r((float) s1r_m.at<double>(0) / s1r_m.at<double>(2), (float) s1r_m.at<double>(1) / s1r_m.at<double>(2), 1);
		LOG_INFO("s1r: {}", s1r);
		cv::circle(cv_imgDebug, cv::Point2f(s1r.x, s1r.y), 20, cv::Scalar(255, 0, 0), 5);
		cv::Mat s2r_m = KPose2 * S;
		LOG_INFO("s2r_m: {}", s2r_m);
		cv::Point3f s2r((float) s2r_m.at<double>(0) / s2r_m.at<double>(2), (float) s2r_m.at<double>(1) / s2r_m.at<double>(2), 1);
		LOG_INFO("s2r: {}", s2r);
		cv::circle(cv_imgDebug, cv::Point2f(s2r.x + img1_width, s2r.y), 20, cv::Scalar(255, 0, 0), 5);

	}
	return 0;
}


void drawTriangulation(	const std::vector<Keyline> & keylines1,
						const std::vector<Keyline> & keylines2,
						const std::vector<DescriptorMatch> & matches,
						const Transform3Df & pose1,
						const Transform3Df & pose2,
						const CamCalibration & intrinsicParams,
						const std::vector<Edge3Df> & lines3D,
						const std::vector<int> & indices,
						const SRef<Image> & img1, const SRef<Image> & img2,
						SRef<Image> & imgDebug)
{
	if (imgDebug == nullptr)
	{
		imgDebug = xpcf::utils::make_shared<Image>(img1->getWidth() + img2->getWidth(), std::max(img1->getHeight(), img2->getHeight()), img1->getImageLayout(), img1->getPixelOrder(), img1->getDataType());
	}
	else if ((imgDebug->getWidth() != img1->getWidth() + img2->getWidth()) || (imgDebug->getHeight() != std::max(img1->getHeight(), img2->getHeight())))
	{
		imgDebug->setSize(img1->getWidth() + img2->getWidth(), std::max(img1->getHeight(), img2->getHeight()));
	}
	cv::Mat cv_img1, cv_img2, cv_imgDebug;
	cv_img1 = SolAROpenCVHelper::mapToOpenCV(img1);
	cv_img2 = SolAROpenCVHelper::mapToOpenCV(img2);
	cv_imgDebug = SolAROpenCVHelper::mapToOpenCV(imgDebug);
	cv_imgDebug.setTo(0);
	int img1_width = img1->getWidth();
	cv_img1.copyTo(cv_imgDebug(cv::Rect(0, 0, img1_width, img1->getHeight())));
	cv_img2.copyTo(cv_imgDebug(cv::Rect(img1_width, 0, img2->getWidth(), img2->getHeight())));

	for (unsigned i = 0; i < indices.size(); i++)
	{
		Keyline kl1 = keylines1[matches[indices[i]].getIndexInDescriptorA()];
		Keyline kl2 = keylines2[matches[indices[i]].getIndexInDescriptorB()];
		Edge3Df line3D = lines3D[i];

		// Draw l1 and l2 on their respective image
		cv::line(cv_imgDebug, cv::Point2f(kl1.getStartPointX(), kl1.getStartPointY()), cv::Point2f(kl1.getEndPointX(), kl1.getEndPointY()), cv::Scalar(0, 255, 0), 3);
		cv::line(cv_imgDebug, cv::Point2f(kl2.getStartPointX() + img1_width, kl2.getStartPointY()), cv::Point2f(kl2.getEndPointX() + img1_width, kl2.getEndPointY()), cv::Scalar(0, 255, 0), 3);

		// Reproject 3D triangulated line in both images
		cv::Mat_<double> m_camMatrix;
		m_camMatrix.create(3, 3);
		m_camMatrix.at<double>(0, 0) = (double)intrinsicParams(0, 0);
		m_camMatrix.at<double>(0, 1) = (double)intrinsicParams(0, 1);
		m_camMatrix.at<double>(0, 2) = (double)intrinsicParams(0, 2);
		m_camMatrix.at<double>(1, 0) = (double)intrinsicParams(1, 0);
		m_camMatrix.at<double>(1, 1) = (double)intrinsicParams(1, 1);
		m_camMatrix.at<double>(1, 2) = (double)intrinsicParams(1, 2);
		m_camMatrix.at<double>(2, 0) = (double)intrinsicParams(2, 0);
		m_camMatrix.at<double>(2, 1) = (double)intrinsicParams(2, 1);
		m_camMatrix.at<double>(2, 2) = (double)intrinsicParams(2, 2);

		cv::Matx44d Pose1(pose1(0, 0), pose1(0, 1), pose1(0, 2), pose1(0, 3),
			pose1(1, 0), pose1(1, 1), pose1(1, 2), pose1(1, 3),
			pose1(2, 0), pose1(2, 1), pose1(2, 2), pose1(2, 3),
			pose1(3, 0), pose1(3, 1), pose1(3, 2), pose1(3, 3));

		cv::Matx44d Pose2(pose2(0, 0), pose2(0, 1), pose2(0, 2), pose2(0, 3),
			pose2(1, 0), pose2(1, 1), pose2(1, 2), pose2(1, 3),
			pose2(2, 0), pose2(2, 1), pose2(2, 2), pose2(2, 3),
			pose2(3, 0), pose2(3, 1), pose2(3, 2), pose2(3, 3));

		cv::Mat P1 = cv::Mat(Pose1);
		cv::Mat P2 = cv::Mat(Pose2);

		cv::Mat pose1Inv = P1.inv();
		cv::Mat pose2Inv = P2.inv();

		cv::Mat KPose1 = m_camMatrix * (pose1Inv.rowRange(0, 3));
		cv::Mat KPose2 = m_camMatrix * (pose2Inv.rowRange(0, 3));

		cv::Mat startPoint	= (cv::Mat_<double>(4, 1) << line3D.p1.getX(), line3D.p1.getY(), line3D.p1.getZ(), 1);
		cv::Mat endPoint	= (cv::Mat_<double>(4, 1) << line3D.p2.getX(), line3D.p2.getY(), line3D.p2.getZ(), 1);


		cv::Mat s1 = KPose1 * startPoint;
		cv::Mat e1 = KPose1 * endPoint;
		cv::Mat s2 = KPose2 * startPoint;
		cv::Mat e2 = KPose2 * endPoint;

		cv::Point2f start2D1((float) s1.at<double>(0) / s1.at<double>(2), (float) s1.at<double>(1) / s1.at<double>(2));
		cv::Point2f end2D1((float) e1.at<double>(0) / e1.at<double>(2), (float) e1.at<double>(1) / e1.at<double>(2));
		cv::Point2f start2D2((float) s2.at<double>(0) / s2.at<double>(2) + img1_width, (float) s2.at<double>(1) / s2.at<double>(2));
		cv::Point2f end2D2((float) e2.at<double>(0) / e2.at<double>(2) + img1_width, (float) e2.at<double>(1) / e2.at<double>(2));

		cv::line(cv_imgDebug, start2D1, end2D1, cv::Scalar(255, 0, 0), 2);
		cv::line(cv_imgDebug, start2D2, end2D2, cv::Scalar(255, 0, 0), 2);
	}


}

int main(int argc, char *argv[])
{
#if NDEBUG
    boost::log::core::get()->set_logging_enabled(false);
#endif
	LOG_ADD_LOG_TO_CONSOLE();

	try
	{
		/* instantiate component manager*/
		/* this is needed in dynamic mode */
		SRef<xpcf::IComponentManager> xpcfComponentManager = xpcf::getComponentManagerInstance();

		if(xpcfComponentManager->load("SolARKeylineTriangulator_config.xml") != org::bcom::xpcf::_SUCCESS)
		{
			LOG_ERROR("Failed to load the configuration file SolARKeylineTriangulator_config.xml");
			return -1;
		}

		// declare and create components
        LOG_INFO("Start creating components");

		auto camera = xpcfComponentManager->resolve<input::devices::ICamera>();
		auto descriptorsExtractor = xpcfComponentManager->resolve<features::IDescriptorsExtractorBinary>();
		auto descriptorMatcher = xpcfComponentManager->create<SolARDescriptorMatcherBinaryOpencv>()->bindTo<features::IDescriptorMatcher>();
		auto matchesFilter = xpcfComponentManager->resolve<features::IMatchesFilter>();
		auto triangulator = xpcfComponentManager->resolve<solver::map::ITriangulator>();
		auto overlay2D = xpcfComponentManager->resolve<display::I2DOverlay>();
		auto overlay3D = xpcfComponentManager->create<SolAR3DOverlayBoxOpencv>()->bindTo<display::I3DOverlay>();
		auto viewer = xpcfComponentManager->create<SolARImageViewerOpencv>("release")->bindTo<display::IImageViewer>();
		auto viewerDebug = xpcfComponentManager->create<SolARImageViewerOpencv>("debug")->bindTo<display::IImageViewer>();
		auto viewer3D = xpcfComponentManager->resolve<display::I3DPointsViewer>();
		// Fiducial Marker
		auto binaryMarker = xpcfComponentManager->resolve<input::files::IMarker2DSquaredBinary>();
		auto imageFilterBinary = xpcfComponentManager->resolve<image::IImageFilter>();
		auto imageConvertor = xpcfComponentManager->resolve<image::IImageConvertor>();
		auto contoursExtractor = xpcfComponentManager->resolve<features::IContoursExtractor>();
		auto contoursFilter = xpcfComponentManager->resolve<features::IContoursFilter>();
		auto perspectiveController = xpcfComponentManager->resolve<image::IPerspectiveController>();
		auto patternDescriptorExtractor = xpcfComponentManager->resolve<features::IDescriptorsExtractorSBPattern>();
		auto patternMatcher = xpcfComponentManager->create<SolARDescriptorMatcherRadiusOpencv>()->bindTo<features::IDescriptorMatcher>();
		auto patternReIndexer = xpcfComponentManager->resolve<features::ISBPatternReIndexer>();
		auto img2worldMapper = xpcfComponentManager->resolve<geom::IImage2WorldMapper>();
		auto pnp = xpcfComponentManager->resolve<solver::pose::I3DTransformFinderFrom2D3D>();

        LOG_DEBUG("Components created!");

		// Init with camera intrinsics
		pnp->setCameraParameters(camera->getIntrinsicsParameters(), camera->getDistorsionParameters());
		triangulator->setCameraParameters(camera->getIntrinsicsParameters(), camera->getDistorsionParameters());
		overlay3D->setCameraParameters(camera->getIntrinsicsParameters(), camera->getDistorsionParameters());

		// Components initialisation for marker detection
		SRef<DescriptorBuffer> markerPatternDescriptor;
		binaryMarker->loadMarker();
		patternDescriptorExtractor->extract(binaryMarker->getPattern(), markerPatternDescriptor);
		LOG_DEBUG("Marker pattern:\n {}", binaryMarker->getPattern().getPatternMatrix());
		int patternSize = binaryMarker->getPattern().getSize();
		patternDescriptorExtractor->bindTo<xpcf::IConfigurable>()->getProperty("patternSize")->setIntegerValue(patternSize);
		patternReIndexer->bindTo<xpcf::IConfigurable>()->getProperty("sbPatternSize")->setIntegerValue(patternSize);
		img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("digitalWidth")->setIntegerValue(patternSize);
		img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("digitalHeight")->setIntegerValue(patternSize);
		img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("worldWidth")->setFloatingValue(binaryMarker->getSize().width);
		img2worldMapper->bindTo<xpcf::IConfigurable>()->getProperty("worldHeight")->setFloatingValue(binaryMarker->getSize().height);

		bool stop = false;
		int count = 0;
		clock_t start, end;
		xpcf::DropBuffer<SRef<Image>> m_markerPoseBuffer;
		xpcf::DropBuffer<std::pair<SRef<Image>, Transform3Df>> m_detectionBuffer;
		xpcf::DropBuffer<std::tuple<SRef<Image>, Transform3Df, std::vector<Keyline>, SRef<DescriptorBuffer>>> m_triangulationBuffer;
		xpcf::DropBuffer<SRef<Image>> m_displayBuffer;
		xpcf::DropBuffer<SRef<Image>> m_displayDebugBuffer;
		xpcf::DropBuffer<std::pair<std::vector<Edge3Df>, Transform3Df>> m_display3DBuffer;

		// Pose estimation using fiducial marker
		auto detectFiducialMarker = [&imageConvertor, &imageFilterBinary, &contoursExtractor, &contoursFilter, &perspectiveController,
			&patternDescriptorExtractor, &patternMatcher, &markerPatternDescriptor, &patternReIndexer, &img2worldMapper, &pnp, &overlay3D](SRef<Image>& image, Transform3Df &pose) {
			SRef<Image>                     greyImage, binaryImage;
			std::vector<Contour2Df>   contours;
			std::vector<Contour2Df>   filtered_contours;
			std::vector<SRef<Image>>        patches;
			std::vector<Contour2Df>   recognizedContours;
			SRef<DescriptorBuffer>          recognizedPatternsDescriptors;
			std::vector<DescriptorMatch>    patternMatches;
			std::vector<Point2Df>     pattern2DPoints;
			std::vector<Point2Df>     img2DPoints;
			std::vector<Point3Df>     pattern3DPoints;

			bool marker_found = false;
			// Convert Image from RGB to grey
			imageConvertor->convert(image, greyImage, Image::ImageLayout::LAYOUT_GREY);
			for (int num_threshold = 0; !marker_found && num_threshold < NB_THRESHOLD; num_threshold++)
			{
				// Compute the current Threshold valu for image binarization
				int threshold = MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD)*((float)num_threshold / (float)(NB_THRESHOLD - 1));
				// Convert Image from grey to black and white
				imageFilterBinary->bindTo<xpcf::IConfigurable>()->getProperty("min")->setIntegerValue(threshold);
				imageFilterBinary->bindTo<xpcf::IConfigurable>()->getProperty("max")->setIntegerValue(255);
				// Convert Image from grey to black and white
				imageFilterBinary->filter(greyImage, binaryImage);
				// Extract contours from binary image
				contoursExtractor->extract(binaryImage, contours);
				// Filter 4 edges contours to find those candidate for marker contours
				contoursFilter->filter(contours, filtered_contours);
				// Create one warpped and cropped image by contour
				perspectiveController->correct(binaryImage, filtered_contours, patches);
				// test if this last image is really a squared binary marker, and if it is the case, extract its descriptor
				if (patternDescriptorExtractor->extract(patches, filtered_contours, recognizedPatternsDescriptors, recognizedContours) != FrameworkReturnCode::_ERROR_)
				{
					// From extracted squared binary pattern, match the one corresponding to the squared binary marker
					if (patternMatcher->match(markerPatternDescriptor, recognizedPatternsDescriptors, patternMatches) == features::IDescriptorMatcher::DESCRIPTORS_MATCHER_OK)
					{
						// Reindex the pattern to create two vector of points, the first one corresponding to marker corner, the second one corresponding to the poitsn of the contour
						patternReIndexer->reindex(recognizedContours, patternMatches, pattern2DPoints, img2DPoints);
						// Compute the 3D position of each corner of the marker
						img2worldMapper->map(pattern2DPoints, pattern3DPoints);
						// Compute the pose of the camera using a Perspective n Points algorithm using only the 4 corners of the marker
						if (pnp->estimate(img2DPoints, pattern3DPoints, pose) == FrameworkReturnCode::_SUCCESS)
						{
							marker_found = true;
						}
					}
				}
			}
			return marker_found;
		};

		// Camera image capture task
		auto fnCapture = [&]()
		{
			SRef<Image> image;
			if (camera->getNextImage(image) != SolAR::FrameworkReturnCode::_SUCCESS)
			{
				stop = true;
				return;
			}
			m_markerPoseBuffer.push(image);
		};

		auto fnMarkerPose = [&]()
		{
			SRef<Image> image;
			if (!m_markerPoseBuffer.tryPop(image))
			{
				xpcf::DelegateTask::yield();
				return;
			}
			Transform3Df pose;
			if (detectFiducialMarker(image, pose))
			{
				m_detectionBuffer.push(std::make_pair(image, pose));
				overlay3D->draw(pose, image);
				LOG_DEBUG("found marker");
			}
			m_displayBuffer.push(image);
		};

		// Features extraction task
		auto fnDetection = [&]()
		{
			std::pair<SRef<Image>, Transform3Df> bufferOutput;
			if (!m_detectionBuffer.tryPop(bufferOutput))
			{
				xpcf::DelegateTask::yield();
				return;
			}
			SRef<Image> image = bufferOutput.first;
			Transform3Df markerPose = bufferOutput.second;
			std::vector<Keyline> keylines;
			SRef<DescriptorBuffer> descriptors;
			descriptorsExtractor->compute(image, keylines, descriptors);
			m_triangulationBuffer.push(std::make_tuple(image, markerPose, keylines, descriptors));
		};
		 
		// Triangulation task
		SRef<Image> previousImage;
		Transform3Df previousPose;
		std::vector<Keyline> previousKeylines;
		SRef<DescriptorBuffer> previousDescriptors;
		SRef<Image> imgDebug;
		bool initDone = false;
		auto fnTriangulation = [&]()
		{
			std::tuple<SRef<Image>, Transform3Df, std::vector<Keyline>, SRef<DescriptorBuffer>> bufferOutput;
			if (!m_triangulationBuffer.tryPop(bufferOutput))
			{
				xpcf::DelegateTask::yield();
				return;
			}
			SRef<Image> image = std::get<0>(bufferOutput);
			Transform3Df pose = std::get<1>(bufferOutput);
			std::vector<Keyline> keylines = std::get<2>(bufferOutput);
			SRef<DescriptorBuffer> descriptors = std::get<3>(bufferOutput);
			std::vector<DescriptorMatch> matches;
			std::vector<DescriptorMatch> outMatches;
			std::vector<Edge3Df> lines3D;
			std::vector<int> indices;
			double error;
			// Init values on first frame
			if (!initDone)
			{
				initDone = true;
				previousImage = image;
				previousPose = pose;
				previousKeylines = keylines;
				previousDescriptors = descriptors;
			}
			else
			{
				float disPoses = std::sqrtf(std::powf(pose(0, 3) - previousPose(0, 3), 2.f) + std::powf(pose(1, 3) - previousPose(1, 3), 2.f) +
					std::powf(pose(2, 3) - previousPose(2, 3), 2.f));

				if (disPoses > 1.0f)
				{
					// Matching
					descriptorMatcher->match(descriptors, previousDescriptors, matches);
					LOG_INFO("matches size: {}", matches.size());
					// Filter out obvious outliers
					matchesFilter->filter(matches, outMatches, keylines, previousKeylines);
					LOG_INFO("outMatches size: {}", outMatches.size());
					// Triangulate
					//triangulate(keylines, previousKeylines, outMatches, pose, previousPose, camera->getIntrinsicsParameters(), lines3D, indices, image, previousImage, imgDebug);
					error = triangulator->triangulate(keylines, previousKeylines, outMatches, pose, previousPose, lines3D, indices);
					LOG_INFO("lines3D size: {}", lines3D.size());
					//LOG_INFO("reproj error: {}", error);
					drawTriangulation(keylines, previousKeylines, outMatches, pose, previousPose, camera->getIntrinsicsParameters(), lines3D, indices, image, previousImage, imgDebug);
					m_displayDebugBuffer.push(imgDebug);

					m_display3DBuffer.push(std::make_pair(lines3D, pose));

					// Push current data to previous data
					previousImage = image;
					previousPose = pose;
					previousKeylines = keylines;
					previousDescriptors = descriptors;
				}
			}
		};

		// Display task
		auto fnDisplay = [&]()
		{
			SRef<Image> image;
			if (!m_displayBuffer.tryPop(image))
			{
				xpcf::DelegateTask::yield();
				return;
			}
			count++;
			if (viewer->display(image) == FrameworkReturnCode::_STOP)
				stop = true;
		};

		// Display Debug task
		auto fnDisplayDebug = [&]()
		{
			SRef<Image> image;
			if (!m_displayDebugBuffer.tryPop(image))
			{
				xpcf::DelegateTask::yield();
				return;
			}
			if (viewerDebug->display(image) == FrameworkReturnCode::_STOP)
				stop = true;
		};

		// Display 3D lines task
		Transform3Df poseView;
		std::vector<Edge3Df> lines3Dview;
		bool viewerInit = false;
		auto fnDisplay3D = [&]()
		{
			std::pair<std::vector<Edge3Df>, Transform3Df> bufferOutput;
			if (!m_display3DBuffer.tryPop(bufferOutput))
			{
				if (viewerInit)
					if (viewer3D->display(lines3Dview, poseView) == FrameworkReturnCode::_STOP)
						stop = true;
				xpcf::DelegateTask::yield();
				return;
			}
			if (!viewerInit) viewerInit = true;
			
			lines3Dview = bufferOutput.first;
			poseView = bufferOutput.second;
			if (viewer3D->display(lines3Dview, poseView) == FrameworkReturnCode::_STOP)
				stop = true;
		};

		// Init camera
		if (camera->start() != FrameworkReturnCode::_SUCCESS)
		{
			LOG_ERROR("Camera cannot start");
			return -1;
		}
		// Create tasks
		xpcf::DelegateTask taskCapture(fnCapture);
		xpcf::DelegateTask taskMarkerPose(fnMarkerPose);
		xpcf::DelegateTask taskDetection(fnDetection);
		xpcf::DelegateTask taskTriangulation(fnTriangulation);
		// Start tasks
		taskCapture.start();
		taskMarkerPose.start();
		taskDetection.start();
		taskTriangulation.start();
		// Main loop, press escape key to exit
		start = clock();
		while (!stop)
		{
			fnDisplay();
			fnDisplayDebug();
			fnDisplay3D();
		}
		// Stop tasks
		taskCapture.stop();
		taskMarkerPose.stop();
		taskDetection.stop();
		taskTriangulation.stop();
		// Time measure
		end = clock();
		double duration = double(end - start) / CLOCKS_PER_SEC;
		printf("\n\nElasped time is %.2lf seconds.\n", duration);
		printf("Number of processed frames per second : %8.2f\n\n", count / duration);
	}
	catch (xpcf::Exception &e)
	{
		LOG_ERROR("{}", e.what());
		return -1;
	}
	return 0;
}
