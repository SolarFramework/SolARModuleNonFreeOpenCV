/**
 * @copyright Copyright (c) 2020 B-com http://www.b-com.com/
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

#include "xpcf/xpcf.h"
#include "xpcf/threading/BaseTask.h"
#include "xpcf/threading/DropBuffer.h"
#include "core/Log.h"

#include "SolARModuleOpencv_traits.h"
#include "SolARModuleNonFreeOpencv_traits.h"

#include "api/display/I2DOverlay.h"
#include "api/display/I3DOverlay.h"
#include "api/display/I3DPointsViewer.h"
#include "api/display/IImageViewer.h"
#include "api/features/IContoursExtractor.h"
#include "api/features/IContoursFilter.h"
#include "api/features/IDescriptorMatcher.h"
#include "api/features/IDescriptorsExtractorBinary.h"
#include "api/features/IDescriptorsExtractorSBPattern.h"
#include "api/features/IKeylineDetector.h"
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

#include <boost/log/core.hpp>

using namespace SolAR;
using namespace SolAR::api;
using namespace SolAR::datastructure;
using namespace SolAR::MODULES::OPENCV;
using namespace SolAR::MODULES::NONFREEOPENCV;

namespace xpcf = org::bcom::xpcf;

constexpr int MIN_THRESHOLD{ -1 };
constexpr int MAX_THRESHOLD{ 220 };
constexpr int NB_THRESHOLD{ 8 };

int main(int argc, char *argv[])
{
#if NDEBUG
	boost::log::core::get()->set_logging_enabled(false);
#endif
	LOG_ADD_LOG_TO_CONSOLE();

	try
	{
		/*
		 * Initialization
		 */

		/* Instantiate component manager */
		SRef<xpcf::IComponentManager> xpcfComponentManager = xpcf::getComponentManagerInstance();

		std::string configFile = "SolARTest_ModuleNonFreeOpenCV_KeylineTriangulator_conf.xml";
		if (xpcfComponentManager->load("SolARTest_ModuleNonFreeOpenCV_KeylineTriangulator_conf.xml") != org::bcom::xpcf::_SUCCESS)
		{
			LOG_ERROR("Failed to load the configuration file {}", configFile);
			return -1;
		}

		/* Declare and create components */
		LOG_INFO("Start creating components");
        // Input
		auto camera = xpcfComponentManager->resolve<input::devices::ICamera>();
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
        // Processing
        auto keylineDetector = xpcfComponentManager->resolve<features::IKeylineDetector>();
		auto descriptorsExtractor = xpcfComponentManager->resolve<features::IDescriptorsExtractorBinary>();
        auto descriptorMatcher = xpcfComponentManager->create<SolARDescriptorMatcherBinaryOpencv>()->bindTo<features::IDescriptorMatcher>();
		auto matchesFilter = xpcfComponentManager->resolve<features::IMatchesFilter>();
		auto triangulator = xpcfComponentManager->resolve<solver::map::ITriangulator>();
        // Display
		auto overlay2D = xpcfComponentManager->resolve<display::I2DOverlay>();
		auto overlay3D = xpcfComponentManager->resolve<display::I3DOverlay>();
		auto viewer3D = xpcfComponentManager->resolve<display::I3DPointsViewer>();
		auto viewer = xpcfComponentManager->resolve<display::IImageViewer>();
        LOG_DEBUG("Components created!");

        /* Components initialization */
        // Component injection
        descriptorsExtractor->setDetector(keylineDetector);
        // Init camera intrinsics
        const CamCalibration camIntrinsics = camera->getIntrinsicsParameters();
        const CamDistortion camDistortion = camera->getDistortionParameters();
        overlay3D->setCameraParameters(camIntrinsics, camDistortion);
        pnp->setCameraParameters(camIntrinsics, camDistortion);
        triangulator->setCameraParameters(camIntrinsics, camDistortion);
        // Marker detection
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

        /* Global variables initialization */
		// Timers and FPS counter
        bool stop = false;
		int count = 0;
		clock_t start, end;
        // Buffers
		xpcf::DropBuffer<SRef<Image>> m_captureBuffer;
		xpcf::DropBuffer<SRef<Image>> m_displayBuffer;
		xpcf::DropBuffer<SRef<Frame>> m_featuresBuffer;
		xpcf::DropBuffer<std::pair<SRef<Image>, Transform3Df>> m_markerDetectionBuffer;
        using Buffer3D = std::tuple<std::vector<SRef<CloudLine>>, Transform3Df, Transform3Df>;
        xpcf::DropBuffer<Buffer3D> m_display3DBuffer;
		// Triangulation
		bool initDone{ false };
		SRef<Frame> previousFrame;
		// 3D viewer
		bool viewerInit{ false };
		std::vector<SRef<CloudLine>> lines3Dview;
		Transform3Df previousPose3Dview;
		Transform3Df currentPose3Dview;

        /*
         * Tasks and functions
         */
    
        /* Fiducial marker detection */
        auto detectFiducialMarker = [&](const SRef<Image>& image, Transform3Df &pose)
		{
			SRef<Image>                     greyImage, binaryImage;
			std::vector<Contour2Df>         contours;
			std::vector<Contour2Df>         filtered_contours;
			std::vector<SRef<Image>>        patches;
			std::vector<Contour2Df>         recognizedContours;
			SRef<DescriptorBuffer>          recognizedPatternsDescriptors;
			std::vector<DescriptorMatch>    patternMatches;
			std::vector<Point2Df>           pattern2DPoints;
			std::vector<Point2Df>           img2DPoints;
			std::vector<Point3Df>           pattern3DPoints;

			// Convert Image from RGB to grey
			imageConvertor->convert(image, greyImage, Image::ImageLayout::LAYOUT_GREY);
			for (int num_threshold = 0; num_threshold < NB_THRESHOLD; num_threshold++)
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
				// Test if this last image is really a squared binary marker, and if it is the case, extract its descriptor
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
							return true;
					}
				}
			}
			return false;
		};

        /* Camera image capture task */
		auto fnCapture = [&]()
		{
			SRef<Image> image;
			if (camera->getNextImage(image) != SolAR::FrameworkReturnCode::_SUCCESS)
			{
				stop = true;
				return;
			}
			m_captureBuffer.push(image);
		};

        /* Marker Pose Estimation task */
		auto fnMarkerPose = [&]()
		{
			SRef<Image> image, imageView;
			if (!m_captureBuffer.tryPop(image))
			{
				xpcf::DelegateTask::yield();
				return;
			}
			imageView = image->copy();
			Transform3Df pose;
			if (detectFiducialMarker(image, pose))
			{
				m_markerDetectionBuffer.push({ image, pose });
				overlay3D->draw(pose, imageView);
			}
			m_displayBuffer.push(imageView);
		};

		/* Features extraction task */
		auto fnDetection = [&]()
		{
			std::pair<SRef<Image>, Transform3Df> bufferOutput;
			if (!m_markerDetectionBuffer.tryPop(bufferOutput))
			{
				xpcf::DelegateTask::yield();
				return;
			}
			SRef<Image> image = bufferOutput.first;
			Transform3Df markerPose = bufferOutput.second;
			std::vector<Keyline> keylines;
			SRef<DescriptorBuffer> descriptors;
			descriptorsExtractor->compute(image, keylines, descriptors);
			SRef<Frame> frame = xpcf::utils::make_shared<Frame>(keylines, descriptors, image, markerPose);
			m_featuresBuffer.push(frame);
		};

		/* Feature matcher */
		auto matchFeatures = [&](const SRef<Frame> & frame1, const SRef<Frame> & frame2, std::vector<DescriptorMatch> & matches, bool filter = true)
		{
			// Matching
			descriptorMatcher->match(frame1->getDescriptorsLine(), frame2->getDescriptorsLine(), matches);
			LOG_INFO("matches size: {}", matches.size());
			if (filter)
			{
				// Filter out obvious outliers (using fundamental matrix)
				//matchesFilter->filter(matches, matches, frame1->getKeylines(), frame2->getKeylines());
				matchesFilter->filter(matches, matches, frame1->getKeylines(), frame2->getKeylines(), frame1->getPose(), frame2->getPose(), camIntrinsics);
				LOG_INFO("Filtered matches size: {}", matches.size());
			}
		};

		/* Pose difference check */
		auto checkPoseDiff = [&](const SRef<Frame> & frame1, const SRef<Frame> & frame2, float minDistance = 0.5f) -> bool
		{
			auto pose1 = frame1->getPose();
			auto pose2 = frame2->getPose();
			float poseDist = std::sqrt(
				std::pow(pose1(0, 3) - pose2(0, 3), 2.f) +
				std::pow(pose1(1, 3) - pose2(1, 3), 2.f) +
				std::pow(pose1(2, 3) - pose2(2, 3), 2.f)
			);
			return poseDist > minDistance;
		};

        /* Line triangulation task */
        auto fnTriangulation = [&]()
        {
            SRef<Frame> currentFrame;
			if (!m_featuresBuffer.tryPop(currentFrame))
			{
				xpcf::DelegateTask::yield();
				return;
			}
            // Skip first frame
            if (!initDone)
            {
                previousFrame = currentFrame;
                initDone = true;
                return;
            }
			else if (checkPoseDiff(previousFrame, currentFrame, 0.5f))
			{
				// Compute matches previous with current frame
				std::vector<DescriptorMatch> matches;
				matchFeatures(previousFrame, currentFrame, matches, true);
				// Triangulate line matches
				std::vector<SRef<CloudLine>> lineCloud;
				float error = triangulator->triangulate(
					previousFrame->getKeylines(), currentFrame->getKeylines(),
					previousFrame->getDescriptorsLine(), currentFrame->getDescriptorsLine(),
					matches, { 0, 1 },
					previousFrame->getPose(), currentFrame->getPose(),
					lineCloud
				);
                LOG_INFO("Triangulated {} lines (error={})", lineCloud.size(), error);
				// Display triangulated 3D lines
				m_display3DBuffer.push({ lineCloud, previousFrame->getPose(), currentFrame->getPose() });
				// Prepare next iteration
				previousFrame = currentFrame;
			}
        };

        /* Display task */
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

        /* Display in 3D viewer task */
        auto fnDisplay3D = [&]()
        {
            Buffer3D bufferOutput;
            if (!m_display3DBuffer.tryPop(bufferOutput))
			{
				if (viewerInit)
					if (viewer3D->display(lines3Dview, currentPose3Dview, { previousPose3Dview, currentPose3Dview }) == FrameworkReturnCode::_STOP)
						stop = true;
				xpcf::DelegateTask::yield();
				return;
			}
			if (!viewerInit)
				viewerInit = true;
            lines3Dview = std::get<0>(bufferOutput);
            previousPose3Dview = std::get<1>(bufferOutput);
            currentPose3Dview = std::get<2>(bufferOutput);
			if (viewer3D->display(lines3Dview, currentPose3Dview, { previousPose3Dview, currentPose3Dview }) == FrameworkReturnCode::_STOP)
				stop = true;
		};

        /*
         * Main loop
         */

        /* Camera initialization */
        if (camera->start() != FrameworkReturnCode::_SUCCESS)
		{
			LOG_ERROR("Camera cannot start");
			return -1;
		}
        /* Create tasks */
		xpcf::DelegateTask taskCapture(fnCapture);
        xpcf::DelegateTask taskMarkerPose(fnMarkerPose);
        xpcf::DelegateTask taskDetection(fnDetection);
		xpcf::DelegateTask taskTriangulation(fnTriangulation);
		xpcf::DelegateTask taskDisplay(fnDisplay);
        /* Start tasks */
        taskCapture.start();
        taskMarkerPose.start();
        taskDetection.start();
		taskTriangulation.start();
		taskDisplay.start();
        /* Start main loop */
        start = clock();
        while (!stop)
        {
            fnDisplay3D();
        }
        /* Stop tasks */
        taskCapture.stop();
        taskMarkerPose.stop();
        taskDetection.stop();
		taskTriangulation.stop();
		taskDisplay.stop();
        /* Stop timers and count FPS */
        end = clock();
		double duration = double(end - start) / CLOCKS_PER_SEC;
		printf("\n\nElasped time is %.2lf seconds.\n", duration);
		printf("Number of processed frames per second : %8.2f\n\n", count / duration);
	}
    /* Exception handling */
	catch (xpcf::Exception &e)
	{
		LOG_ERROR("The following exception has been caught: {}", e.what());
		return -1;
	}
	return 0;
}