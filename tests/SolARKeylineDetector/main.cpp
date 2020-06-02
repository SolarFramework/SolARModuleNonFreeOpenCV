
#include <xpcf/xpcf.h>
#include <boost/log/core.hpp>
#include <core/Log.h>

#include "SolARModuleOpencv_traits.h"

#include "api/display/I2DOverlay.h"
#include "api/display/IImageViewer.h"
#include "api/features/IKeylineDetector.h"
#include "api/image/IImageLoader.h"
#include "api/input/devices/ICamera.h"

#include "SolAROpenCVHelper.h"

#include <opencv2/line_descriptor.hpp>

namespace xpcf=org::bcom::xpcf;

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;

#define WEBCAM 1

/**
 * Declare module.
 */
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

		if(xpcfComponentManager->load("SolARKeylineDetector_config.xml") != org::bcom::xpcf::_SUCCESS)
		{
			LOG_ERROR("Failed to load the configuration file SolARKeylineDetector_config.xml");
			return -1;
		}

		// declare and create components
        LOG_INFO("Start creating components");

		SRef<input::devices::ICamera> camera = xpcfComponentManager->resolve<input::devices::ICamera>();
		SRef<image::IImageLoader> imageLoader = xpcfComponentManager->resolve<image::IImageLoader>();
		SRef<features::IKeylineDetector> keylineDetector = xpcfComponentManager->resolve<features::IKeylineDetector>();
		SRef<display::I2DOverlay> overlay = xpcfComponentManager->resolve<display::I2DOverlay>();
		SRef<display::IImageViewer> viewer = xpcfComponentManager->resolve<display::IImageViewer>();

        LOG_DEBUG("Components created!");

		SRef<Image> image;
		cv::Mat opencvImage;
		std::vector<Keyline> keylines;

#if WEBCAM
		// Init camera
		if (camera->start() != FrameworkReturnCode::_SUCCESS)
		{
			LOG_ERROR("Camera cannot start");
			return -1;
		}
		int count = 0;
		clock_t start, end;
		start = clock();
		// Main loop, press escape key to exit
		while (true)
		{
			// Read image from camera
			if (camera->getNextImage(image) == FrameworkReturnCode::_ERROR_)
				break;
			count++;
			// Detect keylines in image
			keylineDetector->detect(image, keylines);
			// Draw detected keylines
			overlay->drawLines(keylines, image);
			// Display the image with matches in a viewer. If escape key is pressed, exit the loop.
			if (viewer->display(image) == FrameworkReturnCode::_STOP)
			{
				LOG_INFO("End of SolARKeylineDetector test");
				break;
			}
		}
		end = clock();
		double duration = double(end - start) / CLOCKS_PER_SEC;
		printf("\n\nElasped time is %.2lf seconds.\n", duration);
		printf("Number of processed frames per second : %8.2f\n\n", count / duration);
#else
		if (imageLoader->getImage(image) !=  FrameworkReturnCode::_SUCCESS)
		{
			LOG_WARNING("Image ({}) can't be loaded", imageLoader->bindTo<xpcf::IConfigurable>()->getProperty("filePath")->getStringValue());
			return 0;
		}
		// Detect keylines in image
		keylineDetector->detect(image, keylines);
		LOG_INFO("Detected {} lines.", keylines.size());
		// Draw detected keylines
		overlay->drawLines(keylines, image);
        // Display the image with matches in a viewer. If escape key is pressed, exit the loop.
        while (true)
            if (viewer->display(image) == FrameworkReturnCode::_STOP)
                break;
        LOG_INFO("End of SolARKeylineDetector test");
#endif
	}
	catch (xpcf::Exception &e)
	{
		LOG_ERROR("{}", e.what());
		return -1;
	}
	return 0;
}
