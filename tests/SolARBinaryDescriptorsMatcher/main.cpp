
#include <xpcf/xpcf.h>
#include <boost/log/core.hpp>
#include <core/Log.h>

#include "SolARModuleOpencv_traits.h"

#include "api/display/IImageViewer.h"
#include "api/display/IMatchesOverlay.h"
#include "api/features/IDescriptorMatcher.h"
#include "api/features/IDescriptorsExtractorBinary.h"
#include "api/features/IMatchesFilter.h"
#include "api/image/IImageLoader.h"
#include "api/input/devices/ICamera.h"

#include "SolAROpenCVHelper.h"

#include <opencv2/line_descriptor.hpp>

namespace xpcf=org::bcom::xpcf;

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;

#define WEBCAM

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

		if(xpcfComponentManager->load("SolARBinaryDescriptorsMatcher_config.xml") != org::bcom::xpcf::_SUCCESS)
		{
			LOG_ERROR("Failed to load the configuration file SolARBinaryDescriptorsMatcher_config.xml");
			return -1;
		}

		// declare and create components
        LOG_INFO("Start creating components");

		SRef<input::devices::ICamera> camera = xpcfComponentManager->resolve<input::devices::ICamera>();
		SRef<image::IImageLoader> imageLoader1 = xpcfComponentManager->resolve<image::IImageLoader>();
		imageLoader1->bindTo<xpcf::IConfigurable>()->configure("SolARBinaryDescriptorsMatcher_config.xml", "image1");
		SRef<image::IImageLoader> imageLoader2 = xpcfComponentManager->resolve<image::IImageLoader>();
		imageLoader2->bindTo<xpcf::IConfigurable>()->configure("SolARBinaryDescriptorsMatcher_config.xml", "image2");
		SRef<features::IDescriptorsExtractorBinary> descriptorsExtractor = xpcfComponentManager->resolve<features::IDescriptorsExtractorBinary>();
		SRef<features::IDescriptorMatcher> descriptorsMatcher = xpcfComponentManager->resolve<features::IDescriptorMatcher>();
		SRef<features::IMatchesFilter> matchesFilter = xpcfComponentManager->resolve<features::IMatchesFilter>();
		SRef<display::IMatchesOverlay> matchesOverlay = xpcfComponentManager->resolve<display::IMatchesOverlay>();
		SRef<display::IImageViewer> viewer = xpcfComponentManager->resolve<display::IImageViewer>();

        LOG_DEBUG("Components created!");

		SRef<Image> image;
		SRef<Image> previousImage;
		SRef<Image> outImage;
		std::vector<Keyline> keylines;
		std::vector<Keyline> previousKeylines;
		SRef<DescriptorBuffer> descriptors;
		SRef<DescriptorBuffer> previousDescriptors;
		std::vector<DescriptorMatch> matches;
		std::vector<DescriptorMatch> outMatches;

#ifdef WEBCAM
		int count = 0;
		clock_t start, end;
		start = clock();
		// Init camera
		if (camera->start() != FrameworkReturnCode::_SUCCESS)
		{
			LOG_ERROR("Camera cannot start");
			return -1;
		}
		// Main loop, press escape key to exit
		bool init = false;
		while (true)
		{
			// Read image from camera
			if (camera->getNextImage(image) == FrameworkReturnCode::_ERROR_)
				break;
			count++;

			if (!init)
			{
				// Init values on first frame
				init = true;
				previousImage = image;
				descriptorsExtractor->compute(previousImage, previousKeylines, previousDescriptors);
				matchesOverlay->draw(image, previousImage, outImage, keylines, previousKeylines, matches);
			}
			else
			{
				// Feature extraction
				descriptorsExtractor->compute(image, keylines, descriptors);
				// Matching
				descriptorsMatcher->match(descriptors, previousDescriptors, matches);
				LOG_INFO("matches size: {}", matches.size());
				// Filter out obvious outliers
				matchesFilter->filter(matches, outMatches, keylines, previousKeylines);
				LOG_INFO("outMatches size: {}", outMatches.size());
				// Draw line correspondances
				matchesOverlay->draw(image, previousImage, outImage, keylines, previousKeylines, outMatches);
				// Push current data to previous data
				previousImage = image;
				previousKeylines = keylines;
				previousDescriptors = descriptors;
			}
			// Display the image with matches in a viewer. If escape key is pressed, exit the loop.
			if (viewer->display(outImage) == FrameworkReturnCode::_STOP)
			{
				LOG_INFO("End of SolARBinaryDescriptorsMatcher test");
				break;
			}
		}
		end = clock();
		double duration = double(end - start) / CLOCKS_PER_SEC;
		printf("\n\nElasped time is %.2lf seconds.\n", duration);
		printf("Number of processed frames per second : %8.2f\n\n", count / duration);
#else // !WEBCAM //
		// Load images
		if (imageLoader1->getImage(image) !=  FrameworkReturnCode::_SUCCESS)
		{
			LOG_WARNING("Image 1 ({}) can't be loaded", imageLoader1->bindTo<xpcf::IConfigurable>()->getProperty("filePath")->getStringValue());
			return 0;
		}
		if (imageLoader2->getImage(previousImage) != FrameworkReturnCode::_SUCCESS)
		{
			LOG_WARNING("Image 2 ({}) can't be loaded", imageLoader2->bindTo<xpcf::IConfigurable>()->getProperty("filePath")->getStringValue());
			return 0;
		}
		// Line Features extraction
		descriptorsExtractor->compute(image, keylines, descriptors);
		descriptorsExtractor->compute(previousImage, previousKeylines, previousDescriptors);
		// Matching
		descriptorsMatcher->match(descriptors, previousDescriptors, matches);
		LOG_INFO("matches size: {}", matches.size());
		// Filter out obvious outliers
		matchesFilter->filter(matches, outMatches, keylines, previousKeylines);
		LOG_INFO("outMatches size: {}", outMatches.size());
		// Draw line correspondances
		matchesOverlay->draw(image, previousImage, outImage, keylines, previousKeylines, outMatches);
        // Display the image with matches in a viewer. If escape key is pressed, exit the loop.
		while (true)
			if (viewer->display(outImage) == FrameworkReturnCode::_STOP)
				break;
        LOG_INFO("End of SolARBinaryDescriptorsMatcher test");
#endif
	}
	catch (xpcf::Exception &e)
	{
		LOG_ERROR("{}", e.what());
		return -1;
	}
	return 0;
}
