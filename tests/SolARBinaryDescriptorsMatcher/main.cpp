
#include <xpcf/xpcf.h>
#include <boost/log/core.hpp>
#include <core/Log.h>

#include "api/display/IImageViewer.h"
#include "api/display/IMatchesOverlay.h"
#include "api/features/IDescriptorMatcher.h"
#include "api/features/IDescriptorsExtractorBinary.h"
#include "api/features/IKeylineDetector.h"
#include "api/features/IMatchesFilter.h"
#include "api/image/IImageLoader.h"
#include "api/input/devices/ICamera.h"

#include "SolARNonFreeOpenCVHelper.h"

namespace xpcf=org::bcom::xpcf;

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;

#define WEBCAM 0

/**
 * Declare module.
 */
int main()
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

#if WEBCAM
		auto camera = xpcfComponentManager->resolve<input::devices::ICamera>();
#else
		auto imageLoader1 = xpcfComponentManager->resolve<image::IImageLoader>("image1");
		auto imageLoader2 = xpcfComponentManager->resolve<image::IImageLoader>("image1");
#endif
		auto keylineDetector = xpcfComponentManager->resolve<features::IKeylineDetector>();
		auto descriptorsExtractor = xpcfComponentManager->resolve<features::IDescriptorsExtractorBinary>();
		auto descriptorsMatcher = xpcfComponentManager->resolve<features::IDescriptorMatcher>();
		auto matchesFilter = xpcfComponentManager->resolve<features::IMatchesFilter>();
		auto matchesOverlay = xpcfComponentManager->resolve<display::IMatchesOverlay>();
		auto viewer = xpcfComponentManager->resolve<display::IImageViewer>();

        LOG_DEBUG("Components created!");
		
		descriptorsExtractor->setDetector(keylineDetector);

		SRef<Image> image;
		SRef<Image> previousImage;
		SRef<Image> outImage;
		std::vector<Keyline> keylines;
		std::vector<Keyline> previousKeylines;
		SRef<DescriptorBuffer> descriptors;
		SRef<DescriptorBuffer> previousDescriptors;
		std::vector<DescriptorMatch> matches;
		std::vector<DescriptorMatch> outMatches;

		int count = 0;
		clock_t start, end;
		start = clock();
#if WEBCAM
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
				LOG_DEBUG("matches size: {}", matches.size());
				// Filter out obvious outliers
				matchesFilter->filter(matches, outMatches, keylines, previousKeylines);
				LOG_DEBUG("outMatches size: {}", outMatches.size());
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
#else // !WEBCAM
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
		count++;
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
