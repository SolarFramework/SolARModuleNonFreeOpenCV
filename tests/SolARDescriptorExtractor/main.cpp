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

#include "xpcf/xpcf.h"

#include "api/image/IImageLoader.h"
#include "api/features/IKeypointDetector.h"
#include "api/features/IDescriptorsExtractor.h"
#include "api/display/I2DOverlay.h"
#include "api/display/IImageViewer.h"
#include "core/Log.h"

#include <boost/log/core.hpp>

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;

namespace xpcf  = org::bcom::xpcf;

int main(int argc,char** argv)
{
#if NDEBUG
    boost::log::core::get()->set_logging_enabled(false);
#endif

    LOG_ADD_LOG_TO_CONSOLE();
    try {

        /* instantiate component manager*/
        /* this is needed in dynamic mode */
        SRef<xpcf::IComponentManager> xpcfComponentManager = xpcf::getComponentManagerInstance();

        if(xpcfComponentManager->load("SolAROpenCVNonFreeDescriptorExtractor_conf.xml")!=org::bcom::xpcf::_SUCCESS)
        {
            LOG_ERROR("Failed to load the configuration file SolAROpenCVNonFreeDescriptorExtractor_conf.xml")
            return -1;
        }

        // declare and create components
        LOG_INFO("Start creating components");

        SRef<image::IImageLoader> imageLoader = xpcfComponentManager->resolve<image::IImageLoader>();
        SRef<features::IKeypointDetector> keypointsDetector = xpcfComponentManager->resolve<features::IKeypointDetector>();
        SRef<features::IDescriptorsExtractor> extractorSIFT = xpcfComponentManager->resolve<features::IDescriptorsExtractor>();
        SRef<display::I2DOverlay> overlay = xpcfComponentManager->resolve<display::I2DOverlay>();
        SRef<display::IImageViewer> viewer = xpcfComponentManager->resolve<display::IImageViewer>();

        if (!imageLoader  || !keypointsDetector || !extractorSIFT || !overlay || !viewer)
        {
            LOG_ERROR("One or more component creations have failed");
            return -1;
        }


        SRef<Image>             testImage;
        std::vector<Keypoint>   keypoints;
        SRef<DescriptorBuffer>  descriptors;

        // components initialisation
            // nothing to do

        // Start
        if (imageLoader->getImage(testImage) != FrameworkReturnCode::_SUCCESS)
        {
           LOG_ERROR("Cannot load image with path {}", imageLoader->bindTo<xpcf::IConfigurable>()->getProperty("filePath")->getStringValue());
           return -1;
        }

        keypointsDetector->detect(testImage, keypoints);
        extractorSIFT->extract(testImage, keypoints, descriptors);
        overlay->drawCircles(keypoints, testImage);

        while (true)
        {
            if (viewer->display(testImage) == FrameworkReturnCode::_STOP)
            {
                LOG_INFO("end of DescriptorsExtractorOpencv test");
                break;
            }
        }
    }
    catch (xpcf::Exception e)
    {
        LOG_ERROR ("The following exception has been catch : {}", e.what());
        return -1;
    }
    return 0;
}


