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

#include "SolARModuleOpencv_traits.h"
#include "SolARModuleNonFreeOpencv_traits.h"
#include "api/image/IImageLoader.h"
#include "api/features/IKeypointDetector.h"
#include "api/features/IDescriptorsExtractor.h"
#include "api/display/I2DOverlay.h"
#include "api/display/IImageViewer.h"
#include "core/Log.h"


#include <boost/log/core.hpp>

using namespace SolAR;
using namespace SolAR::MODULES::OPENCV;
using namespace SolAR::MODULES::NONFREEOPENCV;
using namespace SolAR::datastructure;
using namespace SolAR::api;

namespace xpcf  = org::bcom::xpcf;

int main(int argc,char** argv)
{
#if NDEBUG
    boost::log::core::get()->set_logging_enabled(false);
#endif

    LOG_ADD_LOG_TO_CONSOLE();

    /* instantiate component manager*/
    /* this is needed in dynamic mode */
    SRef<xpcf::IComponentManager> xpcfComponentManager = xpcf::getComponentManagerInstance();

    if(xpcfComponentManager->load("conf_DescriptorExtractor.xml")!=org::bcom::xpcf::_SUCCESS)
    {
        LOG_ERROR("Failed to load the configuration file conf_DescriptorExtractor.xml")
        return -1;
    }

    // declare and create components
    LOG_INFO("Start creating components");

    SRef<image::IImageLoader> imageLoader = xpcfComponentManager->create<SolARImageLoaderOpencv>()->bindTo<image::IImageLoader>();
    SRef<features::IKeypointDetector> keypointsDetector = xpcfComponentManager->create<SolARKeypointDetectorNonFreeOpencv>()->bindTo<features::IKeypointDetector>();
    SRef<features::IDescriptorsExtractor> extractorSIFT = xpcfComponentManager->create<SolARDescriptorsExtractorSIFTOpencv>()->bindTo<features::IDescriptorsExtractor>();
    SRef<display::I2DOverlay> overlay = xpcfComponentManager->create<SolAR2DOverlayOpencv>()->bindTo<display::I2DOverlay>();
    SRef<display::IImageViewer> viewer = xpcfComponentManager->create<SolARImageViewerOpencv>()->bindTo<display::IImageViewer>();

    if (!imageLoader  || !keypointsDetector || !extractorSIFT || !overlay || !viewer)
    {
        LOG_ERROR("One or more component creations have failed");
        return -1;
    }


    SRef<Image>                         testImage;
    std::vector< SRef<Keypoint>>        keypoints;
    SRef<DescriptorBuffer>              descriptors;

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
    return 0;
}


