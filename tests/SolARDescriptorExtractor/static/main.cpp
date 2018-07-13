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

#include <iostream>
#include <string>

// ADD COMPONENTS HEADERS HERE, e.g #include "SolarComponent.h"

#include "SolARImageLoaderOpencv.h"
#include "SolARKeypointDetectorNonFreeOpencv.h"
#include "SolARDescriptorsExtractorSIFTOpencv.h"
#include "SolARImageViewerOpencv.h"
#include "SolAR2DOverlayOpencv.h"

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;
using namespace SolAR::MODULES::NONFREEOPENCV;

namespace xpcf  = org::bcom::xpcf;

int run(int argc, char **argv)
{

    // declarations
    xpcf::utils::uuids::string_generator gen;

    SRef<Image>                                        testImage;
    std::vector< SRef<Keypoint>>        keypoints;
    SRef<DescriptorBuffer>                             descriptors;

    // The escape key to exit the sample
    char escape_key = 27;

    // component creation

    auto imageLoader=xpcf::ComponentFactory::createInstance<SolARImageLoaderOpencv>()->bindTo<image::IImageLoader>();
    auto keypointsDetector=xpcf::ComponentFactory::createInstance<SolARKeypointDetectorNonFreeOpencv>()->bindTo<features::IKeypointDetector>();
    auto extractorSIFT=xpcf::ComponentFactory::createInstance<SolARDescriptorsExtractorSIFTOpencv>()->bindTo<features::IDescriptorsExtractor>();
    auto overlay=xpcf::ComponentFactory::createInstance<SolAR2DOverlayOpencv>()->bindTo<display::I2DOverlay>();
    auto viewer=xpcf::ComponentFactory::createInstance<SolARImageViewerOpencv>()->bindTo<display::IImageViewer>();


    // components initialisation
        // nothing to do

    // Start
    if (imageLoader->loadImage(argv[1], testImage) != FrameworkReturnCode::_SUCCESS)
    {
       LOG_ERROR("Cannot load image with path {}", argv[1]);
       return -1;
    }
    keypointsDetector->detect(testImage, keypoints);
    extractorSIFT->extract(testImage, keypoints, descriptors);

    overlay->drawCircles(keypoints, 3, 1, testImage);

    bool proceed = true;
    while (proceed)
    {
        if (viewer->display("show keypoints", testImage, &escape_key) == FrameworkReturnCode::_STOP)
        {
            proceed = false;
            std::cout << "end of DescriptorsExtractorOpencv test" << std::endl;
        }
    }
    return 0;
}

int printHelp(){
        printf(" usage :\n");
        printf(" exe ImageFilePath configFilePath\n");
        printf(" Escape key to exit");
        return 1;
}

int main(int argc, char **argv){
    if(argc == 2){
        run(argc,argv);
         return 1;
    }
    else
        return(printHelp());
}
