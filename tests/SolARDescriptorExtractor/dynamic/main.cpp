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
#include "xpcf/component/ComponentBase.h"

#include "SolARModuleOpencv_traits.h"
#include "SolARModuleNonFreeOpencv_traits.h"

#include "api/image/IImageLoader.h"
#include "api/display/IImageViewer.h"
#include "api/display/I2DOverlay.h"
#include "api/features/IKeypointDetector.h"
#include "api/features/IDescriptorsExtractor.h"

using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;

namespace xpcf  = org::bcom::xpcf;

int run(int argc,char** argv)
{

    // load libraries
    SRef<xpcf::IComponentManager> xpcfComponentManagerOpenCV = xpcf::getComponentManagerInstance();
     org::bcom::xpcf::XPCFErrorCode returnErrCode = xpcfComponentManagerOpenCV->load("$BCOMDEVROOT/.xpcf/SolAR/xpcf_SolARModuleOpenCV_registry.xml");

    SRef<xpcf::IComponentManager> xpcfComponentManagerNonFreeOpenCV = xpcf::getComponentManagerInstance();
    
    org::bcom::xpcf::XPCFErrorCode returnErrCode2 = xpcfComponentManagerNonFreeOpenCV->load("$BCOMDEVROOT/.xpcf/SolAR/xpcf_SolARModuleNonFreeOpenCV_registry.xml");
    
 // declarations and creation of components
    SRef<image::IImageLoader> imageLoader = xpcfComponentManagerOpenCV->create<SolAR::MODULES::OPENCV::SolARImageLoaderOpencv>()->bindTo<image::IImageLoader>();
    SRef<display::IImageViewer> viewer = xpcfComponentManagerOpenCV->create<SolAR::MODULES::OPENCV::SolARImageViewerOpencv>()->bindTo<display::IImageViewer>();
    SRef<display::I2DOverlay> overlay  = xpcfComponentManagerOpenCV->create<SolAR::MODULES::OPENCV::SolAR2DOverlayOpencv>()->bindTo<display::I2DOverlay>();

    SRef<features::IKeypointDetector> keypointsDetector = xpcfComponentManagerNonFreeOpenCV->create<SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv>()->bindTo<features::IKeypointDetector>();
    SRef<features::IDescriptorsExtractor> extractorSIFT = xpcfComponentManagerNonFreeOpenCV->create<SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSIFTOpencv>()->bindTo<features::IDescriptorsExtractor>();

    

    if (!imageLoader || !keypointsDetector || !extractorSIFT || !viewer || !overlay)
    {
        LOG_ERROR("One or more component creations have failed");
        return -1;
    }

    SRef<Image>                                        testImage;
    std::vector< SRef<Keypoint>>        keypoints;
    SRef<DescriptorBuffer>                             descriptors;

    // The escape key to exit the sample
    char escape_key = 27;


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
        printf(" exe ImageFilePath\n");
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


