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

// ADD COMPONENTS HEADERS HERE


#include "SolARImageLoaderOpencv.h"
#include "SolARImageViewerOpencv.h"

#include "SolARKeypointDetectorNonFreeOpencv.h"
#include "SolARDescriptorsExtractorSIFTOpencv.h"
#include "SolARDescriptorMatcherKNNOpencv.h"
#include "SolARSideBySideOverlayOpencv.h"
#include "SolAR2DOverlayOpencv.h"
#include "SolARHomographyEstimationOpencv.h"
#include "SolAR2DOverlayOpencv.h"

using namespace std;
using namespace SolAR;
using namespace SolAR::datastructure;
using namespace SolAR::api;
using namespace SolAR::MODULES::OPENCV;
using namespace SolAR::MODULES::NONFREEOPENCV;

namespace xpcf  = org::bcom::xpcf;


int run(int argc, char *argv[])
{
    LOG_ADD_LOG_TO_CONSOLE();
    LOG_ADD_LOG_TO_FILE("./SolarHomography.log", "r");


    // components declarations
    xpcf::utils::uuids::string_generator gen;

    // create components
    LOG_INFO("Start creating components");

    auto imageLoader=xpcf::ComponentFactory::createInstance<SolARImageLoaderOpencv>()->bindTo<image::IImageLoader>();
    auto kpDetector=xpcf::ComponentFactory::createInstance<SolARKeypointDetectorNonFreeOpencv>()->bindTo<features::IKeypointDetector>();
    auto descriptorExtractor=xpcf::ComponentFactory::createInstance<SolARDescriptorsExtractorSIFTOpencv>()->bindTo<features::IDescriptorsExtractor>();
    auto matcher=xpcf::ComponentFactory::createInstance<SolARDescriptorMatcherKNNOpencv>()->bindTo<features::IDescriptorMatcher>();  

    auto homographyEstimation=xpcf::ComponentFactory::createInstance<SolARDescriptorMatcherKNNOpencv>()->bindTo<solver::pose::I2DTransformFinder>();  
    auto imageViewer=xpcf::ComponentFactory::createInstance<SolARImageViewerOpencv>()->bindTo<display::IImageViewer>();  
    auto overlaySBSComponent=xpcf::ComponentFactory::createInstance<SolARSideBySideOverlayOpencv>()->bindTo<display::ISideBySideOverlay>();  
    auto overlay2DComponent=xpcf::ComponentFactory::createInstance<SolAR2DOverlayOpencv>()->bindTo<display::I2DOverlay>();  


    LOG_INFO("All components have been created");

    // Declare data structures used to exchange information between components
    SRef<Image> image1, image2;
    SRef<DescriptorBuffer> descriptors1, descriptors2;
    std::vector<DescriptorMatch> matches;
    std::vector< SRef<Keypoint> > ref1Keypoints, ref2Keypoints;
    std::vector <SRef<Point2Df>> matchedRef1Keypoints;
    std::vector <SRef<Point2Df>> matchedRef2Keypoints;


    // load imges
    LOG_INFO("LOAD FIRST IMAGE ");
    imageLoader->loadImage(argv[1], image1);
    if (!image1) {
        LOG_ERROR("cannot load image : {}", argv[1]);
        exit(-3);
    }

    LOG_INFO("LOAD SECOND IMAGE ");
    imageLoader->loadImage(argv[2], image2);
    if (!image2) {
        LOG_ERROR("cannot load image : {}", argv[2]);
        exit(-3);
    }

    LOG_INFO("DETECT FIRST IMAGE KEYPOINTS ");
    kpDetector->detect(image1, ref1Keypoints);

    LOG_INFO("DETECT SECOND IMAGE KEYPOINTS ");
    kpDetector->detect(image2, ref2Keypoints);

    LOG_INFO("COMPUTE FIRST IMAGE DESCRIPTORS ");
    descriptorExtractor->extract(image1, ref1Keypoints, descriptors1);

    LOG_INFO("COMPUTE FIRST IMAGE DESCRIPTORS ");
    descriptorExtractor->extract(image2, ref2Keypoints, descriptors2);

    LOG_INFO("COMPUTE MATCHES ");
    matcher->match(descriptors1, descriptors2, matches);

    LOG_INFO("REINDEX MATCHES ");
    // For OpenCV module testing. A keypoint reindexer component is available in the module "Tools"
    matchedRef1Keypoints.clear();
    matchedRef2Keypoints.clear();

    for( int i = 0; i < matches.size(); i++ )
    {
        matchedRef1Keypoints.push_back(xpcf::utils::make_shared<Point2Df>(ref1Keypoints[ matches[i].getIndexInDescriptorA()]->getX(),ref1Keypoints[ matches[i].getIndexInDescriptorA()]->getY()));
        matchedRef2Keypoints.push_back(xpcf::utils::make_shared<Point2Df>(ref2Keypoints[ matches[i].getIndexInDescriptorB()]->getX(),ref2Keypoints[ matches[i].getIndexInDescriptorB()]->getY()));
    }

    LOG_INFO("FIND HOMOGRAPHY ");
    Transform2Df Hm;

    int res = homographyEstimation->find(matchedRef1Keypoints, matchedRef2Keypoints, Hm); 

    if (res == 0)
    {
        // vector of 4 corners in the marker
        std::vector<SRef <Point2Df>> transformedCorners;
        Vector3f corner0(0, 0, 1);
        Vector3f corner1(image1->getWidth(), 0, 1);
        Vector3f corner2(image1->getWidth(), image1->getHeight(), 1);
        Vector3f corner3(0, image1->getHeight(), 1);

        // For OpenCV module testing. A 2D Transformation component is available in the module "Tools"
        Vector3f transformedCorner0 = Hm*corner0;
        Vector3f transformedCorner1 = Hm*corner1;
        Vector3f transformedCorner2 = Hm*corner2;
        Vector3f transformedCorner3 = Hm*corner3;

        transformedCorners.push_back(xpcf::utils::make_shared<Point2Df>(transformedCorner0[0]/transformedCorner0[2], transformedCorner0[1]/transformedCorner0[2]));
        transformedCorners.push_back(xpcf::utils::make_shared<Point2Df>(transformedCorner1[0]/transformedCorner1[2], transformedCorner1[1]/transformedCorner1[2]));
        transformedCorners.push_back(xpcf::utils::make_shared<Point2Df>(transformedCorner2[0]/transformedCorner2[2], transformedCorner2[1]/transformedCorner2[2]));
        transformedCorners.push_back(xpcf::utils::make_shared<Point2Df>(transformedCorner3[0]/transformedCorner3[2], transformedCorner3[1]/transformedCorner3[2]));

        //draw circles on corners in camera image
        LOG_INFO("DISPLAY HOMOGRAPHY RESULT");
        overlay2DComponent->drawCircles(transformedCorners, 10, 5, image2);
        if (imageViewer->display("target's corners", image2) == SolAR::FrameworkReturnCode::_STOP) {
            LOG_ERROR("Display error with image2");
            exit(-2);
        }
        std::cout << "\n\n\nhit a key within the displayed image to quit \n";

        cv::waitKey(0);

    }

    return 0;
}

int printHelp(){
        printf(" usage :\n");
        printf(" exe firstImagePath secondImagePath\n");
        return 1;
}

int main(int argc, char **argv){
    if(argc == 3){
        run(argc,argv);
         return 1;
    }
    else
        return(printHelp());
}
