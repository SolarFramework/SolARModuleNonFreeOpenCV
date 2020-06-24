#ifndef SOLARMODULENONFREEOPENCV_TRAITS_H
#define SOLARMODULENONFREEOPENCV_TRAITS_H

#endif // SOLARMODULENONFREEOPENCV_TRAITS_H

#include "xpcf/core/traits.h"

namespace SolAR {
namespace MODULES {
/**
 * @namespace SolAR::MODULES::NONFREEOPENCV
 * @brief <B>Provides a set of computer vision components based on OpenCV library (opencv_contrib): https://opencv.org/</B>
 * <B> Warining, the code source of openCV used for this module is not free !</B>
 * <TT>UUID: 28b89d39-41bd-451d-b19e-d25a3d7c5797</TT>
 *
 */
namespace NONFREEOPENCV {
class SolARDescriptorsExtractorSURF128Opencv;
class SolARDescriptorsExtractorSURF64Opencv;
class SolARKeypointDetectorNonFreeOpencv;
}
}
}

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF128Opencv,
                             "fe14a310-d0a2-11e7-8fab-cec278b6b50a",
                             "SolARDescriptorsExtractorSURF128Opencv",
                             "Extracts the SURF descriptors (size 128) for a set of keypoints.")

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF64Opencv,
                             "1a437804-d0a3-11e7-8fab-cec278b6b50a",
                             "SolARDescriptorsExtractorSURF64Opencv",
                             "Extracts the SURF descriptors (size 64) for a set of keypoints.")

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv,
                             "d1f9317c-9519-4671-8ff5-4629773544f2",
                             "SolARKeypointDetectorNonFreeOpencv",
                             "Detects keypoints in an image (based on SIFT or SURF algorithm).")
