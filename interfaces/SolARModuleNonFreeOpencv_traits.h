#ifndef SOLARMODULENONFREEOPENCV_TRAITS_H
#define SOLARMODULENONFREEOPENCV_TRAITS_H

#endif // SOLARMODULENONFREEOPENCV_TRAITS_H

namespace SolAR {
namespace MODULES {
namespace NONFREEOPENCV {

class SolARDescriptorsExtractorSIFTOpencv;
class SolARDescriptorsExtractorSURF128Opencv;
class SolARDescriptorsExtractorSURF64Opencv;
class SolARKeypointDetectorNonFreeOpencv;
}
}
}

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSIFTOpencv,
                             "3787eaa6-d0a0-11e7-8fab-cec278b6b50a",
                             "SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSIFTOpencv")

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF128Opencv,
                             "fe14a310-d0a2-11e7-8fab-cec278b6b50a",
                             "SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF128Opencv")

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF64Opencv,
                             "1a437804-d0a3-11e7-8fab-cec278b6b50a",
                             "SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF64Opencv")

XPCF_DEFINE_COMPONENT_TRAITS(SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv,
                             "d1f9317c-9519-4671-8ff5-4629773544f2",
                             "SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv")
