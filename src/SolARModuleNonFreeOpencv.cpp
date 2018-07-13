/**
 * @copyright Copyright (c) 2015 All Right Reserved, B-com http://www.b-com.com/
 *
 * This file is subject to the B<>Com License.
 * All other rights reserved.
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 */

#include <iostream>

#include "xpcf/module/ModuleFactory.h"

#include "SolARDescriptorsExtractorSIFTOpencv.h"
#include "SolARDescriptorsExtractorSURF128Opencv.h"
#include "SolARDescriptorsExtractorSURF64Opencv.h"
#include "SolARKeypointDetectorNonFreeOpencv.h"


namespace xpcf=org::bcom::xpcf;

XPCF_DECLARE_MODULE("2bce6a98-2ad7-4cd5-bd63-b3ee68fb59c0", "SolARModuleNonFreeOpenCV")

extern "C" XPCF_MODULEHOOKS_API xpcf::XPCFErrorCode XPCF_getComponent(const boost::uuids::uuid& componentUUID,SRef<xpcf::IComponentIntrospect>& interfaceRef)
{

    xpcf::XPCFErrorCode errCode = xpcf::XPCFErrorCode::_FAIL;
    errCode = xpcf::tryCreateComponent<SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSIFTOpencv>(componentUUID,interfaceRef);
  
    if (errCode != xpcf::XPCFErrorCode::_SUCCESS)
    {
        errCode = xpcf::tryCreateComponent<SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF128Opencv>(componentUUID,interfaceRef);
    }

    if (errCode != xpcf::XPCFErrorCode::_SUCCESS)
    {
        errCode = xpcf::tryCreateComponent<SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF64Opencv>(componentUUID,interfaceRef);
    }

    if (errCode != xpcf::XPCFErrorCode::_SUCCESS)
    {
        errCode = xpcf::tryCreateComponent<SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv>(componentUUID,interfaceRef);
    }
    return errCode;
}

XPCF_BEGIN_COMPONENTS_DECLARATION
XPCF_ADD_COMPONENT(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSIFTOpencv)
XPCF_ADD_COMPONENT(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF128Opencv)
XPCF_ADD_COMPONENT(SolAR::MODULES::NONFREEOPENCV::SolARDescriptorsExtractorSURF64Opencv)
XPCF_ADD_COMPONENT(SolAR::MODULES::NONFREEOPENCV::SolARKeypointDetectorNonFreeOpencv)
XPCF_END_COMPONENTS_DECLARATION
