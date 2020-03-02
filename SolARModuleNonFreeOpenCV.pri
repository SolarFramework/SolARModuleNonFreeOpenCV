HEADERS += interfaces/SolAROpencvNonFreeAPI.h \
interfaces/SolARDescriptorsExtractorSURF64Opencv.h \
interfaces/SolARDescriptorsExtractorSURF128Opencv.h \
interfaces/SolARDescriptorsExtractorSIFTOpencv.h \
interfaces/SolARKeypointDetectorNonFreeOpencv.h \
interfaces/SolARModuleNonFreeOpencv_traits.h

SOURCES += src/SolARModuleNonFreeOpencv.cpp \
    src/SolARDescriptorsExtractorSIFTOpencv.cpp \
    src/SolARDescriptorsExtractorSURF64Opencv.cpp \
    src/SolARDescriptorsExtractorSURF128Opencv.cpp \
    src/SolARKeypointDetectorNonFreeOpencv.cpp
 
