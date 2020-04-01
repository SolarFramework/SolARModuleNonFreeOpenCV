HEADERS += interfaces/SolAROpencvNonFreeAPI.h \
interfaces/SolARDescriptorsExtractorBinaryOpencv.h \
interfaces/SolARDescriptorsExtractorSURF64Opencv.h \
interfaces/SolARDescriptorsExtractorSURF128Opencv.h \
interfaces/SolARDescriptorsExtractorSIFTOpencv.h \
interfaces/SolARKeylineDetectorOpencv.h \
interfaces/SolARKeypointDetectorNonFreeOpencv.h \
interfaces/SolARModuleNonFreeOpencv_traits.h

SOURCES += src/SolARModuleNonFreeOpencv.cpp \
    src/SolARDescriptorsExtractorBinaryOpencv.cpp \
    src/SolARDescriptorsExtractorSIFTOpencv.cpp \
    src/SolARDescriptorsExtractorSURF64Opencv.cpp \
    src/SolARDescriptorsExtractorSURF128Opencv.cpp \
    src/SolARKeylineDetectorOpencv.cpp \
    src/SolARKeypointDetectorNonFreeOpencv.cpp
 
