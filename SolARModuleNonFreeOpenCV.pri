HEADERS += interfaces/SolAROpencvNonFreeAPI.h \
interfaces/SolARDescriptorMatcherBinaryOpencv.h \
interfaces/SolARDescriptorsExtractorBinaryOpencv.h \
interfaces/SolARDescriptorsExtractorSURF64Opencv.h \
interfaces/SolARDescriptorsExtractorSURF128Opencv.h \
interfaces/SolARKeylineDetectorNonFreeOpencv.h \
interfaces/SolARKeypointDetectorNonFreeOpencv.h \
interfaces/SolARModuleNonFreeOpencv_traits.h \
interfaces/SolARNonFreeOpenCVHelper.h

SOURCES += src/SolARModuleNonFreeOpencv.cpp \
    src/SolARDescriptorMatcherBinaryOpencv.cpp \
    src/SolARDescriptorsExtractorBinaryOpencv.cpp \
    src/SolARDescriptorsExtractorSURF64Opencv.cpp \
    src/SolARDescriptorsExtractorSURF128Opencv.cpp \
    src/SolARKeylineDetectorNonFreeOpencv.cpp \
    src/SolARKeypointDetectorNonFreeOpencv.cpp \
    src/SolARNonFreeOpenCVHelper.cpp
 