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

#ifndef SOLARNONFREEOPENCVHELPER_H
#define SOLARNONFREEOPENCVHELPER_H

#include "opencv2/core.hpp"

#include "SolAROpencvNonFreeAPI.h"
#include "xpcf/api/IComponentManager.h"

#include "core/Messages.h"
#include "datastructure/Image.h"
#include "datastructure/MathDefinitions.h"
#include "datastructure/GeometryDefinitions.h"
#include "datastructure/DescriptorBuffer.h"

namespace SolAR {
using namespace datastructure;
namespace MODULES {
namespace NONFREEOPENCV {

/**
 * @class SolARNonFreeOpenCVHelper
 * @brief A toolbox to convert OpenCV structures to SolAR structures and respectively.
 *
 */

class SOLAROPENCVNONFREE_EXPORT_API SolARNonFreeOpenCVHelper {
public:
    template <class T,int Rows, int Cols>
    static FrameworkReturnCode convertCVMatToSolar(const cv::Mat& openCVMat, Matrix<T, Rows , Cols, 1 > & solarMat);
    template <class T,int Dim>
    static FrameworkReturnCode convertCVMatToSolar(const cv::Mat& openCVMat, Transform<T,Dim> & solarTransform);

    template <class T,int Rows, int Cols>
    static cv::Mat mapToOpenCV (const Matrix<T, Rows , Cols >&  solarMat);

    template <class T,int Dim>
    static cv::Mat mapToOpenCV (const Transform<T,Dim>&  solarTransform);

    static std::vector<cv::Point2i> convertToOpenCV (const Contour2Di &contour);
    static std::vector<cv::Point2f> convertToOpenCV (const Contour2Df &contour);

    static FrameworkReturnCode convertToSolar(cv::Mat&  imgSrc, SRef<Image>& imgDest);

    static void mapToOpenCV (SRef<Image> imgSrc, cv::Mat& imgDest);

    static cv::Mat mapToOpenCV (SRef<Image> imgSrc);
    static uint32_t deduceOpenDescriptorCVType(DescriptorDataType querytype);


	template <class T> inline static constexpr int inferOpenCVType();
 
    static int deduceOpenCVType(SRef<Image> img);

};

template <class T> constexpr int SolARNonFreeOpenCVHelper::inferOpenCVType()
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value || std::is_same<T, int32_t>::value
                   || std::is_same<T, int16_t>::value || std::is_same<T, unsigned char>::value || std::is_same<T, char>::value,
                    "type not allowed to infer openCV type");
    return -1;
}

template <> constexpr int SolARNonFreeOpenCVHelper::inferOpenCVType<float>()
{
    return CV_32F;
}

template <> constexpr int SolARNonFreeOpenCVHelper::inferOpenCVType<double>()
{
    return CV_64F;
}

template <> constexpr int SolARNonFreeOpenCVHelper::inferOpenCVType<int32_t>()
{
    return CV_32S;
}

template <> constexpr int SolARNonFreeOpenCVHelper::inferOpenCVType<int16_t>()
{
    return CV_16S;
}

template <> constexpr int SolARNonFreeOpenCVHelper::inferOpenCVType<unsigned char>()
{
    return CV_8U;
}

template <> constexpr int SolARNonFreeOpenCVHelper::inferOpenCVType<char>()
{
    return CV_8S;
}

template <class T,int Rows, int Cols>
FrameworkReturnCode SolARNonFreeOpenCVHelper::convertCVMatToSolar(const cv::Mat& openCVMat, Matrix<T, Rows , Cols > & solarMat)
{
    if (openCVMat.cols != Cols || openCVMat.rows != Rows || openCVMat.type() != inferOpenCVType<T>()) {
        return FrameworkReturnCode::_ERROR_;
    }
    Matrix<T, Rows , Cols > mat(reinterpret_cast<T*>( openCVMat.data));
    solarMat = mat;

    return FrameworkReturnCode::_SUCCESS;
}

template <class T,int Dim>
FrameworkReturnCode SolARNonFreeOpenCVHelper::convertCVMatToSolar(const cv::Mat& openCVMat, Transform<T,Dim> & solarTransform)
{
    if (openCVMat.cols != solarTransform.cols() || openCVMat.rows != solarTransform.rows()+1 || openCVMat.type() != inferOpenCVType<T>()) {
        return FrameworkReturnCode::_ERROR_;
    }
    Matrix<T, Dim+1 , Dim+1 > transform(reinterpret_cast<T*>( openCVMat.data));
    solarTransform = transform;

    return FrameworkReturnCode::_SUCCESS;
}

template <class T, int Rows, int Cols>
cv::Mat SolARNonFreeOpenCVHelper::mapToOpenCV (const Matrix<T, Rows , Cols>&  solarMat)
{
    int type = inferOpenCVType<T>(); // typeid ??
    cv::Mat mat(solarMat.rows(),solarMat.cols(),type,(void *)solarMat.data());
    return mat;
}

template <class T,int Dim>
cv::Mat SolARNonFreeOpenCVHelper::mapToOpenCV (const Transform<T,Dim>&  solarTransform)
{
    int type = inferOpenCVType<T>(); // typeid ??
    cv::Mat mat(Dim+1,Dim+1,type,(void *)solarTransform.data());
    return mat;
}

}
}
}

#endif // SOLARNONFREEOPENCVHELPER_H