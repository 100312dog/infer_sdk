#ifndef FACE_ALIGN_H
#define FACE_ALIGN_H

#include "common.h"

namespace infer_sdk{
    class FaceAlign {
    public:
        static cv::Mat get_aligned_face(const cv::Mat& img, const std::vector<Kp>& kps, const cv::Size& dst_size);

    private:
        static void cal_B(const cv::Size& dst_size);
        static cv::Mat get_affine_matrix(const std::vector<Kp>& kps, const cv::Size& dst_size);
        static cv::Mat B_;
        static std::once_flag flag_;
    };

#endif //FACE_ALIGN_H
}

