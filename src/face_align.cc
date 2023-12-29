#include "face_align.h"

namespace infer_sdk{
    using namespace std;
    cv::Mat FaceAlign::B_;
    std::once_flag FaceAlign::flag_;

    void FaceAlign::cal_B(const cv::Size& dst_size) {
        // standard face kps for 112(w) x 112(h), looks like
        //     0       1
        //         2
        //      3     4
        B_ = (cv::Mat_<float>(10, 1) <<
                38.2946, 51.6963,
                73.5318, 51.5014,
                56.0252, 71.7366,
                41.5493, 92.3655,
                70.7299, 92.2041);
        if (dst_size.height != 112 || dst_size.width != 112) {
            float dst_ratio = static_cast<float>(dst_size.width) / static_cast<float>(dst_size.height);
            float src_ratio = 96.f / 112.f;
            if (src_ratio < dst_ratio) {
                // pad width
                float scale = static_cast<float>(dst_size.height) / 112.f;
                int pad_width = (dst_size.width - 96.f * scale) / 2.f;
                B_ *= scale;
                for (int i = 0; i < 10; i += 2) {
                    B_.at<float>(i, 0) += pad_width;
                }
            } else {
                // pad height
                float scale = static_cast<float>(dst_size.width) / 96.f;
                int pad_height = (dst_size.height - 112.f * scale) / 2.f;
                B_ *= scale;
                for (int i = 1; i < 10; i += 2) {
                    B_.at<float>(i, 0) += pad_height;
                }
            }
            std::cout << B_ << std::endl;
        }
    }

    cv::Mat FaceAlign::get_affine_matrix(const vector<Kp>& kps, const cv::Size& dst_size){
        // solve AX = B, here A(10,4), X(4,1), B(10,1)
        cv::Mat A(10, 4, CV_32F);
        call_once(flag_, [dst_size](){cal_B(dst_size);});
        for(int i = 0; i < A.rows; i++){
            if(i % 2 == 0){
                A.at<float>(i, 0) = kps[i / 2].x_;
                A.at<float>(i, 1) = kps[i / 2].y_;
                A.at<float>(i, 2) = 1;
                A.at<float>(i, 3) = 0;
            }else{
                A.at<float>(i, 0) = kps[i / 2].y_;
                A.at<float>(i, 1) = -kps[i / 2].x_;
                A.at<float>(i, 2) = 0;
                A.at<float>(i, 3) = 1;
            }
        }
        cv::Mat X(4, 1, CV_32F);
        X = (A.t() * A).inv() * A.t() * B_;
        cv::Mat M = (cv::Mat_<float>(2, 3) <<
                X.at<float>(0, 0), X.at<float>(1, 0), X.at<float>(2, 0),
                -X.at<float>(1, 0), X.at<float>(0, 0), X.at<float>(3, 0));
        return M;
    }

    cv::Mat FaceAlign::get_aligned_face(const cv::Mat& img, const vector<Kp>& kps, const cv::Size& dst_size){
        auto M = get_affine_matrix(kps, dst_size);
        cv::Mat aligned_face;
        cv::warpAffine(img, aligned_face, M, dst_size);
        return aligned_face;
    }

}
