#ifndef RTMPOSE_H
#define RTMPOSE_H

#include "infer_impl.h"
#include "common.h"

namespace infer_sdk{
    class RTMPose : public InferImpl<cv::Mat, std::vector<Kp>> {
    public:
        explicit RTMPose(const KpAttr& kp_attr);
    private:
        void preprocess(const cv::Mat& img) override;
        void postprocess() override;
        ResizeAttr resize_attr_;
        KpAttr kp_attr_;
        struct RTMposeAttr{
            std::vector<float>mean_ {123.675, 116.28, 103.53};
            std::vector<float>std_ {58.395, 57.12, 57.375};
            int Nx_, Ny_;
            int num_kps_;
            float simcc_split_ratio_ = 2.f;
        } rtmpose_attr_;
    };
}


#endif
