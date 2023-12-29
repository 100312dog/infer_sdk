#ifndef SCRFD_H
#define SCRFD_H

#include "infer_impl.h"
#include "common.h"

namespace infer_sdk{
    class SCRFD: public InferImpl<cv::Mat, std::vector<Face>>{
    public:
        explicit SCRFD(const DetAttr& det_attr);
    private:
        void preprocess(const cv::Mat& img) override;
        void postprocess() override;
        ResizeAttr resize_attr_;
        DetAttr det_attr_;
        struct SCRFDAttr{
            std::vector<float>mean_ {127.5, 127.5, 127.5};
            std::vector<float>std_ {128.0, 128.0, 128.0};
            std::vector<int> strides {8, 16, 32};
            int num_anchors = 2, num_kps = 5;
        } scrfd_attr_;
        std::vector<Face> output_before_nms_;
    };

}


#endif
