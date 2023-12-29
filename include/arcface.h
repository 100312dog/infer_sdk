#ifndef ARCFACE_H
#define ARCFACE_H

#include "infer_impl.h"
#include "common.h"

namespace infer_sdk{
    class ARCFace: public InferImpl<cv::Mat, cv::Mat>{
    public:
        explicit ARCFace(const FeatAttr& feat_attr);
    private:
        void preprocess(const cv::Mat& img) override;
        void postprocess() override;
        FeatAttr feat_attr_;
        ResizeAttr resize_attr_;
        struct ARCFACEAttr{
            std::vector<float>mean_ {127.5, 127.5, 127.5};
            std::vector<float>std_ {127.5, 127.5, 127.5};
        } arcface_attr_;
    };

}


#endif
