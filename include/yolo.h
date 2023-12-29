#ifndef YOLO_H
#define YOLO_H

#include "infer_impl.h"
#include "common.h"

namespace infer_sdk{
    class Yolo: public InferImpl<cv::Mat, std::vector<BBox>>{
    public:
        explicit Yolo(const DetAttr& det_attr);
    private:
        void preprocess(const cv::Mat& img) override;
        void postprocess() override;
        ResizeAttr resize_attr_;
        DetAttr det_attr_;
        struct YoloAttr{
            int anchors_per_branch_;
            int preds_per_anchor_, num_classes_;
        } yolo_attr_;
        std::vector<BBox> output_before_nms_;
    };

}



#endif