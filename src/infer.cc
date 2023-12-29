#include "infer.h"

namespace infer_sdk{
    using namespace std;
    using namespace cv;

    DetInfer::DetInfer(const DetAttr &det_attr) {
        SPDLOG_INFO("Creating {}", alg_type_to_string(det_attr.alg_type_));
        switch (det_attr.alg_type_) {
            case AlgType::YOLOV5:
            case AlgType::YOLOV7:
            case AlgType::YOLOX:
                infer_impl_ = std::make_shared<Yolo>(det_attr);
                break;
        }
    }

    vector<BBox> DetInfer::infer(const Mat &img, const string& label, cudaStream_t stream){
        return infer(img, vector<string>({label}), stream)[label];
    }

    map<string, vector<BBox>> DetInfer::infer(const cv::Mat &img, const vector<string>& labels, cudaStream_t stream){
        auto boxes = infer(img, stream);
        map<string, vector<BBox>> filtered_boxes;
        for(const auto& box: boxes){
            for(const auto& label: labels){
                if(box.label_ == label){
                    filtered_boxes[label].emplace_back(box);
                }
            }
        }
        return filtered_boxes;
    }

    FaceDetInfer::FaceDetInfer(const DetAttr &det_attr) {
        SPDLOG_INFO("Creating {}", alg_type_to_string(det_attr.alg_type_));
        switch (det_attr.alg_type_) {
            case AlgType::SCRFD:
                infer_impl_ = std::make_shared<SCRFD>(det_attr);
                break;
        }
    }

    KpInfer::KpInfer(const KpAttr &kp_attr) {
        SPDLOG_INFO("Creating {}", alg_type_to_string(kp_attr.alg_type_));
        switch (kp_attr.alg_type_) {
            case AlgType::RTMPOSE:
                infer_impl_ = std::make_shared<RTMPose>(kp_attr);
                break;
        }
    }

    FeatInfer::FeatInfer(const FeatAttr &feat_attr){
        SPDLOG_INFO("Creating {}", alg_type_to_string(feat_attr.alg_type_));
        switch (feat_attr.alg_type_) {
            case AlgType::ARCFACE:
                infer_impl_ = std::make_shared<ARCFace>(feat_attr);
                break;
        }
    }

    cv::Size FeatInfer::get_input_size() {
        auto backend = infer_impl_->backend_;
        return {backend->inputs(0)->dim(3), backend->inputs(0)->dim(2)};
    }

    int FeatInfer::get_feat_len() {
        auto backend = infer_impl_->backend_;
        return backend->outputs(0)->dim(1);
    }
}
