#include "yolo.h"

//#include <utility>

namespace infer_sdk{
    using namespace cv;
    using namespace std;
    using json = nlohmann::json;

#ifdef TENSORRT_BACKEND
    Yolo::Yolo(const DetAttr& det_attr): det_attr_(det_attr) {
        backend_ = make_shared<TensorRTBackend>(det_attr_.engine_path_);
        assert(backend_->num_inputs() == 1);
        assert(backend_->num_outputs() == 3);
        // output dim(1) -> 3 * (num_classes + 5)
        yolo_attr_.preds_per_anchor_ = backend_->outputs(0)->dim(1) / 3;
        yolo_attr_.num_classes_ = yolo_attr_.preds_per_anchor_ - 5;
        assert(yolo_attr_.num_classes_ == det_attr_.labels_.size());
        auto input = backend_->inputs(0);
        // b c h w
        resize_attr_.input_h_ = input->dim(2);
        resize_attr_.input_w_ = input->dim(3);
        if(det_attr_.alg_type_ == AlgType::YOLOX)
            yolo_attr_.anchors_per_branch_ = 1;
        else
            yolo_attr_.anchors_per_branch_ = 3;
    }
#else
    Yolo::Yolo(const json &cfg) {
        det_attr_.parse(cfg);
        backend_ = make_shared<RKNNBackend>(det_attr_.engine_path_);
        assert(backend_->num_inputs() == 1);
        assert(backend_->num_outputs() == 3);
        // output dim(1) -> 3 * (num_classes + 5)
        yolo_attr_.preds_per_anchor_ = backend_->outputs(0)->dim(1) / 3;
        yolo_attr_.num_classes_ = yolo_attr_.preds_per_anchor_ - 5;
        assert(yolo_attr_.num_classes_ == det_attr_.labels_.size());
        auto input = backend_->inputs(0);
#ifdef RKNN2_BACKEND
        // b h w c
        yolo_attr_.input_height_ = input->dim(1);
        yolo_attr_.input_width_ = input->dim(2);
#else
        // b c h w
        yolo_attr_.input_height_ = input->dim(2);
        yolo_attr_.input_width_ = input->dim(3);
#endif
        if(det_attr_.type_ == DetType::YOLOX)
            yolo_attr_.anchors_per_branch_ = 1;
        else
            yolo_attr_.anchors_per_branch_ = 3;
    }
#endif

    void Yolo::preprocess(const Mat& img){
        resize_attr_.ori_img_w_ = img.cols;
        resize_attr_.ori_img_h_ = img.rows;
        auto input = backend_->inputs(0);
#ifndef TENSORRT_BACKEND
        Mat resized_img(yolo_attr_.input_height_, yolo_attr_.input_width_, CV_8UC3, input->data_ptr<uint8_t>());
#else
        Mat resized_img(resize_attr_.input_h_, resize_attr_.input_w_, CV_8UC3);
#endif
        resize_padding(img, resized_img,
                       resize_attr_.scale_x_, resize_attr_.scale_y_,
                       resize_attr_.pad_top_, resize_attr_.pad_bottom_,
                       resize_attr_.pad_left_, resize_attr_.pad_right_);
#ifdef TENSORRT_BACKEND
        auto input_ptr = input->data_ptr<float>();

        for(int i = 2; i >= 0 ; i--){
            auto img_ptr = resized_img.data + i;
            for(int j = 0; j < input->size() / 3; j++){
                *input_ptr = (float) (*img_ptr) / 255.f;
                img_ptr += 3;
                input_ptr++;
            }
        }
#endif
    }

    void Yolo::postprocess() {
        output_before_nms_.clear();
        output_.clear();
        int stride, grid_h, grid_w, grid_len;
        int* anchor;
        float *optr;

        for (int s = 0; s < backend_->num_outputs(); s++){
            optr = backend_->outputs(s)->data_ptr<float>();
            anchor = det_attr_.anchors_.data() + s * 2 * yolo_attr_.anchors_per_branch_;
            grid_h = backend_->outputs(s)->dim(2);
            grid_w = backend_->outputs(s)->dim(3);
            assert(resize_attr_.input_h_ / grid_h == resize_attr_.input_w_ / grid_w);
            stride = resize_attr_.input_h_ / grid_h;
            grid_len = grid_h * grid_w;
            for (int a = 0; a < yolo_attr_.anchors_per_branch_; a++) {
                for (int i = 0; i < grid_h; i++) {

                    for (int j = 0; j < grid_w; j++) {

                        float bbox_confidence = optr[(yolo_attr_.preds_per_anchor_ * a + 4) * grid_len + i * grid_w + j];

                        if (bbox_confidence >= det_attr_.obj_thr_) {
                            int offset = (yolo_attr_.preds_per_anchor_ * a) * grid_len + i * grid_w + j;
                            float *in_ptr = optr + offset;

                            float maxClassProbs = in_ptr[5 * grid_len];
                            int maxClassId = 0;
                            for (int k = 1; k < yolo_attr_.num_classes_; ++k)
                            {
                                float prob = in_ptr[(5 + k) * grid_len];
                                if (prob > maxClassProbs)
                                {
                                    maxClassId = k;
                                    maxClassProbs = prob;
                                }
                            }

                            float limit_score;
                            if (det_attr_.alg_type_ == AlgType::YOLOX) {
                                limit_score = maxClassProbs;
                            } else {
                                limit_score = bbox_confidence * maxClassProbs;
                            }
                            if (limit_score > det_attr_.conf_thr_) {
                                float bbox_x, bbox_y, bbox_w, bbox_h;
                                if (det_attr_.alg_type_ == AlgType::YOLOX) {
                                    bbox_x = *in_ptr;
                                    bbox_y = (in_ptr[grid_len]);
                                    bbox_w = exp(in_ptr[2 * grid_len]) * (float) stride;
                                    bbox_h = exp(in_ptr[3 * grid_len]) * (float) stride;
                                } else {
                                    bbox_x = *in_ptr * 2.0f - 0.5f;
                                    bbox_y = (in_ptr[grid_len]) * 2.0f - 0.5f;
                                    bbox_w = (in_ptr[2 * grid_len]) * 2.0f;
                                    bbox_h = (in_ptr[3 * grid_len]) * 2.0f;
                                    bbox_w *= bbox_w;
                                    bbox_h *= bbox_h;
                                }
                                bbox_x = (bbox_x + (float) j) * (float) stride;
                                bbox_y = (bbox_y + (float) i) * (float) stride;
                                bbox_w *= (float) anchor[a * 2];
                                bbox_h *= (float) anchor[a * 2 + 1];
                                bbox_x -= (bbox_w / 2.0f);
                                bbox_y -= (bbox_h / 2.0f);

                                BBox_<float> bbox_float(bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h, bbox_confidence, det_attr_.labels_[maxClassId]);
                                map_bbox_to_origin_plane(bbox_float, resize_attr_);
                                BBox bbox = static_cast<BBox>(bbox_float);
                                output_before_nms_.emplace_back(bbox);
                            }
                        }
                    }
                }
            }
        }
        if (det_attr_.alg_type_ == AlgType::YOLOX)
            cpu_nms(output_before_nms_, output_, det_attr_.nms_thr_, false);
        else
            cpu_nms(output_before_nms_, output_, det_attr_.nms_thr_);
    }

}