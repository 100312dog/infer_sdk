#include "scrfd.h"

namespace infer_sdk {
    using namespace cv;
    using namespace std;
    using json = nlohmann::json;

    SCRFD::SCRFD(const DetAttr &det_attr) : det_attr_(det_attr) {
        backend_ = make_shared<TensorRTBackend>(det_attr_.engine_path_);
        assert(backend_->num_inputs() == 1);
        assert(backend_->num_outputs() == 9);
        auto input = backend_->inputs(0);
        resize_attr_.input_h_ = input->dim(2);
        resize_attr_.input_w_ = input->dim(3);
        for(int s = 0; s < scrfd_attr_.strides.size(); s++){
            auto bbox_buffer = backend_->outputs(s * 3);
            auto kps_buffer = backend_->outputs(s * 3 + 1);
            auto score_buffer = backend_->outputs(s * 3 + 2);
            assert(bbox_buffer->dim(1) == scrfd_attr_.num_anchors * 4);
            assert(kps_buffer->dim(1) == scrfd_attr_.num_anchors * scrfd_attr_.num_kps * 2);
            assert(score_buffer->dim(1) == scrfd_attr_.num_anchors);
            assert(score_buffer->dim(2) * scrfd_attr_.strides[s] == resize_attr_.input_h_
                   && bbox_buffer->dim(3) * scrfd_attr_.strides[s] == resize_attr_.input_w_);
            assert(kps_buffer->dim(2) * scrfd_attr_.strides[s] == resize_attr_.input_h_
                   && kps_buffer->dim(3) * scrfd_attr_.strides[s] == resize_attr_.input_w_);
            assert(score_buffer->dim(2) * scrfd_attr_.strides[s] == resize_attr_.input_h_
                   && score_buffer->dim(3) * scrfd_attr_.strides[s] == resize_attr_.input_w_);
        }
    }

    void SCRFD::preprocess(const Mat &img) {
        resize_attr_.ori_img_w_ = img.cols;
        resize_attr_.ori_img_h_ = img.rows;
        auto input = backend_->inputs(0);
        Mat resized_img(resize_attr_.input_h_, resize_attr_.input_w_, CV_8UC3);
        resize(img, resized_img, resize_attr_.scale_x_, resize_attr_.scale_y_);
        auto input_ptr = input->data_ptr<float>();
        // bgr2rgb + normalization;
        for (int i = 0; i < 3; i++) {
            auto img_ptr = resized_img.data + 2 - i;
            for (int j = 0; j < input->size() / 3; j++) {
                *input_ptr = ((float) (*img_ptr) - scrfd_attr_.mean_[i]) / scrfd_attr_.std_[i];
                img_ptr += 3;
                input_ptr++;
            }
        }
    }

    void SCRFD::postprocess() {
        output_before_nms_.clear();
        output_.clear();

        int grid_h, grid_w;
        float score, left, top, right, bottom;
        // strides
        for (int s = 0; s < 3; s++) {
            grid_h = resize_attr_.input_h_ / scrfd_attr_.strides[s];
            grid_w = resize_attr_.input_w_ / scrfd_attr_.strides[s];

            auto bbox_buffer = backend_->outputs(s * 3);
            auto kps_buffer = backend_->outputs(s * 3 + 1);
            auto score_buffer = backend_->outputs(s * 3 + 2);

            // num anchors
            for(int a = 0; a < scrfd_attr_.num_anchors; a++){
                // grid h
                for (int i = 0; i < grid_h; i++){
                    // grid w
                    for (int j = 0; j < grid_w; j++){
                        score = score_buffer->at<float>({0, a, i, j});
                        if(score > det_attr_.conf_thr_){
                            left = (j - bbox_buffer->at<float>({0, a * 4, i, j})) * scrfd_attr_.strides[s];
                            top = (i - bbox_buffer->at<float>({0, a * 4 + 1, i, j})) * scrfd_attr_.strides[s];
                            right = (j + bbox_buffer->at<float>({0, a * 4 + 2, i, j})) * scrfd_attr_.strides[s];
                            bottom = (i + bbox_buffer->at<float>({0, a * 4 + 3, i, j})) * scrfd_attr_.strides[s];
                            vector<Kp_<float>> kps_float(scrfd_attr_.num_kps);
                            for(int l = 0; l < scrfd_attr_.num_kps; l++){
                                kps_float[l].x_ = (j + kps_buffer->at<float>({0, a * 10 + 2 * l, i, j})) * scrfd_attr_.strides[s];
                                kps_float[l].y_ = (i + kps_buffer->at<float>({0, a * 10 + 2 * l + 1, i, j})) * scrfd_attr_.strides[s];
                            }
                            Face_<float> face_float(BBox_<float>(left, top, right, bottom, score, "face"), kps_float);
                            map_face_to_origin_plane(face_float, resize_attr_);
                            Face face = static_cast<Face>(face_float);
                            output_before_nms_.emplace_back(face);
                        }
                    }
                }
            }
        }
        cpu_nms(output_before_nms_, output_, det_attr_.nms_thr_);
    }


}