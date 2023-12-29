#include "rtmpose.h"

#include <utility>

namespace infer_sdk{
    using namespace cv;
    using namespace std;
    using json = nlohmann::json;

    RTMPose::RTMPose(const KpAttr& kp_attr): kp_attr_(std::move(kp_attr)){
        backend_ = make_shared<TensorRTBackend>(kp_attr_.engine_path_);
        assert(backend_->num_inputs() == 1);
        assert(backend_->num_outputs() == 2);
        resize_attr_.input_h_ = backend_->inputs(0)->dim(2);
        resize_attr_.input_w_ = backend_->inputs(0)->dim(3);
        // b, num_joints, w*simcc_split_ratio
        rtmpose_attr_.num_kps_ = backend_->outputs(0)->dim(1);
        rtmpose_attr_.Nx_ = backend_->outputs(0)->dim(2);
        rtmpose_attr_.Ny_ = backend_->outputs(1)->dim(2);
        assert(rtmpose_attr_.Nx_ == resize_attr_.input_w_ * rtmpose_attr_.simcc_split_ratio_);
        assert(rtmpose_attr_.Ny_ == resize_attr_.input_h_ * rtmpose_attr_.simcc_split_ratio_);
    }

    void RTMPose::preprocess(const Mat& img){
        resize_attr_.ori_img_w_ = img.cols;
        resize_attr_.ori_img_h_ = img.rows;
        auto input = backend_->inputs(0);
#ifndef TENSORRT_BACKEND
        Mat resized_img(yolo_attr_.input_height_, yolo_attr_.input_width_, CV_8UC3, input->data_ptr<uint8_t>());
#else
        Mat resized_img(resize_attr_.input_h_, resize_attr_.input_w_, CV_8UC3);
#endif
        resize(img, resized_img, resize_attr_.scale_x_, resize_attr_.scale_y_);

#ifdef TENSORRT_BACKEND
        auto input_ptr = input->data_ptr<float>();
        // bgr2rgb + normalization;
        for(int i = 0; i < 3; i++){
            auto img_ptr = resized_img.data + 2 - i;
            for(int j = 0; j < input->size() / 3; j++){
                *input_ptr = ((float) (*img_ptr) - rtmpose_attr_.mean_[i]) / rtmpose_attr_.std_[i];
                img_ptr += 3;
                input_ptr++;
            }
        }
#endif
    }

    void RTMPose::postprocess() {
        output_.clear();
        output_.resize(rtmpose_attr_.num_kps_);
        auto simcc_x = backend_->outputs(0);
        auto simcc_y = backend_->outputs(1);

        for (int i = 0; i < rtmpose_attr_.num_kps_; i++) {
            cv::Mat mat_x = cv::Mat(rtmpose_attr_.Nx_, 1, CV_32FC1, simcc_x->data_ptr<float>() + i * rtmpose_attr_.Nx_);
            cv::Mat mat_y = cv::Mat(rtmpose_attr_.Ny_, 1, CV_32FC1, simcc_y->data_ptr<float>() + i * rtmpose_attr_.Ny_);
            double min_val_x, max_val_x, min_val_y, max_val_y;
            cv::Point min_loc_x, max_loc_x, min_loc_y, max_loc_y;
            cv::minMaxLoc(mat_x, &min_val_x, &max_val_x, &min_loc_x, &max_loc_x);
            cv::minMaxLoc(mat_y, &min_val_y, &max_val_y, &min_loc_y, &max_loc_y);
            float s = max_val_x > max_val_y ? max_val_y : max_val_x;
//            int x = s > kp_attr_.conf_thr_ ? max_loc_x.y : -1;
//            int y = s > kp_attr_.conf_thr_ ? max_loc_y.y : -1;
            output_[i].x_ = static_cast<int>(max_loc_x.y / rtmpose_attr_.simcc_split_ratio_ * resize_attr_.scale_x_);
            output_[i].y_ =static_cast<int>(max_loc_y.y / rtmpose_attr_.simcc_split_ratio_ * resize_attr_.scale_y_);
            output_[i].conf_ = s;
        }
    }
}