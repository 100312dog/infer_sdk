#include "arcface.h"

namespace infer_sdk {
    using namespace cv;
    using namespace std;
    using json = nlohmann::json;

    ARCFace::ARCFace(const FeatAttr& feat_attr): feat_attr_(feat_attr) {
        backend_ = make_shared<TensorRTBackend>(feat_attr_.engine_path_);
        assert(backend_->num_inputs() == 1);
        assert(backend_->num_outputs() == 1);
        auto input = backend_->inputs(0);
        resize_attr_.input_h_ = input->dim(2);
        resize_attr_.input_w_ = input->dim(3);
    }

    void ARCFace::preprocess(const Mat &img) {
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
                *input_ptr = ((float) (*img_ptr) - arcface_attr_.mean_[i]) / arcface_attr_.std_[i];
                img_ptr += 3;
                input_ptr++;
            }
        }
    }

    void ARCFace::postprocess() {
        output_ = cv::Mat(1, 512, CV_32F, backend_->outputs(0)->data_ptr<float>());
        // when computing similarity  a / |a| * b / |b| = cos(theta), ranging in [-1, 1].
        cv::normalize(output_, output_, 1, 0, NORM_L2);
    }

}