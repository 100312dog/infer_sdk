#ifndef INFER_H
#define INFER_H

#include "yolo.h"
#include "rtmpose.h"
#include "arcface.h"
#include "scrfd.h"
#include <map>

namespace infer_sdk {
    template<typename Input, typename Output>
    class Infer {
    public:
        Output infer(const Input &input, cudaStream_t stream=nullptr) {
            return infer_impl_->infer(input, stream);
        };
        virtual ~Infer() = default;
    protected:
        std::shared_ptr<InferImpl<Input, Output>> infer_impl_;
    };

    class DetInfer : public Infer<cv::Mat, std::vector<BBox>> {
    public:
        explicit DetInfer(const DetAttr &det_attr);
        using Infer<cv::Mat, std::vector<BBox>>::infer;
        std::vector<BBox> infer(const cv::Mat &img, const std::string& label, cudaStream_t stream=nullptr);
        std::map<std::string, std::vector<BBox>> infer(const cv::Mat &img, const std::vector<std::string>& labels, cudaStream_t stream=nullptr);
    };

    class FaceDetInfer : public Infer<cv::Mat, std::vector<Face>> {
    public:
        explicit FaceDetInfer(const DetAttr &det_attr);
    };

    class KpInfer : public Infer<cv::Mat, std::vector<Kp>> {
    public:
        explicit KpInfer(const KpAttr &kp_attr);
    };

    class FeatInfer : public Infer<cv::Mat, cv::Mat> {
    public:
        explicit FeatInfer(const FeatAttr &feat_attr);
        cv::Size get_input_size();
        int get_feat_len();
    };

}
#endif
