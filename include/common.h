#ifndef COMMON_H
#define COMMON_H

#ifdef TENSORRT_BACKEND
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <string>
#include <dirent.h>

#define CUDA_RUNTIME_CHECK(ret) cuda_runtime_check(ret)

namespace infer_sdk
{

    enum class AlgType
    {
        // det
        YOLOV5,
        YOLOV7,
        YOLOX,
        RTMDET,
        NANODET,
        SCRFD,
        // kp
        TINYPOSE,
        RTMPOSE,
        // feat
        ARCFACE
    };

    // todo algtype string operation
    inline std::string alg_type_to_string(const AlgType alg_type)
    {
        switch (alg_type)
        {
            case AlgType::YOLOV5:
                return "YOLOV5";
            case AlgType::YOLOV7:
                return "YOLOV5";
            case AlgType::YOLOX:
                return "YOLOX";
            case AlgType::RTMDET:
                return "RTMDET";
            case AlgType::NANODET:
                return "NANODET";
            case AlgType::SCRFD:
                return "SCRFD";
            case AlgType::TINYPOSE:
                return "TINYPOSE";
            case AlgType::RTMPOSE:
                return "RTMPOSE";
            case AlgType::ARCFACE:
                return "ARCFACE";
        }
    }

    enum class DataType
    {
        kFLOAT,
        kHALF,
        kBOOL,
        kINT8,
        kUINT8,
        kINT32,
        kINT64,
    };

    enum class DeviceType{
        HOST,
        DEVICE
    };

    struct ResizeAttr{
        float scale_x_, scale_y_;
        int ori_img_w_, ori_img_h_;
        int input_w_, input_h_;
        int pad_top_ = 0, pad_bottom_ = 0, pad_left_ = 0, pad_right_ = 0;
    };

    struct DrawAttr{
        cv::Scalar color_ {0, 255, 0};
        int line_width_ = 2;
        int point_radius_ = 3;
        int font_ = cv::FONT_HERSHEY_SIMPLEX;
        float font_scale_ = 0.8;
        int label_offset_x_ = 0, label_offset_y_ = 20;
        bool with_label_ = true;
        bool with_id_ = false;
    };


    template<typename T>
    class BBox_;
    using BBox = BBox_<int>;
    
    template<typename T>
    class BBox_ {
    public:
        T left_, top_, right_, bottom_;
        float conf_;
        std::string label_;
        int id_;

        BBox_() = default;

        BBox_(T left, T top, T right, T bottom, float conf, const std::string& label)
                : left_(left), top_(top), right_(right), bottom_(bottom),
                conf_(conf), label_(label) {};

        explicit operator BBox() const{
            return {static_cast<int>(left_),
                    static_cast<int>(top_),
                    static_cast<int>(right_),
                    static_cast<int>(bottom_),
                    conf_,
                    label_};
        }

        T area() const{
            auto area = (right_ - left_) * (bottom_ - top_);
            return std::max<T>(0, area);
        }
    };


    template<typename T>
    class Kp_;
    using Kp =  Kp_<int>;
    
    template<typename T>
    class Kp_ {
    public:
        T x_;
        T y_;
        float conf_;

        Kp_() = default;

        Kp_(T x, T y, float conf) : x_(x), y_(y), conf_(conf) {}

        explicit operator Kp() const{
            return {static_cast<int>(x_),
                    static_cast<int>(y_),
                    conf_};
        }
    };
    
    template<typename T>
    class Face_;
    using Face = Face_<int>;
    
    template<typename T>
    class Face_ {
    public:
        BBox_<T> bbox_;
        std::vector<Kp_<T>> kps_;

        Face_() = default;

        Face_(const BBox_<T>& bbox, const std::vector<Kp_<T>>& kps) : bbox_(bbox), kps_(kps) {}

        explicit operator Face() const{
            BBox box = static_cast<BBox>(bbox_);
            std::vector<Kp> kps(kps_.size());
            std::transform(kps_.begin(), kps_.end(), kps.begin(),
                           [](const auto& kp){return static_cast<Kp>(kp);});
            return {box, kps};
        }
    };

    inline size_t element_size(const DataType data_type)
    {
        switch (data_type)
        {
        case DataType::kFLOAT:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kBOOL:
        case DataType::kUINT8:
        case DataType::kINT8:
            return 1;
        case DataType::kINT32:
            return 4;
        case DataType::kINT64:
            return 8;
        default:
            return 0;
        }
    }

    inline DataType data_type(const nvinfer1::DataType data_type)
    {
        switch (data_type)
        {
        case nvinfer1::DataType::kINT32:
            return DataType::kINT32;
        case nvinfer1::DataType::kFLOAT:
            return DataType::kFLOAT;
        case nvinfer1::DataType::kHALF:
            return DataType::kHALF;
        case nvinfer1::DataType::kBOOL: 
            return DataType::kBOOL;
//        case nvinfer1::DataType::kUINT8:
//            return DataType::kUINT8;
        case nvinfer1::DataType::kINT8:
            return DataType::kINT8;
        }
    }

    inline std::string data_type_to_string(const DataType data_type)
    {
        switch (data_type)
        {
        case DataType::kFLOAT:
            return "FP32 ";
        case DataType::kHALF:
            return "FP16 ";
        case DataType::kUINT8:
            return "UINT8";
        case DataType::kINT8:
            return "INT8 ";
        case DataType::kINT32:
            return "INT32";
        case DataType::kINT64:
            return "INT64";
        case DataType::kBOOL:
            return "BOOL ";
        default:
            return "Unknown";
        }
    }

    inline std::string shape_to_string(const std::vector<int>& shape)
    {
        std::string output("[");
        if (shape.size() == 0)
        {
            return output + std::string("]");
        }
        for (int i = 0; i < shape.size() - 1; ++i)
        {
            output += std::to_string(shape[i]) + std::string(", ");
        }
        output += std::to_string(shape[shape.size() - 1]) + std::string("]");
        return output;
    }

    inline bool cuda_runtime_check(cudaError_t e){
        if (e != cudaSuccess)
        {
            SPDLOG_ERROR("CUDA runtime API error ,{}, {}", cudaGetErrorString(e), cudaGetErrorName(e));
            return false;
        }
        return true;
    };

    inline void from_json(const nlohmann::json &j, AlgType &alg_type){
        std::string str = j.get<std::string>();
        if (str == "YOLOV5")
            alg_type = AlgType::YOLOV5;
        else if (str == "YOLOV7")
            alg_type = AlgType::YOLOV7;
        else if (str == "YOLOX")
            alg_type = AlgType::YOLOX;
        else if (str == "RTMDET")
            alg_type = AlgType::RTMDET;
        else if (str == "NANODET")
            alg_type = AlgType::NANODET;
        else if (str == "SCRFD")
            alg_type = AlgType::SCRFD;
        else if (str == "TINYPOSE")
            alg_type = AlgType::TINYPOSE;
        else if (str == "RTMPOSE")
            alg_type = AlgType::RTMPOSE;
        else if (str == "ARCFACE")
            alg_type = AlgType::ARCFACE;
        else
            SPDLOG_ERROR("Invalid AlgType: {}", str.c_str());
    }


    nlohmann::json load_json(const std::string &json_path);


    void resize(const cv::Mat& origin_img, cv::Mat& dst_img, float &scale_x, float &scale_y);

    void resize_padding(const cv::Mat &origin_img, cv::Mat &dst_img,
                        float &scale_x, float &scale_y,
                        int &pad_top, int &pad_bottom, int &pad_left, int &pad_right);

    template<typename T>
    inline T clamp(T val, T min, T max){
        return val > min ? (val < max ? val : max) : min;
    }

    template<typename T>
    inline void limit_bbox_within_img(BBox_<T>& bbox, int width, int height){
        bbox.left_ = clamp<T>(bbox.left_, 0, width - 1);
        bbox.right_ = clamp<T>(bbox.right_, 0, width - 1);
        bbox.top_ = clamp<T>(bbox.top_, 0, height - 1);
        bbox.bottom_ = clamp<T>(bbox.bottom_, 0, height - 1);
    }

    template<typename T>
    inline void limit_kp_within_img(Kp_<T>& kp, int width, int height){
        kp.x_ = clamp<T>(kp.x_, 0, width - 1);
        kp.y_ = clamp<T>(kp.y_, 0, height - 1);
    }

    template<typename T>
    void map_bbox_to_origin_plane(BBox_<T>& bbox, const ResizeAttr& resize_attr_){
        bbox.left_ = (bbox.left_ - resize_attr_.pad_left_) * resize_attr_.scale_x_;
        bbox.top_ = (bbox.top_ - resize_attr_.pad_top_) * resize_attr_.scale_y_;
        bbox.right_ = (bbox.right_ - resize_attr_.pad_right_) * resize_attr_.scale_x_;
        bbox.bottom_ = (bbox.bottom_ - resize_attr_.pad_bottom_) * resize_attr_.scale_y_;
        limit_bbox_within_img<T>(bbox, resize_attr_.ori_img_w_, resize_attr_.ori_img_h_);
    }


    template<typename T>
    void map_kps_to_origin_plane(std::vector<Kp_<T>>& kps, const ResizeAttr& resize_attr_){
        for(auto& kp: kps){
            kp.x_ = (kp.x_ - resize_attr_.pad_left_) * resize_attr_.scale_x_;
            kp.y_ = (kp.y_ - resize_attr_.pad_top_) * resize_attr_.scale_y_;
            limit_kp_within_img<T>(kp, resize_attr_.ori_img_w_, resize_attr_.ori_img_h_);
        }
    }

    inline void map_bbox_kps_to_origin_plane(std::vector<Kp>& kps, const BBox& bbox){
        for(auto& kp:kps){
            kp.x_ = bbox.left_ + kp.x_;
            kp.y_ = bbox.top_ + kp.y_;
        }
    }

    template<typename T>
    void map_face_to_origin_plane(Face_<T>& face, const ResizeAttr& resize_attr_){
        map_bbox_to_origin_plane<T>(face.bbox_, resize_attr_);
        map_kps_to_origin_plane<T>(face.kps_, resize_attr_);
    }

    float iou(const BBox &a, const BBox &b);

    void cpu_nms(std::vector<BBox> &bboxes, std::vector<BBox> &output, float threshold, bool by_class=true);
    void cpu_nms(std::vector<Face> &faces, std::vector<Face> &output, float threshold);

    inline void put_text_on_bbox(cv::Mat &img, const BBox &bbox, const std::string &text, const DrawAttr& draw_attr={}) {
        cv::putText(img, text, cv::Point(bbox.left_ + draw_attr.label_offset_x_, bbox.top_ + draw_attr.label_offset_y_),
                    draw_attr.font_,
                    draw_attr.font_scale_,
                    draw_attr.color_,
                    2);
    }

    inline void draw_bbox(cv::Mat &img, const BBox &bbox, const DrawAttr& draw_attr={}) {
        cv::rectangle(img, cv::Point(bbox.left_, bbox.top_), cv::Point(bbox.right_, bbox.bottom_), draw_attr.color_, draw_attr.line_width_);
        if(draw_attr.with_label_ || draw_attr.with_id_){
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2);
            if(draw_attr.with_id_)
                stream << bbox.id_ << " ";
            if(draw_attr.with_label_)
                stream << bbox.label_ << " ";
            stream << bbox.conf_;
            put_text_on_bbox(img, bbox, stream.str(), draw_attr);
        }
    }

    void draw_bboxes(cv::Mat &img, const std::vector<BBox> &bboxes, const DrawAttr& draw_attr={});

    inline void draw_kps(cv::Mat &img, const std::vector<Kp> &kps, const DrawAttr& draw_attr={}){
        for(const auto &kp:kps){
            cv::circle(img, cv::Point(kp.x_, kp.y_), draw_attr.point_radius_, draw_attr.color_, -1);
        }
    }

    inline void draw_face(cv::Mat &img, const Face &face, const DrawAttr& draw_attr={}){
        draw_bbox(img, face.bbox_, draw_attr);
        draw_kps(img, face.kps_, draw_attr);
    }

    void draw_faces(cv::Mat &img, const std::vector<Face> &faces, const DrawAttr& draw_attr={});

    void draw_coco_kps(cv::Mat& img, const std::vector<Kp> &kps, const DrawAttr& draw_attr={});

    void crop_img_with_padding_bbox(const cv::Mat &img, const BBox &bbox,
                                   cv::Mat &crop_img, BBox &padding_bbox,
                                   float expand_ratio);

    std::vector<std::string> get_files_in_folder(const std::string& folder_path);

    std::string get_current_time();
}

#endif