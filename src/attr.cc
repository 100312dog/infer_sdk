#include "attr.h"

namespace infer_sdk
{
    using namespace std;
    using json = nlohmann::json;

    void DetAttr::parse(const json &cfg)
    {
        try
        {
            auto det_cfg = cfg.at("det_cfg");
            InferAttr::parse(det_cfg);
            switch(alg_type_){
                case AlgType::YOLOV5:
                    obj_thr_ = det_cfg.at("obj_thr");
                    if (det_cfg.contains("anchors")){
                        anchors_ = det_cfg.get<vector<int>>();
                    }
                    else{
                        anchors_ = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
                        SPDLOG_INFO("YOLOV5 anchor set to default [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]");
                    }
                    break;
                case AlgType::YOLOV7:
                    obj_thr_ = det_cfg.at("obj_thr");
                    if (det_cfg.contains("anchors")){
                        anchors_ = det_cfg.get<vector<int>>();
                    }
                    else{
                        anchors_ = {12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
                        SPDLOG_INFO("YOLOV7 anchor set to default [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]");
                    }
                    break;
                case AlgType::YOLOX:
                    obj_thr_ = det_cfg.at("obj_thr");
                    anchors_ = vector<int>(6, 1);
                    break;
            }
            conf_thr_ = det_cfg.at("conf_thr");
            nms_thr_ = det_cfg.at("nms_thr");
            if(det_cfg.contains("labels")){
                labels_ = det_cfg.at("labels").get<vector<string>>();
            }
        }
        catch (exception &e)
        {
            SPDLOG_ERROR("Error parsing DetAttr");
            throw;
        }
    };

    void KpAttr::parse(const json &cfg)
    {
        try
        {
            auto kp_cfg = cfg.at("kp_cfg");
            InferAttr::parse(kp_cfg);
        }
        catch (exception &e)
        {
            SPDLOG_ERROR("Error parsing KpAttr");
            throw;
        }
    }

    void FeatAttr::parse(const json &cfg)
    {
        try
        {
            auto feat_cfg = cfg.at("feat_cfg");
            InferAttr::parse(feat_cfg);
        }
        catch (exception &e)
        {
            SPDLOG_ERROR("Error parsing FeatAttr");
            throw;
        }
    }

    void StatAttr::parse(const nlohmann::json &cfg) {
        try
        {
            auto stat_cfg = cfg.at("stat_cfg");
            frame_window_size_ = stat_cfg.at("frame_window_size");
            count_thr_ = stat_cfg.at("count_thr");
            labels_ = stat_cfg.at("labels").get<vector<string>>();
        }
        catch (exception &e)
        {
            SPDLOG_ERROR("Error parsing StatAttr");
            throw;
        }
    }
    
    void FallDownAttr::parse(const json &cfg)
    {
        try
        {
            auto falldown_cfg = cfg.at("fall_down_cfg");
            det_attr_.parse(falldown_cfg);
            kp_attr_.parse(falldown_cfg);
            stat_attr_.parse(falldown_cfg);
        }
        catch (exception &e)
        {
            SPDLOG_ERROR("Error parsing FallDownAttr");
            throw;
        }
    }

    void FaceRecognitionAttr::parse(const json &cfg)
    {
        try
        {
            auto face_recognition_cfg = cfg.at("face_recognition_cfg");
            det_attr_.parse(face_recognition_cfg);
            feat_attr_.parse(face_recognition_cfg);
            feat_database_path_ = face_recognition_cfg.at("feat_database_path");
        }
        catch (exception &e)
        {
            SPDLOG_ERROR("Error parsing FaceRecognitionAttr");
            throw;
        }
    }

}