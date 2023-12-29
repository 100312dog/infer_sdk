#ifndef ATTR_H
#define ATTR_H

#include <json.hpp>
#include "common.h"

namespace infer_sdk
{
    class Attr{
    public:
        virtual void parse(const nlohmann::json &cfg) = 0;
        virtual ~Attr() = default;
    };

    class InferAttr : public Attr{
    public:
        void parse(const nlohmann::json &cfg) override{
            try{
                engine_path_ = cfg.at("engine_path");
                alg_type_ = cfg.at("type").get<AlgType>();
            }catch (std::exception &e){
                SPDLOG_ERROR("Error parsing Attr");
                throw;
            }
        };
        std::string engine_path_;
        AlgType alg_type_;
    };

    class ClsAttr : public InferAttr
    {
    public:
        void parse(const nlohmann::json &cfg) override;
        float conf_thr_;
        std::vector<std::string> labels_;
    };

    class DetAttr : public InferAttr
    {
    public:
        void parse(const nlohmann::json &cfg) override;
        float conf_thr_;
        float obj_thr_;
        float nms_thr_;
        std::vector<int> anchors_;
        std::vector<std::string> labels_;
    };

    class KpAttr : public InferAttr
    {
    public:
        void parse(const nlohmann::json &cfg) override;
    };

    class FeatAttr : public InferAttr
    {
    public:
        void parse(const nlohmann::json &cfg) override;
    };

    class StatAttr : public Attr{
    public:
        void parse(const nlohmann::json &cfg) override;
        int frame_window_size_;
        int count_thr_;
        std::vector<std::string> labels_;
    };

    class FallDownAttr : public Attr
    {
    public:
        void parse(const nlohmann::json &cfg) override;

        DetAttr det_attr_;
        KpAttr kp_attr_;
        StatAttr stat_attr_;
    };

    class FaceRecognitionAttr : public Attr
    {
    public:
        void parse(const nlohmann::json &cfg) override;

        DetAttr det_attr_;
        FeatAttr feat_attr_;
        std::string feat_database_path_;
    };

}

#endif