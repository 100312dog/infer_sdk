#ifndef FACE_DATABASE_H
#define FACE_DATABASE_H

#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <unistd.h>
#include <tuple>

namespace infer_sdk{
    class FeatDatabase{
    public:
        FeatDatabase(const std::string& database_path, int feat_len);
        bool dump_feat(int id, const cv::Mat &feat);
        std::vector<std::tuple<int, float>> query(const cv::Mat &feat, float conf_thr, int top_k=1);
        int num_feats_;
    private:
        void load_database();
        void create_database();
        std::vector<cv::Mat> feats_;
        std::vector<int> ids_;
        int feat_len_;
        std::string database_path_;
    };
}


#endif
