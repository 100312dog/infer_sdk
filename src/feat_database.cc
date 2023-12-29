#include "feat_database.h"

namespace infer_sdk{
    using namespace std;
    FeatDatabase::FeatDatabase(const string &database_path, int feat_len) :
    database_path_(database_path), feat_len_(feat_len) {
        if(access(database_path_.c_str(), R_OK) == 0){
            SPDLOG_INFO("Loading database");
            load_database();
            SPDLOG_INFO("{} feats found", num_feats_);
        }else{
            SPDLOG_INFO("Creating new database");
            create_database();
        }
    }

    void FeatDatabase::load_database() {
        ifstream in_file(database_path_, ios::in | ios::binary);
        if (in_file.is_open()) {
            in_file.read(reinterpret_cast<char *>(&num_feats_), sizeof(int));
            ids_.resize(num_feats_);
            for (int i = 0; i < num_feats_; i++) {
                feats_.emplace_back(1, feat_len_, CV_32F);
                in_file.read(reinterpret_cast<char *>(&ids_[i]), sizeof(int));
                in_file.read(reinterpret_cast<char *>(feats_[i].data), sizeof(float) * feat_len_);
            }
            in_file.close();
        } else {
            SPDLOG_ERROR("Can not open file {}", database_path_);
        }
    }

    void FeatDatabase::create_database() {
        ofstream out_file(database_path_, ios::out | ios::binary);
        if (out_file.is_open()) {
            num_feats_ = 0;
            out_file.write(reinterpret_cast<char *>(&num_feats_), sizeof(int));
            out_file.close();
        } else {
            SPDLOG_ERROR("Can not open file {}", database_path_);
        }
    }

    bool FeatDatabase::dump_feat(int id, const cv::Mat &feat){
        if(feat.cols != feat_len_){
            SPDLOG_ERROR("Invalid feat len");
            return false;
        }
        fstream io_file(database_path_, ios::in | ios::out | ios::binary);
        if (io_file.is_open()) {
            num_feats_++;
            feats_.emplace_back(feat.clone());
            ids_.emplace_back(id);
            io_file.seekp(0);
            io_file.write(reinterpret_cast<char *>(&num_feats_), sizeof(int));
            io_file.seekp(0,ios::end);
            io_file.write(reinterpret_cast<char *>(&id), sizeof(int));
            io_file.write(reinterpret_cast<char *>(feat.data), sizeof(float) * feat_len_);
            io_file.close();
            return true;
        } else {
            SPDLOG_ERROR("Can not open file {}", database_path_);
            return false;
        }
    }

    vector<tuple<int, float>> FeatDatabase::query(const cv::Mat &query_feat, float conf_thr, int top_k){
        assert(top_k > 0);
        vector<tuple<int, float>> res;
        int max;
        for(int i = 0; i < num_feats_; i++){
            auto feat = feats_[i];
            int id = ids_[i];
            float score = query_feat.dot(feat);
            if(score < conf_thr)
                continue;
            if(res.size() < top_k){
                res.emplace_back(id, score);
                sort(res.begin(), res.end(), [](const auto& r1, const auto& r2){
                    return get<1>(r1) > get<1>(r2);});
                max = get<1>(res[res.size() - 1]);
            }else{
                if(score > max){
                    res.pop_back();
                    res.emplace_back(id, score);
                    sort(res.begin(), res.end(), [](const auto& r1, const auto& r2){
                        return get<1>(r1) > get<1>(r2);});
                    max = get<1>(res[res.size() - 1]);
                }
            }
        }
        return res;
    }

}