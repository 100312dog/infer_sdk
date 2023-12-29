#include "stat.h"

namespace infer_sdk{
    using namespace std;
    using json = nlohmann::json;

    json Stat::update(const vector<bool>& current_frame_result) {
        // update multi_frame_result
        for (int i=0; i < stat_attr_.labels_.size(); i++) {
            if(multi_frame_result_[i].size() < stat_attr_.frame_window_size_){
                multi_frame_result_[i].push_back(current_frame_result[i]);
            }
            else{
                multi_frame_result_[i].pop_front();
                multi_frame_result_[i].push_back(current_frame_result[i]);
            }
        }

        std::vector<int> multi_frame_count(stat_attr_.labels_.size());

        // count for multi_frame_result
        for(int i=0; i < stat_attr_.labels_.size(); i++){
            int count = 0;
            for (auto result: multi_frame_result_[i]){
                if(result)
                    count++;
            }
            multi_frame_count[i] = count;
        }
        auto max_count = max_element(multi_frame_count.begin(),multi_frame_count.end());
        int max_count_idx = max_count - multi_frame_count.begin();
        if (*max_count > stat_attr_.count_thr_)
            return json({{"label", stat_attr_.labels_[max_count_idx]},
                         {"count", *max_count}});
        else
            return {};
    }

    void Stat::clear(){
        multi_frame_result_.clear();
        multi_frame_result_.resize(stat_attr_.labels_.size());
    }

    int Stat::get_num_labels(){
        return stat_attr_.labels_.size();
    }

    json MultiStat::update(int track_id, const std::vector<bool>& current_frame_result){
        std::shared_ptr<Stat> stat;
        if (stats_.count(track_id) == 0) {
            // if stat_collector of track_id does not exist, create one.
            stat = std::make_shared<Stat>(stat_attr_);
            stats_.insert(make_pair(track_id, stat));
        } else
            stat = stats_[track_id];

        return stat->update(current_frame_result);
    }

    void MultiStat::clear(){
        stats_.clear();
    }

    int MultiStat::get_num_labels(){
        return stat_attr_.labels_.size();
    }

}
