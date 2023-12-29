//
// Created by fsw on 23-3-27.
//

#ifndef STAT_H
#define STAT_H

#include <deque>
#include "attr.h"

namespace infer_sdk{
    class Stat{
    public:
        Stat(const StatAttr& stat_attr): stat_attr_(stat_attr), multi_frame_result_(stat_attr.labels_.size()){};
        nlohmann::json update(const std::vector<bool>& current_frame_result);
        void clear();
        int get_num_labels();

    private:
        std::vector<std::deque<bool>> multi_frame_result_;
        StatAttr stat_attr_;
    };

    class MultiStat{
    public:
        MultiStat(const StatAttr& stat_attr): stat_attr_(stat_attr){};
        nlohmann::json update(int track_id, const std::vector<bool>& current_frame_result);
        void clear();
        int get_num_labels();

    private:
        std::map<int, std::shared_ptr<Stat>> stats_;
        StatAttr stat_attr_;
    };
}

#endif
