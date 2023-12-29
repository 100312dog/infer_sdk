#ifndef TASK_H
#define TASK_H

#include "infer.h"
#include "face_align.h"
#include "feat_database.h"
#include "BYTETracker.h"
#include "stat.h"

namespace infer_sdk{
    class Task{
    public:
        virtual nlohmann::json infer(const cv::Mat &img) = 0;
        virtual ~Task() = default;
        //to do cudastream pool
    };

    class FaceRecognition : public Task {
    public:
        FaceRecognition(const nlohmann::json& cfg);
        nlohmann::json infer(const cv::Mat & img) override;
        bool register_face(int id, const cv::Mat& img);
        cv::Mat result_img_;
    private:
        std::shared_ptr<FaceDetInfer> det_infer_;
        std::shared_ptr<FeatInfer> feat_infer_;
        std::unique_ptr<FeatDatabase> feat_database_;
        std::unique_ptr<byte_track::BYTETracker> tracker_;
        friend class FeatDatabase;
    };
}


#endif //TASK_H
