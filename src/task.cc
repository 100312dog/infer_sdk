#include "task.h"

namespace infer_sdk{
    using namespace cv;
    using namespace std;
    using json = nlohmann::json;


    FaceRecognition::FaceRecognition(const json& cfg){
        SPDLOG_INFO("Initializing FaceRecognition Task");
        FaceRecognitionAttr face_recognition_attr;
        face_recognition_attr.parse(cfg);
        det_infer_ = make_shared<FaceDetInfer>(face_recognition_attr.det_attr_);
        feat_infer_ = make_shared<FeatInfer>(face_recognition_attr.feat_attr_);
        feat_database_ = make_unique<FeatDatabase>(face_recognition_attr.feat_database_path_, feat_infer_->get_feat_len());
        tracker_ = make_unique<byte_track::BYTETracker>();
    }

    map<int, string> id_name_mapping = {{0, "ccc"}, {1, "fwy"}, {2, "fyc"}, {3, "wlq"}};

    nlohmann::json FaceRecognition::infer(const cv::Mat &img){
        if (feat_database_->num_feats_ == 0){
            SPDLOG_ERROR("Empty feat database.");
            return {};
        }
        result_img_ = img.clone();
        auto faces = det_infer_->infer(img);
        DrawAttr draw_attr;
        draw_attr.with_label_ = false;
        draw_faces(result_img_, faces, draw_attr);
        faces = tracker_->update(faces);
        if (faces.empty())
            return {};

        // draw faces
//        draw_attr.color_ = {0, 0, 255};
//        draw_attr.with_label_ = true;
//        draw_attr.with_id_ = true;
//        draw_faces(result_img_, faces, draw_attr);

        auto max_face = max_element(faces.begin(), faces.end(), [](const auto &face1, const auto &face2) {
            return face1.bbox_.area() < face2.bbox_.area();
        });
//        draw_face(result_img_, *max_face);
        auto aligned_max_face = FaceAlign::get_aligned_face(img, (*max_face).kps_, feat_infer_->get_input_size());
        auto max_face_feat = feat_infer_->infer(aligned_max_face);
        auto match = feat_database_->query(max_face_feat, 0.3, 1);

        // draw id
        string text;
        if(match.empty()){
            text = "Unknown";
        }else{
            stringstream stream;
            stream << fixed << setprecision(2);
            stream << id_name_mapping[get<0>(match[0])] << ": " << get<1>(match[0]);
            text = stream.str();
        }
        put_text_on_bbox(result_img_, max_face->bbox_, text, draw_attr);

        if(match.empty())
            return {};

        return json({{"id", get<0>(match[0])},
                     {"score", get<1>(match[0])}});
    }

    bool FaceRecognition::register_face(int id, const cv::Mat& img){
        auto faces = det_infer_->infer(img);
        if (faces.empty()){
            SPDLOG_INFO("Failed to register face. No face found.");
            return false;
        }
        auto max_face = max_element(faces.begin(), faces.end(), [](const auto &face1, const auto &face2) {
            return face1.bbox_.area() < face2.bbox_.area();
        });
        auto aligned_max_face = FaceAlign::get_aligned_face(img, (*max_face).kps_, feat_infer_->get_input_size());
        auto max_face_feat = feat_infer_->infer(aligned_max_face);
        return feat_database_->dump_feat(id, max_face_feat);
    }

}