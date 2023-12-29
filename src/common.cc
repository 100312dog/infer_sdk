#include "common.h"

namespace infer_sdk
{

    using namespace std;
    using json = nlohmann::json;



    // void from_json(const json &j, ClsType &cls_type) {
    //     string str = j.get<string>();
    //     if (str == "img")
    //         cls_type = ClsType::img;
    //     else if (str == "KEYPOINT")
    //         cls_type = ClsType::KEYPOINT;
    //     else if (str == "ARCFace")
    //         cls_type = ClsType::ARCFace;
    //     else
    //         INFOE("Invalid ClsType: %s must be img, KEYPOINT or ARCFace", str.c_str());
    // }

    json load_json(const string &json_path)
    {
        // Open the JSON file
        ifstream file(json_path);
        if (!file.is_open())
        {
            cerr << "Error opening file\n";
        }

        // Parse the JSON file
        json json_data;
        try
        {
            file >> json_data;
            file.close();
        }
        catch (json::parse_error &e)
        {
            cerr << "Error parsing JSON: " << e.what() << "\n";
        }
        return json_data;
    }


    void resize(const cv::Mat& origin_img, cv::Mat& dst_img, float &scale_x, float &scale_y){
        cv::Size origin_size = origin_img.size();
        cv::Size target_size = dst_img.size();

        if(origin_size != target_size){
            scale_x = (float) origin_size.width / (float) target_size.width;
            scale_y = (float) origin_size.height / (float) target_size.height;
            cv::resize(origin_img, dst_img, target_size, 0, 0, cv::InterpolationFlags::INTER_NEAREST);
        }else{
            scale_x = scale_y = 1;
            origin_img.copyTo(dst_img);
        }
    }

    void resize_padding(const cv::Mat &origin_img, cv::Mat &dst_img,
                        float &scale_x, float &scale_y,
                        int &pad_top, int &pad_bottom, int &pad_left, int &pad_right) {

        cv::Size origin_size = origin_img.size();
        cv::Size target_size = dst_img.size();
        if (origin_size != target_size){
            float target_wh_ratio = (float) target_size.width / (float) target_size.height;
            float input_wh_ratio = (float) origin_size.width / (float) origin_size.height;
            int new_height, new_width;

            if (input_wh_ratio >= target_wh_ratio){
                // pad height
                scale_y = scale_x = (float) origin_size.width / (float) target_size.width;
                new_height = (int)((float) origin_size.height / scale_y);
                new_width = target_size.width;
                pad_top = (target_size.height - new_height) / 2;
                pad_bottom = target_size.height - new_height - pad_top;
                pad_left = pad_right =  0;
            }else{
                // pad width
                scale_x = scale_y = (float) origin_size.height / (float) target_size.height;
                new_height = target_size.height;
                new_width = (int)((float) origin_size.width / scale_x);
                pad_top = pad_bottom = 0;
                pad_left = (target_size.width - new_width) / 2;
                pad_right = target_size.width - new_width - pad_left;
            }

            cv::Mat resized_img;
            cv::resize(origin_img, resized_img, cv::Size(new_width, new_height), 0, 0, cv::InterpolationFlags::INTER_NEAREST);
            if (pad_top != 0 | pad_bottom != 0 | pad_left != 0 | pad_right != 0)
                cv::copyMakeBorder(resized_img, dst_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
            else
                resized_img.copyTo(dst_img);
        }else{
            scale_x = scale_y = 1;
            pad_top = pad_bottom = pad_left = pad_right = 0;
            origin_img.copyTo(dst_img);
        }
    }

    float iou(const BBox &a, const BBox &b) {
        int cleft = max(a.left_, b.left_);
        int ctop = max(a.top_, b.top_);
        int cright = min(a.right_, b.right_);
        int cbottom = min(a.bottom_, b.bottom_);

        int c_area = max(cright - cleft, 0) * max(cbottom - ctop, 0);
        if (c_area == 0)
            return 0.f;

        int a_area = max(0, a.right_ - a.left_) * max(0, a.bottom_ - a.top_);
        int b_area = max(0, b.right_ - b.left_) * max(0, b.bottom_ - b.top_);
        return static_cast<float>(c_area) / static_cast<float>(a_area + b_area - c_area);
    }

    void cpu_nms(vector<BBox> &bboxes, vector<BBox> &output, float threshold, bool by_class) {
        assert(output.empty());

        std::sort(bboxes.begin(), bboxes.end(), [](const BBox& a, const BBox& b) {
            return a.conf_ > b.conf_;
        });

        vector<bool> remove_flags(bboxes.size());
        for (int i = 0; i < bboxes.size(); ++i) {

            if (remove_flags[i]) continue;

            auto& a = bboxes[i];
            output.emplace_back(a);

            for (int j = i + 1; j < bboxes.size(); ++j) {
                if (remove_flags[j]) continue;

                auto& b = bboxes[j];
                if (by_class & (b.label_ != a.label_))
                    continue;
                else{
                    if (iou(a, b) >= threshold)
                        remove_flags[j] = true;
                }
            }
        }
    }


    void cpu_nms(vector<Face> &faces, vector<Face> &output, float threshold) {
        assert(output.empty());

        std::sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) {
            return a.bbox_.conf_ > b.bbox_.conf_;
        });

        vector<bool> remove_flags(faces.size());
        for (int i = 0; i < faces.size(); ++i) {

            if (remove_flags[i]) continue;

            auto& a = faces[i];
            output.emplace_back(a);

            for (int j = i + 1; j < faces.size(); ++j) {
                if (remove_flags[j]) continue;

                auto& b = faces[j];
                if (iou(a.bbox_, b.bbox_) >= threshold)
                    remove_flags[j] = true;
            }
        }
    }

    void crop_img_with_padding_bbox(const cv::Mat &img, const BBox &bbox,
                                   cv::Mat &crop_img, BBox &padding_bbox,
                                   float expand_ratio) {
        int center_x = (bbox.left_ + bbox.right_) / 2.;
        int center_y = (bbox.top_ + bbox.bottom_) / 2.;
        int half_h = (bbox.bottom_ - bbox.top_) / 2.;
        int half_w = (bbox.right_ - bbox.left_) / 2.;

        // adjust h or w to keep img ratio, expand the shorter edge
        if (half_h * 3 > half_w * 4) {
            half_w = static_cast<int>(half_h * 0.75);
        } else {
            half_h = static_cast<int>(half_w * 4 / 3);
        }

        padding_bbox.left_ = center_x - static_cast<int>(half_w * (1 + expand_ratio));
        padding_bbox.top_ = center_y - static_cast<int>(half_h * (1 + expand_ratio));
        padding_bbox.right_ = static_cast<int>(center_x + half_w * (1 + expand_ratio));
        padding_bbox.bottom_ = static_cast<int>(center_y + half_h * (1 + expand_ratio));
        padding_bbox.conf_ = bbox.conf_;
        padding_bbox.label_ = bbox.label_;

        limit_bbox_within_img(padding_bbox, img.cols, img.rows);
        crop_img = img(cv::Range(padding_bbox.top_, padding_bbox.bottom_ + 1),
                         cv::Range(padding_bbox.left_, padding_bbox.right_ + 1));
    }

    void draw_bboxes(cv::Mat &img, const vector<BBox> &bboxes, const DrawAttr& draw_attr){
        for (const auto &bbox: bboxes) {
            draw_bbox(img, bbox, draw_attr);
        }
    }

    void draw_faces(cv::Mat &img, const vector<Face> &faces, const DrawAttr& draw_attr){
        for (const auto &face: faces) {
            draw_face(img, face, draw_attr);
        }
    }

    void draw_coco_kps(cv::Mat& img, const vector<Kp>& kps, const DrawAttr& draw_attr) {
        assert(kps.size() == 17);
        const int edge[17][2] = {{0,  1},
                                 {0,  2},
                                 {1,  3},
                                 {2,  4},
                                 {3,  5},
                                 {4,  6},
                                 {5,  7},
                                 {6,  8},
                                 {7,  9},
                                 {8,  10},
                                 {5,  11},
                                 {6,  12},
                                 {11, 13},
                                 {12, 14},
                                 {13, 15},
                                 {14, 16},
                                 {11, 12}};
        // draw points
        draw_kps(img, kps);
        // draw lines
        for (int i = 0; i < 17; i++) {
            int x_start = int(kps[edge[i][0]].x_);
            int y_start = int(kps[edge[i][0]].y_);
            int x_end = int(kps[edge[i][1]].x_);
            int y_end = int(kps[edge[i][1]].y_);
            cv::line(img, cv::Point2d(x_start, y_start), cv::Point2d(x_end, y_end), draw_attr.color_, draw_attr.line_width_);
        }
    }

    vector<std::string> get_files_in_folder(const string& folder_path) {
        std::vector<std::string> files;
        DIR* dir = opendir(folder_path.c_str());
        if (dir != nullptr) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                if (entry->d_type == DT_REG) {
                    files.push_back(folder_path + "/" + entry->d_name);
                }
            }
            closedir(dir);
        }
        std::sort(files.begin(), files.end());
        return files;
    }

    std::string get_current_time(){
        auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
        auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now);

        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm *parts = std::localtime(&now_c);

        char time_string[100];
//        std::strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", parts);

        std::strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S.", parts);
        std::string timestamp = time_string + std::to_string((now_us - now_ms).count()).substr(0, 3);
        return {timestamp};
    }

}