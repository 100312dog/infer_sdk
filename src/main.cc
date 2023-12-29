#include "task.h"
#include "parser.h"
#include <sstream>

// must contain imread. or missing library img_codec ?

using namespace std;
using json = nlohmann::json;


cv::Mat task_infer(const string &task_string, infer_sdk::Task *task, const cv::Mat& image, const string& obj, bool clear=false) {
    if (task_string == "face"){
        auto face_task = dynamic_cast<infer_sdk::FaceRecognition *>(task);
        json result = face_task->infer(image);
        return face_task->result_img_;
    }else if (task_string == "falldown"){
        auto falldown_task = dynamic_cast<infer_sdk::FallDown *>(task);
        json result = falldown_task->infer(image);
        return falldown_task->result_img_;
    }
}

infer_sdk::Task *create_task(const string &task_type, json &cfg) {
    if (task_type == "falldown")
        return new infer_sdk::FallDown(cfg);
    else if (task_type == "face"){
        auto face_task = new infer_sdk::FaceRecognition(cfg);
//        auto image_paths = infer_sdk::get_files_in_folder("data/faces");
//        for (int i = 0; i < image_paths.size(); i++) {
//            face_task->register_face(i, cv::imread(image_paths[i]));
//        }
        return face_task;
    }
    else
        SPDLOG_ERROR("unknown task");
}


void demo(infer_sdk::Parser* parser, json& cfg) {
    cv::VideoCapture cap;
    if (parser->mode == "video"){
        cap = cv::VideoCapture(parser->path);
        if (!cap.isOpened()) {
            SPDLOG_ERROR("Failed to open video: " + parser->path);
            return;
        }
    }else if(parser->mode == "camera") {
        // set resolution for camera
        int width, height;
        try{
            istringstream iss(parser->resolution);
            string width_str, height_str;
            getline(iss, width_str, 'x');
            getline(iss, height_str);
            width = std::stoi(width_str);
            height = std::stoi(height_str);
        }catch (std::exception & e){
            SPDLOG_ERROR("Wrong resolution");
        }
        cap = cv::VideoCapture(parser->cam_id);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        if (!cap.isOpened()) {
            SPDLOG_ERROR("Failed to open camera: " + to_string(parser->cam_id));
            return;
        }
    }
    cv::VideoWriter result_video_writer, origin_video_writer;
    if(parser->record){
        string time = infer_sdk::get_current_time();
        cv::Size2i frame_size = {int(cap.get(cv::CAP_PROP_FRAME_WIDTH)), int(cap.get(cv::CAP_PROP_FRAME_HEIGHT))};
        string result_filename = parser->save_dir + "result-" + time + ".mp4";
        result_video_writer.open(result_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame_size, true);
        if(parser->mode == "camera"){
            string origin_filename = parser->save_dir + "origin-" + time + ".mp4";
            origin_video_writer.open(origin_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame_size, true);
        }
    }

    infer_sdk::Task *task = create_task(parser->task, cfg);

    if (parser->mode == "video" || parser->mode == "camera") {

        cv::Mat frame;
        while (true) {

            cap >> frame;
            if (frame.empty()) {
                break;
            }

            if(parser->record && parser->mode == "camera"){
                origin_video_writer.write(frame);
            }

            auto task_result = task_infer(parser->task, task, frame, parser->obj);

            if(parser->record){
                result_video_writer.write(task_result);
            }

            if (parser->show) {
                imshow("Frame", task_result);

                char c = (char) cv::waitKey(1);
                if (c == 27) {
                    break;
                }
            }
        }

        cap.release();
        if(parser->record){
            result_video_writer.release();
            if(parser->mode == "camera"){
                origin_video_writer.release();
            }
        }
        if (parser->show)
            cv::destroyAllWindows();
    }

//    } else if (parser->mode == "folder") {
//        auto image_paths = kiwi::get_files_in_folder(cmd_parser->path);
//        for (auto image_path: image_paths) {
//            auto image = cv::imread(image_path);
//
//            auto start = kiwi::timestamp_now_float();
//            auto task_result = task_infer(cmd_parser->task, task, image, cmd_parser->obj, true);
//            auto end = kiwi::timestamp_now_float();
//            if(cfg["print_info"])
//                cout << "total:" << end - start << endl;
//
//            if (cmd_parser->show){
//                imshow("Frame", task_result);
//
//                char c = (char) cv::waitKey(1);
//                if (c == 27) {
//                    break;
//                }
//            }
//        }
//        if (cmd_parser->show)
//            cv::destroyAllWindows();
//    }

    delete task;
}



int main(int argc, char **argv) {
    auto parser = new infer_sdk::Parser();
    parser->parse(argc, argv);
    auto cfg = infer_sdk::load_json("cv_cfg.json");
    demo(parser, cfg);
    return 0;
}
