#include "parser.h"

namespace infer_sdk{
    using json = nlohmann::json;
    Parser::Parser() {
        task_constraint = std::make_shared<TCLAP::ValuesConstraint<std::string>>(std::vector<std::string>{"find","face","emotion","fitness","action","falldown"});
        taskArg = std::make_shared<TCLAP::ValueArg<std::string>>("t", "task", "task type", true, "", task_constraint.get());
        cmd->add(*taskArg);

//    auto mode_constraint = TCLAP::ValuesConstraint<std::string>((std::vector<std::string>{"folder", "video", "camera"}));
        mode_constraint = std::make_shared<TCLAP::ValuesConstraint<std::string>>(std::vector<std::string>{"folder", "video", "camera"});
        modeArg = std::make_shared<TCLAP::ValueArg<std::string>>("m", "mode", "mode type", true, "", mode_constraint.get());
        cmd->add(*modeArg);

        pathArg = std::make_shared<TCLAP::ValueArg<std::string>>("p", "path", "video path or folder path", false, "", "string");
        cmd->add(*pathArg);

        savedirArg = std::make_shared<TCLAP::ValueArg<std::string>>("", "savedir", "record save dir", false, "", "string");
        cmd->add(*savedirArg);

        objArg = std::make_shared<TCLAP::ValueArg<std::string>>("", "obj", "find_obj for find mode", false, "", "string");
        cmd->add(*objArg);

        camidArg = std::make_shared<TCLAP::ValueArg<int>>("", "camid", "camid for video mode", false, 0 , "int");
        cmd->add(*camidArg);

        recordArg = std::make_shared<TCLAP::ValueArg<bool>>("r", "record", "record for camera mode", false, false, "bool");
        cmd->add(*recordArg);

        showArg = std::make_shared<TCLAP::ValueArg<bool>>("s", "show", "imshow by opencv", false, true, "bool");
        cmd->add(*showArg);

        resolutionArg = std::make_shared<TCLAP::ValueArg<std::string>>("", "resolution", "resolution for camera mode", false, "1280x720" , "string");
        cmd->add(*resolutionArg);
    }

    void Parser::parse(int argc, char **argv) {
        cmd->parse(argc, argv);

        task = taskArg->getValue();
        mode = modeArg->getValue();
        path = pathArg->getValue();
        save_dir = savedirArg->getValue();
        if (!save_dir.empty() &&
            !(save_dir.substr(save_dir.size() - 1) == "/" || save_dir.substr(save_dir.size() - 1) == "\\")){
            save_dir += "/";
        }
        obj = objArg->getValue();
        cam_id = camidArg->getValue();
        record = recordArg->getValue();
        show = showArg->getValue();
        resolution = resolutionArg->getValue();

        if (mode == "folder" | mode == "video")
            if (path.empty())
                throw std::runtime_error("please specify the path for " + mode + " mode");

        if (task == "find")
            if (obj.empty())
                throw std::runtime_error("please specify the obj to find");
    }

}