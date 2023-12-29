#ifndef PARSER_H
#define PARSER_H

#include <tclap/CmdLine.h>
#include <memory>
#include <json.hpp>

namespace infer_sdk{
    class Parser{
    public:
        Parser();
        void parse(int argc, char **argv);

    std::string task, mode, path, save_dir, obj, resolution;
    int cam_id;
    bool record, show;

    private:
        std::shared_ptr<TCLAP::CmdLine> cmd = std::make_shared<TCLAP::CmdLine>("", ' ', "1.0"); // description seperator version
        std::shared_ptr<TCLAP::ValuesConstraint<std::string>> task_constraint, mode_constraint;
        std::shared_ptr<TCLAP::ValueArg<std::string>> taskArg, modeArg, pathArg, savedirArg, objArg, resolutionArg;
        std::shared_ptr<TCLAP::ValueArg<int>> camidArg;
        std::shared_ptr<TCLAP::ValueArg<bool>> recordArg, showArg;
    };
}

#endif
