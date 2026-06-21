#ifndef VCP_UI_COMMAND_BUILDER_H
#define VCP_UI_COMMAND_BUILDER_H

#include "pipeline_run_config.h"

#include <string>
#include <vector>

namespace vcpui {

struct BuiltCommand {
    std::vector<std::string> args;
    std::string preview;
    std::wstring win32CommandLine;
};

BuiltCommand build_command(const PipelineRunConfig &config);
std::string quote_arg_for_preview(const std::string &arg);
std::wstring quote_arg_for_win32(const std::wstring &arg);

}  // namespace vcpui

#endif
