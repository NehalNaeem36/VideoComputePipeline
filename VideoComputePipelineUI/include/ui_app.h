#ifndef VCP_UI_UI_APP_H
#define VCP_UI_UI_APP_H

#include "command_builder.h"
#include "process_runner.h"

#include <string>

namespace vcpui {

class UiApp {
public:
    int run();

private:
    void render();
    void render_run_config_tab();
    void render_monitor_tab();
    void render_logs_tab();
    void render_help_tab();
    void render_tooltip(const char *text);
    void mark_custom();
    void start_pipeline();
    void stop_pipeline();
    void refresh_command();
    void normalize_default_paths();

    PipelineRunConfig config_{};
    BuiltCommand command_{};
    ProcessRunner runner_;
    std::string logFilter_;
    bool autoScroll_ = true;
    bool validationEnabled_ = true;
};

}  // namespace vcpui

#endif
