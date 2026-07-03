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
    void refresh_input_files_if_needed();
    void sync_input_path_from_selection();
    void refresh_model_files_if_needed();
    void sync_model_path_from_selection();
    void sync_output_artifact_paths();
    void refresh_labels_if_needed();
    bool class_id_selected(int class_id) const;
    void set_class_id_selected(int class_id, bool selected);

    PipelineRunConfig config_{};
    BuiltCommand command_{};
    ProcessRunner runner_;
    std::vector<std::string> inputFiles_;
    std::string loadedInputFolderPath_;
    std::vector<std::string> modelFiles_;
    std::string loadedModelFolderPath_;
    Runtime loadedModelRuntime_ = Runtime::Auto;
    std::vector<std::string> classLabels_;
    std::string loadedLabelsPath_;
    std::string classSearch_;
    std::string logFilter_;
    std::string logSaveStatus_;
    bool autoScroll_ = true;
    bool validationEnabled_ = true;
    bool logShowErrors_ = true;
    bool logShowWarnings_ = true;
    bool logShowTensorRt_ = true;
    bool logShowExecutionPlan_ = true;
};

}  // namespace vcpui

#endif
