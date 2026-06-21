#ifndef VCP_UI_PROGRESS_PARSER_H
#define VCP_UI_PROGRESS_PARSER_H

#include <cstdint>
#include <string>

namespace vcpui {

struct RunProgress {
    bool running = false;
    int framesProcessed = 0;
    int totalFrames = 0;
    double timestampMs = 0.0;
    double durationMs = 0.0;
    double progress = 0.0;
    double fps = 0.0;
    double speed = 0.0;
    double elapsedSeconds = 0.0;
    double etaSeconds = 0.0;
    int totalDetections = 0;
    std::uint64_t outputBytes = 0;
    std::string lastStatusLine;
};

bool parse_progress_line(const std::string &line, RunProgress &progress);
void update_elapsed_progress(RunProgress &progress, double elapsedSeconds);
std::string format_duration(double seconds);

}  // namespace vcpui

#endif
