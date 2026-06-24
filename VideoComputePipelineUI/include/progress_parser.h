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
    double estimatedTotalSeconds = 0.0;
    int totalDetections = 0;
    std::uint64_t outputBytes = 0;
    int executionMode = -1;
    int batchSize = 0;
    int scheduleBatchSize = 0;
    int backendBatchSize = 0;
    int inflightBatches = 0;
    int totalActiveFrames = 0;
    int activeFrameCapacity = 0;
    int framesPerUploadBatch = 0;
    int framesPerDownloadBatch = 0;
    int inferenceContextCount = 0;
    int inferenceLaneCount = 0;
    bool pipelineOverlapEnabled = false;
    bool parallelInferenceEnabled = false;
    double vramBudgetMb = 0.0;
    double estimatedBatchMb = 0.0;
    double unusedVramBudgetMb = 0.0;
    std::string fallbackReason;
    std::string lastStatusLine;
};

bool parse_progress_line(const std::string &line, RunProgress &progress);
void update_elapsed_progress(RunProgress &progress, double elapsedSeconds);
std::string format_duration(double seconds);

}  // namespace vcpui

#endif
