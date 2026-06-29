/*
 * Progress parser module: extracts frame progress, FPS, execution-plan fields,
 * and timing estimates from pipeline log lines. ProcessRunner updates this
 * state as logs arrive so the Monitor tab can stay responsive.
 */
#include "progress_parser.h"

#include <algorithm>
#include <iomanip>
#include <regex>
#include <sstream>

namespace vcpui {

namespace {

double parse_double(const std::string &text, double fallback = 0.0) {
    try {
        return std::stod(text);
    } catch (...) {
        return fallback;
    }
}

int parse_int(const std::string &text, int fallback = 0) {
    try {
        return std::stoi(text);
    } catch (...) {
        return fallback;
    }
}

bool parse_bool(const std::string &text, bool fallback = false) {
    if (text == "true" || text == "1" || text == "yes") {
        return true;
    }
    if (text == "false" || text == "0" || text == "no") {
        return false;
    }
    return fallback;
}

std::string trim_copy(const std::string &value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\r' || value[begin] == '\n')) {
        ++begin;
    }
    while (end > begin && (value[end - 1u] == ' ' || value[end - 1u] == '\t' || value[end - 1u] == '\r' || value[end - 1u] == '\n')) {
        --end;
    }
    return value.substr(begin, end - begin);
}

}  // namespace

bool parse_progress_line(const std::string &line, RunProgress &progress) {
    progress.lastStatusLine = line;

    if (line.rfind("PROGRESS ", 0) == 0) {
        std::istringstream input(line.substr(9));
        std::string token;
        while (input >> token) {
            const size_t eq = token.find('=');
            if (eq == std::string::npos) {
                continue;
            }
            const std::string key = token.substr(0, eq);
            const std::string value = token.substr(eq + 1);
            if (key == "frame") progress.framesProcessed = parse_int(value, progress.framesProcessed);
            else if (key == "total_frames") progress.totalFrames = parse_int(value, progress.totalFrames);
            else if (key == "timestamp_ms") progress.timestampMs = parse_double(value, progress.timestampMs);
            else if (key == "duration_ms") progress.durationMs = parse_double(value, progress.durationMs);
            else if (key == "fps") progress.fps = parse_double(value, progress.fps);
            else if (key == "speed") progress.speed = parse_double(value, progress.speed);
            else if (key == "detections") progress.totalDetections = parse_int(value, progress.totalDetections);
            else if (key == "output_bytes") progress.outputBytes = (std::uint64_t)parse_double(value, (double)progress.outputBytes);
        }
        if (progress.totalFrames > 0) {
            progress.progress = std::clamp((double)progress.framesProcessed / (double)progress.totalFrames, 0.0, 1.0);
        } else if (progress.durationMs > 0.0) {
            progress.progress = std::clamp(progress.timestampMs / progress.durationMs, 0.0, 1.0);
        }
        return true;
    }

    static const std::regex planField(R"(^\s*([a-zA-Z0-9_]+):\s*(.*?)\s*$)");
    std::smatch planMatch;
    if (std::regex_match(line, planMatch, planField)) {
        const std::string key = planMatch[1].str();
        const std::string value = trim_copy(planMatch[2].str());
        if (key == "execution_mode") progress.executionMode = parse_int(value, progress.executionMode);
        else if (key == "batch_size") progress.batchSize = parse_int(value, progress.batchSize);
        else if (key == "schedule_batch_size") progress.scheduleBatchSize = parse_int(value, progress.scheduleBatchSize);
        else if (key == "backend_batch_size") progress.backendBatchSize = parse_int(value, progress.backendBatchSize);
        else if (key == "inflight_batches") progress.inflightBatches = parse_int(value, progress.inflightBatches);
        else if (key == "total_active_frames") progress.totalActiveFrames = parse_int(value, progress.totalActiveFrames);
        else if (key == "active_frame_capacity") progress.activeFrameCapacity = parse_int(value, progress.activeFrameCapacity);
        else if (key == "frames_per_upload_batch") progress.framesPerUploadBatch = parse_int(value, progress.framesPerUploadBatch);
        else if (key == "frames_per_download_batch") progress.framesPerDownloadBatch = parse_int(value, progress.framesPerDownloadBatch);
        else if (key == "inference_context_count") progress.inferenceContextCount = parse_int(value, progress.inferenceContextCount);
        else if (key == "inference_lane_count") progress.inferenceLaneCount = parse_int(value, progress.inferenceLaneCount);
        else if (key == "pipeline_overlap_enabled") progress.pipelineOverlapEnabled = parse_bool(value, progress.pipelineOverlapEnabled);
        else if (key == "parallel_inference_enabled") progress.parallelInferenceEnabled = parse_bool(value, progress.parallelInferenceEnabled);
        else if (key == "vram_budget_mb") progress.vramBudgetMb = parse_double(value, progress.vramBudgetMb);
        else if (key == "estimated_batch_mb") progress.estimatedBatchMb = parse_double(value, progress.estimatedBatchMb);
        else if (key == "unused_vram_budget_mb") progress.unusedVramBudgetMb = parse_double(value, progress.unusedVramBudgetMb);
        else if (key == "reason") progress.fallbackReason = value;
        else return false;
        return true;
    }

    static const std::regex humanProgress(R"(progress:\s+completed\s+([0-9]+).*wall_clock_fps=([0-9.]+))",
                                          std::regex_constants::icase);
    std::smatch match;
    if (std::regex_search(line, match, humanProgress)) {
        progress.framesProcessed = parse_int(match[1].str(), progress.framesProcessed);
        progress.fps = parse_double(match[2].str(), progress.fps);
        if (progress.totalFrames > 0) {
            progress.progress = std::clamp((double)progress.framesProcessed / (double)progress.totalFrames, 0.0, 1.0);
        }
        return true;
    }

    static const std::regex hardwarePipeline(
        R"(hardware detection pipeline:.*batch_capacity=([0-9]+).*inflight_batches=([0-9]+).*inference_workers=([0-9]+))",
        std::regex_constants::icase);
    if (std::regex_search(line, match, hardwarePipeline)) {
        progress.scheduleBatchSize = parse_int(match[1].str(), progress.scheduleBatchSize);
        progress.batchSize = progress.scheduleBatchSize;
        progress.inflightBatches = parse_int(match[2].str(), progress.inflightBatches);
        progress.inferenceLaneCount = parse_int(match[3].str(), progress.inferenceLaneCount);
        progress.inferenceContextCount = progress.inferenceLaneCount;
        progress.totalActiveFrames = progress.scheduleBatchSize * progress.inflightBatches;
        progress.activeFrameCapacity = progress.totalActiveFrames;
        progress.pipelineOverlapEnabled = progress.inflightBatches > 1 || progress.inferenceLaneCount > 1;
        progress.parallelInferenceEnabled = progress.inferenceLaneCount > 1;
        progress.fallbackReason = "physical staged NVDEC -> inference -> output pipeline";
        return true;
    }

    return false;
}

void update_elapsed_progress(RunProgress &progress, double elapsedSeconds) {
    progress.elapsedSeconds = elapsedSeconds;
    if (progress.progress > 0.0 && progress.progress < 1.0) {
        progress.etaSeconds = elapsedSeconds * ((1.0 - progress.progress) / progress.progress);
        progress.estimatedTotalSeconds = elapsedSeconds / progress.progress;
    } else {
        progress.etaSeconds = 0.0;
        progress.estimatedTotalSeconds = progress.progress >= 1.0 ? elapsedSeconds : 0.0;
    }
}

std::string format_duration(double seconds) {
    if (seconds <= 0.0) {
        return "--";
    }
    const int total = (int)seconds;
    const int h = total / 3600;
    const int m = (total % 3600) / 60;
    const int s = total % 60;
    std::ostringstream out;
    if (h > 0) {
        out << h << ':';
        out << std::setw(2) << std::setfill('0') << m << ':';
    } else {
        out << m << ':';
    }
    out << std::setw(2) << std::setfill('0') << s;
    return out.str();
}

}  // namespace vcpui
