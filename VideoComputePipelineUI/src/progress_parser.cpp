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

    return false;
}

void update_elapsed_progress(RunProgress &progress, double elapsedSeconds) {
    progress.elapsedSeconds = elapsedSeconds;
    if (progress.progress > 0.0 && progress.progress < 1.0) {
        progress.etaSeconds = elapsedSeconds * ((1.0 - progress.progress) / progress.progress);
    } else {
        progress.etaSeconds = 0.0;
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
