#ifndef VCP_UI_PROCESS_RUNNER_H
#define VCP_UI_PROCESS_RUNNER_H

#include "progress_parser.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace vcpui {

enum class ProcessStatus { Idle = 0, Running, Stopping, Completed, Failed };

struct LogLine {
    std::string text;
    bool isError = false;
};

class ProcessRunner {
public:
    ProcessRunner();
    ~ProcessRunner();

    ProcessRunner(const ProcessRunner &) = delete;
    ProcessRunner &operator=(const ProcessRunner &) = delete;

    bool start(const std::wstring &commandLine, const std::wstring &workingDirectory);
    void stop();
    void clear_logs();
    void poll();

    ProcessStatus status() const;
    int exit_code() const;
    std::string last_error() const;
    std::vector<LogLine> logs_snapshot() const;
    RunProgress progress_snapshot() const;
    double elapsed_seconds() const;

private:
    void push_log(const std::string &line, bool isError);
    void set_status(ProcessStatus status);
    void reader_loop(void *pipeHandle, bool isError);
    void wait_loop();
    void close_process_handles();

    mutable std::mutex mutex_;
    std::deque<LogLine> logs_;
    RunProgress progress_;
    ProcessStatus status_ = ProcessStatus::Idle;
    int exitCode_ = 0;
    std::string lastError_;
    double startSeconds_ = 0.0;

#ifdef _WIN32
    PROCESS_INFORMATION processInfo_{};
    HANDLE stdoutRead_ = nullptr;
    HANDLE stderrRead_ = nullptr;
    HANDLE stdoutWrite_ = nullptr;
    HANDLE stderrWrite_ = nullptr;
#endif

    std::thread stdoutThread_;
    std::thread stderrThread_;
    std::thread waitThread_;
};

const char *status_to_string(ProcessStatus status);

}  // namespace vcpui

#endif
