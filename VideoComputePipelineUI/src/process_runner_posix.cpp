/*
 * POSIX process runner module: launches VideoComputePipeline as a child
 * process, captures stdout/stderr asynchronously, tracks status/progress, and
 * provides stop support for the Linux Dear ImGui wrapper.
 */
#include "process_runner.h"

#include "path_utils.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>

namespace vcpui {

namespace {

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string errno_message(const char *prefix) {
    std::string message = prefix ? prefix : "POSIX error";
    message += ": ";
    message += std::strerror(errno);
    return message;
}

void close_fd(int &fd) {
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

}  // namespace

ProcessRunner::ProcessRunner() = default;

ProcessRunner::~ProcessRunner() {
    stop();
    if (stdoutThread_.joinable()) stdoutThread_.join();
    if (stderrThread_.joinable()) stderrThread_.join();
    if (waitThread_.joinable()) waitThread_.join();
    close_process_handles();
}

bool ProcessRunner::start(const std::wstring &commandLine, const std::wstring &workingDirectory) {
    stop();
    if (stdoutThread_.joinable()) stdoutThread_.join();
    if (stderrThread_.joinable()) stderrThread_.join();
    if (waitThread_.joinable()) waitThread_.join();
    close_process_handles();
    clear_logs();

    int stdoutPipe[2] = {-1, -1};
    int stderrPipe[2] = {-1, -1};
    if (pipe(stdoutPipe) != 0 || pipe(stderrPipe) != 0) {
        close_fd(stdoutPipe[0]);
        close_fd(stdoutPipe[1]);
        close_fd(stderrPipe[0]);
        close_fd(stderrPipe[1]);
        lastError_ = errno_message("failed to create subprocess pipes");
        push_log(lastError_, true);
        set_status(ProcessStatus::Failed);
        return false;
    }

    const std::string command = wide_to_utf8(commandLine);
    const std::string cwd = wide_to_utf8(workingDirectory);
    const pid_t pid = fork();
    if (pid < 0) {
        close_fd(stdoutPipe[0]);
        close_fd(stdoutPipe[1]);
        close_fd(stderrPipe[0]);
        close_fd(stderrPipe[1]);
        lastError_ = errno_message("fork failed");
        push_log(lastError_, true);
        set_status(ProcessStatus::Failed);
        return false;
    }

    if (pid == 0) {
        dup2(stdoutPipe[1], STDOUT_FILENO);
        dup2(stderrPipe[1], STDERR_FILENO);
        close(stdoutPipe[0]);
        close(stdoutPipe[1]);
        close(stderrPipe[0]);
        close(stderrPipe[1]);

        if (!cwd.empty() && chdir(cwd.c_str()) != 0) {
            _exit(127);
        }

        execl("/bin/sh", "sh", "-c", command.c_str(), (char *)nullptr);
        _exit(127);
    }

    close_fd(stdoutPipe[1]);
    close_fd(stderrPipe[1]);
    childPid_ = pid;
    stdoutRead_ = stdoutPipe[0];
    stderrRead_ = stderrPipe[0];

    startSeconds_ = now_seconds();
    finishSeconds_ = 0.0;
    exitCode_ = 0;
    progress_ = {};
    progress_.running = true;
    set_status(ProcessStatus::Running);
    stdoutThread_ = std::thread(&ProcessRunner::reader_loop, this, reinterpret_cast<void *>(static_cast<intptr_t>(stdoutRead_)), false);
    stderrThread_ = std::thread(&ProcessRunner::reader_loop, this, reinterpret_cast<void *>(static_cast<intptr_t>(stderrRead_)), true);
    waitThread_ = std::thread(&ProcessRunner::wait_loop, this);
    return true;
}

void ProcessRunner::stop() {
    ProcessStatus current = status();
    if (current == ProcessStatus::Running && childPid_ > 0) {
        set_status(ProcessStatus::Stopping);
        kill(childPid_, SIGTERM);
    }
}

void ProcessRunner::clear_logs() {
    std::lock_guard<std::mutex> lock(mutex_);
    logs_.clear();
}

void ProcessRunner::poll() {
    RunProgress progress = progress_snapshot();
    update_elapsed_progress(progress, elapsed_seconds());
    std::lock_guard<std::mutex> lock(mutex_);
    progress_.elapsedSeconds = progress.elapsedSeconds;
    progress_.etaSeconds = progress.etaSeconds;
}

ProcessStatus ProcessRunner::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

int ProcessRunner::exit_code() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return exitCode_;
}

std::string ProcessRunner::last_error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastError_;
}

std::vector<LogLine> ProcessRunner::logs_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<LogLine>(logs_.begin(), logs_.end());
}

RunProgress ProcessRunner::progress_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return progress_;
}

double ProcessRunner::elapsed_seconds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (startSeconds_ <= 0.0) {
        return 0.0;
    }
    if (finishSeconds_ > 0.0) {
        return finishSeconds_ - startSeconds_;
    }
    return now_seconds() - startSeconds_;
}

void ProcessRunner::push_log(const std::string &line, bool isError) {
    std::lock_guard<std::mutex> lock(mutex_);
    logs_.push_back({line, isError});
    while (logs_.size() > 10000u) {
        logs_.pop_front();
    }
    parse_progress_line(line, progress_);
}

void ProcessRunner::set_status(ProcessStatus status) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
    progress_.running = status == ProcessStatus::Running || status == ProcessStatus::Stopping;
}

void ProcessRunner::reader_loop(void *pipeHandle, bool isError) {
    const int fd = static_cast<int>(reinterpret_cast<intptr_t>(pipeHandle));
    char buffer[4096];
    std::string pending;
    ssize_t readCount = 0;
    while ((readCount = read(fd, buffer, sizeof(buffer))) > 0) {
        pending.append(buffer, buffer + readCount);
        size_t pos = 0;
        while ((pos = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, pos);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            push_log(line, isError);
            pending.erase(0, pos + 1u);
        }
    }
    if (!pending.empty()) {
        push_log(pending, isError);
    }
}

void ProcessRunner::wait_loop() {
    if (childPid_ <= 0) {
        return;
    }

    int status = 0;
    const pid_t waited = waitpid(childPid_, &status, 0);
    int code = 1;
    if (waited > 0) {
        if (WIFEXITED(status)) {
            code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            code = 128 + WTERMSIG(status);
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        exitCode_ = code;
        finishSeconds_ = now_seconds();
        status_ = code == 0 ? ProcessStatus::Completed : ProcessStatus::Failed;
        progress_.elapsedSeconds = finishSeconds_ - startSeconds_;
        progress_.running = false;
        childPid_ = -1;
    }
}

void ProcessRunner::close_process_handles() {
    close_fd(stdoutRead_);
    close_fd(stderrRead_);
}

const char *status_to_string(ProcessStatus status) {
    switch (status) {
        case ProcessStatus::Running: return "running";
        case ProcessStatus::Stopping: return "stopping";
        case ProcessStatus::Completed: return "completed";
        case ProcessStatus::Failed: return "failed";
        case ProcessStatus::Idle:
        default: return "idle";
    }
}

}  // namespace vcpui
