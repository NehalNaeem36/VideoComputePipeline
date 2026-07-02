/*
 * Win32 process runner module: launches VideoComputePipeline.exe as a child
 * process, captures stdout/stderr asynchronously, tracks status/progress, and
 * provides stop support for the UI.
 */
#include "process_runner.h"

#include "path_utils.h"

#include <algorithm>
#include <chrono>

namespace vcpui {

namespace {

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

#ifdef _WIN32
std::string windows_error_message(DWORD error) {
    wchar_t *message = nullptr;
    const DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
    const DWORD length = FormatMessageW(flags,
                                        nullptr,
                                        error,
                                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                        (LPWSTR)&message,
                                        0,
                                        nullptr);
    std::string result = "Windows error " + std::to_string(error);
    if (length > 0 && message) {
        result += ": ";
        result += wide_to_utf8(message);
        while (!result.empty() && (result.back() == '\r' || result.back() == '\n' || result.back() == ' ')) {
            result.pop_back();
        }
    }
    if (message) {
        LocalFree(message);
    }
    return result;
}
#endif

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

#ifdef _WIN32
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    if (!CreatePipe(&stdoutRead_, &stdoutWrite_, &sa, 0) ||
        !CreatePipe(&stderrRead_, &stderrWrite_, &sa, 0)) {
        lastError_ = "failed to create subprocess pipes: " + windows_error_message(GetLastError());
        push_log(lastError_, true);
        set_status(ProcessStatus::Failed);
        return false;
    }
    SetHandleInformation(stdoutRead_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stderrRead_, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = stdoutWrite_;
    startup.hStdError = stderrWrite_;

    std::wstring mutableCommand = commandLine;
    BOOL ok = CreateProcessW(nullptr,
                             mutableCommand.data(),
                             nullptr,
                             nullptr,
                             TRUE,
                             CREATE_NO_WINDOW,
                             nullptr,
                             workingDirectory.empty() ? nullptr : workingDirectory.c_str(),
                             &startup,
                             &processInfo_);

    CloseHandle(stdoutWrite_);
    CloseHandle(stderrWrite_);
    stdoutWrite_ = nullptr;
    stderrWrite_ = nullptr;

    if (!ok) {
        lastError_ = "CreateProcessW failed: " + windows_error_message(GetLastError());
        push_log(lastError_, true);
        push_log("command: " + wide_to_utf8(commandLine), true);
        push_log("working directory: " + wide_to_utf8(workingDirectory), true);
        close_process_handles();
        set_status(ProcessStatus::Failed);
        return false;
    }

    startSeconds_ = now_seconds();
    finishSeconds_ = 0.0;
    exitCode_ = 0;
    progress_ = {};
    progress_.running = true;
    set_status(ProcessStatus::Running);
    stdoutThread_ = std::thread(&ProcessRunner::reader_loop, this, stdoutRead_, false);
    stderrThread_ = std::thread(&ProcessRunner::reader_loop, this, stderrRead_, true);
    waitThread_ = std::thread(&ProcessRunner::wait_loop, this);
    return true;
#else
    (void)commandLine;
    (void)workingDirectory;
    lastError_ = "process runner is Windows-only";
    set_status(ProcessStatus::Failed);
    return false;
#endif
}

void ProcessRunner::stop() {
#ifdef _WIN32
    ProcessStatus current = status();
    if (current == ProcessStatus::Running) {
        set_status(ProcessStatus::Stopping);
        if (processInfo_.hProcess) {
            TerminateProcess(processInfo_.hProcess, 1);
        }
    }
#endif
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
#ifdef _WIN32
    HANDLE pipe = (HANDLE)pipeHandle;
    char buffer[4096];
    std::string pending;
    DWORD read = 0;
    while (ReadFile(pipe, buffer, sizeof(buffer), &read, nullptr) && read > 0) {
        pending.append(buffer, buffer + read);
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
#else
    (void)pipeHandle;
    (void)isError;
#endif
}

void ProcessRunner::wait_loop() {
#ifdef _WIN32
    if (!processInfo_.hProcess) {
        return;
    }
    WaitForSingleObject(processInfo_.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(processInfo_.hProcess, &code);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        exitCode_ = (int)code;
        finishSeconds_ = now_seconds();
        status_ = code == 0 ? ProcessStatus::Completed : ProcessStatus::Failed;
        progress_.elapsedSeconds = finishSeconds_ - startSeconds_;
        progress_.running = false;
    }
#endif
}

void ProcessRunner::close_process_handles() {
#ifdef _WIN32
    if (stdoutRead_) CloseHandle(stdoutRead_);
    if (stderrRead_) CloseHandle(stderrRead_);
    if (stdoutWrite_) CloseHandle(stdoutWrite_);
    if (stderrWrite_) CloseHandle(stderrWrite_);
    if (processInfo_.hThread) CloseHandle(processInfo_.hThread);
    if (processInfo_.hProcess) CloseHandle(processInfo_.hProcess);
    stdoutRead_ = nullptr;
    stderrRead_ = nullptr;
    stdoutWrite_ = nullptr;
    stderrWrite_ = nullptr;
    processInfo_ = {};
#endif
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
