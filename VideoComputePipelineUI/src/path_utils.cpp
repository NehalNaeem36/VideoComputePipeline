/*
 * UI path utilities module: handles path checks, path resolution, file sizing,
 * and UTF-8/Windows string conversion for the desktop wrapper. Command building,
 * validation, monitoring, and process launch code share these helpers.
 */
#include "path_utils.h"

#include <filesystem>
#include <sstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <shlobj.h>
#include <windows.h>
#endif

namespace vcpui {

std::wstring utf8_to_wide(const std::string &text) {
#ifdef _WIN32
    if (text.empty()) {
        return {};
    }
    const int count = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (count <= 0) {
        return {};
    }
    std::wstring result((size_t)count - 1u, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, result.data(), count);
    return result;
#else
    return std::wstring(text.begin(), text.end());
#endif
}

std::string wide_to_utf8(const std::wstring &text) {
#ifdef _WIN32
    if (text.empty()) {
        return {};
    }
    const int count = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (count <= 0) {
        return {};
    }
    std::string result((size_t)count - 1u, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, result.data(), count, nullptr, nullptr);
    return result;
#else
    return std::string(text.begin(), text.end());
#endif
}

bool file_exists(const std::string &path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::u8path(path), ec);
}

bool directory_exists(const std::string &path) {
    std::error_code ec;
    return std::filesystem::is_directory(std::filesystem::u8path(path), ec);
}

std::uint64_t file_size_bytes(const std::string &path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(std::filesystem::u8path(path), ec);
    return ec ? 0u : (std::uint64_t)size;
}

std::string resolve_path(const std::string &baseDirectory, const std::string &path) {
    if (path.empty()) {
        return path;
    }
    std::filesystem::path p = std::filesystem::u8path(path);
    if (p.is_absolute() || baseDirectory.empty()) {
        return p.u8string();
    }
    return (std::filesystem::u8path(baseDirectory) / p).lexically_normal().u8string();
}

std::string parent_directory(const std::string &path) {
    if (path.empty()) {
        return {};
    }
    return std::filesystem::u8path(path).parent_path().u8string();
}

std::string format_bytes(std::uint64_t bytes) {
    static const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = (double)bytes;
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(unit == 0 ? 0 : 2);
    out << value << ' ' << units[unit];
    return out.str();
}

std::string executable_directory() {
#ifdef _WIN32
    std::wstring buffer(32768u, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), (DWORD)buffer.size());
    if (length == 0 || length >= buffer.size()) {
        return std::filesystem::current_path().u8string();
    }
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path().u8string();
#else
    return std::filesystem::current_path().u8string();
#endif
}

bool browse_for_folder(const char *title, std::string &selectedPath) {
#ifdef _WIN32
    BROWSEINFOW browseInfo{};
    std::wstring titleWide = utf8_to_wide(title ? title : "Select folder");
    browseInfo.lpszTitle = titleWide.c_str();
    browseInfo.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_USENEWUI;

    PIDLIST_ABSOLUTE item = SHBrowseForFolderW(&browseInfo);
    if (!item) {
        return false;
    }

    wchar_t path[MAX_PATH];
    const BOOL ok = SHGetPathFromIDListW(item, path);
    CoTaskMemFree(item);
    if (!ok) {
        return false;
    }
    selectedPath = wide_to_utf8(path);
    return true;
#else
    (void)title;
    (void)selectedPath;
    return false;
#endif
}

}  // namespace vcpui
