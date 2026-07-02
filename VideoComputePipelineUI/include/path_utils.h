#ifndef VCP_UI_PATH_UTILS_H
#define VCP_UI_PATH_UTILS_H

#include <cstdint>
#include <string>

namespace vcpui {

std::wstring utf8_to_wide(const std::string &text);
std::string wide_to_utf8(const std::wstring &text);
bool file_exists(const std::string &path);
bool directory_exists(const std::string &path);
std::uint64_t file_size_bytes(const std::string &path);
std::string resolve_path(const std::string &baseDirectory, const std::string &path);
std::string parent_directory(const std::string &path);
std::string format_bytes(std::uint64_t bytes);
std::string executable_directory();

}  // namespace vcpui

#endif
