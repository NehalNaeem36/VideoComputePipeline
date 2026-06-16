#include "utils/logger.h"

#include <stdarg.h>
#include <stdio.h>

static void log_message(const char *level, const char *fmt, va_list args) {
    fprintf(stderr, "[%s] ", level);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

void log_info(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message("INFO", fmt, args);
    va_end(args);
}

void log_warn(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message("WARN", fmt, args);
    va_end(args);
}

void log_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message("ERROR", fmt, args);
    va_end(args);
}
