#include "utils/logger.h"
#include <stdarg.h>

static LogLevel current_level = LOG_INFO;
static FILE *output_file = NULL;

void logger_init(LogLevel level, FILE *output) {
    // TODO: Implement logger initialization
    current_level = level;
    output_file = output ? output : stdout;
}

void logger_debug(const char *format, ...) {
    // TODO: Implement debug logging
}

void logger_info(const char *format, ...) {
    // TODO: Implement info logging
}

void logger_warn(const char *format, ...) {
    // TODO: Implement warning logging
}

void logger_error(const char *format, ...) {
    // TODO: Implement error logging
}

void logger_shutdown(void) {
    // TODO: Implement cleanup
}
