#ifndef VIDEOCOMPUTEPIPELINE_UTILS_LOGGER_H
#define VIDEOCOMPUTEPIPELINE_UTILS_LOGGER_H

#include <stdio.h>

/**
 * Log levels
 */
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
} LogLevel;

/**
 * Initialize logger
 */
void logger_init(LogLevel level, FILE *output);

/**
 * Log debug message
 */
void logger_debug(const char *format, ...);

/**
 * Log info message
 */
void logger_info(const char *format, ...);

/**
 * Log warning message
 */
void logger_warn(const char *format, ...);

/**
 * Log error message
 */
void logger_error(const char *format, ...);

/**
 * Shutdown logger
 */
void logger_shutdown(void);

#endif // VIDEOCOMPUTEPIPELINE_UTILS_LOGGER_H
