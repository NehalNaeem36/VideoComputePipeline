#ifndef VIDEOCOMPUTEPIPELINE_BENCHMARK_TIMER_H
#define VIDEOCOMPUTEPIPELINE_BENCHMARK_TIMER_H

#include <stdint.h>

/**
 * High-resolution timer
 */
typedef struct {
    uint64_t start_time;
    uint64_t end_time;
    double elapsed_ms;
} Timer;

/**
 * Create and start timer
 */
Timer* timer_create(void);

/**
 * Start timing
 */
void timer_start(Timer *timer);

/**
 * Stop timing and calculate elapsed time
 */
double timer_stop(Timer *timer);

/**
 * Get elapsed time in milliseconds
 */
double timer_get_elapsed_ms(Timer *timer);

/**
 * Free timer
 */
void timer_destroy(Timer *timer);

#endif // VIDEOCOMPUTEPIPELINE_BENCHMARK_TIMER_H
