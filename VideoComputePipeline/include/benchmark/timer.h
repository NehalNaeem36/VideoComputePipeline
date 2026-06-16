#ifndef VIDEOCOMPUTEPIPELINE_BENCHMARK_TIMER_H
#define VIDEOCOMPUTEPIPELINE_BENCHMARK_TIMER_H

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct {
#ifdef _WIN32
    LARGE_INTEGER start;
    LARGE_INTEGER frequency;
#else
    clock_t start;
#endif
} Timer;

void timer_start(Timer *timer);
double timer_stop_ms(Timer *timer);

#endif
