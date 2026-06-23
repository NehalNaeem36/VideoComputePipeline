/*
 * Timer module: provides small high-resolution timing helpers used by pipeline,
 * benchmark, video, and inference stages. It owns timing mechanics so other
 * modules only measure stage boundaries.
 */
#include "benchmark/timer.h"

void timer_start(Timer *timer) {
    if (!timer) {
        return;
    }

#ifdef _WIN32
    QueryPerformanceFrequency(&timer->frequency);
    QueryPerformanceCounter(&timer->start);
#else
    timer->start = clock();
#endif
}

double timer_stop_ms(Timer *timer) {
    if (!timer) {
        return 0.0;
    }

#ifdef _WIN32
    LARGE_INTEGER end;
    QueryPerformanceCounter(&end);
    return ((double)(end.QuadPart - timer->start.QuadPart) * 1000.0) /
           (double)timer->frequency.QuadPart;
#else
    const clock_t end = clock();
    return ((double)(end - timer->start) * 1000.0) / (double)CLOCKS_PER_SEC;
#endif
}
