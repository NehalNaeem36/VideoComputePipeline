#include "benchmark/benchmark.h"
#include <stdlib.h>

BenchmarkStats* benchmark_create(size_t max_frames) {
    // TODO: Implement stats allocation
    return NULL;
}

void benchmark_record_frame(BenchmarkStats *stats, uint64_t frame_number, double cpu_ms, double gpu_ms) {
    // TODO: Implement frame timing recording
}

void benchmark_calculate_stats(BenchmarkStats *stats) {
    // TODO: Calculate averages and totals
}

void benchmark_destroy(BenchmarkStats *stats) {
    // TODO: Implement deallocation
}
