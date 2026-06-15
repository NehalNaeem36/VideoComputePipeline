#ifndef VIDEOCOMPUTEPIPELINE_BENCHMARK_BENCHMARK_H
#define VIDEOCOMPUTEPIPELINE_BENCHMARK_BENCHMARK_H

#include <stddef.h>
#include <stdint.h>

/**
 * Per-frame benchmark data
 */
typedef struct {
    uint64_t frame_number;
    double cpu_time_ms;
    double gpu_time_ms;
} FrameBenchmark;

/**
 * Aggregate benchmark statistics
 */
typedef struct {
    FrameBenchmark *samples;
    size_t num_samples;
    double total_cpu_time;
    double total_gpu_time;
    double avg_cpu_time;
    double avg_gpu_time;
} BenchmarkStats;

/**
 * Create benchmark stats
 */
BenchmarkStats* benchmark_create(size_t max_frames);

/**
 * Record per-frame timing
 */
void benchmark_record_frame(BenchmarkStats *stats, uint64_t frame_number, double cpu_ms, double gpu_ms);

/**
 * Calculate statistics
 */
void benchmark_calculate_stats(BenchmarkStats *stats);

/**
 * Free benchmark stats
 */
void benchmark_destroy(BenchmarkStats *stats);

#endif // VIDEOCOMPUTEPIPELINE_BENCHMARK_BENCHMARK_H
