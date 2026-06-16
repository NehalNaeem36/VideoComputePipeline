#ifndef VIDEOCOMPUTEPIPELINE_BENCHMARK_H
#define VIDEOCOMPUTEPIPELINE_BENCHMARK_H

#include <stddef.h>

typedef struct {
    int frame_index;
    double decode_ms;
    double process_ms;
    double upload_ms;
    double kernel_ms;
    double download_ms;
    double encode_ms;
    double total_ms;
} FrameTiming;

typedef struct {
    FrameTiming *items;
    size_t count;
    size_t capacity;
} Benchmark;

void benchmark_init(Benchmark *bench);
int benchmark_add_frame_result(Benchmark *bench, const FrameTiming *timing);
int benchmark_write_csv(const Benchmark *bench, const char *path);
void benchmark_print_summary(const Benchmark *bench);
void benchmark_free(Benchmark *bench);

#endif
