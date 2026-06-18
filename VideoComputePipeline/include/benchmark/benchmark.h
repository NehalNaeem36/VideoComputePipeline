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
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
} FrameTiming;

typedef struct {
    FrameTiming *items;
    size_t count;
    size_t capacity;
    double total_ms;
    double decode_ms;
    double process_ms;
    double upload_ms;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double wall_clock_ms;
    void *csv_file;
} Benchmark;

void benchmark_init(Benchmark *bench);
void benchmark_set_wall_clock_ms(Benchmark *bench, double wall_clock_ms);
int benchmark_open_csv(Benchmark *bench, const char *path);
int benchmark_add_frame_result(Benchmark *bench, const FrameTiming *timing);
int benchmark_write_csv(const Benchmark *bench, const char *path);
int benchmark_close_csv(Benchmark *bench);
void benchmark_print_summary(const Benchmark *bench);
void benchmark_print_detection_summary(const Benchmark *bench);
void benchmark_free(Benchmark *bench);

#endif
