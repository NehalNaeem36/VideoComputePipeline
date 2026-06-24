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
    double overlay_ms;
    double mux_write_ms;
    int batch_size;
    int schedule_batch_size;
    int backend_batch_size;
    int inflight_batches;
    int total_active_frames;
    int active_frame_capacity;
    int frames_per_upload_batch;
    int frames_per_download_batch;
    int execution_mode;
    int inference_context_count;
    int inference_lane_count;
    double vram_budget_mb;
    double estimated_batch_mb;
    double unused_vram_budget_mb;
    char runtime_backend[32];
    char model_format[32];
    char model_adapter[32];
    char backend_device[16];
    char input_layout[16];
    char input_dtype[16];
    char output_device[16];
    char precision[16];
    int video_width;
    int video_height;
    double video_fps;
    size_t frame_bytes;
    double backend_inference_ms;
    size_t raw_frame_upload_bytes;
    size_t raw_frame_download_bytes;
    size_t metadata_download_bytes;
    size_t detections_count;
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
    double download_ms;
    double encode_ms;
    double overlay_ms;
    double mux_write_ms;
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
