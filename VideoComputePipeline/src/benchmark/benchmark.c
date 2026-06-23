/*
 * Benchmark module: records per-frame timing rows and aggregate wall-clock
 * metrics. Pipeline runners feed FrameTiming data here after processing,
 * while report modules consume the CSV output for summaries.
 */
#include "benchmark/benchmark.h"
#include "utils/file_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void benchmark_init(Benchmark *bench) {
    if (!bench) {
        return;
    }

    bench->items = NULL;
    bench->count = 0;
    bench->capacity = 0;
    bench->total_ms = 0.0;
    bench->decode_ms = 0.0;
    bench->process_ms = 0.0;
    bench->upload_ms = 0.0;
    bench->preprocess_ms = 0.0;
    bench->inference_ms = 0.0;
    bench->postprocess_ms = 0.0;
    bench->download_ms = 0.0;
    bench->encode_ms = 0.0;
    bench->overlay_ms = 0.0;
    bench->mux_write_ms = 0.0;
    bench->wall_clock_ms = 0.0;
    bench->csv_file = NULL;
}

void benchmark_set_wall_clock_ms(Benchmark *bench, double wall_clock_ms) {
    if (!bench || wall_clock_ms < 0.0) {
        return;
    }

    bench->wall_clock_ms = wall_clock_ms;
}

static int write_csv_header(FILE *file) {
    return fprintf(file, "frame_index,decode_ms,process_ms,upload_ms,kernel_ms,download_ms,encode_ms,total_ms,preprocess_ms,inference_ms,postprocess_ms,overlay_ms,mux_write_ms,batch_size,inflight_batches,total_active_frames,frames_per_upload_batch,frames_per_download_batch,execution_mode,inference_context_count,vram_budget_mb,estimated_batch_mb,runtime_backend,model_format,model_adapter,backend_device,input_layout,input_dtype,output_device,precision,video_width,video_height,video_fps,frame_bytes,backend_inference_ms,raw_frame_upload_bytes,raw_frame_download_bytes,metadata_download_bytes,detections_count\n") < 0 ? -1 : 0;
}

static int write_csv_row(FILE *file, const FrameTiming *t) {
    return fprintf(file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%.6f,%zu,%.6f,%zu,%zu,%zu,%zu\n",
                   t->frame_index,
                   t->decode_ms,
                   t->process_ms,
                   t->upload_ms,
                   t->kernel_ms,
                   t->download_ms,
                   t->encode_ms,
                   t->total_ms,
                   t->preprocess_ms,
                   t->inference_ms,
                   t->postprocess_ms,
                   t->overlay_ms,
                   t->mux_write_ms,
                   t->batch_size,
                   t->inflight_batches,
                   t->total_active_frames,
                   t->frames_per_upload_batch,
                   t->frames_per_download_batch,
                   t->execution_mode,
                   t->inference_context_count,
                   t->vram_budget_mb,
                   t->estimated_batch_mb,
                   t->runtime_backend,
                   t->model_format,
                   t->model_adapter,
                   t->backend_device,
                   t->input_layout,
                   t->input_dtype,
                   t->output_device,
                   t->precision,
                   t->video_width,
                   t->video_height,
                   t->video_fps,
                   t->frame_bytes,
                   t->backend_inference_ms,
                   t->raw_frame_upload_bytes,
                   t->raw_frame_download_bytes,
                   t->metadata_download_bytes,
                   t->detections_count) < 0 ? -1 : 0;
}

int benchmark_open_csv(Benchmark *bench, const char *path) {
    if (!bench || !path || bench->csv_file) {
        return -1;
    }

    if (create_parent_directory_if_missing /* module: utils/file_utils */ (path) != 0) {
        return -1;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return -1;
    }

    if (write_csv_header /* module: benchmark/benchmark */ (file) != 0) {
        fclose(file);
        return -1;
    }

    bench->csv_file = file;
    return 0;
}

static int append_in_memory(Benchmark *bench, const FrameTiming *timing) {
    if (bench->count == bench->capacity) {
        const size_t new_capacity = bench->capacity == 0 ? 256 : bench->capacity * 2;
        FrameTiming *new_items = (FrameTiming *)realloc(bench->items, new_capacity * sizeof(*new_items));
        if (!new_items) {
            return -1;
        }
        bench->items = new_items;
        bench->capacity = new_capacity;
    }

    bench->items[bench->count++] = *timing;
    return 0;
}

int benchmark_add_frame_result(Benchmark *bench, const FrameTiming *timing) {
    if (!bench || !timing) {
        return -1;
    }

    if (bench->csv_file) {
        if (write_csv_row /* module: benchmark/benchmark */ ((FILE *)bench->csv_file, timing) != 0) {
            return -1;
        }
        bench->count++;
        if (bench->count % 300u == 0u) {
            fflush((FILE *)bench->csv_file);
        }
    } else if (append_in_memory /* module: benchmark/benchmark */ (bench, timing) != 0) {
        return -1;
    }

    bench->total_ms += timing->total_ms;
    bench->decode_ms += timing->decode_ms;
    bench->process_ms += timing->process_ms;
    bench->upload_ms += timing->upload_ms;
    bench->preprocess_ms += timing->preprocess_ms;
    bench->inference_ms += timing->inference_ms;
    bench->postprocess_ms += timing->postprocess_ms;
    bench->download_ms += timing->download_ms;
    bench->encode_ms += timing->encode_ms;
    bench->overlay_ms += timing->overlay_ms;
    bench->mux_write_ms += timing->mux_write_ms;
    return 0;
}

int benchmark_write_csv(const Benchmark *bench, const char *path) {
    if (!bench || !path) {
        return -1;
    }

    if (create_parent_directory_if_missing /* module: utils/file_utils */ (path) != 0) {
        return -1;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return -1;
    }

    if (write_csv_header /* module: benchmark/benchmark */ (file) != 0) {
        fclose(file);
        return -1;
    }
    for (size_t i = 0; i < bench->count; ++i) {
        if (write_csv_row /* module: benchmark/benchmark */ (file, &bench->items[i]) != 0) {
            fclose(file);
            return -1;
        }
    }

    fclose(file);
    return 0;
}

int benchmark_close_csv(Benchmark *bench) {
    if (!bench || !bench->csv_file) {
        return 0;
    }

    FILE *file = (FILE *)bench->csv_file;
    bench->csv_file = NULL;
    return fclose(file) == 0 ? 0 : -1;
}

void benchmark_print_summary(const Benchmark *bench) {
    if (!bench || bench->count == 0) {
        printf("Benchmark: no frames recorded\n");
        return;
    }

    printf("Benchmark summary:\n");
    printf("  frames: %zu\n", bench->count);
    printf("  total_ms: %.3f\n", bench->total_ms);
    printf("  avg_total_ms: %.3f\n", bench->total_ms / (double)bench->count);
    printf("  avg_process_ms: %.3f\n", bench->process_ms / (double)bench->count);
    if (bench->total_ms > 0.0) {
        printf("  processed_fps: %.3f\n", (double)bench->count * 1000.0 / bench->total_ms);
    }
    if (bench->wall_clock_ms > 0.0) {
        printf("  wall_clock_ms: %.3f\n", bench->wall_clock_ms);
        printf("  wall_clock_fps: %.3f\n", (double)bench->count * 1000.0 / bench->wall_clock_ms);
    }
}

void benchmark_print_detection_summary(const Benchmark *bench) {
    if (!bench || bench->count == 0) {
        printf("Detection benchmark: no frames recorded\n");
        return;
    }

    printf("Detection benchmark summary:\n");
    printf("  frames: %zu\n", bench->count);
    printf("  wall_clock_ms: %.3f\n", bench->wall_clock_ms);
    if (bench->wall_clock_ms > 0.0) {
        printf("  wall_clock_fps: %.3f\n", (double)bench->count * 1000.0 / bench->wall_clock_ms);
    }
    printf("  avg_decode_ms: %.3f\n", bench->decode_ms / (double)bench->count);
    printf("  avg_upload_ms: %.3f\n", bench->upload_ms / (double)bench->count);
    printf("  avg_preprocess_ms: %.3f\n", bench->preprocess_ms / (double)bench->count);
    printf("  avg_inference_ms: %.3f\n", bench->inference_ms / (double)bench->count);
    printf("  avg_download_ms: %.3f\n", bench->download_ms / (double)bench->count);
    printf("  avg_postprocess_ms: %.3f\n", bench->postprocess_ms / (double)bench->count);
    printf("  avg_overlay_ms: %.3f\n", bench->overlay_ms / (double)bench->count);
    printf("  avg_encode_ms: %.3f\n", bench->encode_ms / (double)bench->count);
    printf("  avg_mux_write_ms: %.3f\n", bench->mux_write_ms / (double)bench->count);
    printf("  avg_total_ms: %.3f\n", bench->total_ms / (double)bench->count);
    if (bench->total_ms > 0.0) {
        printf("  latency_equivalent_fps: %.3f\n", (double)bench->count * 1000.0 / bench->total_ms);
    }
}

void benchmark_free(Benchmark *bench) {
    if (!bench) {
        return;
    }

    benchmark_close_csv /* module: benchmark/benchmark */ (bench);
    free(bench->items);
    benchmark_init /* module: benchmark/benchmark */ (bench);
}
