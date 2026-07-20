/*
 * Benchmark module: records per-frame timing rows and aggregate wall-clock
 * metrics. Pipeline runners feed FrameTiming data here after processing,
 * while report modules consume the CSV output for summaries.
 */
#include "benchmark/benchmark.h"
#include "utils/c_runtime.h"
#include "utils/file_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCHMARK_CSV_BUFFER_SIZE (4u * 1024u * 1024u)

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
    bench->csv_buffer = NULL;
    bench->schema = BENCHMARK_SCHEMA_FILTER;
}

void benchmark_set_wall_clock_ms(Benchmark *bench, double wall_clock_ms) {
    if (!bench || wall_clock_ms < 0.0) {
        return;
    }

    bench->wall_clock_ms = wall_clock_ms;
}

static int write_filter_csv_header(FILE *file) {
    return fprintf(file, "frame_index,decode_ms,upload_ms,kernel_ms,download_ms,process_ms,encode_ms,total_ms\n") < 0 ? -1 : 0;
}

static int write_filter_csv_row(FILE *file, const FrameTiming *t) {
    return fprintf(file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                   t->frame_index,
                   t->decode_ms,
                   t->upload_ms,
                   t->kernel_ms,
                   t->download_ms,
                   t->process_ms,
                   t->encode_ms,
                   t->total_ms) < 0 ? -1 : 0;
}

static int write_detection_csv_header(FILE *file) {
    return fprintf(file, "frame_index,decode_ms,raw_frame_upload_bytes,upload_ms,preprocess_ms,inference_ms,backend_inference_ms,metadata_download_bytes,download_ms,postprocess_ms,detections_count,overlay_ms,encode_ms,mux_write_ms,total_ms,runtime_backend,backend_device,precision,model_format,model_adapter,input_layout,input_dtype,output_device,video_width,video_height,video_fps,frame_bytes,raw_frame_download_bytes,schedule_batch_size,backend_batch_size,batch_size,inflight_batches,total_active_frames,active_frame_capacity,frames_per_upload_batch,frames_per_download_batch,execution_mode,inference_context_count,inference_lane_count,vram_budget_mb,estimated_batch_mb,unused_vram_budget_mb,decode_queue_wait_ms,inference_queue_wait_ms,inference_lane_wait_ms,output_reorder_wait_ms,metadata_queue_wait_ms,encode_queue_wait_ms,end_to_end_latency_ms,stage_sum_ms\n") < 0 ? -1 : 0;
}

static int write_detection_csv_row(FILE *file, const FrameTiming *t) {
    return fprintf(file, "%d,%.6f,%zu,%.6f,%.6f,%.6f,%.6f,%zu,%.6f,%.6f,%zu,%.6f,%.6f,%.6f,%.6f,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%.6f,%zu,%zu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                   t->frame_index,
                   t->decode_ms,
                   t->raw_frame_upload_bytes,
                   t->upload_ms,
                   t->preprocess_ms,
                   t->inference_ms,
                   t->backend_inference_ms,
                   t->metadata_download_bytes,
                   t->download_ms,
                   t->postprocess_ms,
                   t->detections_count,
                   t->overlay_ms,
                   t->encode_ms,
                   t->mux_write_ms,
                   t->total_ms,
                   t->runtime_backend,
                   t->backend_device,
                   t->precision,
                   t->model_format,
                   t->model_adapter,
                   t->input_layout,
                   t->input_dtype,
                   t->output_device,
                   t->video_width,
                   t->video_height,
                   t->video_fps,
                   t->frame_bytes,
                   t->raw_frame_download_bytes,
                   t->schedule_batch_size,
                   t->backend_batch_size,
                   t->batch_size,
                   t->inflight_batches,
                   t->total_active_frames,
                   t->active_frame_capacity,
                   t->frames_per_upload_batch,
                   t->frames_per_download_batch,
                   t->execution_mode,
                   t->inference_context_count,
                   t->inference_lane_count,
                   t->vram_budget_mb,
                   t->estimated_batch_mb,
                   t->unused_vram_budget_mb,
                   t->decode_queue_wait_ms,
                   t->inference_queue_wait_ms,
                   t->inference_lane_wait_ms,
                   t->output_reorder_wait_ms,
                   t->metadata_queue_wait_ms,
                   t->encode_queue_wait_ms,
                   t->end_to_end_latency_ms,
                   t->stage_sum_ms) < 0 ? -1 : 0;
}

static int write_csv_header(FILE *file, BenchmarkSchema schema) {
    return schema == BENCHMARK_SCHEMA_DETECTION
               ? write_detection_csv_header /* module: benchmark/benchmark */ (file)
               : write_filter_csv_header /* module: benchmark/benchmark */ (file);
}

static int write_csv_row(FILE *file, const FrameTiming *t, BenchmarkSchema schema) {
    return schema == BENCHMARK_SCHEMA_DETECTION
               ? write_detection_csv_row /* module: benchmark/benchmark */ (file, t)
               : write_filter_csv_row /* module: benchmark/benchmark */ (file, t);
}

int benchmark_open_csv(Benchmark *bench, const char *path) {
    return benchmark_open_csv_with_schema /* module: benchmark/benchmark */ (bench, path, BENCHMARK_SCHEMA_FILTER);
}

int benchmark_open_csv_with_schema(Benchmark *bench, const char *path, BenchmarkSchema schema) {
    if (!bench || !path || bench->csv_file) {
        return -1;
    }

    if (create_parent_directory_if_missing /* module: utils/file_utils */ (path) != 0) {
        return -1;
    }

    FILE *file = NULL;
    if (vcp_fopen /* module: utils/c_runtime */ (&file, path, "w") != 0) {
        return -1;
    }

    bench->csv_buffer = (char *)malloc(BENCHMARK_CSV_BUFFER_SIZE);
    if (bench->csv_buffer) {
        setvbuf(file, bench->csv_buffer, _IOFBF, BENCHMARK_CSV_BUFFER_SIZE);
    }

    if (write_csv_header /* module: benchmark/benchmark */ (file, schema) != 0) {
        fclose(file);
        free(bench->csv_buffer);
        bench->csv_buffer = NULL;
        return -1;
    }

    bench->csv_file = file;
    bench->schema = schema;
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
        if (write_csv_row /* module: benchmark/benchmark */ ((FILE *)bench->csv_file, timing, bench->schema) != 0) {
            return -1;
        }
        bench->count++;
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

    FILE *file = NULL;
    if (vcp_fopen /* module: utils/c_runtime */ (&file, path, "w") != 0) {
        return -1;
    }

    if (write_csv_header /* module: benchmark/benchmark */ (file, bench->schema) != 0) {
        fclose(file);
        return -1;
    }
    for (size_t i = 0; i < bench->count; ++i) {
        if (write_csv_row /* module: benchmark/benchmark */ (file, &bench->items[i], bench->schema) != 0) {
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
    const int result = fclose(file) == 0 ? 0 : -1;
    free(bench->csv_buffer);
    bench->csv_buffer = NULL;
    return result;
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
    printf("  avg_stage_sum_ms: %.3f\n", bench->total_ms / (double)bench->count);
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
