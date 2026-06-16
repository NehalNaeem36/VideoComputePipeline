#include "pipeline/pipeline_runner.h"

#include "benchmark/benchmark.h"
#include "benchmark/timer.h"
#include "cpu/cpu_filters.h"
#include "gpu/gpu_filters.h"
#include "utils/logger.h"
#include "video/video_reader.h"
#include "video/video_writer.h"

#include <string.h>

static int process_cpu(FilterType filter, const Frame *input, Frame *output) {
    switch (filter) {
        case FILTER_GRAYSCALE:
            return cpu_grayscale(input, output);
        case FILTER_BLUR_3X3:
            return cpu_blur3x3(input, output);
        case FILTER_BLUR_5X5:
            return cpu_blur5x5(input, output);
        case FILTER_BLUR_9X9:
            return cpu_blur9x9(input, output);
        default:
            return -1;
    }
}

static int process_gpu(FilterType filter, GPUFilterContext *gpu, const Frame *input, Frame *output) {
    switch (filter) {
        case FILTER_GRAYSCALE:
            return gpu_grayscale(gpu, input, output);
        case FILTER_BLUR_3X3:
            return gpu_blur3x3(gpu, input, output);
        case FILTER_BLUR_5X5:
            return gpu_blur5x5(gpu, input, output);
        case FILTER_BLUR_9X9:
            return gpu_blur9x9(gpu, input, output);
        default:
            return -1;
    }
}

int pipeline_run(const PipelineConfig *config) {
    if (!config) {
        return -1;
    }

    int status = -1;
    VideoReader reader;
    VideoWriter writer;
    Benchmark benchmark;
    GPUFilterContext gpu;
    int gpu_initialized = 0;

    memset(&reader, 0, sizeof(reader));
    memset(&writer, 0, sizeof(writer));
    memset(&gpu, 0, sizeof(gpu));
    benchmark_init(&benchmark);

    if (video_reader_open(&reader, config->input_path) != 0) {
        log_error("failed to open input video: %s", config->input_path);
        goto cleanup;
    }

    const VideoInfo *info = video_reader_get_info(&reader);
    if (!info) {
        log_error("failed to read input video info");
        goto cleanup;
    }

    if (video_writer_open(&writer, config->output_path, info->width, info->height, info->fps) != 0) {
        log_error("failed to open output video: %s", config->output_path);
        goto cleanup;
    }

    if (config->mode == PROCESS_GPU) {
        if (gpu_filters_init(&gpu) != 0) {
            log_error("failed to initialize GPU filters");
            goto cleanup;
        }
        gpu_initialized = 1;
        opencl_context_print_info(&gpu.ctx);
    }

    int frame_count = 0;
    for (;;) {
        Frame input;
        Frame output;
        FrameTiming timing;
        Timer total_timer;
        Timer stage_timer;

        frame_init(&input);
        frame_init(&output);
        memset(&timing, 0, sizeof(timing));

        timer_start(&stage_timer);
        const int read_result = video_reader_read_frame(&reader, &input);
        timing.decode_ms = timer_stop_ms(&stage_timer);

        if (read_result == 0) {
            frame_free(&input);
            frame_free(&output);
            break;
        }
        if (read_result < 0) {
            log_error("failed while decoding frame");
            frame_free(&input);
            frame_free(&output);
            goto cleanup;
        }

        timer_start(&total_timer);
        timing.frame_index = input.index;

        timer_start(&stage_timer);
        int process_result = 0;
        if (config->mode == PROCESS_GPU) {
            process_result = process_gpu(config->filter, &gpu, &input, &output);
            timing.upload_ms = gpu.last_upload_ms;
            timing.kernel_ms = gpu.last_kernel_ms;
            timing.download_ms = gpu.last_download_ms;
        } else {
            process_result = process_cpu(config->filter, &input, &output);
        }
        timing.process_ms = timer_stop_ms(&stage_timer);

        if (process_result != 0) {
            log_error("failed while processing frame %d", input.index);
            frame_free(&input);
            frame_free(&output);
            goto cleanup;
        }

        timer_start(&stage_timer);
        if (video_writer_write_frame(&writer, &output) != 0) {
            log_error("failed while encoding frame %d", output.index);
            frame_free(&input);
            frame_free(&output);
            goto cleanup;
        }
        timing.encode_ms = timer_stop_ms(&stage_timer);
        timing.total_ms = timer_stop_ms(&total_timer) + timing.decode_ms;

        if (config->enable_benchmark) {
            if (benchmark_add_frame_result(&benchmark, &timing) != 0) {
                log_error("failed to record benchmark data");
                frame_free(&input);
                frame_free(&output);
                goto cleanup;
            }
        }

        frame_free(&input);
        frame_free(&output);
        frame_count++;

        if (config->max_frames > 0 && frame_count >= config->max_frames) {
            break;
        }
    }

    if (video_writer_flush(&writer) != 0) {
        log_warn("video writer flush reported an error");
    }

    if (config->enable_benchmark) {
        if (benchmark_write_csv(&benchmark, config->benchmark_path) != 0) {
            log_error("failed to write benchmark CSV: %s", config->benchmark_path);
            goto cleanup;
        }
        benchmark_print_summary(&benchmark);
    }

    status = 0;

cleanup:
    if (gpu_initialized) {
        gpu_filters_release(&gpu);
    }
    benchmark_free(&benchmark);
    video_writer_close(&writer);
    video_reader_close(&reader);
    return status;
}
