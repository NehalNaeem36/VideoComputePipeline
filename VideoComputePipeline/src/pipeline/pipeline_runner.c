#include "pipeline/pipeline_runner.h"

#include "benchmark/benchmark.h"
#include "benchmark/timer.h"
#include "cpu/cpu_filters.h"
#include "gpu/gpu_filters.h"
#include "utils/logger.h"
#include "video/video_reader.h"
#include "video/video_writer.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct {
    Frame frame;
    FrameTiming timing;
} PipelinePacket;

typedef struct {
    PipelinePacket *items;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
    int closed;
#ifdef _WIN32
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE not_empty;
    CONDITION_VARIABLE not_full;
#else
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
#endif
} PacketQueue;

typedef struct PendingNode {
    PipelinePacket packet;
    struct PendingNode *next;
} PendingNode;

typedef struct {
    const PipelineConfig *config;
    VideoReader reader;
    VideoWriter writer;
    Benchmark benchmark;
    GPUFilterContext gpu;
    PacketQueue raw_queue;
    PacketQueue processed_queue;
    int gpu_initialized;
    int processor_workers;
    int failed;
#ifdef _WIN32
    volatile LONG active_processors;
#else
    int active_processors;
    pthread_mutex_t active_lock;
#endif
} PipelineContext;

static void packet_init(PipelinePacket *packet) {
    if (!packet) {
        return;
    }
    frame_init(&packet->frame);
    memset(&packet->timing, 0, sizeof(packet->timing));
}

static void packet_free(PipelinePacket *packet) {
    if (!packet) {
        return;
    }
    frame_free(&packet->frame);
    memset(&packet->timing, 0, sizeof(packet->timing));
}

static int packet_move(PipelinePacket *dst, PipelinePacket *src) {
    if (!dst || !src) {
        return -1;
    }

    packet_free(dst);
    dst->timing = src->timing;
    if (frame_move(&dst->frame, &src->frame) != 0) {
        return -1;
    }
    memset(&src->timing, 0, sizeof(src->timing));
    return 0;
}

static int packet_queue_init(PacketQueue *queue, size_t capacity) {
    if (!queue || capacity == 0) {
        return -1;
    }

    memset(queue, 0, sizeof(*queue));
    queue->items = (PipelinePacket *)calloc(capacity, sizeof(*queue->items));
    if (!queue->items) {
        return -1;
    }

    queue->capacity = capacity;
    for (size_t i = 0; i < capacity; ++i) {
        packet_init(&queue->items[i]);
    }

#ifdef _WIN32
    InitializeCriticalSection(&queue->lock);
    InitializeConditionVariable(&queue->not_empty);
    InitializeConditionVariable(&queue->not_full);
#else
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->not_empty, NULL);
    pthread_cond_init(&queue->not_full, NULL);
#endif
    return 0;
}

static void packet_queue_close(PacketQueue *queue) {
    if (!queue) {
        return;
    }

#ifdef _WIN32
    EnterCriticalSection(&queue->lock);
    queue->closed = 1;
    WakeAllConditionVariable(&queue->not_empty);
    WakeAllConditionVariable(&queue->not_full);
    LeaveCriticalSection(&queue->lock);
#else
    pthread_mutex_lock(&queue->lock);
    queue->closed = 1;
    pthread_cond_broadcast(&queue->not_empty);
    pthread_cond_broadcast(&queue->not_full);
    pthread_mutex_unlock(&queue->lock);
#endif
}

static void packet_queue_free(PacketQueue *queue) {
    if (!queue) {
        return;
    }

    packet_queue_close(queue);
    if (queue->items) {
        for (size_t i = 0; i < queue->capacity; ++i) {
            packet_free(&queue->items[i]);
        }
    }
    free(queue->items);
    queue->items = NULL;
    queue->capacity = 0;
    queue->count = 0;

#ifdef _WIN32
    DeleteCriticalSection(&queue->lock);
#else
    pthread_cond_destroy(&queue->not_empty);
    pthread_cond_destroy(&queue->not_full);
    pthread_mutex_destroy(&queue->lock);
#endif
}

static int packet_queue_push(PacketQueue *queue, PipelinePacket *packet) {
    if (!queue || !packet) {
        return -1;
    }

#ifdef _WIN32
    EnterCriticalSection(&queue->lock);
    while (!queue->closed && queue->count == queue->capacity) {
        SleepConditionVariableCS(&queue->not_full, &queue->lock, INFINITE);
    }
    if (queue->closed) {
        LeaveCriticalSection(&queue->lock);
        return 0;
    }
    if (packet_move(&queue->items[queue->tail], packet) != 0) {
        LeaveCriticalSection(&queue->lock);
        return -1;
    }
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    WakeConditionVariable(&queue->not_empty);
    LeaveCriticalSection(&queue->lock);
#else
    pthread_mutex_lock(&queue->lock);
    while (!queue->closed && queue->count == queue->capacity) {
        pthread_cond_wait(&queue->not_full, &queue->lock);
    }
    if (queue->closed) {
        pthread_mutex_unlock(&queue->lock);
        return 0;
    }
    if (packet_move(&queue->items[queue->tail], packet) != 0) {
        pthread_mutex_unlock(&queue->lock);
        return -1;
    }
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->lock);
#endif
    return 1;
}

static int packet_queue_pop(PacketQueue *queue, PipelinePacket *packet) {
    if (!queue || !packet) {
        return -1;
    }

#ifdef _WIN32
    EnterCriticalSection(&queue->lock);
    while (!queue->closed && queue->count == 0) {
        SleepConditionVariableCS(&queue->not_empty, &queue->lock, INFINITE);
    }
    if (queue->count == 0 && queue->closed) {
        LeaveCriticalSection(&queue->lock);
        return 0;
    }
    packet_move(packet, &queue->items[queue->head]);
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    WakeConditionVariable(&queue->not_full);
    LeaveCriticalSection(&queue->lock);
#else
    pthread_mutex_lock(&queue->lock);
    while (!queue->closed && queue->count == 0) {
        pthread_cond_wait(&queue->not_empty, &queue->lock);
    }
    if (queue->count == 0 && queue->closed) {
        pthread_mutex_unlock(&queue->lock);
        return 0;
    }
    packet_move(packet, &queue->items[queue->head]);
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->lock);
#endif
    return 1;
}

static void pipeline_fail(PipelineContext *ctx) {
    if (!ctx) {
        return;
    }
    ctx->failed = 1;
    packet_queue_close(&ctx->raw_queue);
    packet_queue_close(&ctx->processed_queue);
}

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
        case FILTER_BLUR_13X13:
            return cpu_blur13x13(input, output);
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
        case FILTER_BLUR_13X13:
            return gpu_blur13x13(gpu, input, output);
        default:
            return -1;
    }
}

static void processor_finished(PipelineContext *ctx) {
#ifdef _WIN32
    if (InterlockedDecrement(&ctx->active_processors) == 0) {
        packet_queue_close(&ctx->processed_queue);
    }
#else
    pthread_mutex_lock(&ctx->active_lock);
    ctx->active_processors--;
    if (ctx->active_processors == 0) {
        packet_queue_close(&ctx->processed_queue);
    }
    pthread_mutex_unlock(&ctx->active_lock);
#endif
}

#ifdef _WIN32
static DWORD WINAPI decoder_thread_main(LPVOID arg)
#else
static void *decoder_thread_main(void *arg)
#endif
{
    PipelineContext *ctx = (PipelineContext *)arg;
    int global_frame_id = 0;

    for (;;) {
        PipelinePacket packet;
        Timer timer;
        packet_init(&packet);

        if (ctx->config->max_frames > 0 && global_frame_id >= ctx->config->max_frames) {
            packet_free(&packet);
            break;
        }

        timer_start(&timer);
        const int read_result = video_reader_read_frame(&ctx->reader, &packet.frame);
        packet.timing.decode_ms = timer_stop_ms(&timer);

        if (read_result == 0) {
            packet_free(&packet);
            break;
        }
        if (read_result < 0) {
            packet_free(&packet);
            log_error("decoder worker failed");
            pipeline_fail(ctx);
            break;
        }
        if (ctx->failed) {
            packet_free(&packet);
            break;
        }

        packet.frame.index = global_frame_id;
        packet.timing.frame_index = global_frame_id;
        global_frame_id++;

        if (packet_queue_push(&ctx->raw_queue, &packet) < 0) {
            packet_free(&packet);
            pipeline_fail(ctx);
            break;
        }
        packet_free(&packet);
    }

    packet_queue_close(&ctx->raw_queue);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

#ifdef _WIN32
static DWORD WINAPI processor_thread_main(LPVOID arg)
#else
static void *processor_thread_main(void *arg)
#endif
{
    PipelineContext *ctx = (PipelineContext *)arg;

    for (;;) {
        PipelinePacket input;
        PipelinePacket output;
        Timer timer;
        int result = 0;

        packet_init(&input);
        packet_init(&output);

        result = packet_queue_pop(&ctx->raw_queue, &input);
        if (result == 0) {
            packet_free(&input);
            packet_free(&output);
            break;
        }
        if (result < 0 || ctx->failed) {
            packet_free(&input);
            packet_free(&output);
            pipeline_fail(ctx);
            break;
        }

        timer_start(&timer);
        if (ctx->config->mode == PROCESS_GPU) {
            result = process_gpu(ctx->config->filter, &ctx->gpu, &input.frame, &output.frame);
            output.timing.upload_ms = ctx->gpu.last_upload_ms;
            output.timing.kernel_ms = ctx->gpu.last_kernel_ms;
            output.timing.download_ms = ctx->gpu.last_download_ms;
        } else {
            result = process_cpu(ctx->config->filter, &input.frame, &output.frame);
        }
        output.timing = input.timing;
        output.timing.process_ms = timer_stop_ms(&timer);
        if (ctx->config->mode == PROCESS_GPU) {
            output.timing.upload_ms = ctx->gpu.last_upload_ms;
            output.timing.kernel_ms = ctx->gpu.last_kernel_ms;
            output.timing.download_ms = ctx->gpu.last_download_ms;
        }

        if (result != 0) {
            log_error("processor worker failed on frame %d", input.frame.index);
            packet_free(&input);
            packet_free(&output);
            pipeline_fail(ctx);
            break;
        }

        output.frame.index = input.frame.index;
        output.timing.frame_index = input.frame.index;

        if (packet_queue_push(&ctx->processed_queue, &output) < 0) {
            packet_free(&input);
            packet_free(&output);
            pipeline_fail(ctx);
            break;
        }

        packet_free(&input);
        packet_free(&output);
    }

    processor_finished(ctx);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static int pending_insert(PendingNode **head, PipelinePacket *packet) {
    PendingNode *node = (PendingNode *)calloc(1, sizeof(*node));
    if (!node) {
        return -1;
    }
    packet_init(&node->packet);
    packet_move(&node->packet, packet);

    if (!*head || node->packet.frame.index < (*head)->packet.frame.index) {
        node->next = *head;
        *head = node;
        return 0;
    }

    PendingNode *cur = *head;
    while (cur->next && cur->next->packet.frame.index < node->packet.frame.index) {
        cur = cur->next;
    }
    node->next = cur->next;
    cur->next = node;
    return 0;
}

static int pending_take(PendingNode **head, int frame_id, PipelinePacket *out) {
    PendingNode *prev = NULL;
    PendingNode *cur = *head;
    while (cur) {
        if (cur->packet.frame.index == frame_id) {
            if (prev) {
                prev->next = cur->next;
            } else {
                *head = cur->next;
            }
            packet_move(out, &cur->packet);
            packet_free(&cur->packet);
            free(cur);
            return 1;
        }
        prev = cur;
        cur = cur->next;
    }
    return 0;
}

static void pending_free_all(PendingNode *head) {
    while (head) {
        PendingNode *next = head->next;
        packet_free(&head->packet);
        free(head);
        head = next;
    }
}

static int write_packet(PipelineContext *ctx, PipelinePacket *packet) {
    Timer timer;
    timer_start(&timer);
    if (video_writer_write_frame(&ctx->writer, &packet->frame) != 0) {
        return -1;
    }
    packet->timing.encode_ms = timer_stop_ms(&timer);
    packet->timing.total_ms = packet->timing.decode_ms + packet->timing.process_ms + packet->timing.encode_ms;
    if (ctx->config->enable_benchmark) {
        if (benchmark_add_frame_result(&ctx->benchmark, &packet->timing) != 0) {
            return -1;
        }
    }
    return 0;
}

static int write_available_ordered(PipelineContext *ctx, PendingNode **pending, int *next_frame_id) {
    for (;;) {
        PipelinePacket packet;
        packet_init(&packet);
        const int found = pending_take(pending, *next_frame_id, &packet);
        if (!found) {
            packet_free(&packet);
            return 0;
        }
        if (write_packet(ctx, &packet) != 0) {
            packet_free(&packet);
            return -1;
        }
        (*next_frame_id)++;
        packet_free(&packet);
    }
}

#ifdef _WIN32
static DWORD WINAPI encoder_thread_main(LPVOID arg)
#else
static void *encoder_thread_main(void *arg)
#endif
{
    PipelineContext *ctx = (PipelineContext *)arg;
    PendingNode *pending = NULL;
    int next_frame_id = 0;

    for (;;) {
        PipelinePacket packet;
        packet_init(&packet);

        const int result = packet_queue_pop(&ctx->processed_queue, &packet);
        if (result == 0) {
            packet_free(&packet);
            break;
        }
        if (result < 0 || ctx->failed) {
            packet_free(&packet);
            pipeline_fail(ctx);
            break;
        }

        if (pending_insert(&pending, &packet) != 0 ||
            write_available_ordered(ctx, &pending, &next_frame_id) != 0) {
            packet_free(&packet);
            log_error("encoder worker failed");
            pipeline_fail(ctx);
            break;
        }

        packet_free(&packet);
    }

    if (!ctx->failed && write_available_ordered(ctx, &pending, &next_frame_id) != 0) {
        pipeline_fail(ctx);
    }

    if (pending) {
        log_error("encoder stopped with out-of-order frames still pending");
        pipeline_fail(ctx);
        pending_free_all(pending);
    }

    if (!ctx->failed && video_writer_flush(&ctx->writer) != 0) {
        log_warn("video writer flush reported an error");
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

int pipeline_run(const PipelineConfig *config) {
    if (!config) {
        return -1;
    }

    PipelineContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.config = config;
    benchmark_init(&ctx.benchmark);

#ifndef _WIN32
    pthread_mutex_init(&ctx.active_lock, NULL);
#endif

    int effective_frame_slots = config->frame_slots;
    int effective_decoder_threads = config->decoder_threads;
    int effective_encoder_threads = config->encoder_threads;
    int effective_processor_workers = config->processor_workers;
    if (config->lossless_output) {
        if (effective_frame_slots > 1) {
            effective_frame_slots = 1;
        }
        if (effective_decoder_threads > 2) {
            effective_decoder_threads = 2;
        }
        if (effective_encoder_threads > 2) {
            effective_encoder_threads = 2;
        }
        if (config->mode == PROCESS_CPU && effective_processor_workers > 1) {
            effective_processor_workers = 1;
        }
    }

    const int processor_workers = config->mode == PROCESS_GPU ? 1 : effective_processor_workers;
    ctx.processor_workers = processor_workers > 0 ? processor_workers : 1;
#ifdef _WIN32
    ctx.active_processors = ctx.processor_workers;
#else
    ctx.active_processors = ctx.processor_workers;
#endif

    if (packet_queue_init(&ctx.raw_queue, (size_t)effective_frame_slots) != 0 ||
        packet_queue_init(&ctx.processed_queue, (size_t)effective_frame_slots) != 0) {
        log_error("failed to initialize pipeline queues");
        packet_queue_free(&ctx.raw_queue);
        packet_queue_free(&ctx.processed_queue);
        return -1;
    }

    if (video_reader_open_with_threads(&ctx.reader, config->input_path, effective_decoder_threads) != 0) {
        log_error("failed to open input video: %s", config->input_path);
        packet_queue_free(&ctx.raw_queue);
        packet_queue_free(&ctx.processed_queue);
        return -1;
    }

    const VideoInfo *info = video_reader_get_info(&ctx.reader);
    if (!info) {
        log_error("failed to read input video info");
        video_reader_close(&ctx.reader);
        packet_queue_free(&ctx.raw_queue);
        packet_queue_free(&ctx.processed_queue);
        return -1;
    }

    if (video_writer_open_with_options(&ctx.writer,
                                       config->output_path,
                                       info->width,
                                       info->height,
                                       info->fps,
                                       effective_encoder_threads,
                                       config->encoder_name,
                                       config->lossless_output) != 0) {
        log_error("failed to open output video: %s", config->output_path);
        video_reader_close(&ctx.reader);
        packet_queue_free(&ctx.raw_queue);
        packet_queue_free(&ctx.processed_queue);
        return -1;
    }

    if (config->mode == PROCESS_GPU) {
        if (gpu_filters_init(&ctx.gpu) != 0) {
            log_error("failed to initialize GPU filters");
            video_writer_close(&ctx.writer);
            video_reader_close(&ctx.reader);
            packet_queue_free(&ctx.raw_queue);
            packet_queue_free(&ctx.processed_queue);
            return -1;
        }
        ctx.gpu_initialized = 1;
        opencl_context_print_info(&ctx.gpu.ctx);
    }

    if (config->lossless_output &&
        (effective_frame_slots != config->frame_slots ||
         effective_decoder_threads != config->decoder_threads ||
         effective_encoder_threads != config->encoder_threads ||
         effective_processor_workers != config->processor_workers)) {
        log_info("lossless memory profile: frame_slots=%d decoder_threads=%d encoder_threads=%d processor_workers=%d",
                 effective_frame_slots,
                 effective_decoder_threads,
                 effective_encoder_threads,
                 ctx.processor_workers);
    }

    log_info("pipeline workers: decoder_stage=1 ffmpeg_decoder_threads=%d processor_workers=%d encoder_stage=1 ffmpeg_encoder_threads=%d",
             effective_decoder_threads,
             ctx.processor_workers,
             effective_encoder_threads);

#ifdef _WIN32
    HANDLE decoder_thread = CreateThread(NULL, 0, decoder_thread_main, &ctx, 0, NULL);
    HANDLE encoder_thread = CreateThread(NULL, 0, encoder_thread_main, &ctx, 0, NULL);
    HANDLE *processor_threads = (HANDLE *)calloc((size_t)ctx.processor_workers, sizeof(*processor_threads));
    if (!decoder_thread || !encoder_thread || !processor_threads) {
        pipeline_fail(&ctx);
    } else {
        for (int i = 0; i < ctx.processor_workers; ++i) {
            processor_threads[i] = CreateThread(NULL, 0, processor_thread_main, &ctx, 0, NULL);
            if (!processor_threads[i]) {
                pipeline_fail(&ctx);
            }
        }
    }

    if (decoder_thread) {
        WaitForSingleObject(decoder_thread, INFINITE);
    }
    if (processor_threads) {
        for (int i = 0; i < ctx.processor_workers; ++i) {
            if (processor_threads[i]) {
                WaitForSingleObject(processor_threads[i], INFINITE);
            }
        }
    }
    if (encoder_thread) {
        WaitForSingleObject(encoder_thread, INFINITE);
    }

    if (decoder_thread) {
        CloseHandle(decoder_thread);
    }
    if (encoder_thread) {
        CloseHandle(encoder_thread);
    }
    if (processor_threads) {
        for (int i = 0; i < ctx.processor_workers; ++i) {
            if (processor_threads[i]) {
                CloseHandle(processor_threads[i]);
            }
        }
        free(processor_threads);
    }
#else
    pthread_t decoder_thread;
    pthread_t encoder_thread;
    pthread_t *processor_threads = (pthread_t *)calloc((size_t)ctx.processor_workers, sizeof(*processor_threads));
    if (!processor_threads ||
        pthread_create(&decoder_thread, NULL, decoder_thread_main, &ctx) != 0 ||
        pthread_create(&encoder_thread, NULL, encoder_thread_main, &ctx) != 0) {
        pipeline_fail(&ctx);
    } else {
        for (int i = 0; i < ctx.processor_workers; ++i) {
            if (pthread_create(&processor_threads[i], NULL, processor_thread_main, &ctx) != 0) {
                pipeline_fail(&ctx);
            }
        }
        pthread_join(decoder_thread, NULL);
        for (int i = 0; i < ctx.processor_workers; ++i) {
            pthread_join(processor_threads[i], NULL);
        }
        pthread_join(encoder_thread, NULL);
    }
    free(processor_threads);
#endif

    if (ctx.config->enable_benchmark && !ctx.failed) {
        if (benchmark_write_csv(&ctx.benchmark, ctx.config->benchmark_path) != 0) {
            log_error("failed to write benchmark CSV: %s", ctx.config->benchmark_path);
            ctx.failed = 1;
        } else {
            benchmark_print_summary(&ctx.benchmark);
        }
    }

    if (ctx.gpu_initialized) {
        gpu_filters_release(&ctx.gpu);
    }
    benchmark_free(&ctx.benchmark);
    video_writer_close(&ctx.writer);
    video_reader_close(&ctx.reader);
    packet_queue_free(&ctx.raw_queue);
    packet_queue_free(&ctx.processed_queue);
#ifndef _WIN32
    pthread_mutex_destroy(&ctx.active_lock);
#endif

    return ctx.failed ? -1 : 0;
}
