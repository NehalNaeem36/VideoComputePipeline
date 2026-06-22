#include "pipeline/pipeline_execution_plan.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

static void make_config(PipelineConfig *config) {
    pipeline_config_default /* module: pipeline/pipeline_config */ (config);
    config->task = PIPELINE_TASK_DETECT;
    config->enable_auto_tune = 1;
    config->batch_size_mode = BATCH_SETTING_AUTO;
    config->inflight_batches_mode = BATCH_SETTING_AUTO;
    config->target_fps = 60.0f;
}

static void make_hardware(HardwareProfile *hardware) {
    hardware_profile_init /* module: pipeline/hardware_profile */ (hardware);
    hardware->cuda_available = 1;
    hardware->total_vram_bytes = (size_t)4u * 1024u * 1024u * 1024u;
    hardware->free_vram_after_engine_bytes = (size_t)3u * 1024u * 1024u * 1024u;
    hardware->async_engine_count = 2;
    hardware->h2d_pinned_gbps = 10.0;
    hardware->d2h_pinned_gbps = 10.0;
}

int main(void) {
    PipelineConfig config;
    PipelineConfig default_config;
    HardwareProfile hardware;
    InferenceBatchCapability capability;
    VideoProfile video;
    PipelineExecutionPlan plan;

    make_hardware(&hardware);
    memset(&capability, 0, sizeof(capability));
    capability.min_batch_size = 1;
    capability.max_batch_size = 1;
    capability.input_bytes_per_frame = (size_t)3u * 640u * 640u * 4u;
    capability.output_bytes_per_frame = 25200u * 85u * 4u;

    video_profile_init /* module: pipeline/video_profile */ (&video);
    video.width = 3840;
    video.height = 2160;
    video.fps = 60.0;
    video.working_format = FRAME_FORMAT_NV12;
    video.frame_bytes = (size_t)3840u * 2160u * 3u / 2u;
    video.requires_raw_upload = 0;
    video.requires_raw_download = 0;

    pipeline_config_default /* module: pipeline/pipeline_config */ (&default_config);
    default_config.task = PIPELINE_TASK_DETECT;
    TEST_ASSERT(pipeline_execution_plan_build /* module: pipeline/pipeline_execution_plan */ (&plan, &default_config, &video, &hardware, &capability) == 0);
    TEST_ASSERT(plan.execution_mode == 0);
    TEST_ASSERT(plan.batch_size == 1);
    TEST_ASSERT(plan.inference_context_count == 1);

    make_config(&config);
    TEST_ASSERT(pipeline_execution_plan_build /* module: pipeline/pipeline_execution_plan */ (&plan, &config, &video, &hardware, &capability) == 0);
    TEST_ASSERT(plan.batch_size == 1);
    TEST_ASSERT(plan.frames_per_upload_batch == 0);
    TEST_ASSERT(plan.frames_per_download_batch == 0);
    TEST_ASSERT(plan.vram_budget_bytes <= (size_t)(4.0 * 1024.0 * 1024.0 * 1024.0 * DEFAULT_VRAM_BUDGET_RATIO));

    capability.max_batch_size = 8;
    capability.supports_parallel_contexts = 1;
    capability.max_parallel_contexts = 3;
    video.width = 1280;
    video.height = 720;
    video.frame_bytes = (size_t)1280u * 720u * 3u / 2u;
    video.requires_raw_upload = 1;
    TEST_ASSERT(pipeline_execution_plan_build /* module: pipeline/pipeline_execution_plan */ (&plan, &config, &video, &hardware, &capability) == 0);
    TEST_ASSERT(plan.batch_size >= 1);
    TEST_ASSERT(plan.batch_size <= 8);
    TEST_ASSERT(plan.frames_per_upload_batch == plan.batch_size);
    TEST_ASSERT(plan.transfer_batching_enabled == (plan.batch_size > 1));

    config.parallel_inference_mode = PIPELINE_FEATURE_ON;
    config.inference_contexts_mode = BATCH_SETTING_MANUAL;
    config.inference_contexts = 2;
    config.pipeline_overlap_mode = PIPELINE_FEATURE_ON;
    TEST_ASSERT(pipeline_execution_plan_build /* module: pipeline/pipeline_execution_plan */ (&plan, &config, &video, &hardware, &capability) == 0);
    TEST_ASSERT(plan.pipeline_overlap_enabled == 1);
    TEST_ASSERT(plan.parallel_inference_enabled == 1);
    TEST_ASSERT(plan.inference_context_count == 2);
    TEST_ASSERT(plan.execution_mode == 3);
    return 0;
}
