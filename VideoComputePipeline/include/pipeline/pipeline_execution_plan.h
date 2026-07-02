#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_EXECUTION_PLAN_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_EXECUTION_PLAN_H

#include "inference/inference_engine.h"
#include "pipeline/hardware_profile.h"
#include "pipeline/pipeline_config.h"
#include "pipeline/video_profile.h"

#include <stddef.h>

typedef struct {
    int execution_mode;
    int batch_size;
    int schedule_batch_size;
    int backend_batch_size;
    int inflight_batches;
    int total_active_frames;
    int active_frame_capacity;
    int frames_per_upload_batch;
    int frames_per_download_batch;
    size_t vram_budget_bytes;
    size_t estimated_batch_bytes;
    size_t unused_vram_budget_bytes;
    int use_pinned_memory;
    int use_contiguous_batch_buffers;
    int use_async_copies;
    int supports_true_backend_batching;
    int true_backend_batching_enabled;
    int transfer_batching_enabled;
    int pipeline_overlap_enabled;
    int parallel_inference_enabled;
    int inference_context_count;
    int inference_lane_count;
    int inference_serialized;
    size_t pinned_host_batch_bytes;
    size_t device_raw_batch_bytes;
    size_t device_output_batch_bytes;
    int prefer_gpu_resident_path;
    int allow_full_frame_download;
    char fallback_reason[256];
} PipelineExecutionPlan;

void pipeline_execution_plan_init(PipelineExecutionPlan *plan);
int pipeline_execution_plan_build(PipelineExecutionPlan *plan,
                                  const PipelineConfig *config,
                                  const VideoProfile *video,
                                  const HardwareProfile *hardware,
                                  const InferenceBatchCapability *capability);
void pipeline_execution_plan_print(const PipelineExecutionPlan *plan);

#endif
