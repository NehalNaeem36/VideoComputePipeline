/*
 * Pipeline execution plan module: chooses detection batch size, in-flight work,
 * transfer behavior, and backend inference context usage from config, video metadata,
 * inference capability, and hardware profile data.
 */
#include "pipeline/pipeline_execution_plan.h"
#include "utils/c_runtime.h"

#include <stdio.h>
#include <string.h>

static size_t max_size(size_t a, size_t b) {
    return a > b ? a : b;
}

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

void pipeline_execution_plan_init(PipelineExecutionPlan *plan) {
    if (!plan) {
        return;
    }
    memset(plan, 0, sizeof(*plan));
    plan->execution_mode = 0;
    plan->batch_size = 1;
    plan->schedule_batch_size = 1;
    plan->backend_batch_size = 1;
    plan->inflight_batches = 1;
    plan->total_active_frames = 1;
    plan->active_frame_capacity = 1;
    plan->inference_context_count = 1;
    plan->inference_lane_count = 1;
    plan->inference_serialized = 1;
    vcp_copy_string /* module: utils/c_runtime */ (plan->fallback_reason,
                                                   sizeof(plan->fallback_reason),
                                                   "default single-frame execution");
}

static size_t estimate_per_frame_gpu_bytes(const VideoProfile *video, const InferenceBatchCapability *capability) {
    const size_t input_bytes = capability ? capability->input_bytes_per_frame : 0u;
    const size_t output_bytes = capability ? capability->output_bytes_per_frame : 0u;
    return video->frame_bytes + input_bytes + output_bytes + max_size(video->frame_bytes / 4u, 4u * 1024u * 1024u);
}

static int candidate_fits_transfer_budget(int candidate,
                                          const PipelineConfig *config,
                                          const VideoProfile *video,
                                          const HardwareProfile *hardware) {
    const double fps = config->target_fps > 0.0f ? (double)config->target_fps : (video->fps > 0.0 ? video->fps : 30.0);
    const double batch_time_budget_ms = (1000.0 / fps) * (double)candidate;
    const double transfer_budget_ms = batch_time_budget_ms * 0.25;
    double transfer_ms = 0.0;

    if (!hardware || hardware->h2d_pinned_gbps <= 0.0 || hardware->d2h_pinned_gbps <= 0.0) {
        return 1;
    }

    if (video->requires_raw_upload) {
        const double usable = hardware->h2d_pinned_gbps * 0.70;
        const double bytes = (double)video->frame_bytes * (double)candidate;
        transfer_ms += (bytes / (usable * 1000000000.0)) * 1000.0 + 0.05;
    }
    if (video->requires_raw_download) {
        const double usable = hardware->d2h_pinned_gbps * 0.70;
        const double bytes = (double)video->frame_bytes * (double)candidate;
        transfer_ms += (bytes / (usable * 1000000000.0)) * 1000.0 + 0.05;
    }
    return transfer_ms <= transfer_budget_ms;
}

int pipeline_execution_plan_build(PipelineExecutionPlan *plan,
                                  const PipelineConfig *config,
                                  const VideoProfile *video,
                                  const HardwareProfile *hardware,
                                  const InferenceBatchCapability *capability) {
    static const int candidates[] = {1, 2, 3, 4, 6, 8};
    size_t budget_from_total = 0u;
    size_t reserve = 512u * 1024u * 1024u;
    size_t budget_from_free = 0u;
    size_t per_frame_gpu_bytes = 0u;
    int best = 1;

    if (!plan || !config || !video) {
        return -1;
    }

    pipeline_execution_plan_init(plan);
    plan->prefer_gpu_resident_path = config->decoder_mode == VIDEO_DECODER_NVDEC ? 1 : 0;
    plan->allow_full_frame_download = video->requires_raw_download ? 1 : 0;
    plan->frames_per_upload_batch = video->requires_raw_upload ? 1 : 0;
    plan->frames_per_download_batch = video->requires_raw_download ? 1 : 0;
    plan->use_pinned_memory = video->requires_raw_upload || video->requires_raw_download;
    plan->use_contiguous_batch_buffers = plan->use_pinned_memory;
    plan->use_async_copies = hardware && hardware->async_engine_count > 0;
    plan->supports_true_backend_batching = capability && capability->supports_true_batching;
    plan->inference_context_count = 1;
    plan->inference_serialized = 1;

    if (hardware && hardware->cuda_available && hardware->total_vram_bytes > 0u) {
        budget_from_total = (size_t)((double)hardware->total_vram_bytes * (double)config->vram_budget_ratio);
        if (config->vram_reserve_mb > 0) {
            reserve = (size_t)config->vram_reserve_mb * 1024u * 1024u;
        } else {
            reserve = max_size(reserve, (size_t)((double)hardware->total_vram_bytes * 0.20));
        }
        if (hardware->free_vram_after_engine_bytes > reserve) {
            budget_from_free = hardware->free_vram_after_engine_bytes - reserve;
        }
        plan->vram_budget_bytes = min_size(budget_from_total, budget_from_free);
    }

    per_frame_gpu_bytes = estimate_per_frame_gpu_bytes(video, capability);
    if (plan->vram_budget_bytes == 0u) {
        plan->vram_budget_bytes = per_frame_gpu_bytes * 2u;
    }

    if (config->batch_size_mode == BATCH_SETTING_MANUAL) {
        best = config->batch_size;
    } else if (config->enable_auto_tune) {
        for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
            const int candidate = candidates[i];
            const size_t estimate = per_frame_gpu_bytes * (size_t)candidate + 16u * 1024u * 1024u;
            if (estimate > plan->vram_budget_bytes) {
                continue;
            }
            if (!candidate_fits_transfer_budget(candidate, config, video, hardware)) {
                continue;
            }
            best = candidate;
        }
    }

    if (!config->enable_auto_tune && config->batch_size_mode != BATCH_SETTING_MANUAL) {
        snprintf(plan->fallback_reason, sizeof(plan->fallback_reason), "auto-tune disabled; preserving single-frame behavior");
    } else if (!plan->supports_true_backend_batching && best > 1) {
        snprintf(plan->fallback_reason, sizeof(plan->fallback_reason), "schedule batch exceeds backend batch; running single-frame inference lanes");
    } else {
        snprintf(plan->fallback_reason, sizeof(plan->fallback_reason), "selected by execution planner");
    }

    if (best < 1) {
        best = 1;
    }
    plan->batch_size = best;
    plan->schedule_batch_size = best;
    plan->backend_batch_size = 1;
    if (capability && capability->supports_true_batching && capability->max_batch_size > 1) {
        plan->backend_batch_size = best < capability->max_batch_size ? best : capability->max_batch_size;
    }
    plan->true_backend_batching_enabled = plan->backend_batch_size > 1;
    if (config->inflight_batches_mode == BATCH_SETTING_MANUAL) {
        plan->inflight_batches = config->inflight_batches;
    } else if (config->enable_auto_tune && plan->vram_budget_bytes >= per_frame_gpu_bytes * (size_t)best * 4u) {
        plan->inflight_batches = 2;
    } else {
        plan->inflight_batches = 1;
    }
    if (plan->inflight_batches < 1) {
        plan->inflight_batches = 1;
    }
    plan->total_active_frames = plan->batch_size * plan->inflight_batches;
    plan->active_frame_capacity = plan->total_active_frames;
    plan->frames_per_upload_batch = video->requires_raw_upload ? plan->batch_size : 0;
    plan->frames_per_download_batch = video->requires_raw_download ? plan->batch_size : 0;
    plan->estimated_batch_bytes = per_frame_gpu_bytes * (size_t)plan->batch_size + 16u * 1024u * 1024u;
    plan->unused_vram_budget_bytes = plan->vram_budget_bytes > plan->estimated_batch_bytes
                                          ? plan->vram_budget_bytes - plan->estimated_batch_bytes
                                          : 0u;
    plan->transfer_batching_enabled = (video->requires_raw_upload || video->requires_raw_download) && plan->batch_size > 1;
    plan->pinned_host_batch_bytes = plan->transfer_batching_enabled ? video->frame_bytes * (size_t)plan->batch_size : 0u;
    plan->device_raw_batch_bytes = video->requires_raw_upload ? video->frame_bytes * (size_t)plan->batch_size : 0u;
    plan->device_output_batch_bytes = capability ? capability->output_bytes_per_frame * (size_t)plan->batch_size : 0u;

    if (config->pipeline_overlap_mode == PIPELINE_FEATURE_ON) {
        plan->pipeline_overlap_enabled = 1;
    } else if (config->pipeline_overlap_mode == PIPELINE_FEATURE_AUTO) {
        plan->pipeline_overlap_enabled = config->enable_auto_tune && plan->total_active_frames > 1 && plan->use_async_copies;
    }

    if (config->parallel_inference_mode != PIPELINE_FEATURE_OFF && capability && capability->supports_parallel_contexts) {
        int requested_contexts = 1;
        int max_contexts = capability->max_parallel_contexts > 0 ? capability->max_parallel_contexts : 1;
        if (hardware && !hardware->concurrent_kernels && config->parallel_inference_mode == PIPELINE_FEATURE_AUTO) {
            max_contexts = 1;
        }
        if (config->inference_contexts_mode == BATCH_SETTING_MANUAL) {
            requested_contexts = config->inference_contexts;
        } else if (config->parallel_inference_mode == PIPELINE_FEATURE_ON || config->parallel_inference_mode == PIPELINE_FEATURE_AUTO) {
            requested_contexts = plan->batch_size > 1 ? plan->batch_size : plan->inflight_batches;
        }
        if (requested_contexts > max_contexts) {
            requested_contexts = max_contexts;
        }
        if (requested_contexts > plan->total_active_frames) {
            requested_contexts = plan->total_active_frames;
        }
        if (requested_contexts > 1) {
            plan->parallel_inference_enabled = 1;
            plan->inference_context_count = requested_contexts;
            plan->inference_lane_count = requested_contexts;
            plan->inference_serialized = 0;
        }
    }

    if (config->parallel_inference_mode == PIPELINE_FEATURE_ON && !plan->parallel_inference_enabled) {
        snprintf(plan->fallback_reason, sizeof(plan->fallback_reason), "parallel inference requested but unsupported by current runtime/resources");
    }

    if (plan->parallel_inference_enabled) {
        plan->execution_mode = 3;
    } else if (plan->pipeline_overlap_enabled) {
        plan->execution_mode = 2;
    } else if (plan->transfer_batching_enabled) {
        plan->execution_mode = 1;
    } else {
        plan->execution_mode = 0;
    }
    return 0;
}

static double mib(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

void pipeline_execution_plan_print(const PipelineExecutionPlan *plan) {
    if (!plan) {
        return;
    }
    printf("Execution plan:\n");
    printf("  execution_mode: %d\n", plan->execution_mode);
    printf("  batch_size: %d\n", plan->batch_size);
    printf("  schedule_batch_size: %d\n", plan->schedule_batch_size);
    printf("  backend_batch_size: %d\n", plan->backend_batch_size);
    printf("  inflight_batches: %d\n", plan->inflight_batches);
    printf("  total_active_frames: %d\n", plan->total_active_frames);
    printf("  active_frame_capacity: %d\n", plan->active_frame_capacity);
    printf("  frames_per_upload_batch: %d\n", plan->frames_per_upload_batch);
    printf("  frames_per_download_batch: %d\n", plan->frames_per_download_batch);
    printf("  vram_budget_mb: %.3f\n", mib(plan->vram_budget_bytes));
    printf("  estimated_batch_mb: %.3f\n", mib(plan->estimated_batch_bytes));
    printf("  unused_vram_budget_mb: %.3f\n", mib(plan->unused_vram_budget_bytes));
    printf("  use_pinned_memory: %s\n", plan->use_pinned_memory ? "true" : "false");
    printf("  use_contiguous_batch_buffers: %s\n", plan->use_contiguous_batch_buffers ? "true" : "false");
    printf("  use_async_copies: %s\n", plan->use_async_copies ? "true" : "false");
    printf("  supports_true_backend_batching: %s\n", plan->supports_true_backend_batching ? "true" : "false");
    printf("  true_backend_batching_enabled: %s\n", plan->true_backend_batching_enabled ? "true" : "false");
    printf("  transfer_batching_enabled: %s\n", plan->transfer_batching_enabled ? "true" : "false");
    printf("  pipeline_overlap_enabled: %s\n", plan->pipeline_overlap_enabled ? "true" : "false");
    printf("  parallel_inference_enabled: %s\n", plan->parallel_inference_enabled ? "true" : "false");
    printf("  inference_context_count: %d\n", plan->inference_context_count);
    printf("  inference_lane_count: %d\n", plan->inference_lane_count);
    printf("  inference_serialized: %s\n", plan->inference_serialized ? "true" : "false");
    printf("  pinned_host_batch_mb: %.3f\n", mib(plan->pinned_host_batch_bytes));
    printf("  device_raw_batch_mb: %.3f\n", mib(plan->device_raw_batch_bytes));
    printf("  device_output_batch_mb: %.3f\n", mib(plan->device_output_batch_bytes));
    printf("  prefer_gpu_resident_path: %s\n", plan->prefer_gpu_resident_path ? "true" : "false");
    printf("  allow_full_frame_download: %s\n", plan->allow_full_frame_download ? "true" : "false");
    printf("  reason: %s\n", plan->fallback_reason);
}
