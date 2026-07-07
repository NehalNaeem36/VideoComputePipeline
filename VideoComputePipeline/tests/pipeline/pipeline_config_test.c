#include "pipeline/pipeline_config.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

static int pipeline_config_test_defaults(void) {
    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(strcmp(config.input_path, DEFAULT_INPUT_PATH) == 0);
    TEST_ASSERT(strcmp(config.output_path, DEFAULT_OUTPUT_PATH) == 0);
    TEST_ASSERT(strcmp(config.benchmark_path, DEFAULT_BENCHMARK_PATH) == 0);
    TEST_ASSERT(strcmp(config.detections_path, DEFAULT_DETECTIONS_PATH) == 0);
    TEST_ASSERT(strcmp(config.model_path, DEFAULT_MODEL_PATH) == 0);
    TEST_ASSERT(strcmp(config.labels_path, DEFAULT_LABELS_PATH) == 0);
    TEST_ASSERT(strcmp(config.encoder_name, DEFAULT_ENCODER_NAME) == 0);
    TEST_ASSERT(config.task == PIPELINE_TASK_FILTER);
    TEST_ASSERT(config.mode == PROCESS_CPU);
    TEST_ASSERT(config.filter == FILTER_GRAYSCALE);
    TEST_ASSERT(config.max_frames == DEFAULT_MAX_FRAMES);
    TEST_ASSERT(config.enable_benchmark == DEFAULT_ENABLE_BENCHMARK);
    TEST_ASSERT(config.lossless_output == DEFAULT_LOSSLESS_OUTPUT);
    TEST_ASSERT(config.memory_profile == MEMORY_PROFILE_AUTO);
    TEST_ASSERT(config.memory_budget_mb == DEFAULT_MEMORY_BUDGET_MB);
    TEST_ASSERT(config.frame_slots == DEFAULT_FRAME_SLOTS);
    TEST_ASSERT(config.decoder_threads == DEFAULT_DECODER_THREADS);
    TEST_ASSERT(config.encoder_threads == DEFAULT_ENCODER_THREADS);
    TEST_ASSERT(config.processor_workers == DEFAULT_PROCESSOR_WORKERS);
    TEST_ASSERT(config.confidence_threshold == DEFAULT_DETECTION_CONFIDENCE);
    TEST_ASSERT(config.iou_threshold == DEFAULT_DETECTION_IOU_THRESHOLD);
    TEST_ASSERT(config.inference_input_size == DEFAULT_INFERENCE_INPUT_SIZE);
    TEST_ASSERT(config.inference_input_width == DEFAULT_INFERENCE_INPUT_SIZE);
    TEST_ASSERT(config.inference_input_height == DEFAULT_INFERENCE_INPUT_SIZE);
    TEST_ASSERT(config.detection_class_count == DEFAULT_DETECTION_CLASS_COUNT);
    TEST_ASSERT(config.class_filter_id_count == 0);
    TEST_ASSERT(config.class_filter_name_count == 0);
    TEST_ASSERT(config.progress_interval == DEFAULT_PROGRESS_INTERVAL);
    TEST_ASSERT(strcmp(config.ffmpeg_log_level, DEFAULT_FFMPEG_LOG_LEVEL) == 0);
    TEST_ASSERT(config.decoder_mode == VIDEO_DECODER_CPU);
    TEST_ASSERT(config.decoder_fallback == DECODER_FALLBACK_CPU);
    TEST_ASSERT(config.output_format == OUTPUT_FORMAT_AUTO);
    TEST_ASSERT(config.draw_boxes == DEFAULT_DRAW_BOXES);
    TEST_ASSERT(config.box_thickness == DEFAULT_BOX_THICKNESS);
    TEST_ASSERT(config.box_confidence == DEFAULT_BOX_CONFIDENCE);
    TEST_ASSERT(config.batch_size_mode == BATCH_SETTING_MANUAL);
    TEST_ASSERT(config.batch_size == DEFAULT_BATCH_SIZE);
    TEST_ASSERT(config.inflight_batches_mode == BATCH_SETTING_MANUAL);
    TEST_ASSERT(config.inflight_batches == DEFAULT_INFLIGHT_BATCHES);
    TEST_ASSERT(config.enable_auto_tune == DEFAULT_AUTO_TUNE);
    TEST_ASSERT(config.profile_hardware_only == DEFAULT_PROFILE_HARDWARE_ONLY);
    TEST_ASSERT(config.target_fps == DEFAULT_TARGET_FPS);
    TEST_ASSERT(config.vram_budget_ratio == DEFAULT_VRAM_BUDGET_RATIO);
    TEST_ASSERT(config.vram_reserve_mb == DEFAULT_VRAM_RESERVE_MB);
    TEST_ASSERT(config.pipeline_overlap_mode == PIPELINE_FEATURE_AUTO);
    TEST_ASSERT(config.parallel_inference_mode == PIPELINE_FEATURE_AUTO);
    TEST_ASSERT(config.inference_contexts_mode == BATCH_SETTING_AUTO);
    TEST_ASSERT(config.inference_contexts == DEFAULT_INFERENCE_CONTEXTS);
    TEST_ASSERT(config.runtime == INFERENCE_RUNTIME_AUTO);
    TEST_ASSERT(config.backend_device == BACKEND_DEVICE_CUDA);
    TEST_ASSERT(config.model_type == MODEL_TYPE_AUTO);
    TEST_ASSERT(config.allow_host_backend == DEFAULT_ALLOW_HOST_BACKEND);
    TEST_ASSERT(config.list_backends == DEFAULT_LIST_BACKENDS);
    TEST_ASSERT(config.model_info == DEFAULT_MODEL_INFO);
    return 0;
}

static int pipeline_config_test_parse_args(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--input", "input.mp4",
        "--output", "output.mp4",
        "--benchmark", "bench.csv",
        "--task", "detect",
        "--model", "models/custom.engine",
        "--labels", "models/custom.names",
        "--detections", "benchmarks/detections.csv",
        "--confidence", "0.35",
        "--iou-threshold", "0.50",
        "--input-size", "512",
        "--class-ids", "0,2,5",
        "--classes", "person,traffic light",
        "--inference-backend", "tensorrt",
        "--precision", "fp16",
        "--encoder", "h264_nvenc",
        "--output-format", "mkv",
        "--lossless",
        "--mode", "gpu",
        "--filter", "blur13x13",
        "--max-frames", "12",
        "--frame-slots", "8",
        "--decoder-threads", "3",
        "--encoder-threads", "5",
        "--processor-workers", "6",
        "--memory-profile", "manual",
        "--memory-budget-mb", "512",
        "--progress-interval", "42",
        "--ffmpeg-log-level", "error",
        "--decoder", "nvdec",
        "--decoder-fallback", "none",
        "--draw-boxes",
        "--box-thickness", "4",
        "--box-confidence", "0.40",
        "--batch-size", "4",
        "--inflight-batches", "2",
        "--auto-tune",
        "--target-fps", "60",
        "--vram-budget-ratio", "0.35",
        "--vram-reserve-mb", "768",
        "--pipeline-overlap", "on",
        "--parallel-inference", "off",
        "--inference-contexts", "2",
        "--runtime", "tensorrt",
        "--backend-device", "cuda",
        "--model-type", "yolov5",
        "--allow-host-backend",
        "--no-benchmark"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(strcmp(config.input_path, "input.mp4") == 0);
    TEST_ASSERT(strcmp(config.output_path, "output.mkv") == 0);
    TEST_ASSERT(strcmp(config.benchmark_path, "bench.csv") == 0);
    TEST_ASSERT(strcmp(config.detections_path, "benchmarks/detections.csv") == 0);
    TEST_ASSERT(strcmp(config.model_path, "models/custom.engine") == 0);
    TEST_ASSERT(strcmp(config.labels_path, "models/custom.names") == 0);
    TEST_ASSERT(strcmp(config.encoder_name, "h264_nvenc") == 0);
    TEST_ASSERT(config.output_format == OUTPUT_FORMAT_MKV);
    TEST_ASSERT(config.task == PIPELINE_TASK_DETECT);
    TEST_ASSERT(config.mode == PROCESS_GPU);
    TEST_ASSERT(config.filter == FILTER_BLUR_13X13);
    TEST_ASSERT(config.max_frames == 12);
    TEST_ASSERT(config.lossless_output == 1);
    TEST_ASSERT(config.memory_profile == MEMORY_PROFILE_MANUAL);
    TEST_ASSERT(config.memory_budget_mb == 512);
    TEST_ASSERT(config.frame_slots == 8);
    TEST_ASSERT(config.decoder_threads == 3);
    TEST_ASSERT(config.encoder_threads == 5);
    TEST_ASSERT(config.processor_workers == 6);
    TEST_ASSERT(config.enable_benchmark == 0);
    TEST_ASSERT(config.confidence_threshold > 0.34f && config.confidence_threshold < 0.36f);
    TEST_ASSERT(config.iou_threshold > 0.49f && config.iou_threshold < 0.51f);
    TEST_ASSERT(config.inference_input_size == 512);
    TEST_ASSERT(config.inference_input_width == 512);
    TEST_ASSERT(config.inference_input_height == 512);
    TEST_ASSERT(config.class_filter_id_count == 3);
    TEST_ASSERT(config.class_filter_ids[0] == 0);
    TEST_ASSERT(config.class_filter_ids[1] == 2);
    TEST_ASSERT(config.class_filter_ids[2] == 5);
    TEST_ASSERT(config.class_filter_name_count == 2);
    TEST_ASSERT(strcmp(config.class_filter_names[0], "person") == 0);
    TEST_ASSERT(strcmp(config.class_filter_names[1], "traffic light") == 0);
    TEST_ASSERT(strcmp(config.inference_backend, "tensorrt") == 0);
    TEST_ASSERT(strcmp(config.inference_precision, "fp16") == 0);
    TEST_ASSERT(config.progress_interval == 42);
    TEST_ASSERT(strcmp(config.ffmpeg_log_level, "error") == 0);
    TEST_ASSERT(config.decoder_mode == VIDEO_DECODER_NVDEC);
    TEST_ASSERT(config.decoder_fallback == DECODER_FALLBACK_NONE);
    TEST_ASSERT(config.draw_boxes == 1);
    TEST_ASSERT(config.box_thickness == 4);
    TEST_ASSERT(config.box_confidence > 0.39f && config.box_confidence < 0.41f);
    TEST_ASSERT(config.batch_size_mode == BATCH_SETTING_AUTO);
    TEST_ASSERT(config.inflight_batches_mode == BATCH_SETTING_AUTO);
    TEST_ASSERT(config.enable_auto_tune == 1);
    TEST_ASSERT(config.target_fps > 59.9f && config.target_fps < 60.1f);
    TEST_ASSERT(config.vram_budget_ratio > 0.34f && config.vram_budget_ratio < 0.36f);
    TEST_ASSERT(config.vram_reserve_mb == 768);
    TEST_ASSERT(config.pipeline_overlap_mode == PIPELINE_FEATURE_ON);
    TEST_ASSERT(config.parallel_inference_mode == PIPELINE_FEATURE_OFF);
    TEST_ASSERT(config.inference_contexts_mode == BATCH_SETTING_MANUAL);
    TEST_ASSERT(config.inference_contexts == 2);
    TEST_ASSERT(config.runtime == INFERENCE_RUNTIME_TENSORRT);
    TEST_ASSERT(config.backend_device == BACKEND_DEVICE_CUDA);
    TEST_ASSERT(config.model_type == MODEL_TYPE_YOLOV5);
    TEST_ASSERT(config.allow_host_backend == 1);
    return 0;
}

static int pipeline_config_test_parse_fp32_precision(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--precision", "fp32"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(strcmp(config.inference_precision, "fp32") == 0);
    return 0;
}

static int pipeline_config_test_no_progress(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--no-progress"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(config.progress_interval == 0);
    return 0;
}

static int pipeline_config_test_parse_error_message(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--confidence", "2"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0);
    TEST_ASSERT(strstr(pipeline_config_last_error /* module: pipeline/pipeline_config */ (), "--confidence must be between 0 and 1") != NULL);
    return 0;
}

static int pipeline_config_test_missing_value_error_message(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--input-size"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0);
    TEST_ASSERT(strstr(pipeline_config_last_error /* module: pipeline/pipeline_config */ (), "missing value for --input-size") != NULL);
    return 0;
}

static int pipeline_config_test_rectangular_input_size(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--input-width", "1280",
        "--input-height", "736"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(config.inference_input_width == 1280);
    TEST_ASSERT(config.inference_input_height == 736);
    TEST_ASSERT(config.inference_input_size == 0);
    return 0;
}

static int pipeline_config_test_bad_class_ids(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--class-ids", "0,nope"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0);
    TEST_ASSERT(strstr(pipeline_config_last_error /* module: pipeline/pipeline_config */ (), "--class-ids") != NULL);
    return 0;
}

static int pipeline_config_test_manual_batch_options(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--batch-size", "3",
        "--inflight-batches", "2"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(config.batch_size_mode == BATCH_SETTING_MANUAL);
    TEST_ASSERT(config.batch_size == 3);
    TEST_ASSERT(config.inflight_batches_mode == BATCH_SETTING_MANUAL);
    TEST_ASSERT(config.inflight_batches == 2);
    return 0;
}

static int pipeline_config_test_bad_batch_size(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--batch-size", "0"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0);
    TEST_ASSERT(strstr(pipeline_config_last_error /* module: pipeline/pipeline_config */ (), "--batch-size") != NULL);
    return 0;
}

static int pipeline_config_test_runtime_auto_onnx(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--task", "detect",
        "--runtime", "auto",
        "--model", "models/model.onnx",
        "--model-info"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(config.runtime == INFERENCE_RUNTIME_AUTO);
    TEST_ASSERT(config.model_info == 1);
    TEST_ASSERT(strcmp(config.model_path, "models/model.onnx") == 0);
    return 0;
}

static int pipeline_config_test_runtime_mismatch_fails(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--task", "detect",
        "--runtime", "tensorrt",
        "--model", "models/model.onnx"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0);
    TEST_ASSERT(strstr(pipeline_config_last_error /* module: pipeline/pipeline_config */ (), "model extension selects onnxruntime") != NULL);
    return 0;
}

static int pipeline_config_test_list_backends_skips_model_validation(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--task", "detect",
        "--runtime", "tensorrt",
        "--model", "models/model.onnx",
        "--list-backends"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(config.list_backends == 1);
    return 0;
}

static int pipeline_config_test_detect_summary(void) {
    PipelineConfig config;
    char summary[4096];

    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);
    config.task = PIPELINE_TASK_DETECT;
    config.class_filter_id_count = 1;
    config.class_filter_ids[0] = 0;
    config.class_filter_name_count = 1;
    strcpy(config.class_filter_names[0], "car");

    TEST_ASSERT(pipeline_config_format_summary /* module: pipeline/pipeline_config */ (&config, summary, sizeof(summary)) == 0);
    TEST_ASSERT(strstr(summary, "task: detect") != NULL);
    TEST_ASSERT(strstr(summary, "decoder: cpu") != NULL);
    TEST_ASSERT(strstr(summary, "runtime: auto") != NULL);
    TEST_ASSERT(strstr(summary, "backend_device: cuda") != NULL);
    TEST_ASSERT(strstr(summary, "inference_device: cuda") != NULL);
    TEST_ASSERT(strstr(summary, "model_type: auto") != NULL);
    TEST_ASSERT(strstr(summary, "model_path:") != NULL);
    TEST_ASSERT(strstr(summary, "labels_path:") != NULL);
    TEST_ASSERT(strstr(summary, "detections_path:") != NULL);
    TEST_ASSERT(strstr(summary, "confidence:") != NULL);
    TEST_ASSERT(strstr(summary, "input_size:") != NULL);
    TEST_ASSERT(strstr(summary, "class_filter: ids=0 names=car") != NULL);
    TEST_ASSERT(strstr(summary, "execution_plan:") != NULL);
    TEST_ASSERT(strstr(summary, "batch_size:") != NULL);
    TEST_ASSERT(strstr(summary, "auto_tune:") != NULL);
    TEST_ASSERT(strstr(summary, "pipeline_overlap:") != NULL);
    TEST_ASSERT(strstr(summary, "parallel_inference:") != NULL);
    TEST_ASSERT(strstr(summary, "inference_contexts:") != NULL);
    TEST_ASSERT(strstr(summary, "output_mode: csv-only") != NULL);
    TEST_ASSERT(strstr(summary, "raw_upload_bridge: true") != NULL);
    TEST_ASSERT(strstr(summary, "raw_download_bridge: false") != NULL);
    TEST_ASSERT(strstr(summary, "\n  filter:") == NULL);
    TEST_ASSERT(strstr(summary, "encoder:") == NULL);
    TEST_ASSERT(strstr(summary, "output_path:") == NULL);
    TEST_ASSERT(strstr(summary, "lossless_output:") == NULL);
    return 0;
}

static int pipeline_config_test_detect_summary_with_boxes(void) {
    PipelineConfig config;
    char summary[4096];

    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);
    config.task = PIPELINE_TASK_DETECT;
    config.decoder_mode = VIDEO_DECODER_NVDEC;
    config.draw_boxes = 1;

    TEST_ASSERT(pipeline_config_format_summary /* module: pipeline/pipeline_config */ (&config, summary, sizeof(summary)) == 0);
    TEST_ASSERT(strstr(summary, "decoder: nvdec") != NULL);
    TEST_ASSERT(strstr(summary, "output_mode: cpu-annotated-video") != NULL);
    TEST_ASSERT(strstr(summary, "raw_upload_bridge: false") != NULL);
    TEST_ASSERT(strstr(summary, "draw_boxes: true") != NULL);
    TEST_ASSERT(strstr(summary, "output_path:") != NULL);
    TEST_ASSERT(strstr(summary, "box_thickness:") != NULL);
    return 0;
}

static int pipeline_config_test_detect_summary_nvdec_cpu_bridge(void) {
    PipelineConfig config;
    char summary[4096];

    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);
    config.task = PIPELINE_TASK_DETECT;
    config.decoder_mode = VIDEO_DECODER_NVDEC;
    config.backend_device = BACKEND_DEVICE_CPU;

    TEST_ASSERT(pipeline_config_format_summary /* module: pipeline/pipeline_config */ (&config, summary, sizeof(summary)) == 0);
    TEST_ASSERT(strstr(summary, "decoder: nvdec") != NULL);
    TEST_ASSERT(strstr(summary, "backend_device: cpu") != NULL);
    TEST_ASSERT(strstr(summary, "inference_device: cpu") != NULL);
    TEST_ASSERT(strstr(summary, "output_mode: csv-only") != NULL);
    TEST_ASSERT(strstr(summary, "raw_upload_bridge: false") != NULL);
    TEST_ASSERT(strstr(summary, "raw_download_bridge: true") != NULL);
    return 0;
}

static int pipeline_config_test_filter_summary(void) {
    PipelineConfig config;
    char summary[4096];

    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_format_summary /* module: pipeline/pipeline_config */ (&config, summary, sizeof(summary)) == 0);
    TEST_ASSERT(strstr(summary, "task: filter") != NULL);
    TEST_ASSERT(strstr(summary, "output_path:") != NULL);
    TEST_ASSERT(strstr(summary, "encoder:") != NULL);
    TEST_ASSERT(strstr(summary, "mode:") != NULL);
    TEST_ASSERT(strstr(summary, "filter:") != NULL);
    TEST_ASSERT(strstr(summary, "model_path:") == NULL);
    TEST_ASSERT(strstr(summary, "confidence:") == NULL);
    TEST_ASSERT(strstr(summary, "detections_path:") == NULL);
    TEST_ASSERT(strstr(summary, "execution_plan:") == NULL);
    TEST_ASSERT(strstr(summary, "batch_size:") == NULL);
    return 0;
}

int main(void) {
    printf("Running pipeline_config tests...\n");
    if (pipeline_config_test_defaults() != 0) {
        return 1;
    }
    if (pipeline_config_test_parse_args() != 0) {
        return 1;
    }
    if (pipeline_config_test_parse_fp32_precision() != 0) {
        return 1;
    }
    if (pipeline_config_test_no_progress() != 0) {
        return 1;
    }
    if (pipeline_config_test_parse_error_message() != 0) {
        return 1;
    }
    if (pipeline_config_test_missing_value_error_message() != 0) {
        return 1;
    }
    if (pipeline_config_test_rectangular_input_size() != 0) {
        return 1;
    }
    if (pipeline_config_test_bad_class_ids() != 0) {
        return 1;
    }
    if (pipeline_config_test_manual_batch_options() != 0) {
        return 1;
    }
    if (pipeline_config_test_bad_batch_size() != 0) {
        return 1;
    }
    if (pipeline_config_test_runtime_auto_onnx() != 0) {
        return 1;
    }
    if (pipeline_config_test_runtime_mismatch_fails() != 0) {
        return 1;
    }
    if (pipeline_config_test_list_backends_skips_model_validation() != 0) {
        return 1;
    }
    if (pipeline_config_test_detect_summary() != 0) {
        return 1;
    }
    if (pipeline_config_test_detect_summary_with_boxes() != 0) {
        return 1;
    }
    if (pipeline_config_test_detect_summary_nvdec_cpu_bridge() != 0) {
        return 1;
    }
    if (pipeline_config_test_filter_summary() != 0) {
        return 1;
    }
    return 0;
}
