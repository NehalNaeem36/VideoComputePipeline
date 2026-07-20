/*
 * Pipeline config module: owns command-line parsing, defaults, validation, and
 * task-aware configuration summaries. main.c builds a PipelineConfig here before
 * handing execution to the pipeline runner.
 */
#include "pipeline/pipeline_config.h"
#include "inference/backend_registry.h"
#include "utils/c_runtime.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_parse_error[256];

static void clear_parse_error(void) {
    g_parse_error[0] = '\0';
}

static void set_parse_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_parse_error, sizeof(g_parse_error), fmt, args);
    va_end(args);
}

static int copy_path(char *dst, size_t dst_size, const char *src) {
    return vcp_copy_string /* module: utils/c_runtime */ (dst, dst_size, src);
}

static const char *mode_to_string(ProcessMode mode) {
    return mode == PROCESS_GPU ? "gpu" : "cpu";
}

static const char *task_to_string(PipelineTask task) {
    return task == PIPELINE_TASK_DETECT ? "detect" : "filter";
}

static const char *filter_to_string(FilterType filter) {
    switch (filter) {
        case FILTER_GRAYSCALE:
            return "grayscale";
        case FILTER_BLUR_3X3:
            return "blur3x3";
        case FILTER_BLUR_5X5:
            return "blur5x5";
        case FILTER_BLUR_9X9:
            return "blur9x9";
        case FILTER_BLUR_13X13:
            return "blur13x13";
        default:
            return "unknown";
    }
}

static int parse_mode(const char *value, ProcessMode *mode) {
    if (strcmp(value, "cpu") == 0) {
        *mode = PROCESS_CPU;
        return 0;
    }

    if (strcmp(value, "gpu") == 0) {
        *mode = PROCESS_GPU;
        return 0;
    }

    return -1;
}

static int parse_task(const char *value, PipelineTask *task) {
    if (strcmp(value, "filter") == 0) {
        *task = PIPELINE_TASK_FILTER;
        return 0;
    }

    if (strcmp(value, "detect") == 0) {
        *task = PIPELINE_TASK_DETECT;
        return 0;
    }

    return -1;
}

static int parse_filter(const char *value, FilterType *filter) {

    /* parses arguments containing video filter and stores numeric equivalent */
    if (strcmp(value, "grayscale") == 0) {
        *filter = FILTER_GRAYSCALE;
        return 0;
    }

    if (strcmp(value, "blur3x3") == 0) {
        *filter = FILTER_BLUR_3X3;
        return 0;
    }

    if (strcmp(value, "blur5x5") == 0) {
        *filter = FILTER_BLUR_5X5;
        return 0;
    }

    if (strcmp(value, "blur9x9") == 0) {
        *filter = FILTER_BLUR_9X9;
        return 0;
    }

    if (strcmp(value, "blur13x13") == 0) {
        *filter = FILTER_BLUR_13X13;
        return 0;
    }

    return -1;
}

static const char *memory_profile_to_string(MemoryProfile profile) {
    switch (profile) {
        case MEMORY_PROFILE_AUTO:
            return "auto";
        case MEMORY_PROFILE_LOW:
            return "low";
        case MEMORY_PROFILE_BALANCED:
            return "balanced";
        case MEMORY_PROFILE_MANUAL:
            return "manual";
        default:
            return "unknown";
    }
}

static const char *decoder_mode_to_string(VideoDecoderMode mode) {
    return mode == VIDEO_DECODER_NVDEC ? "nvdec" : "cpu";
}

static const char *decoder_fallback_to_string(DecoderFallbackMode fallback) {
    return fallback == DECODER_FALLBACK_NONE ? "none" : "cpu";
}

static int detection_encoder_is_nvenc(const char *encoder_name) {
    return encoder_name && strcmp(encoder_name, "h264_nvenc") == 0;
}

static const char *detection_output_mode_summary(const PipelineConfig *config) {
    if (!config || !config->draw_boxes) {
        return "csv-only";
    }
    return detection_encoder_is_nvenc /* module: pipeline/pipeline_config */ (config->encoder_name)
               ? "cuda-annotated-video"
               : "cpu-annotated-video";
}

static int detection_requires_raw_upload(const PipelineConfig *config) {
    return config &&
           config->decoder_mode == VIDEO_DECODER_CPU &&
           config->backend_device == BACKEND_DEVICE_CUDA;
}

static int detection_requires_raw_download(const PipelineConfig *config) {
    return config &&
           config->decoder_mode == VIDEO_DECODER_NVDEC &&
           config->backend_device == BACKEND_DEVICE_CPU;
}

static const char *output_format_to_string(OutputFormat format) {
    switch (format) {
        case OUTPUT_FORMAT_MP4:
            return "mp4";
        case OUTPUT_FORMAT_MKV:
            return "mkv";
        case OUTPUT_FORMAT_AUTO:
        default:
            return "auto";
    }
}

static int parse_output_format(const char *value, OutputFormat *format) {
    if (strcmp(value, "auto") == 0) {
        *format = OUTPUT_FORMAT_AUTO;
        return 0;
    }
    if (strcmp(value, "mp4") == 0) {
        *format = OUTPUT_FORMAT_MP4;
        return 0;
    }
    if (strcmp(value, "mkv") == 0) {
        *format = OUTPUT_FORMAT_MKV;
        return 0;
    }
    return -1;
}

static int apply_output_format_extension(PipelineConfig *config) {
    const char *extension = NULL;
    char *last_dot = NULL;
    char *last_slash = NULL;
    char *last_backslash = NULL;
    char *path_end = NULL;
    size_t prefix_len = 0;
    size_t extension_len = 0;

    if (!config || config->output_format == OUTPUT_FORMAT_AUTO) {
        return 0;
    }

    extension = config->output_format == OUTPUT_FORMAT_MKV ? ".mkv" : ".mp4";
    last_dot = strrchr(config->output_path, '.');
    last_slash = strrchr(config->output_path, '/');
    last_backslash = strrchr(config->output_path, '\\');
    path_end = config->output_path + strlen(config->output_path);
    if (!last_dot || (last_slash && last_dot < last_slash) || (last_backslash && last_dot < last_backslash)) {
        last_dot = path_end;
    }

    prefix_len = (size_t)(last_dot - config->output_path);
    extension_len = strlen(extension);
    if (prefix_len + extension_len >= sizeof(config->output_path)) {
        set_parse_error /* module: pipeline/pipeline_config */ ("output path is too long after applying --output-format");
        return -1;
    }

    memcpy(config->output_path + prefix_len, extension, extension_len + 1u);
    return 0;
}

static int parse_memory_profile(const char *value, MemoryProfile *profile) {
    if (strcmp(value, "auto") == 0) {
        *profile = MEMORY_PROFILE_AUTO;
        return 0;
    }
    if (strcmp(value, "low") == 0) {
        *profile = MEMORY_PROFILE_LOW;
        return 0;
    }
    if (strcmp(value, "balanced") == 0) {
        *profile = MEMORY_PROFILE_BALANCED;
        return 0;
    }
    if (strcmp(value, "manual") == 0) {
        *profile = MEMORY_PROFILE_MANUAL;
        return 0;
    }
    return -1;
}

static int parse_decoder_mode(const char *value, VideoDecoderMode *mode) {
    if (strcmp(value, "cpu") == 0) {
        *mode = VIDEO_DECODER_CPU;
        return 0;
    }
    if (strcmp(value, "nvdec") == 0) {
        *mode = VIDEO_DECODER_NVDEC;
        return 0;
    }
    return -1;
}

static int parse_decoder_fallback(const char *value, DecoderFallbackMode *fallback) {
    if (strcmp(value, "cpu") == 0) {
        *fallback = DECODER_FALLBACK_CPU;
        return 0;
    }
    if (strcmp(value, "none") == 0) {
        *fallback = DECODER_FALLBACK_NONE;
        return 0;
    }
    return -1;
}

static int parse_int_value(const char *value, int *out) {
    char *end = NULL;
    long parsed = 0;

    if (!value || !out) {
        return -1;
    }

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < -2147483647L - 1L || parsed > 2147483647L) {
        return -1;
    }

    *out = (int)parsed;
    return 0;
}

static int parse_float_value(const char *value, float *out) {
    char *end = NULL;
    float parsed = 0.0f;

    if (!value || !out) {
        return -1;
    }

    errno = 0;
    parsed = strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0') {
        return -1;
    }

    *out = parsed;
    return 0;
}

static int parse_batch_setting(const char *value, BatchSettingMode *mode, int *out) {
    if (!value || !mode || !out) {
        return -1;
    }
    if (strcmp(value, "auto") == 0) {
        *mode = BATCH_SETTING_AUTO;
        *out = 1;
        return 0;
    }
    if (parse_int_value /* module: pipeline/pipeline_config */ (value, out) != 0 || *out <= 0) {
        return -1;
    }
    *mode = BATCH_SETTING_MANUAL;
    return 0;
}

static int parse_feature_mode(const char *value, PipelineFeatureMode *mode) {
    if (!value || !mode) {
        return -1;
    }
    if (strcmp(value, "auto") == 0) {
        *mode = PIPELINE_FEATURE_AUTO;
        return 0;
    }
    if (strcmp(value, "on") == 0) {
        *mode = PIPELINE_FEATURE_ON;
        return 0;
    }
    if (strcmp(value, "off") == 0) {
        *mode = PIPELINE_FEATURE_OFF;
        return 0;
    }
    return -1;
}

static const char *feature_mode_to_string(PipelineFeatureMode mode) {
    switch (mode) {
        case PIPELINE_FEATURE_ON:
            return "on";
        case PIPELINE_FEATURE_OFF:
            return "off";
        case PIPELINE_FEATURE_AUTO:
        default:
            return "auto";
    }
}

static int parse_ffmpeg_log_level(const char *value) {
    return value &&
           (strcmp(value, "quiet") == 0 ||
            strcmp(value, "error") == 0 ||
            strcmp(value, "warning") == 0 ||
            strcmp(value, "info") == 0 ||
            strcmp(value, "debug") == 0);
}

static void trim_span(const char **begin, const char **end) {
    while (*begin < *end && (**begin == ' ' || **begin == '\t' || **begin == '\r' || **begin == '\n')) {
        ++(*begin);
    }
    while (*end > *begin && ((*end)[-1] == ' ' || (*end)[-1] == '\t' || (*end)[-1] == '\r' || (*end)[-1] == '\n')) {
        --(*end);
    }
}

static int add_class_filter_id(PipelineConfig *config, int class_id) {
    if (!config || class_id < 0) {
        return -1;
    }
    for (int i = 0; i < config->class_filter_id_count; ++i) {
        if (config->class_filter_ids[i] == class_id) {
            return 0;
        }
    }
    if (config->class_filter_id_count >= VCP_MAX_CLASS_FILTERS) {
        return -1;
    }
    config->class_filter_ids[config->class_filter_id_count++] = class_id;
    return 0;
}

static int add_class_filter_name(PipelineConfig *config, const char *begin, const char *end) {
    size_t len = 0;
    if (!config || !begin || !end || end < begin) {
        return -1;
    }
    trim_span /* module: pipeline/pipeline_config */ (&begin, &end);
    len = (size_t)(end - begin);
    if (len == 0) {
        return 0;
    }
    if (len >= VCP_MAX_CLASS_NAME_LENGTH || config->class_filter_name_count >= VCP_MAX_CLASS_FILTERS) {
        return -1;
    }
    for (int i = 0; i < config->class_filter_name_count; ++i) {
        if (strlen(config->class_filter_names[i]) == len &&
            strncmp(config->class_filter_names[i], begin, len) == 0) {
            return 0;
        }
    }
    memcpy(config->class_filter_names[config->class_filter_name_count], begin, len);
    config->class_filter_names[config->class_filter_name_count][len] = '\0';
    config->class_filter_name_count++;
    return 0;
}

static int parse_class_id_list(PipelineConfig *config, const char *value) {
    const char *cursor = value;
    const char *token = value;
    if (!config || !value || value[0] == '\0') {
        return -1;
    }
    for (;;) {
        if (*cursor == ',' || *cursor == '\0') {
            char temp[32];
            const char *begin = token;
            const char *end = cursor;
            int id = -1;
            size_t len = 0;
            trim_span /* module: pipeline/pipeline_config */ (&begin, &end);
            len = (size_t)(end - begin);
            if (len == 0 || len >= sizeof(temp)) {
                return -1;
            }
            memcpy(temp, begin, len);
            temp[len] = '\0';
            if (parse_int_value /* module: pipeline/pipeline_config */ (temp, &id) != 0 ||
                add_class_filter_id /* module: pipeline/pipeline_config */ (config, id) != 0) {
                return -1;
            }
            if (*cursor == '\0') {
                break;
            }
            token = cursor + 1;
        }
        ++cursor;
    }
    return 0;
}

static int parse_class_name_list(PipelineConfig *config, const char *value) {
    const char *cursor = value;
    const char *token = value;
    if (!config || !value || value[0] == '\0') {
        return -1;
    }
    for (;;) {
        if (*cursor == ',' || *cursor == '\0') {
            if (add_class_filter_name /* module: pipeline/pipeline_config */ (config, token, cursor) != 0) {
                return -1;
            }
            if (*cursor == '\0') {
                break;
            }
            token = cursor + 1;
        }
        ++cursor;
    }
    return 0;
}

static int require_value(int argc, char **argv, int index, const char *option) {
    (void)argv;
    if (index + 1 >= argc) {
        set_parse_error /* module: pipeline/pipeline_config */ ("missing value for %s", option);
        return -1;
    }
    return 0;
}

void pipeline_config_default(PipelineConfig *config) {
    if (!config) {
        return;
    }

    copy_path /* module: pipeline/pipeline_config */ (config->input_path, sizeof(config->input_path), DEFAULT_INPUT_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->output_path, sizeof(config->output_path), DEFAULT_OUTPUT_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->benchmark_path, sizeof(config->benchmark_path), DEFAULT_BENCHMARK_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->detections_path, sizeof(config->detections_path), DEFAULT_DETECTIONS_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->model_path, sizeof(config->model_path), DEFAULT_MODEL_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->labels_path, sizeof(config->labels_path), DEFAULT_LABELS_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->inference_backend, sizeof(config->inference_backend), DEFAULT_INFERENCE_BACKEND);
    copy_path /* module: pipeline/pipeline_config */ (config->inference_precision, sizeof(config->inference_precision), DEFAULT_INFERENCE_PRECISION);
    copy_path /* module: pipeline/pipeline_config */ (config->encoder_name, sizeof(config->encoder_name), DEFAULT_ENCODER_NAME);
    config->task = PIPELINE_TASK_FILTER;
    config->mode = PROCESS_CPU;
    config->filter = FILTER_GRAYSCALE;
    config->max_frames = DEFAULT_MAX_FRAMES;
    config->enable_benchmark = DEFAULT_ENABLE_BENCHMARK;
    config->lossless_output = DEFAULT_LOSSLESS_OUTPUT;
    config->memory_profile = MEMORY_PROFILE_AUTO;
    config->memory_budget_mb = DEFAULT_MEMORY_BUDGET_MB;
    config->frame_slots = DEFAULT_FRAME_SLOTS;
    config->decoder_threads = DEFAULT_DECODER_THREADS;
    config->encoder_threads = DEFAULT_ENCODER_THREADS;
    config->processor_workers = DEFAULT_PROCESSOR_WORKERS;
    config->confidence_threshold = DEFAULT_DETECTION_CONFIDENCE;
    config->iou_threshold = DEFAULT_DETECTION_IOU_THRESHOLD;
    config->inference_input_size = DEFAULT_INFERENCE_INPUT_SIZE;
    config->inference_input_width = DEFAULT_INFERENCE_INPUT_SIZE;
    config->inference_input_height = DEFAULT_INFERENCE_INPUT_SIZE;
    config->detection_class_count = DEFAULT_DETECTION_CLASS_COUNT;
    config->class_filter_id_count = 0;
    memset(config->class_filter_ids, 0, sizeof(config->class_filter_ids));
    config->class_filter_name_count = 0;
    memset(config->class_filter_names, 0, sizeof(config->class_filter_names));
    config->max_detections_per_frame = DEFAULT_MAX_DETECTIONS_PER_FRAME;
    config->progress_interval = DEFAULT_PROGRESS_INTERVAL;
    copy_path /* module: pipeline/pipeline_config */ (config->ffmpeg_log_level, sizeof(config->ffmpeg_log_level), DEFAULT_FFMPEG_LOG_LEVEL);
    config->decoder_mode = VIDEO_DECODER_CPU;
    config->decoder_fallback = DECODER_FALLBACK_CPU;
    config->output_format = OUTPUT_FORMAT_AUTO;
    config->draw_boxes = DEFAULT_DRAW_BOXES;
    config->box_thickness = DEFAULT_BOX_THICKNESS;
    config->box_confidence = DEFAULT_BOX_CONFIDENCE;
    config->batch_size_mode = BATCH_SETTING_MANUAL;
    config->batch_size = DEFAULT_BATCH_SIZE;
    config->inflight_batches_mode = BATCH_SETTING_MANUAL;
    config->inflight_batches = DEFAULT_INFLIGHT_BATCHES;
    config->enable_auto_tune = DEFAULT_AUTO_TUNE;
    config->profile_hardware_only = DEFAULT_PROFILE_HARDWARE_ONLY;
    config->target_fps = DEFAULT_TARGET_FPS;
    config->vram_budget_ratio = DEFAULT_VRAM_BUDGET_RATIO;
    config->vram_reserve_mb = DEFAULT_VRAM_RESERVE_MB;
    config->pipeline_overlap_mode = PIPELINE_FEATURE_AUTO;
    config->parallel_inference_mode = PIPELINE_FEATURE_AUTO;
    config->inference_contexts_mode = BATCH_SETTING_AUTO;
    config->inference_contexts = DEFAULT_INFERENCE_CONTEXTS;
    config->runtime = INFERENCE_RUNTIME_AUTO;
    config->backend_device = BACKEND_DEVICE_CUDA;
    config->model_type = MODEL_TYPE_AUTO;
    config->allow_host_backend = DEFAULT_ALLOW_HOST_BACKEND;
    config->list_backends = DEFAULT_LIST_BACKENDS;
    config->model_info = DEFAULT_MODEL_INFO;
}

int pipeline_config_parse_args(PipelineConfig *config, int argc, char **argv) {
    if (!config) {
        return -1;
    }
    clear_parse_error /* module: pipeline/pipeline_config */ ();

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if (strcmp(arg, "--input") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->input_path, sizeof(config->input_path), argv[++i]) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --input is too long");
                return -1;
            }
        } else if (strcmp(arg, "--output") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->output_path, sizeof(config->output_path), argv[++i]) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --output is too long");
                return -1;
            }
        } else if (strcmp(arg, "--benchmark") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->benchmark_path, sizeof(config->benchmark_path), argv[++i]) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --benchmark is too long");
                return -1;
            }
        } else if (strcmp(arg, "--task") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_task /* module: pipeline/pipeline_config */ (value, &config->task) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown task: %s", value);
                return -1;
            }
        } else if (strcmp(arg, "--model") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->model_path, sizeof(config->model_path), argv[++i]) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --model is too long");
                return -1;
            }
        } else if (strcmp(arg, "--labels") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->labels_path, sizeof(config->labels_path), argv[++i]) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --labels is too long");
                return -1;
            }
        } else if (strcmp(arg, "--detections") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->detections_path, sizeof(config->detections_path), argv[++i]) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --detections is too long");
                return -1;
            }
        } else if (strcmp(arg, "--confidence") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_float_value /* module: pipeline/pipeline_config */ (argv[++i], &config->confidence_threshold) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--confidence must be a number between 0 and 1");
                return -1;
            }
            if (config->confidence_threshold < 0.0f || config->confidence_threshold > 1.0f) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--confidence must be between 0 and 1");
                return -1;
            }
        } else if (strcmp(arg, "--iou-threshold") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_float_value /* module: pipeline/pipeline_config */ (argv[++i], &config->iou_threshold) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--iou-threshold must be a number between 0 and 1");
                return -1;
            }
            if (config->iou_threshold < 0.0f || config->iou_threshold > 1.0f) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--iou-threshold must be between 0 and 1");
                return -1;
            }
        } else if (strcmp(arg, "--input-size") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->inference_input_size) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--input-size must be a positive integer");
                return -1;
            }
            if (config->inference_input_size <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--input-size must be greater than 0");
                return -1;
            }
            config->inference_input_width = config->inference_input_size;
            config->inference_input_height = config->inference_input_size;
        } else if (strcmp(arg, "--input-width") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->inference_input_width) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--input-width must be a positive integer");
                return -1;
            }
            if (config->inference_input_width <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--input-width must be greater than 0");
                return -1;
            }
            config->inference_input_size = config->inference_input_width == config->inference_input_height ? config->inference_input_width : 0;
        } else if (strcmp(arg, "--input-height") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->inference_input_height) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--input-height must be a positive integer");
                return -1;
            }
            if (config->inference_input_height <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--input-height must be greater than 0");
                return -1;
            }
            config->inference_input_size = config->inference_input_width == config->inference_input_height ? config->inference_input_width : 0;
        } else if (strcmp(arg, "--class-ids") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_class_id_list /* module: pipeline/pipeline_config */ (config, value) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--class-ids must be a comma-separated list of non-negative class IDs");
                return -1;
            }
        } else if (strcmp(arg, "--classes") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_class_name_list /* module: pipeline/pipeline_config */ (config, value) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--classes must be a comma-separated list of class names shorter than %d characters", VCP_MAX_CLASS_NAME_LENGTH);
                return -1;
            }
        } else if (strcmp(arg, "--inference-backend") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *backend = argv[++i];
            if (inference_runtime_parse /* module: inference/backend_registry */ (backend, &config->runtime) != 0 ||
                config->runtime == INFERENCE_RUNTIME_AUTO) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown inference backend: %s", backend);
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->inference_backend, sizeof(config->inference_backend), backend) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --inference-backend is too long");
                return -1;
            }
        } else if (strcmp(arg, "--runtime") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *runtime = argv[++i];
            if (inference_runtime_parse /* module: inference/backend_registry */ (runtime, &config->runtime) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown runtime: %s", runtime);
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->inference_backend,
                                                                  sizeof(config->inference_backend),
                                                                  inference_runtime_to_string /* module: inference/backend_registry */ (config->runtime)) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --runtime is too long");
                return -1;
            }
        } else if (strcmp(arg, "--backend-device") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *device = argv[++i];
            if (backend_device_parse /* module: inference/backend_registry */ (device, &config->backend_device) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown backend device: %s", device);
                return -1;
            }
        } else if (strcmp(arg, "--allow-host-backend") == 0) {
            config->allow_host_backend = 1;
        } else if (strcmp(arg, "--list-backends") == 0) {
            config->list_backends = 1;
        } else if (strcmp(arg, "--model-info") == 0) {
            config->model_info = 1;
        } else if (strcmp(arg, "--model-type") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *model_type = argv[++i];
            if (model_type_parse /* module: inference/backend_registry */ (model_type, &config->model_type) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown model type: %s", model_type);
                return -1;
            }
        } else if (strcmp(arg, "--precision") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *precision = argv[++i];
            if (strcmp(precision, "fp16") != 0 && strcmp(precision, "fp32") != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown precision: %s", precision);
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->inference_precision, sizeof(config->inference_precision), precision) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --precision is too long");
                return -1;
            }
        } else if (strcmp(arg, "--encoder") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *encoder = argv[++i];
            if (strcmp(encoder, "libx264") != 0 &&
                strcmp(encoder, "libx264rgb") != 0 &&
                strcmp(encoder, "h264_nvenc") != 0 &&
                strcmp(encoder, "mpeg4") != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown encoder: %s", encoder);
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->encoder_name, sizeof(config->encoder_name), encoder) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --encoder is too long");
                return -1;
            }
        } else if (strcmp(arg, "--output-format") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_output_format /* module: pipeline/pipeline_config */ (value, &config->output_format) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown output format: %s", value);
                return -1;
            }
        } else if (strcmp(arg, "--mode") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_mode /* module: pipeline/pipeline_config */ (value, &config->mode) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown mode: %s", value);
                return -1;
            }
        } else if (strcmp(arg, "--filter") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_filter /* module: pipeline/pipeline_config */ (value, &config->filter) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown filter: %s", value);
                return -1;
            }
        } else if (strcmp(arg, "--max-frames") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->max_frames) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--max-frames must be a non-negative integer");
                return -1;
            }
            if (config->max_frames < 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--max-frames must be 0 or greater");
                return -1;
            }
        } else if (strcmp(arg, "--frame-slots") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->frame_slots) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--frame-slots must be a positive integer");
                return -1;
            }
            if (config->frame_slots <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--frame-slots must be greater than 0");
                return -1;
            }
        } else if (strcmp(arg, "--decoder-threads") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->decoder_threads) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--decoder-threads must be a positive integer");
                return -1;
            }
            if (config->decoder_threads <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--decoder-threads must be greater than 0");
                return -1;
            }
        } else if (strcmp(arg, "--encoder-threads") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->encoder_threads) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--encoder-threads must be a positive integer");
                return -1;
            }
            if (config->encoder_threads <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--encoder-threads must be greater than 0");
                return -1;
            }
        } else if (strcmp(arg, "--processor-workers") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->processor_workers) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--processor-workers must be a positive integer");
                return -1;
            }
            if (config->processor_workers <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--processor-workers must be greater than 0");
                return -1;
            }
        } else if (strcmp(arg, "--no-benchmark") == 0) {
            config->enable_benchmark = 0;
        } else if (strcmp(arg, "--lossless") == 0) {
            config->lossless_output = 1;
        } else if (strcmp(arg, "--lossy") == 0) {
            config->lossless_output = 0;
        } else if (strcmp(arg, "--memory-profile") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_memory_profile /* module: pipeline/pipeline_config */ (value, &config->memory_profile) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown memory profile: %s", value);
                return -1;
            }
        } else if (strcmp(arg, "--memory-budget-mb") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->memory_budget_mb) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--memory-budget-mb must be a non-negative integer");
                return -1;
            }
            if (config->memory_budget_mb < 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--memory-budget-mb must be 0 or greater");
                return -1;
            }
        } else if (strcmp(arg, "--progress-interval") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->progress_interval) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--progress-interval must be a non-negative integer");
                return -1;
            }
            if (config->progress_interval < 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--progress-interval must be 0 or greater");
                return -1;
            }
        } else if (strcmp(arg, "--no-progress") == 0) {
            config->progress_interval = 0;
        } else if (strcmp(arg, "--ffmpeg-log-level") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *level = argv[++i];
            if (!parse_ffmpeg_log_level /* module: pipeline/pipeline_config */ (level)) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown FFmpeg log level: %s", level);
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->ffmpeg_log_level, sizeof(config->ffmpeg_log_level), level) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("value for --ffmpeg-log-level is too long");
                return -1;
            }
        } else if (strcmp(arg, "--decoder") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_decoder_mode /* module: pipeline/pipeline_config */ (value, &config->decoder_mode) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown decoder: %s", value);
                return -1;
            }
        } else if (strcmp(arg, "--decoder-fallback") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            const char *value = argv[++i];
            if (parse_decoder_fallback /* module: pipeline/pipeline_config */ (value, &config->decoder_fallback) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("unknown decoder fallback: %s", value);
                return -1;
            }
        } else if (strcmp(arg, "--no-decoder-fallback") == 0) {
            config->decoder_fallback = DECODER_FALLBACK_NONE;
        } else if (strcmp(arg, "--draw-boxes") == 0) {
            config->draw_boxes = 1;
        } else if (strcmp(arg, "--no-draw-boxes") == 0) {
            config->draw_boxes = 0;
        } else if (strcmp(arg, "--box-thickness") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->box_thickness) != 0 || config->box_thickness <= 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--box-thickness must be a positive integer");
                return -1;
            }
        } else if (strcmp(arg, "--box-confidence") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_float_value /* module: pipeline/pipeline_config */ (argv[++i], &config->box_confidence) != 0 ||
                config->box_confidence < 0.0f ||
                config->box_confidence > 1.0f) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--box-confidence must be a number between 0 and 1");
                return -1;
            }
        } else if (strcmp(arg, "--batch-size") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_batch_setting /* module: pipeline/pipeline_config */ (argv[++i], &config->batch_size_mode, &config->batch_size) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--batch-size must be auto or a positive integer");
                return -1;
            }
        } else if (strcmp(arg, "--inflight-batches") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_batch_setting /* module: pipeline/pipeline_config */ (argv[++i], &config->inflight_batches_mode, &config->inflight_batches) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--inflight-batches must be auto or a positive integer");
                return -1;
            }
        } else if (strcmp(arg, "--auto-tune") == 0) {
            config->enable_auto_tune = 1;
            config->batch_size_mode = BATCH_SETTING_AUTO;
            config->inflight_batches_mode = BATCH_SETTING_AUTO;
        } else if (strcmp(arg, "--target-fps") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_float_value /* module: pipeline/pipeline_config */ (argv[++i], &config->target_fps) != 0 || config->target_fps < 0.0f) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--target-fps must be a non-negative number");
                return -1;
            }
        } else if (strcmp(arg, "--vram-budget-ratio") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_float_value /* module: pipeline/pipeline_config */ (argv[++i], &config->vram_budget_ratio) != 0 ||
                config->vram_budget_ratio <= 0.0f ||
                config->vram_budget_ratio > 1.0f) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--vram-budget-ratio must be greater than 0 and at most 1");
                return -1;
            }
        } else if (strcmp(arg, "--vram-reserve-mb") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_int_value /* module: pipeline/pipeline_config */ (argv[++i], &config->vram_reserve_mb) != 0 || config->vram_reserve_mb < 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--vram-reserve-mb must be a non-negative integer");
                return -1;
            }
        } else if (strcmp(arg, "--profile-hardware") == 0) {
            config->profile_hardware_only = 1;
            config->enable_auto_tune = 1;
        } else if (strcmp(arg, "--pipeline-overlap") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_feature_mode /* module: pipeline/pipeline_config */ (argv[++i], &config->pipeline_overlap_mode) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--pipeline-overlap must be auto, on, or off");
                return -1;
            }
        } else if (strcmp(arg, "--parallel-inference") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_feature_mode /* module: pipeline/pipeline_config */ (argv[++i], &config->parallel_inference_mode) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--parallel-inference must be auto, on, or off");
                return -1;
            }
        } else if (strcmp(arg, "--inference-contexts") == 0) {
            if (require_value /* module: pipeline/pipeline_config */ (argc, argv, i, arg) != 0) {
                return -1;
            }
            if (parse_batch_setting /* module: pipeline/pipeline_config */ (argv[++i], &config->inference_contexts_mode, &config->inference_contexts) != 0) {
                set_parse_error /* module: pipeline/pipeline_config */ ("--inference-contexts must be auto or a positive integer");
                return -1;
            }
        } else {
            set_parse_error /* module: pipeline/pipeline_config */ ("unknown option: %s", arg);
            return -1;
        }
    }

    if (config->task == PIPELINE_TASK_DETECT && !config->list_backends && !config->model_info) {
        char validation[256] = {0};
        if (inference_runtime_validate_model_path /* module: inference/backend_registry */ (config->runtime,
                                                                                          config->model_path,
                                                                                          validation,
                                                                                          sizeof(validation)) != 0) {
            set_parse_error /* module: pipeline/pipeline_config */ ("%s", validation);
            return -1;
        }
    }

    return apply_output_format_extension /* module: pipeline/pipeline_config */ (config);
}

const char *pipeline_config_last_error(void) {
    return g_parse_error[0] != '\0' ? g_parse_error : "invalid command-line arguments";
}

static int append_summary(char *buffer, size_t buffer_size, size_t *offset, const char *fmt, ...) {
    int written = 0;
    va_list args;

    if (!buffer || !offset || *offset >= buffer_size) {
        return -1;
    }

    va_start(args, fmt);
    written = vsnprintf(buffer + *offset, buffer_size - *offset, fmt, args);
    va_end(args);
    if (written < 0 || (size_t)written >= buffer_size - *offset) {
        if (buffer_size > 0) {
            buffer[buffer_size - 1u] = '\0';
        }
        return -1;
    }

    *offset += (size_t)written;
    return 0;
}

int pipeline_config_format_summary(const PipelineConfig *config, char *buffer, size_t buffer_size) {
    size_t offset = 0;

    if (!config || !buffer || buffer_size == 0) {
        return -1;
    }

    buffer[0] = '\0';
    if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "task: %s\n", task_to_string /* module: pipeline/pipeline_config */ (config->task)) != 0 ||
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "input:\n") != 0 ||
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  input_path: %s\n", config->input_path) != 0 ||
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  max_frames: %d\n", config->max_frames) != 0 ||
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  decoder_threads: %d\n", config->decoder_threads) != 0) {
        return -1;
    }

    if (config->task == PIPELINE_TASK_DETECT) {
        if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "inference:\n") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  decoder: %s\n", decoder_mode_to_string /* module: pipeline/pipeline_config */ (config->decoder_mode)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  decoder_fallback: %s\n", decoder_fallback_to_string /* module: pipeline/pipeline_config */ (config->decoder_fallback)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  runtime: %s\n", inference_runtime_to_string /* module: inference/backend_registry */ (config->runtime)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  backend_device: %s\n", backend_device_to_string /* module: inference/backend_registry */ (config->backend_device)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  inference_device: %s\n", backend_device_to_string /* module: inference/backend_registry */ (config->backend_device)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  allow_host_backend: %s\n", config->allow_host_backend ? "true" : "false") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  model_type: %s\n", model_type_to_string /* module: inference/backend_registry */ (config->model_type)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  precision: %s\n", config->inference_precision) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  model_path: %s\n", config->model_path) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  labels_path: %s\n", config->labels_path) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  input_size: %dx%d\n", config->inference_input_width, config->inference_input_height) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  confidence: %.3f\n", config->confidence_threshold) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  iou_threshold: %.3f\n", config->iou_threshold) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  class_filter: ") != 0) {
            return -1;
        }
        if (config->class_filter_id_count == 0 && config->class_filter_name_count == 0) {
            if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "all\n") != 0) {
                return -1;
            }
        } else {
            for (int i = 0; i < config->class_filter_id_count; ++i) {
                if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "%s%d", i > 0 ? "," : "ids=", config->class_filter_ids[i]) != 0) {
                    return -1;
                }
            }
            for (int i = 0; i < config->class_filter_name_count; ++i) {
                if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "%s%s", i > 0 ? "," : (config->class_filter_id_count > 0 ? " names=" : "names="), config->class_filter_names[i]) != 0) {
                    return -1;
                }
            }
            if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "\n") != 0) {
                return -1;
            }
        }
        if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "execution_plan:\n") != 0) {
            return -1;
        }
        if (config->batch_size_mode == BATCH_SETTING_MANUAL) {
            if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  batch_size: %d\n", config->batch_size) != 0) {
                return -1;
            }
        } else if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  batch_size: auto\n") != 0) {
            return -1;
        }
        if (config->inflight_batches_mode == BATCH_SETTING_MANUAL) {
            if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  inflight_batches: %d\n", config->inflight_batches) != 0) {
                return -1;
            }
        } else if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  inflight_batches: auto\n") != 0) {
            return -1;
        }
        if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  auto_tune: %s\n", config->enable_auto_tune ? "true" : "false") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  profile_hardware_only: %s\n", config->profile_hardware_only ? "true" : "false") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  target_fps: %.3f\n", config->target_fps) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  vram_budget_ratio: %.3f\n", config->vram_budget_ratio) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  vram_reserve_mb: %d\n", config->vram_reserve_mb) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  pipeline_overlap: %s\n", feature_mode_to_string /* module: pipeline/pipeline_config */ (config->pipeline_overlap_mode)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  parallel_inference: %s\n", feature_mode_to_string /* module: pipeline/pipeline_config */ (config->parallel_inference_mode)) != 0) {
            return -1;
        }
        if (config->inference_contexts_mode == BATCH_SETTING_MANUAL) {
            if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  inference_contexts: %d\n", config->inference_contexts) != 0) {
                return -1;
            }
        } else if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  inference_contexts: auto\n") != 0) {
            return -1;
        }
        if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "outputs:\n") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  output_mode: %s\n", detection_output_mode_summary /* module: pipeline/pipeline_config */ (config)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  raw_upload_bridge: %s\n", detection_requires_raw_upload /* module: pipeline/pipeline_config */ (config) ? "true" : "false") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  raw_download_bridge: %s\n", detection_requires_raw_download /* module: pipeline/pipeline_config */ (config) ? "true" : "false") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  detections_path: %s\n", config->detections_path) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  draw_boxes: %s\n", config->draw_boxes ? "true" : "false") != 0) {
            return -1;
        }
        if (config->draw_boxes &&
            (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  output_path: %s\n", config->output_path) != 0 ||
             append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  output_format: %s\n", output_format_to_string /* module: pipeline/pipeline_config */ (config->output_format)) != 0 ||
             append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  encoder: %s\n", config->encoder_name) != 0 ||
             append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  box_thickness: %d\n", config->box_thickness) != 0 ||
             append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  box_confidence: %.3f\n", config->box_confidence) != 0)) {
            return -1;
        }
    } else {
        if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "processing:\n") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  mode: %s\n", mode_to_string /* module: pipeline/pipeline_config */ (config->mode)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  filter: %s\n", filter_to_string /* module: pipeline/pipeline_config */ (config->filter)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "output:\n") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  output_path: %s\n", config->output_path) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  output_format: %s\n", output_format_to_string /* module: pipeline/pipeline_config */ (config->output_format)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  encoder: %s\n", config->encoder_name) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  lossless_output: %s\n", config->lossless_output ? "true" : "false") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "workers:\n") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  frame_slots: %d\n", config->frame_slots) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  encoder_threads: %d\n", config->encoder_threads) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  processor_workers: %d\n", config->processor_workers) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "memory:\n") != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  memory_profile: %s\n", memory_profile_to_string /* module: pipeline/pipeline_config */ (config->memory_profile)) != 0 ||
            append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  memory_budget_mb: %d\n", config->memory_budget_mb) != 0) {
            return -1;
        }
    }

    if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "benchmark:\n") != 0 ||
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  enabled: %s\n", config->enable_benchmark ? "true" : "false") != 0) {
        return -1;
    }
    if (config->enable_benchmark &&
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  benchmark_path: %s\n", config->benchmark_path) != 0) {
        return -1;
    }

    if (append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "runtime:\n") != 0 ||
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  progress_interval: %d\n", config->progress_interval) != 0 ||
        append_summary /* module: pipeline/pipeline_config */ (buffer, buffer_size, &offset, "  ffmpeg_log_level: %s\n", config->ffmpeg_log_level) != 0) {
        return -1;
    }

    return 0;
}

void pipeline_config_print(const PipelineConfig *config) {

/*   A function that prints the pipeline configuration */

    if (!config) {
        return;
    }

    char summary[4096];
    if (pipeline_config_format_summary /* module: pipeline/pipeline_config */ (config, summary, sizeof(summary)) == 0) {
        fputs(summary, stdout);
    }
}
