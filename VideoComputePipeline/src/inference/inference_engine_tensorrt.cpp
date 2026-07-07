/*
 * TensorRT inference engine module: implements the opaque C inference API with
 * CUDA streams, TensorRT execution contexts, preprocess buffers, and YOLO
 * postprocess integration. Detection runners pass CPU NV12 or CUDA NV12 frames
 * here and receive compact DetectionResult data.
 */
#include "inference/inference_engine.h"

#include "cuda_preprocess.h"
#include "inference/backend_registry.h"
#include "inference_engine_internal.hpp"
#include "yolo_postprocess.h"

#ifdef VCP_ENABLE_TENSORRT
#include <NvInfer.h>
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef VCP_ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

#ifdef VCP_ENABLE_TENSORRT
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

Logger g_logger;
#endif

thread_local std::string g_last_error = "no error";

#ifdef VCP_ENABLE_TENSORRT
size_t volume(const nvinfer1::Dims &dims) {
    if (dims.nbDims <= 0) {
        return 0;
    }
    size_t value = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return 0;
        }
        value *= (size_t)dims.d[i];
    }
    return value;
}

size_t trt_element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 0;
    }
}

std::vector<char> read_file(const char *path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return {};
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> data((size_t)size);
    file.read(data.data(), size);
    return file ? data : std::vector<char>{};
}
#endif

size_t shape_volume(const std::vector<int64_t> &shape) {
    size_t value = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        value *= (size_t)dim;
    }
    return value;
}

size_t tensor_element_size(TensorDataType type) {
    switch (type) {
        case TENSOR_DTYPE_FP32: return sizeof(float);
        case TENSOR_DTYPE_FP16: return sizeof(uint16_t);
        case TENSOR_DTYPE_INT32: return sizeof(int32_t);
        case TENSOR_DTYPE_INT8: return sizeof(int8_t);
        default: return 0;
    }
}

std::string trim_copy(const std::string &value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && std::isspace((unsigned char)value[begin])) {
        ++begin;
    }
    while (end > begin && std::isspace((unsigned char)value[end - 1u])) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string lower_copy(std::string value) {
    for (char &ch : value) {
        ch = (char)std::tolower((unsigned char)ch);
    }
    return value;
}

std::vector<std::string> read_labels(const char *path) {
    std::ifstream file(path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        std::string label = trim_copy(line);
        if (!label.empty()) {
            labels.push_back(label);
        }
    }
    return labels;
}

int add_resolved_class_id_to_config(InferenceConfig &config, int class_id) {
    if (class_id < 0 || class_id >= config.class_count) {
        g_last_error = "class filter ID is outside the configured label range";
        return -1;
    }
    for (int i = 0; i < config.class_filter_id_count; ++i) {
        if (config.class_filter_ids[i] == class_id) {
            return 0;
        }
    }
    if (config.class_filter_id_count >= VCP_MAX_CLASS_FILTERS) {
        g_last_error = "too many class filters";
        return -1;
    }
    config.class_filter_ids[config.class_filter_id_count++] = class_id;
    return 0;
}

int resolve_class_configuration_for_config(InferenceConfig &config) {
    std::vector<std::string> labels = read_labels(config.labels_path);
    if (config.class_count <= 0) {
        if (labels.empty()) {
            g_last_error = "class count is not configured and labels file is empty or unreadable";
            return -1;
        }
        config.class_count = (int)labels.size();
    }

    const int initial_id_count = config.class_filter_id_count;
    int initial_ids[VCP_MAX_CLASS_FILTERS] = {0};
    std::copy(config.class_filter_ids, config.class_filter_ids + initial_id_count, initial_ids);
    config.class_filter_id_count = 0;
    for (int i = 0; i < initial_id_count; ++i) {
        if (add_resolved_class_id_to_config(config, initial_ids[i]) != 0) {
            return -1;
        }
    }

    for (int i = 0; i < config.class_filter_name_count; ++i) {
        const std::string wanted = lower_copy(trim_copy(config.class_filter_names[i]));
        int found = -1;
        for (size_t label_index = 0; label_index < labels.size(); ++label_index) {
            if (lower_copy(labels[label_index]) == wanted) {
                found = (int)label_index;
                break;
            }
        }
        if (found < 0) {
            g_last_error = "class filter name was not found in labels file: " + std::string(config.class_filter_names[i]);
            return -1;
        }
        if (add_resolved_class_id_to_config(config, found) != 0) {
            return -1;
        }
    }
    return 0;
}

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start, end) != cudaSuccess) {
        return 0.0f;
    }
    return ms;
}

struct CudaSlot {
#ifdef VCP_ENABLE_TENSORRT
    std::unique_ptr<nvinfer1::IExecutionContext> context;
#endif
    int slot_index = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t upload_done = nullptr;
    cudaEvent_t preprocess_done = nullptr;
    cudaEvent_t inference_done = nullptr;
    cudaEvent_t output_done = nullptr;
    uint8_t *d_nv12 = nullptr;
    void *d_input = nullptr;
    void *d_output = nullptr;
    std::vector<uint8_t> h_output_raw;
    std::vector<float> h_output_float;
    std::vector<int> output_dims;
};

#ifdef VCP_ENABLE_TENSORRT
class TensorRTYoloEngine : public InferenceEngineImpl {
public:
    explicit TensorRTYoloEngine(const InferenceConfig &cfg) : config_(cfg) {}
    ~TensorRTYoloEngine() { release(); }

    int init() {
        const InferenceRuntime selected_runtime = config_.runtime == INFERENCE_RUNTIME_AUTO
                                                      ? inference_runtime_from_model_path(config_.model_path)
                                                      : config_.runtime;
        char validation[256] = {0};
        if (inference_runtime_validate_model_path(config_.runtime,
                                                  config_.model_path,
                                                  validation,
                                                  sizeof(validation)) != 0) {
            g_last_error = validation;
            return -1;
        }
        if (selected_runtime != INFERENCE_RUNTIME_TENSORRT) {
            BackendRegistryInfo info{};
            inference_backend_registry_get(selected_runtime, &info);
            g_last_error = info.diagnostic[0] != '\0'
                               ? info.diagnostic
                               : "selected runtime is not handled by the TensorRT compatibility backend";
            return -1;
        }
        if (config_.backend_device != BACKEND_DEVICE_CUDA) {
            g_last_error = "TensorRT backend requires --backend-device cuda";
            return -1;
        }

        if (config_.model_path[0] == '\0') {
            g_last_error = "TensorRT model path is empty";
            return -1;
        }

        std::vector<char> model = read_file(config_.model_path);
        if (model.empty()) {
            g_last_error = "failed to read TensorRT engine file";
            return -1;
        }

        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        if (!runtime_) {
            g_last_error = "failed to create TensorRT runtime";
            return -1;
        }

        engine_.reset(runtime_->deserializeCudaEngine(model.data(), model.size()));
        if (!engine_) {
            g_last_error = "failed to deserialize TensorRT engine";
            return -1;
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            g_last_error = "failed to create TensorRT execution context";
            return -1;
        }

        if (resolve_class_configuration() != 0) {
            return -1;
        }

        if (discover_io() != 0 || allocate_static_buffers() != 0) {
            return -1;
        }

        return 0;
    }

    int run(const Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        if (!frame || !result || !timing || frame->format != FRAME_FORMAT_NV12 || !frame->planes[0] || !frame->planes[1]) {
            g_last_error = "invalid NV12 frame for inference";
            return -1;
        }

        if (ensure_frame_buffers(frame->width, frame->height) != 0) {
            return -1;
        }

        CudaSlot &slot = slots_[(size_t)frame->index % slots_.size()];
        const size_t y_size = (size_t)frame->width * (size_t)frame->height;
        const uint8_t *d_y = slot.d_nv12;
        const uint8_t *d_uv = slot.d_nv12 + y_size;

        const float scale = std::min((float)config_.input_width / (float)frame->width,
                                     (float)config_.input_height / (float)frame->height);
        const float resized_w = (float)frame->width * scale;
        const float resized_h = (float)frame->height * scale;
        const float pad_x = ((float)config_.input_width - resized_w) * 0.5f;
        const float pad_y = ((float)config_.input_height - resized_h) * 0.5f;

        cudaEventRecord(slot.start, slot.stream);
        if (cudaMemcpy2DAsync(slot.d_nv12,
                              (size_t)frame->width,
                              frame->planes[0],
                              frame->linesize[0],
                              (size_t)frame->width,
                              (size_t)frame->height,
                              cudaMemcpyHostToDevice,
                              slot.stream) != cudaSuccess ||
            cudaMemcpy2DAsync(slot.d_nv12 + y_size,
                              (size_t)frame->width,
                              frame->planes[1],
                              frame->linesize[1],
                              (size_t)frame->width,
                              (size_t)frame->height / 2u,
                              cudaMemcpyHostToDevice,
                              slot.stream) != cudaSuccess) {
            g_last_error = "failed to upload NV12 frame";
            return -1;
        }
        cudaEventRecord(slot.upload_done, slot.stream);

        cudaError_t preprocess_result = cudaSuccess;
        if (input_type_ == nvinfer1::DataType::kHALF) {
            preprocess_result = vcp_cuda_preprocess_nv12_to_nchw_fp16(d_y,
                                                                      d_uv,
                                                                      frame->width,
                                                                      frame->height,
                                                                      (size_t)frame->width,
                                                                      (size_t)frame->width,
                                                                      config_.input_width,
                                                                      config_.input_height,
                                                                      scale,
                                                                      pad_x,
                                                                      pad_y,
                                                                      reinterpret_cast<__half *>(slot.d_input),
                                                                      slot.stream);
        } else {
            preprocess_result = vcp_cuda_preprocess_nv12_to_nchw_fp32(d_y,
                                                                      d_uv,
                                                                      frame->width,
                                                                      frame->height,
                                                                      (size_t)frame->width,
                                                                      (size_t)frame->width,
                                                                      config_.input_width,
                                                                      config_.input_height,
                                                                      scale,
                                                                      pad_x,
                                                                      pad_y,
                                                                      reinterpret_cast<float *>(slot.d_input),
                                                                      slot.stream);
        }
        if (preprocess_result != cudaSuccess) {
            g_last_error = "failed to launch CUDA NV12 preprocess kernel";
            return -1;
        }
        cudaEventRecord(slot.preprocess_done, slot.stream);

        if (!slot.context ||
            !slot.context->setTensorAddress(input_name_.c_str(), slot.d_input) ||
            !slot.context->setTensorAddress(output_name_.c_str(), slot.d_output) ||
            !slot.context->enqueueV3(slot.stream)) {
            g_last_error = "TensorRT enqueueV3 failed";
            return -1;
        }
        cudaEventRecord(slot.inference_done, slot.stream);

        if (cudaMemcpyAsync(slot.h_output_raw.data(),
                            slot.d_output,
                            output_bytes_,
                            cudaMemcpyDeviceToHost,
                            slot.stream) != cudaSuccess) {
            g_last_error = "failed to copy TensorRT output to host";
            return -1;
        }
        cudaEventRecord(slot.output_done, slot.stream);
        if (cudaStreamSynchronize(slot.stream) != cudaSuccess) {
            g_last_error = "CUDA inference stream synchronization failed";
            return -1;
        }

        timing->upload_ms = elapsed_event_ms(slot.start, slot.upload_done);
        timing->preprocess_ms = elapsed_event_ms(slot.upload_done, slot.preprocess_done);
        timing->inference_ms = elapsed_event_ms(slot.preprocess_done, slot.inference_done);
        timing->download_ms = elapsed_event_ms(slot.inference_done, slot.output_done);

        auto post_start = std::chrono::high_resolution_clock::now();
        convert_output_to_float(slot);
        YoloPostprocessConfig post_config{};
        post_config.frame_index = frame->index;
        post_config.src_width = frame->width;
        post_config.src_height = frame->height;
        post_config.input_width = config_.input_width;
        post_config.input_height = config_.input_height;
        post_config.class_count = config_.class_count;
        post_config.class_filter_id_count = config_.class_filter_id_count;
        std::copy(config_.class_filter_ids,
                  config_.class_filter_ids + config_.class_filter_id_count,
                  post_config.class_filter_ids);
        post_config.confidence_threshold = config_.confidence_threshold;
        post_config.iou_threshold = config_.iou_threshold;
        post_config.scale = scale;
        post_config.pad_x = pad_x;
        post_config.pad_y = pad_y;
        int dims[8] = {0};
        for (int i = 0; i < output_dims_.nbDims && i < 8; ++i) {
            dims[i] = output_dims_.d[i];
        }
        if (yolo_postprocess(slot.h_output_float.data(), slot.h_output_float.size(), dims, output_dims_.nbDims, &post_config, result) != 0) {
            g_last_error = "YOLOv5 postprocess failed";
            return -1;
        }
        auto post_end = std::chrono::high_resolution_clock::now();
        timing->postprocess_ms = std::chrono::duration<double, std::milli>(post_end - post_start).count();
        return 0;
    }

    int run_device(const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        if (!frame || !result || !timing || !cuda_nv12_frame_is_valid(frame)) {
            g_last_error = "invalid CUDA NV12 frame for inference";
            return -1;
        }

        CudaSlot &slot = slots_[(size_t)frame->index % slots_.size()];
        const float scale = std::min((float)config_.input_width / (float)frame->width,
                                     (float)config_.input_height / (float)frame->height);
        const float resized_w = (float)frame->width * scale;
        const float resized_h = (float)frame->height * scale;
        const float pad_x = ((float)config_.input_width - resized_w) * 0.5f;
        const float pad_y = ((float)config_.input_height - resized_h) * 0.5f;

        cudaEventRecord(slot.start, slot.stream);
        cudaEventRecord(slot.upload_done, slot.stream);
        cudaError_t preprocess_result = cudaSuccess;
        if (input_type_ == nvinfer1::DataType::kHALF) {
            preprocess_result = vcp_cuda_preprocess_nv12_to_nchw_fp16(frame->d_y,
                                                                      frame->d_uv,
                                                                      frame->width,
                                                                      frame->height,
                                                                      frame->y_pitch,
                                                                      frame->uv_pitch,
                                                                      config_.input_width,
                                                                      config_.input_height,
                                                                      scale,
                                                                      pad_x,
                                                                      pad_y,
                                                                      reinterpret_cast<__half *>(slot.d_input),
                                                                      slot.stream);
        } else {
            preprocess_result = vcp_cuda_preprocess_nv12_to_nchw_fp32(frame->d_y,
                                                                      frame->d_uv,
                                                                      frame->width,
                                                                      frame->height,
                                                                      frame->y_pitch,
                                                                      frame->uv_pitch,
                                                                      config_.input_width,
                                                                      config_.input_height,
                                                                      scale,
                                                                      pad_x,
                                                                      pad_y,
                                                                      reinterpret_cast<float *>(slot.d_input),
                                                                      slot.stream);
        }
        if (preprocess_result != cudaSuccess) {
            g_last_error = "failed to launch CUDA device NV12 preprocess kernel";
            return -1;
        }
        cudaEventRecord(slot.preprocess_done, slot.stream);

        if (!slot.context ||
            !slot.context->setTensorAddress(input_name_.c_str(), slot.d_input) ||
            !slot.context->setTensorAddress(output_name_.c_str(), slot.d_output) ||
            !slot.context->enqueueV3(slot.stream)) {
            g_last_error = "TensorRT enqueueV3 failed";
            return -1;
        }
        cudaEventRecord(slot.inference_done, slot.stream);

        if (cudaMemcpyAsync(slot.h_output_raw.data(),
                            slot.d_output,
                            output_bytes_,
                            cudaMemcpyDeviceToHost,
                            slot.stream) != cudaSuccess) {
            g_last_error = "failed to copy TensorRT output to host";
            return -1;
        }
        cudaEventRecord(slot.output_done, slot.stream);
        if (cudaStreamSynchronize(slot.stream) != cudaSuccess) {
            g_last_error = "CUDA inference stream synchronization failed";
            return -1;
        }

        timing->upload_ms = 0.0;
        timing->preprocess_ms = elapsed_event_ms(slot.upload_done, slot.preprocess_done);
        timing->inference_ms = elapsed_event_ms(slot.preprocess_done, slot.inference_done);
        timing->download_ms = elapsed_event_ms(slot.inference_done, slot.output_done);

        auto post_start = std::chrono::high_resolution_clock::now();
        convert_output_to_float(slot);
        YoloPostprocessConfig post_config{};
        post_config.frame_index = frame->index;
        post_config.src_width = frame->width;
        post_config.src_height = frame->height;
        post_config.input_width = config_.input_width;
        post_config.input_height = config_.input_height;
        post_config.class_count = config_.class_count;
        post_config.class_filter_id_count = config_.class_filter_id_count;
        std::copy(config_.class_filter_ids,
                  config_.class_filter_ids + config_.class_filter_id_count,
                  post_config.class_filter_ids);
        post_config.confidence_threshold = config_.confidence_threshold;
        post_config.iou_threshold = config_.iou_threshold;
        post_config.scale = scale;
        post_config.pad_x = pad_x;
        post_config.pad_y = pad_y;
        int dims[8] = {0};
        for (int i = 0; i < output_dims_.nbDims && i < 8; ++i) {
            dims[i] = output_dims_.d[i];
        }
        if (yolo_postprocess(slot.h_output_float.data(), slot.h_output_float.size(), dims, output_dims_.nbDims, &post_config, result) != 0) {
            g_last_error = "YOLOv5 postprocess failed";
            return -1;
        }
        auto post_end = std::chrono::high_resolution_clock::now();
        timing->postprocess_ms = std::chrono::duration<double, std::milli>(post_end - post_start).count();
        return 0;
    }

    int get_batch_capability(InferenceBatchCapability *capability) const override {
        if (!capability) {
            g_last_error = "invalid batch capability output";
            return -1;
        }

        std::memset(capability, 0, sizeof(*capability));
        capability->min_batch_size = 1;
        capability->max_batch_size = 1;
        capability->supports_dynamic_batch = supports_dynamic_batch_ ? 1 : 0;
        capability->supports_true_batching = 0;
        capability->supports_parallel_contexts = slots_.size() > 1 ? 1 : 0;
        capability->max_parallel_contexts = (int)slots_.size();
        capability->selected_context_count = selected_context_count_;
        capability->input_width = config_.input_width;
        capability->input_height = config_.input_height;
        capability->input_bytes_per_frame = input_bytes_per_frame_;
        capability->output_bytes_per_frame = output_bytes_per_frame_;
        std::snprintf(capability->description,
                      sizeof(capability->description),
                      "TensorRT single-frame enqueue with %d parallel context slots",
                      capability->max_parallel_contexts);
        return 0;
    }

    int run_batch(FrameBatch *batch) override {
        if (!batch || batch->use_cuda_frames || batch->valid_frames < 0 || batch->valid_frames > batch->capacity) {
            g_last_error = "invalid CPU FrameBatch for inference";
            return -1;
        }
        if (batch->valid_frames > 0 &&
            ensure_frame_buffers(batch->cpu_frames[0].width, batch->cpu_frames[0].height) != 0) {
            return -1;
        }
        if (selected_context_count_ <= 1 || batch->valid_frames <= 1) {
            for (int i = 0; i < batch->valid_frames; ++i) {
                if (run(&batch->cpu_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                    return -1;
                }
            }
            return 0;
        }

        for (int begin = 0; begin < batch->valid_frames; begin += selected_context_count_) {
            const int end = std::min(batch->valid_frames, begin + selected_context_count_);
            std::vector<std::thread> workers;
            std::atomic<int> failed{0};
            workers.reserve((size_t)(end - begin));
            for (int i = begin; i < end; ++i) {
                workers.emplace_back([this, batch, i, &failed]() {
                    if (run(&batch->cpu_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                        failed.store(1);
                    }
                });
            }
            for (std::thread &worker : workers) {
                worker.join();
            }
            if (failed.load() != 0) {
                g_last_error = "parallel CPU batch inference worker failed";
                return -1;
            }
        }
        return 0;
    }

    int run_device_batch(FrameBatch *batch) override {
        if (!batch || !batch->use_cuda_frames || batch->valid_frames < 0 || batch->valid_frames > batch->capacity) {
            g_last_error = "invalid CUDA FrameBatch for inference";
            return -1;
        }
        if (selected_context_count_ <= 1 || batch->valid_frames <= 1) {
            for (int i = 0; i < batch->valid_frames; ++i) {
                if (run_device(&batch->cuda_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                    return -1;
                }
            }
            return 0;
        }

        for (int begin = 0; begin < batch->valid_frames; begin += selected_context_count_) {
            const int end = std::min(batch->valid_frames, begin + selected_context_count_);
            std::vector<std::thread> workers;
            std::atomic<int> failed{0};
            workers.reserve((size_t)(end - begin));
            for (int i = begin; i < end; ++i) {
                workers.emplace_back([this, batch, i, &failed]() {
                    if (run_device(&batch->cuda_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                        failed.store(1);
                    }
                });
            }
            for (std::thread &worker : workers) {
                worker.join();
            }
            if (failed.load() != 0) {
                g_last_error = "parallel CUDA batch inference worker failed";
                return -1;
            }
        }
        return 0;
    }

    int set_parallel_contexts(int context_count) override {
        if (context_count < 1) {
            g_last_error = "parallel inference context count must be positive";
            return -1;
        }
        if (context_count > (int)slots_.size()) {
            g_last_error = "requested parallel inference contexts exceed allocated CUDA slots";
            return -1;
        }
        selected_context_count_ = context_count;
        return 0;
    }

private:
    int add_resolved_class_id(int class_id) {
        if (class_id < 0 || class_id >= config_.class_count) {
            g_last_error = "class filter ID is outside the configured label range";
            return -1;
        }
        for (int i = 0; i < config_.class_filter_id_count; ++i) {
            if (config_.class_filter_ids[i] == class_id) {
                return 0;
            }
        }
        if (config_.class_filter_id_count >= VCP_MAX_CLASS_FILTERS) {
            g_last_error = "too many class filters";
            return -1;
        }
        config_.class_filter_ids[config_.class_filter_id_count++] = class_id;
        return 0;
    }

    int resolve_class_configuration() {
        std::vector<std::string> labels = read_labels(config_.labels_path);
        if (config_.class_count <= 0) {
            if (labels.empty()) {
                g_last_error = "class count is not configured and labels file is empty or unreadable";
                return -1;
            }
            config_.class_count = (int)labels.size();
        }

        const int initial_id_count = config_.class_filter_id_count;
        config_.class_filter_id_count = 0;
        for (int i = 0; i < initial_id_count; ++i) {
            if (add_resolved_class_id(config_.class_filter_ids[i]) != 0) {
                return -1;
            }
        }

        for (int i = 0; i < config_.class_filter_name_count; ++i) {
            const std::string wanted = lower_copy(trim_copy(config_.class_filter_names[i]));
            int found = -1;
            for (size_t label_index = 0; label_index < labels.size(); ++label_index) {
                if (lower_copy(labels[label_index]) == wanted) {
                    found = (int)label_index;
                    break;
                }
            }
            if (found < 0) {
                g_last_error = "class filter name was not found in labels file: " + std::string(config_.class_filter_names[i]);
                return -1;
            }
            if (add_resolved_class_id(found) != 0) {
                return -1;
            }
        }

        return 0;
    }

    int discover_io() {
        const int nb_tensors = engine_->getNbIOTensors();
        for (int i = 0; i < nb_tensors; ++i) {
            const char *name = engine_->getIOTensorName(i);
            if (!name) {
                continue;
            }
            const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT && input_name_.empty()) {
                input_name_ = name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_name_.empty()) {
                output_name_ = name;
            }
        }

        if (input_name_.empty() || output_name_.empty()) {
            g_last_error = "TensorRT engine must have one input and one output tensor";
            return -1;
        }

        input_type_ = engine_->getTensorDataType(input_name_.c_str());
        if (input_type_ != nvinfer1::DataType::kHALF && input_type_ != nvinfer1::DataType::kFLOAT) {
            g_last_error = "TensorRT input must be FP16 or FP32 for this backend";
            return -1;
        }

        nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
        bool dynamic = false;
        int declared_batch = 1;
        for (int i = 0; i < input_dims.nbDims; ++i) {
            dynamic = dynamic || input_dims.d[i] < 0;
        }
        if (input_dims.nbDims == 4 && input_dims.d[0] > 0) {
            declared_batch = input_dims.d[0];
        }
        if (dynamic) {
            if (!context_->setInputShape(input_name_.c_str(), nvinfer1::Dims4{1, 3, config_.input_height, config_.input_width})) {
                g_last_error = "failed to set TensorRT dynamic input shape";
                return -1;
            }
            input_dims = context_->getTensorShape(input_name_.c_str());
            supports_dynamic_batch_ = true;
            max_supported_batch_size_ = 8;
        } else {
            max_supported_batch_size_ = declared_batch > 0 ? declared_batch : 1;
        }

        if (input_dims.nbDims != 4 ||
            input_dims.d[0] != 1 ||
            input_dims.d[1] != 3 ||
            input_dims.d[2] != config_.input_height ||
            input_dims.d[3] != config_.input_width) {
            g_last_error = "unsupported TensorRT engine input shape";
            return -1;
        }

        output_dims_ = context_->getTensorShape(output_name_.c_str());
        output_elements_ = volume(output_dims_);
        output_type_ = engine_->getTensorDataType(output_name_.c_str());
        output_element_size_ = trt_element_size(output_type_);
        output_bytes_ = output_elements_ * output_element_size_;
        if (output_elements_ == 0 || output_element_size_ == 0 || (output_type_ != nvinfer1::DataType::kFLOAT && output_type_ != nvinfer1::DataType::kHALF)) {
            g_last_error = "unsupported TensorRT engine output shape or type";
            return -1;
        }
        input_bytes_per_frame_ = (size_t)3 * (size_t)config_.input_width * (size_t)config_.input_height *
                                 (input_type_ == nvinfer1::DataType::kHALF ? sizeof(__half) : sizeof(float));
        output_bytes_per_frame_ = output_bytes_;
        return 0;
    }

    int allocate_static_buffers() {
        slots_.resize(3);
        const size_t input_element_size = input_type_ == nvinfer1::DataType::kHALF ? sizeof(__half) : sizeof(float);
        const size_t input_bytes = (size_t)3 * (size_t)config_.input_width * (size_t)config_.input_height * input_element_size;
        for (size_t i = 0; i < slots_.size(); ++i) {
            CudaSlot &slot = slots_[i];
            slot.slot_index = (int)i;
            slot.context.reset(engine_->createExecutionContext());
            if (!slot.context) {
                g_last_error = "failed to create TensorRT slot execution context";
                return -1;
            }
            if (supports_dynamic_batch_ &&
                !slot.context->setInputShape(input_name_.c_str(), nvinfer1::Dims4{1, 3, config_.input_height, config_.input_width})) {
                g_last_error = "failed to set TensorRT slot dynamic input shape";
                return -1;
            }
            if (cudaStreamCreate(&slot.stream) != cudaSuccess ||
                cudaEventCreate(&slot.start) != cudaSuccess ||
                cudaEventCreate(&slot.upload_done) != cudaSuccess ||
                cudaEventCreate(&slot.preprocess_done) != cudaSuccess ||
                cudaEventCreate(&slot.inference_done) != cudaSuccess ||
                cudaEventCreate(&slot.output_done) != cudaSuccess ||
                cudaMalloc(&slot.d_input, input_bytes) != cudaSuccess ||
                cudaMalloc(&slot.d_output, output_bytes_) != cudaSuccess) {
                g_last_error = "failed to allocate CUDA/TensorRT static buffers";
                return -1;
            }
            slot.h_output_raw.resize(output_bytes_);
            slot.h_output_float.resize(output_elements_);
        }
        return 0;
    }

    int ensure_frame_buffers(int width, int height) {
        if (width == frame_width_ && height == frame_height_) {
            return 0;
        }
        if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0) {
            g_last_error = "NV12 frame dimensions must be positive and even";
            return -1;
        }

        const size_t bytes = (size_t)width * (size_t)height * 3u / 2u;
        for (CudaSlot &slot : slots_) {
            if (slot.d_nv12) {
                cudaFree(slot.d_nv12);
                slot.d_nv12 = nullptr;
            }
            if (cudaMalloc((void **)&slot.d_nv12, bytes) != cudaSuccess) {
                g_last_error = "failed to allocate CUDA NV12 frame buffer";
                return -1;
            }
        }
        frame_width_ = width;
        frame_height_ = height;
        return 0;
    }

    void convert_output_to_float(CudaSlot &slot) {
        if (output_type_ == nvinfer1::DataType::kFLOAT) {
            std::copy_n(reinterpret_cast<const float *>(slot.h_output_raw.data()), output_elements_, slot.h_output_float.data());
            return;
        }

        const __half *half_data = reinterpret_cast<const __half *>(slot.h_output_raw.data());
        for (size_t i = 0; i < output_elements_; ++i) {
            slot.h_output_float[i] = __half2float(half_data[i]);
        }
    }

    void release() {
        for (CudaSlot &slot : slots_) {
            if (slot.d_nv12) cudaFree(slot.d_nv12);
            if (slot.d_input) cudaFree(slot.d_input);
            if (slot.d_output) cudaFree(slot.d_output);
            if (slot.start) cudaEventDestroy(slot.start);
            if (slot.upload_done) cudaEventDestroy(slot.upload_done);
            if (slot.preprocess_done) cudaEventDestroy(slot.preprocess_done);
            if (slot.inference_done) cudaEventDestroy(slot.inference_done);
            if (slot.output_done) cudaEventDestroy(slot.output_done);
            if (slot.stream) cudaStreamDestroy(slot.stream);
        }
        slots_.clear();
        context_.reset();
        engine_.reset();
        runtime_.reset();
    }

    InferenceConfig config_{};
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string input_name_;
    std::string output_name_;
    nvinfer1::Dims output_dims_{};
    nvinfer1::DataType input_type_{nvinfer1::DataType::kHALF};
    nvinfer1::DataType output_type_{nvinfer1::DataType::kFLOAT};
    size_t output_elements_ = 0;
    size_t output_element_size_ = 0;
    size_t output_bytes_ = 0;
    size_t input_bytes_per_frame_ = 0;
    size_t output_bytes_per_frame_ = 0;
    int max_supported_batch_size_ = 1;
    int selected_context_count_ = 1;
    bool supports_dynamic_batch_ = false;
    int frame_width_ = 0;
    int frame_height_ = 0;
    std::vector<CudaSlot> slots_;
};
#endif

#if defined(VCP_ENABLE_ONNXRUNTIME) || defined(VCP_ENABLE_LIBTORCH)
class CudaYoloRuntimeEngine : public InferenceEngineImpl {
public:
    explicit CudaYoloRuntimeEngine(const InferenceConfig &cfg) : config_(cfg) {}
    ~CudaYoloRuntimeEngine() override { release_common(); }

    int run(const Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        if (!frame || !result || !timing || frame->format != FRAME_FORMAT_NV12 || !frame->planes[0] || !frame->planes[1]) {
            g_last_error = "invalid NV12 frame for inference";
            return -1;
        }
        if (ensure_frame_buffers(frame->width, frame->height) != 0) {
            return -1;
        }

        CudaSlot &slot = slots_[(size_t)frame->index % slots_.size()];
        const size_t y_size = (size_t)frame->width * (size_t)frame->height;
        const uint8_t *d_y = slot.d_nv12;
        const uint8_t *d_uv = slot.d_nv12 + y_size;

        const float scale = std::min((float)config_.input_width / (float)frame->width,
                                     (float)config_.input_height / (float)frame->height);
        const float resized_w = (float)frame->width * scale;
        const float resized_h = (float)frame->height * scale;
        const float pad_x = ((float)config_.input_width - resized_w) * 0.5f;
        const float pad_y = ((float)config_.input_height - resized_h) * 0.5f;

        cudaEventRecord(slot.start, slot.stream);
        if (cudaMemcpy2DAsync(slot.d_nv12,
                              (size_t)frame->width,
                              frame->planes[0],
                              frame->linesize[0],
                              (size_t)frame->width,
                              (size_t)frame->height,
                              cudaMemcpyHostToDevice,
                              slot.stream) != cudaSuccess ||
            cudaMemcpy2DAsync(slot.d_nv12 + y_size,
                              (size_t)frame->width,
                              frame->planes[1],
                              frame->linesize[1],
                              (size_t)frame->width,
                              (size_t)frame->height / 2u,
                              cudaMemcpyHostToDevice,
                              slot.stream) != cudaSuccess) {
            g_last_error = "failed to upload NV12 frame";
            return -1;
        }
        cudaEventRecord(slot.upload_done, slot.stream);
        return run_preprocessed(slot, frame->index, frame->width, frame->height, d_y, d_uv, (size_t)frame->width, (size_t)frame->width, scale, pad_x, pad_y, result, timing);
    }

    int run_device(const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        if (!frame || !result || !timing || !cuda_nv12_frame_is_valid(frame)) {
            g_last_error = "invalid CUDA NV12 frame for inference";
            return -1;
        }

        CudaSlot &slot = slots_[(size_t)frame->index % slots_.size()];
        const float scale = std::min((float)config_.input_width / (float)frame->width,
                                     (float)config_.input_height / (float)frame->height);
        const float resized_w = (float)frame->width * scale;
        const float resized_h = (float)frame->height * scale;
        const float pad_x = ((float)config_.input_width - resized_w) * 0.5f;
        const float pad_y = ((float)config_.input_height - resized_h) * 0.5f;

        cudaEventRecord(slot.start, slot.stream);
        cudaEventRecord(slot.upload_done, slot.stream);
        const int status = run_preprocessed(slot,
                                            frame->index,
                                            frame->width,
                                            frame->height,
                                            frame->d_y,
                                            frame->d_uv,
                                            frame->y_pitch,
                                            frame->uv_pitch,
                                            scale,
                                            pad_x,
                                            pad_y,
                                            result,
                                            timing);
        timing->upload_ms = 0.0;
        return status;
    }

    int get_batch_capability(InferenceBatchCapability *capability) const override {
        if (!capability) {
            g_last_error = "invalid batch capability output";
            return -1;
        }

        std::memset(capability, 0, sizeof(*capability));
        capability->min_batch_size = 1;
        capability->max_batch_size = max_batch_size_;
        capability->supports_dynamic_batch = supports_dynamic_batch_ ? 1 : 0;
        capability->supports_true_batching = 0;
        capability->supports_parallel_contexts = slots_.size() > 1 ? 1 : 0;
        capability->max_parallel_contexts = (int)slots_.size();
        capability->selected_context_count = selected_context_count_;
        capability->input_width = config_.input_width;
        capability->input_height = config_.input_height;
        capability->input_bytes_per_frame = input_bytes_;
        capability->output_bytes_per_frame = output_bytes_;
        std::snprintf(capability->description,
                      sizeof(capability->description),
                      "%s single-frame execution with %d inference lane slots",
                      backend_name(),
                      capability->max_parallel_contexts);
        return 0;
    }

    int set_parallel_contexts(int context_count) override {
        if (context_count < 1) {
            g_last_error = "parallel inference context count must be positive";
            return -1;
        }
        if (context_count > (int)slots_.size()) {
            g_last_error = std::string(backend_name()) + " requested inference lanes exceed allocated CUDA slots";
            return -1;
        }
        selected_context_count_ = context_count;
        return 0;
    }

    int run_batch(FrameBatch *batch) override {
        if (!batch || batch->use_cuda_frames || batch->valid_frames < 0 || batch->valid_frames > batch->capacity) {
            g_last_error = "invalid CPU FrameBatch for inference";
            return -1;
        }
        if (selected_context_count_ <= 1 || batch->valid_frames <= 1) {
            for (int i = 0; i < batch->valid_frames; ++i) {
                if (run(&batch->cpu_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                    return -1;
                }
            }
            return 0;
        }

        for (int begin = 0; begin < batch->valid_frames; begin += selected_context_count_) {
            const int end = std::min(batch->valid_frames, begin + selected_context_count_);
            std::vector<std::thread> workers;
            std::atomic<int> failed{0};
            workers.reserve((size_t)(end - begin));
            for (int i = begin; i < end; ++i) {
                workers.emplace_back([this, batch, i, &failed]() {
                    if (run(&batch->cpu_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                        failed.store(1);
                    }
                });
            }
            for (std::thread &worker : workers) {
                worker.join();
            }
            if (failed.load() != 0) {
                g_last_error = std::string(backend_name()) + " parallel CPU batch inference worker failed";
                return -1;
            }
        }
        return 0;
    }

    int run_device_batch(FrameBatch *batch) override {
        if (!batch || !batch->use_cuda_frames || batch->valid_frames < 0 || batch->valid_frames > batch->capacity) {
            g_last_error = "invalid CUDA FrameBatch for inference";
            return -1;
        }
        if (selected_context_count_ <= 1 || batch->valid_frames <= 1) {
            for (int i = 0; i < batch->valid_frames; ++i) {
                if (run_device(&batch->cuda_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                    return -1;
                }
            }
            return 0;
        }

        for (int begin = 0; begin < batch->valid_frames; begin += selected_context_count_) {
            const int end = std::min(batch->valid_frames, begin + selected_context_count_);
            std::vector<std::thread> workers;
            std::atomic<int> failed{0};
            workers.reserve((size_t)(end - begin));
            for (int i = begin; i < end; ++i) {
                workers.emplace_back([this, batch, i, &failed]() {
                    if (run_device(&batch->cuda_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                        failed.store(1);
                    }
                });
            }
            for (std::thread &worker : workers) {
                worker.join();
            }
            if (failed.load() != 0) {
                g_last_error = std::string(backend_name()) + " parallel CUDA batch inference worker failed";
                return -1;
            }
        }
        return 0;
    }

protected:
    virtual const char *backend_name() const = 0;
    virtual int run_backend(CudaSlot &slot, FrameTiming *timing) = 0;

    int init_common() {
        if (config_.backend_device != BACKEND_DEVICE_CUDA) {
            g_last_error = std::string(backend_name()) + " backend currently requires --backend-device cuda";
            return -1;
        }
        if (resolve_class_configuration_for_config(config_) != 0) {
            return -1;
        }
        if (input_dtype_ != TENSOR_DTYPE_FP16 && input_dtype_ != TENSOR_DTYPE_FP32) {
            input_dtype_ = config_.use_fp16 ? TENSOR_DTYPE_FP16 : TENSOR_DTYPE_FP32;
        }
        input_bytes_ = (size_t)3 * (size_t)config_.input_width * (size_t)config_.input_height * tensor_element_size(input_dtype_);
        if (input_bytes_ == 0) {
            g_last_error = "invalid model input tensor size";
            return -1;
        }
        slots_.resize(3);
        for (size_t i = 0; i < slots_.size(); ++i) {
            CudaSlot &slot = slots_[i];
            slot.slot_index = (int)i;
            if (cudaStreamCreate(&slot.stream) != cudaSuccess ||
                cudaEventCreate(&slot.start) != cudaSuccess ||
                cudaEventCreate(&slot.upload_done) != cudaSuccess ||
                cudaEventCreate(&slot.preprocess_done) != cudaSuccess ||
                cudaEventCreate(&slot.inference_done) != cudaSuccess ||
                cudaEventCreate(&slot.output_done) != cudaSuccess ||
                cudaMalloc(&slot.d_input, input_bytes_) != cudaSuccess) {
                g_last_error = "failed to allocate CUDA inference input buffers";
                return -1;
            }
            if (output_bytes_ > 0 && cudaMalloc(&slot.d_output, output_bytes_) != cudaSuccess) {
                g_last_error = "failed to allocate CUDA inference output buffers";
                return -1;
            }
            if (output_bytes_ > 0) {
                slot.h_output_raw.resize(output_bytes_);
                slot.h_output_float.resize(output_elements_);
            }
        }
        return 0;
    }

    int set_output_shape(std::vector<int64_t> shape, TensorDataType dtype) {
        if (shape.empty()) {
            g_last_error = "model output shape is empty";
            return -1;
        }
        if (shape[0] < 0) {
            shape[0] = 1;
        }
        const int attributes = 5 + config_.class_count;
        for (int64_t &dim : shape) {
            if (dim < 0) {
                dim = -1;
            }
        }
        if (shape.size() >= 3 && shape[1] < 0 && shape[2] == attributes) {
            const int grid8 = config_.input_width / 8;
            const int grid16 = config_.input_width / 16;
            const int grid32 = config_.input_width / 32;
            shape[1] = (int64_t)(3 * (grid8 * grid8 + grid16 * grid16 + grid32 * grid32));
        } else if (shape.size() >= 3 && shape[2] < 0 && shape[1] == attributes) {
            const int grid8 = config_.input_width / 8;
            const int grid16 = config_.input_width / 16;
            const int grid32 = config_.input_width / 32;
            shape[2] = (int64_t)(3 * (grid8 * grid8 + grid16 * grid16 + grid32 * grid32));
        }

        output_elements_ = shape_volume(shape);
        const size_t element_size = tensor_element_size(dtype);
        if (output_elements_ == 0 || element_size == 0 || (dtype != TENSOR_DTYPE_FP32 && dtype != TENSOR_DTYPE_FP16)) {
            g_last_error = "unsupported model output shape or type";
            return -1;
        }
        output_shape_ = shape;
        output_dtype_ = dtype;
        output_bytes_ = output_elements_ * element_size;
        return 0;
    }

    int run_preprocessed(CudaSlot &slot,
                         int frame_index,
                         int frame_width,
                         int frame_height,
                         const uint8_t *d_y,
                         const uint8_t *d_uv,
                         size_t y_pitch,
                         size_t uv_pitch,
                         float scale,
                         float pad_x,
                         float pad_y,
                         DetectionResult *result,
                         FrameTiming *timing) {
        cudaError_t preprocess_result = cudaSuccess;
        if (input_dtype_ == TENSOR_DTYPE_FP16) {
            preprocess_result = vcp_cuda_preprocess_nv12_to_nchw_fp16(d_y,
                                                                      d_uv,
                                                                      frame_width,
                                                                      frame_height,
                                                                      y_pitch,
                                                                      uv_pitch,
                                                                      config_.input_width,
                                                                      config_.input_height,
                                                                      scale,
                                                                      pad_x,
                                                                      pad_y,
                                                                      reinterpret_cast<__half *>(slot.d_input),
                                                                      slot.stream);
        } else {
            preprocess_result = vcp_cuda_preprocess_nv12_to_nchw_fp32(d_y,
                                                                      d_uv,
                                                                      frame_width,
                                                                      frame_height,
                                                                      y_pitch,
                                                                      uv_pitch,
                                                                      config_.input_width,
                                                                      config_.input_height,
                                                                      scale,
                                                                      pad_x,
                                                                      pad_y,
                                                                      reinterpret_cast<float *>(slot.d_input),
                                                                      slot.stream);
        }
        if (preprocess_result != cudaSuccess) {
            g_last_error = "failed to launch CUDA NV12 preprocess kernel";
            return -1;
        }
        cudaEventRecord(slot.preprocess_done, slot.stream);
        timing->upload_ms = elapsed_event_ms(slot.start, slot.upload_done);
        timing->preprocess_ms = elapsed_event_ms(slot.upload_done, slot.preprocess_done);

        if (run_backend(slot, timing) != 0) {
            return -1;
        }

        auto post_start = std::chrono::high_resolution_clock::now();
        YoloPostprocessConfig post_config{};
        post_config.frame_index = frame_index;
        post_config.src_width = frame_width;
        post_config.src_height = frame_height;
        post_config.input_width = config_.input_width;
        post_config.input_height = config_.input_height;
        post_config.class_count = config_.class_count;
        post_config.class_filter_id_count = config_.class_filter_id_count;
        std::copy(config_.class_filter_ids,
                  config_.class_filter_ids + config_.class_filter_id_count,
                  post_config.class_filter_ids);
        post_config.confidence_threshold = config_.confidence_threshold;
        post_config.iou_threshold = config_.iou_threshold;
        post_config.scale = scale;
        post_config.pad_x = pad_x;
        post_config.pad_y = pad_y;

        if (slot.output_dims.empty()) {
            slot.output_dims.reserve(output_shape_.size());
            for (int64_t dim : output_shape_) {
                slot.output_dims.push_back((int)dim);
            }
        }
        if (yolo_postprocess(slot.h_output_float.data(),
                             slot.h_output_float.size(),
                             slot.output_dims.data(),
                             (int)slot.output_dims.size(),
                             &post_config,
                             result) != 0) {
            g_last_error = "YOLOv5 postprocess failed";
            return -1;
        }
        auto post_end = std::chrono::high_resolution_clock::now();
        timing->postprocess_ms = std::chrono::duration<double, std::milli>(post_end - post_start).count();
        return 0;
    }

    int ensure_frame_buffers(int width, int height) {
        if (width == frame_width_ && height == frame_height_) {
            return 0;
        }
        if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0) {
            g_last_error = "NV12 frame dimensions must be positive and even";
            return -1;
        }

        const size_t bytes = (size_t)width * (size_t)height * 3u / 2u;
        for (CudaSlot &slot : slots_) {
            if (slot.d_nv12) {
                cudaFree(slot.d_nv12);
                slot.d_nv12 = nullptr;
            }
            if (cudaMalloc((void **)&slot.d_nv12, bytes) != cudaSuccess) {
                g_last_error = "failed to allocate CUDA NV12 frame buffer";
                return -1;
            }
        }
        frame_width_ = width;
        frame_height_ = height;
        return 0;
    }

    void convert_output_raw_to_float(CudaSlot &slot) {
        if (output_dtype_ == TENSOR_DTYPE_FP32) {
            std::copy_n(reinterpret_cast<const float *>(slot.h_output_raw.data()), output_elements_, slot.h_output_float.data());
            return;
        }

        const __half *half_data = reinterpret_cast<const __half *>(slot.h_output_raw.data());
        for (size_t i = 0; i < output_elements_; ++i) {
            slot.h_output_float[i] = __half2float(half_data[i]);
        }
    }

    void release_common() {
        for (CudaSlot &slot : slots_) {
            if (slot.d_nv12) cudaFree(slot.d_nv12);
            if (slot.d_input) cudaFree(slot.d_input);
            if (slot.d_output) cudaFree(slot.d_output);
            if (slot.start) cudaEventDestroy(slot.start);
            if (slot.upload_done) cudaEventDestroy(slot.upload_done);
            if (slot.preprocess_done) cudaEventDestroy(slot.preprocess_done);
            if (slot.inference_done) cudaEventDestroy(slot.inference_done);
            if (slot.output_done) cudaEventDestroy(slot.output_done);
            if (slot.stream) cudaStreamDestroy(slot.stream);
        }
        slots_.clear();
    }

    InferenceConfig config_{};
    TensorDataType input_dtype_{TENSOR_DTYPE_FP32};
    TensorDataType output_dtype_{TENSOR_DTYPE_FP32};
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
    size_t output_elements_ = 0;
    int max_batch_size_ = 1;
    int selected_context_count_ = 1;
    bool supports_dynamic_batch_ = false;
    int frame_width_ = 0;
    int frame_height_ = 0;
    std::vector<int64_t> output_shape_;
    std::vector<CudaSlot> slots_;
};
#endif

#ifdef VCP_ENABLE_ONNXRUNTIME
class OnnxRuntimeYoloEngine final : public CudaYoloRuntimeEngine {
public:
    explicit OnnxRuntimeYoloEngine(const InferenceConfig &cfg)
        : CudaYoloRuntimeEngine(cfg),
          env_(ORT_LOGGING_LEVEL_WARNING, "VideoComputePipeline") {}

    int init() {
        if (inference_runtime_validate_model_path(INFERENCE_RUNTIME_ONNXRUNTIME, config_.model_path, validation_, sizeof(validation_)) != 0) {
            g_last_error = validation_;
            return -1;
        }

        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (config_.backend_device == BACKEND_DEVICE_CUDA) {
            try {
                Ort::CUDAProviderOptions cuda_options;
                cuda_options.Update({{"device_id", "0"}});
                const OrtCUDAProviderOptionsV2 *provider_options = cuda_options;
                session_options_.AppendExecutionProvider_CUDA_V2(*provider_options);
            } catch (const Ort::Exception &e) {
                g_last_error = std::string("failed to enable ONNX Runtime CUDA execution provider: ") + e.what();
                return -1;
            }
        }

#ifdef _WIN32
        const std::wstring model_path = widen(config_.model_path);
        sessions_.emplace_back(new Ort::Session(env_, model_path.c_str(), session_options_));
#else
        sessions_.emplace_back(new Ort::Session(env_, config_.model_path, session_options_));
#endif
        allocator_.reset(new Ort::AllocatorWithDefaultOptions());
        if (discover_model() != 0) {
            return -1;
        }
        if (init_common() != 0) {
            return -1;
        }
        return ensure_session_count(slots_.size());
    }

protected:
    const char *backend_name() const override { return "ONNX Runtime"; }

    int run_backend(CudaSlot &slot, FrameTiming *timing) override {
        if (cudaStreamSynchronize(slot.stream) != cudaSuccess) {
            g_last_error = "failed to synchronize CUDA preprocess before ONNX Runtime execution";
            return -1;
        }

        std::array<int64_t, 4> input_shape = {1, 3, config_.input_height, config_.input_width};
        Ort::MemoryInfo cuda_memory("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor(cuda_memory,
                                                           slot.d_input,
                                                           input_bytes_,
                                                           input_shape.data(),
                                                           input_shape.size(),
                                                           input_ort_type_);
        Ort::Value output_tensor = Ort::Value::CreateTensor(cuda_memory,
                                                            slot.d_output,
                                                            output_bytes_,
                                                            output_shape_.data(),
                                                            output_shape_.size(),
                                                            output_ort_type_);

        auto inference_start = std::chrono::high_resolution_clock::now();
        try {
            Ort::Session *session = session_for_slot(slot);
            if (!session) {
                g_last_error = "ONNX Runtime lane session is unavailable";
                return -1;
            }
            Ort::IoBinding binding(*session);
            binding.BindInput(input_name_.c_str(), input_tensor);
            binding.BindOutput(output_name_.c_str(), output_tensor);
            session->Run(Ort::RunOptions{nullptr}, binding);
        } catch (const Ort::Exception &e) {
            g_last_error = std::string("ONNX Runtime inference failed: ") + e.what();
            return -1;
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        timing->inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

        cudaEventRecord(slot.inference_done, slot.stream);
        if (cudaMemcpyAsync(slot.h_output_raw.data(), slot.d_output, output_bytes_, cudaMemcpyDeviceToHost, slot.stream) != cudaSuccess) {
            g_last_error = "failed to copy ONNX Runtime output to host";
            return -1;
        }
        cudaEventRecord(slot.output_done, slot.stream);
        if (cudaStreamSynchronize(slot.stream) != cudaSuccess) {
            g_last_error = "failed to synchronize ONNX Runtime output copy";
            return -1;
        }
        timing->download_ms = elapsed_event_ms(slot.inference_done, slot.output_done);
        convert_output_raw_to_float(slot);
        slot.output_dims.clear();
        for (int64_t dim : output_shape_) {
            slot.output_dims.push_back((int)dim);
        }
        return 0;
    }

private:
#ifdef _WIN32
    static std::wstring widen(const char *value) {
        std::wstring output;
        if (!value) {
            return output;
        }
        while (*value) {
            output.push_back((wchar_t)(unsigned char)*value);
            ++value;
        }
        return output;
    }
#endif

    Ort::Session *session_for_slot(const CudaSlot &slot) {
        if (sessions_.empty()) {
            return nullptr;
        }
        const size_t index = (size_t)slot.slot_index < sessions_.size() ? (size_t)slot.slot_index : 0u;
        return sessions_[index].get();
    }

    int ensure_session_count(size_t count) {
        if (count == 0) {
            return 0;
        }
        try {
            while (sessions_.size() < count) {
#ifdef _WIN32
                const std::wstring model_path = widen(config_.model_path);
                sessions_.emplace_back(new Ort::Session(env_, model_path.c_str(), session_options_));
#else
                sessions_.emplace_back(new Ort::Session(env_, config_.model_path, session_options_));
#endif
            }
        } catch (const Ort::Exception &e) {
            g_last_error = std::string("failed to create ONNX Runtime inference lane session: ") + e.what();
            return -1;
        }
        return 0;
    }

    TensorDataType from_ort_type(ONNXTensorElementDataType type) const {
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return TENSOR_DTYPE_FP32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return TENSOR_DTYPE_FP16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return TENSOR_DTYPE_INT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return TENSOR_DTYPE_INT8;
            default: return TENSOR_DTYPE_UNKNOWN;
        }
    }

    int discover_model() {
        if (sessions_.empty() || !sessions_[0] || !allocator_) {
            g_last_error = "ONNX Runtime session was not initialized";
            return -1;
        }
        Ort::Session &session = *sessions_[0];
        if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) {
            g_last_error = "ONNX Runtime backend expects one input and at least one output";
            return -1;
        }

        auto input_name = session.GetInputNameAllocated(0, *allocator_);
        auto output_name = session.GetOutputNameAllocated(0, *allocator_);
        input_name_ = input_name.get();
        output_name_ = output_name.get();

        const auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = input_info.GetShape();
        if (input_shape.size() != 4 ||
            input_shape[1] != 3 ||
            input_shape[2] != config_.input_height ||
            input_shape[3] != config_.input_width) {
            g_last_error = "unsupported ONNX model input shape; expected 1x3xinput_heightxinput_width";
            return -1;
        }
        max_batch_size_ = input_shape[0] > 0 ? (int)input_shape[0] : 8;
        supports_dynamic_batch_ = input_shape[0] < 0;
        input_ort_type_ = input_info.GetElementType();
        input_dtype_ = from_ort_type(input_ort_type_);
        if (input_dtype_ != TENSOR_DTYPE_FP32 && input_dtype_ != TENSOR_DTYPE_FP16) {
            g_last_error = "ONNX model input must be FP32 or FP16";
            return -1;
        }

        const auto output_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        output_ort_type_ = output_info.GetElementType();
        if (set_output_shape(output_info.GetShape(), from_ort_type(output_ort_type_)) != 0) {
            return -1;
        }
        return 0;
    }

    char validation_[256] = {0};
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::vector<std::unique_ptr<Ort::Session>> sessions_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    std::string input_name_;
    std::string output_name_;
    ONNXTensorElementDataType input_ort_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    ONNXTensorElementDataType output_ort_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
};

class OnnxRuntimeCpuYoloEngine final : public InferenceEngineImpl {
public:
    explicit OnnxRuntimeCpuYoloEngine(const InferenceConfig &cfg)
        : config_(cfg),
          env_(ORT_LOGGING_LEVEL_WARNING, "VideoComputePipelineCPU") {}

    int init() {
        if (inference_runtime_validate_model_path(INFERENCE_RUNTIME_ONNXRUNTIME, config_.model_path, validation_, sizeof(validation_)) != 0) {
            g_last_error = validation_;
            return -1;
        }
        if (resolve_class_configuration_for_config(config_) != 0) {
            return -1;
        }

        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        const std::wstring model_path = widen(config_.model_path);
        session_.reset(new Ort::Session(env_, model_path.c_str(), session_options_));
#else
        session_.reset(new Ort::Session(env_, config_.model_path, session_options_));
#endif
        allocator_.reset(new Ort::AllocatorWithDefaultOptions());
        return discover_model();
    }

    int run(const Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        if (!frame || !result || !timing || frame->format != FRAME_FORMAT_NV12 || !frame->planes[0] || !frame->planes[1]) {
            g_last_error = "ONNX Runtime CPU backend requires CPU NV12 frames";
            return -1;
        }
        if (input_ort_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            g_last_error = "ONNX Runtime CPU backend currently requires an FP32 input tensor model";
            return -1;
        }

        const float scale = std::min((float)config_.input_width / (float)frame->width,
                                     (float)config_.input_height / (float)frame->height);
        const float resized_w = (float)frame->width * scale;
        const float resized_h = (float)frame->height * scale;
        const float pad_x = ((float)config_.input_width - resized_w) * 0.5f;
        const float pad_y = ((float)config_.input_height - resized_h) * 0.5f;

        auto preprocess_start = std::chrono::high_resolution_clock::now();
        preprocess_nv12_to_nchw_fp32(frame, scale, pad_x, pad_y);
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        timing->upload_ms = 0.0;
        timing->preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

        std::array<int64_t, 4> input_shape = {1, 3, config_.input_height, config_.input_width};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                  input_buffer_.data(),
                                                                  input_buffer_.size(),
                                                                  input_shape.data(),
                                                                  input_shape.size());

        auto inference_start = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> outputs;
        try {
            const char *input_names[] = {input_name_.c_str()};
            const char *output_names[] = {output_name_.c_str()};
            outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        } catch (const Ort::Exception &e) {
            g_last_error = std::string("ONNX Runtime CPU inference failed: ") + e.what();
            return -1;
        }
        auto inference_end = std::chrono::high_resolution_clock::now();
        timing->inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
        timing->download_ms = 0.0;

        auto post_start = std::chrono::high_resolution_clock::now();
        if (outputs.empty() || !outputs[0].IsTensor()) {
            g_last_error = "ONNX Runtime CPU output is not a tensor";
            return -1;
        }
        if (copy_output_to_float(outputs[0]) != 0) {
            return -1;
        }

        YoloPostprocessConfig post_config{};
        post_config.frame_index = frame->index;
        post_config.src_width = frame->width;
        post_config.src_height = frame->height;
        post_config.input_width = config_.input_width;
        post_config.input_height = config_.input_height;
        post_config.class_count = config_.class_count;
        post_config.class_filter_id_count = config_.class_filter_id_count;
        std::copy(config_.class_filter_ids,
                  config_.class_filter_ids + config_.class_filter_id_count,
                  post_config.class_filter_ids);
        post_config.confidence_threshold = config_.confidence_threshold;
        post_config.iou_threshold = config_.iou_threshold;
        post_config.scale = scale;
        post_config.pad_x = pad_x;
        post_config.pad_y = pad_y;
        if (yolo_postprocess(output_float_.data(),
                             output_float_.size(),
                             output_dims_int_.data(),
                             (int)output_dims_int_.size(),
                             &post_config,
                             result) != 0) {
            g_last_error = "YOLOv5 CPU postprocess failed";
            return -1;
        }
        auto post_end = std::chrono::high_resolution_clock::now();
        timing->postprocess_ms = std::chrono::duration<double, std::milli>(post_end - post_start).count();
        return 0;
    }

    int run_device(const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        (void)frame;
        (void)result;
        (void)timing;
        g_last_error = "ONNX Runtime CPU backend requires a CPU NV12 bridge before inference";
        return -1;
    }

    int get_batch_capability(InferenceBatchCapability *capability) const override {
        if (!capability) {
            g_last_error = "invalid batch capability output";
            return -1;
        }
        std::memset(capability, 0, sizeof(*capability));
        capability->min_batch_size = 1;
        capability->max_batch_size = 1;
        capability->supports_dynamic_batch = 0;
        capability->supports_true_batching = 0;
        capability->supports_parallel_contexts = 0;
        capability->max_parallel_contexts = 1;
        capability->selected_context_count = 1;
        capability->input_width = config_.input_width;
        capability->input_height = config_.input_height;
        capability->input_bytes_per_frame = input_buffer_.size() * sizeof(float);
        capability->output_bytes_per_frame = output_float_.size() * sizeof(float);
        std::snprintf(capability->description,
                      sizeof(capability->description),
                      "ONNX Runtime CPU single-frame execution");
        return 0;
    }

    int set_parallel_contexts(int context_count) override {
        if (context_count != 1) {
            g_last_error = "ONNX Runtime CPU backend currently supports one inference context";
            return -1;
        }
        return 0;
    }

    int run_batch(FrameBatch *batch) override {
        if (!batch || batch->use_cuda_frames || batch->valid_frames < 0 || batch->valid_frames > batch->capacity) {
            g_last_error = "invalid CPU FrameBatch for ONNX Runtime CPU inference";
            return -1;
        }
        for (int i = 0; i < batch->valid_frames; ++i) {
            if (run(&batch->cpu_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                return -1;
            }
        }
        return 0;
    }

    int run_device_batch(FrameBatch *batch) override {
        (void)batch;
        g_last_error = "ONNX Runtime CPU backend requires CPU FrameBatch input";
        return -1;
    }

private:
#ifdef _WIN32
    static std::wstring widen(const char *value) {
        std::wstring output;
        if (!value) {
            return output;
        }
        while (*value) {
            output.push_back((wchar_t)(unsigned char)*value);
            ++value;
        }
        return output;
    }
#endif

    int discover_model() {
        if (!session_ || !allocator_) {
            g_last_error = "ONNX Runtime CPU session was not initialized";
            return -1;
        }
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() < 1) {
            g_last_error = "ONNX Runtime CPU backend expects one input and at least one output";
            return -1;
        }
        auto input_name = session_->GetInputNameAllocated(0, *allocator_);
        auto output_name = session_->GetOutputNameAllocated(0, *allocator_);
        input_name_ = input_name.get();
        output_name_ = output_name.get();

        const auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = input_info.GetShape();
        if (input_shape.size() != 4 ||
            input_shape[1] != 3 ||
            input_shape[2] != config_.input_height ||
            input_shape[3] != config_.input_width) {
            g_last_error = "unsupported ONNX CPU model input shape; expected 1x3xinput_heightxinput_width";
            return -1;
        }
        input_ort_type_ = input_info.GetElementType();
        if (input_ort_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            g_last_error = "ONNX Runtime CPU backend currently requires FP32 model input";
            return -1;
        }
        input_buffer_.resize((size_t)3 * (size_t)config_.input_height * (size_t)config_.input_width);

        const auto output_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        output_ort_type_ = output_info.GetElementType();
        output_dims_ = output_info.GetShape();
        if (output_dims_.empty()) {
            g_last_error = "ONNX Runtime CPU output shape is empty";
            return -1;
        }
        const int attributes = 5 + config_.class_count;
        if (output_dims_.size() >= 3 && output_dims_[1] < 0 && output_dims_[2] == attributes) {
            const int grid8 = config_.input_width / 8;
            const int grid16 = config_.input_width / 16;
            const int grid32 = config_.input_width / 32;
            output_dims_[1] = (int64_t)(3 * (grid8 * grid8 + grid16 * grid16 + grid32 * grid32));
        } else if (output_dims_.size() >= 3 && output_dims_[2] < 0 && output_dims_[1] == attributes) {
            const int grid8 = config_.input_width / 8;
            const int grid16 = config_.input_width / 16;
            const int grid32 = config_.input_width / 32;
            output_dims_[2] = (int64_t)(3 * (grid8 * grid8 + grid16 * grid16 + grid32 * grid32));
        }
        for (int64_t &dim : output_dims_) {
            if (dim < 0) {
                dim = 1;
            }
        }
        return 0;
    }

    void preprocess_nv12_to_nchw_fp32(const Frame *frame, float scale, float pad_x, float pad_y) {
        const int dst_w = config_.input_width;
        const int dst_h = config_.input_height;
        const size_t channel_size = (size_t)dst_w * (size_t)dst_h;
        float *r_plane = input_buffer_.data();
        float *g_plane = input_buffer_.data() + channel_size;
        float *b_plane = input_buffer_.data() + channel_size * 2u;

        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                const float src_xf = ((float)x - pad_x) / scale;
                const float src_yf = ((float)y - pad_y) / scale;
                const size_t dst_index = (size_t)y * (size_t)dst_w + (size_t)x;
                if (src_xf < 0.0f || src_yf < 0.0f || src_xf > (float)(frame->width - 1) || src_yf > (float)(frame->height - 1)) {
                    r_plane[dst_index] = 114.0f / 255.0f;
                    g_plane[dst_index] = 114.0f / 255.0f;
                    b_plane[dst_index] = 114.0f / 255.0f;
                    continue;
                }
                const int sx = std::max(0, std::min(frame->width - 1, (int)(src_xf + 0.5f)));
                const int sy = std::max(0, std::min(frame->height - 1, (int)(src_yf + 0.5f)));
                const int uv_x = sx & ~1;
                const int uv_y = sy / 2;
                const uint8_t yy = frame->planes[0][(size_t)sy * frame->linesize[0] + (size_t)sx];
                const uint8_t uu = frame->planes[1][(size_t)uv_y * frame->linesize[1] + (size_t)uv_x];
                const uint8_t vv = frame->planes[1][(size_t)uv_y * frame->linesize[1] + (size_t)uv_x + 1u];
                const float yf = (float)yy;
                const float uf = (float)uu - 128.0f;
                const float vf = (float)vv - 128.0f;
                const float rf = std::max(0.0f, std::min(255.0f, yf + 1.402f * vf));
                const float gf = std::max(0.0f, std::min(255.0f, yf - 0.344136f * uf - 0.714136f * vf));
                const float bf = std::max(0.0f, std::min(255.0f, yf + 1.772f * uf));
                r_plane[dst_index] = rf / 255.0f;
                g_plane[dst_index] = gf / 255.0f;
                b_plane[dst_index] = bf / 255.0f;
            }
        }
    }

    int copy_output_to_float(Ort::Value &output) {
        const auto info = output.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        for (int64_t &dim : shape) {
            if (dim < 0) {
                dim = 1;
            }
        }
        const size_t elements = shape_volume(shape);
        if (elements == 0) {
            g_last_error = "ONNX Runtime CPU output tensor is empty";
            return -1;
        }
        output_dims_ = shape;
        output_dims_int_.resize(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] > INT_MAX) {
                g_last_error = "ONNX Runtime CPU output tensor dimension is too large";
                return -1;
            }
            output_dims_int_[i] = (int)shape[i];
        }
        output_float_.resize(elements);
        const ONNXTensorElementDataType type = info.GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float *data = output.GetTensorData<float>();
            std::copy_n(data, elements, output_float_.data());
            return 0;
        }
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const uint16_t *data = output.GetTensorData<uint16_t>();
            for (size_t i = 0; i < elements; ++i) {
                __half h;
                std::memcpy(&h, &data[i], sizeof(h));
                output_float_[i] = __half2float(h);
            }
            return 0;
        }
        g_last_error = "ONNX Runtime CPU output tensor must be FP32 or FP16";
        return -1;
    }

    InferenceConfig config_{};
    char validation_[256] = {0};
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    std::string input_name_;
    std::string output_name_;
    ONNXTensorElementDataType input_ort_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    ONNXTensorElementDataType output_ort_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    std::vector<float> input_buffer_;
    std::vector<float> output_float_;
    std::vector<int64_t> output_dims_;
    std::vector<int> output_dims_int_;
};
#endif

}  // namespace

void vcp_inference_set_last_error(const std::string &message) {
    g_last_error = message;
}

void vcp_inference_set_last_error(const char *message) {
    g_last_error = message ? message : "unknown inference error";
}

struct InferenceEngine {
    InferenceEngineImpl *impl;
};

extern "C" int inference_engine_create(InferenceEngine **engine, const InferenceConfig *config) {
    if (!engine || !config) {
        g_last_error = "invalid inference engine create arguments";
        return -1;
    }

    *engine = nullptr;
    std::unique_ptr<InferenceEngine> wrapper(new InferenceEngine{});
    const InferenceRuntime selected_runtime = config->runtime == INFERENCE_RUNTIME_AUTO
                                                  ? inference_runtime_from_model_path(config->model_path)
                                                  : config->runtime;
    std::unique_ptr<InferenceEngineImpl> impl;

    if (selected_runtime == INFERENCE_RUNTIME_TENSORRT) {
#ifdef VCP_ENABLE_TENSORRT
        std::unique_ptr<TensorRTYoloEngine> trt_impl(new TensorRTYoloEngine(*config));
        if (trt_impl->init() != 0) {
            return -1;
        }
        impl = std::move(trt_impl);
#else
        g_last_error = "TensorRT backend was not compiled. Rebuild with ENABLE_TENSORRT=ON.";
        return -1;
#endif
    } else if (selected_runtime == INFERENCE_RUNTIME_ONNXRUNTIME) {
#ifdef VCP_ENABLE_ONNXRUNTIME
        if (config->backend_device == BACKEND_DEVICE_CPU) {
            std::unique_ptr<OnnxRuntimeCpuYoloEngine> onnx_cpu_impl(new OnnxRuntimeCpuYoloEngine(*config));
            if (onnx_cpu_impl->init() != 0) {
                return -1;
            }
            impl = std::move(onnx_cpu_impl);
        } else {
            std::unique_ptr<OnnxRuntimeYoloEngine> onnx_impl(new OnnxRuntimeYoloEngine(*config));
            if (onnx_impl->init() != 0) {
                return -1;
            }
            impl = std::move(onnx_impl);
        }
#else
        g_last_error = "ONNX Runtime backend was not compiled. Rebuild with ENABLE_ONNXRUNTIME=ON.";
        return -1;
#endif
    } else if (selected_runtime == INFERENCE_RUNTIME_TORCHSCRIPT) {
#ifdef VCP_ENABLE_LIBTORCH
        impl = vcp_create_libtorch_yolo_engine(*config);
        if (!impl) {
            return -1;
        }
#else
        g_last_error = "TorchScript backend was not compiled. Rebuild with ENABLE_LIBTORCH=ON.";
        return -1;
#endif
    } else {
        g_last_error = "could not infer inference runtime from model extension";
        return -1;
    }

    wrapper->impl = impl.release();
    *engine = wrapper.release();
    return 0;
}

extern "C" int inference_engine_run_nv12(InferenceEngine *engine, const Frame *frame, DetectionResult *result, FrameTiming *timing) {
    if (!engine || !engine->impl) {
        g_last_error = "inference engine is not initialized";
        return -1;
    }
    return engine->impl->run(frame, result, timing);
}

extern "C" int inference_engine_get_batch_capability(InferenceEngine *engine, InferenceBatchCapability *capability) {
    if (!engine || !engine->impl) {
        g_last_error = "inference engine is not initialized";
        return -1;
    }
    return engine->impl->get_batch_capability(capability);
}

extern "C" int inference_engine_set_parallel_contexts(InferenceEngine *engine, int context_count) {
    if (!engine || !engine->impl) {
        g_last_error = "inference engine is not initialized";
        return -1;
    }
    return engine->impl->set_parallel_contexts(context_count);
}

extern "C" int inference_engine_run_cuda_nv12(InferenceEngine *engine, const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) {
    if (!engine || !engine->impl) {
        g_last_error = "inference engine is not initialized";
        return -1;
    }
    return engine->impl->run_device(frame, result, timing);
}

extern "C" int inference_engine_run_nv12_batch(InferenceEngine *engine, FrameBatch *batch) {
    if (!engine || !engine->impl) {
        g_last_error = "inference engine is not initialized";
        return -1;
    }
    return engine->impl->run_batch(batch);
}

extern "C" int inference_engine_run_cuda_nv12_batch(InferenceEngine *engine, FrameBatch *batch) {
    if (!engine || !engine->impl) {
        g_last_error = "inference engine is not initialized";
        return -1;
    }
    return engine->impl->run_device_batch(batch);
}

extern "C" void inference_engine_destroy(InferenceEngine *engine) {
    if (!engine) {
        return;
    }
    delete engine->impl;
    delete engine;
}

extern "C" const char *inference_engine_last_error(void) {
    return g_last_error.c_str();
}
