/*
 * TensorRT inference engine module: implements the opaque C inference API with
 * CUDA streams, TensorRT execution contexts, preprocess buffers, and YOLO
 * postprocess integration. Detection runners pass CPU NV12 or CUDA NV12 frames
 * here and receive compact DetectionResult data.
 */
#include "inference/inference_engine.h"

#include "cuda_preprocess.h"
#include "yolo_postprocess.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

Logger g_logger;
thread_local std::string g_last_error = "no error";

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

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start, end) != cudaSuccess) {
        return 0.0f;
    }
    return ms;
}

struct CudaSlot {
    std::unique_ptr<nvinfer1::IExecutionContext> context;
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
};

class TensorRTYoloEngine {
public:
    explicit TensorRTYoloEngine(const InferenceConfig &cfg) : config_(cfg) {}
    ~TensorRTYoloEngine() { release(); }

    int init() {
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

    int run(const Frame *frame, DetectionResult *result, FrameTiming *timing) {
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

    int run_device(const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) {
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

    int get_batch_capability(InferenceBatchCapability *capability) const {
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
                      "%s batch, max_batch=%d",
                      supports_dynamic_batch_ ? "dynamic" : "static",
                      capability->max_batch_size);
        return 0;
    }

    int run_batch(FrameBatch *batch) {
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

    int run_device_batch(FrameBatch *batch) {
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

    int set_parallel_contexts(int context_count) {
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
        for (CudaSlot &slot : slots_) {
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

}  // namespace

struct InferenceEngine {
    TensorRTYoloEngine *impl;
};

extern "C" int inference_engine_create(InferenceEngine **engine, const InferenceConfig *config) {
    if (!engine || !config) {
        g_last_error = "invalid inference engine create arguments";
        return -1;
    }

    *engine = nullptr;
    std::unique_ptr<InferenceEngine> wrapper(new InferenceEngine{});
    std::unique_ptr<TensorRTYoloEngine> impl(new TensorRTYoloEngine(*config));
    if (impl->init() != 0) {
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
