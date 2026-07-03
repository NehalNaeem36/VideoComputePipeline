/*
 * LibTorch TorchScript inference backend: loads exported TorchScript YOLO
 * models through the private inference engine interface. The first
 * implementation supports CPU NV12 preprocessing and GPU-resident NVDEC frames
 * through the shared CUDA NV12-to-NCHW FP32 preprocess kernel.
 */
#include "inference_engine_internal.hpp"

#include "cuda_preprocess.h"
#include "inference/backend_registry.h"
#include "yolo_postprocess.h"

#include <cuda_runtime.h>

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

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

const char *path_extension(const char *path) {
    if (!path) {
        return "";
    }
    const char *dot = std::strrchr(path, '.');
    const char *slash = std::strrchr(path, '/');
    const char *backslash = std::strrchr(path, '\\');
    if (!dot || (slash && dot < slash) || (backslash && dot < backslash)) {
        return "";
    }
    return dot;
}

bool str_ieq(const char *a, const char *b) {
    if (!a || !b) {
        return false;
    }
    while (*a && *b) {
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

std::string raw_pt_export_message(const char *model_path, const std::string &error) {
    std::string message = "failed to load TorchScript model";
    if (model_path && str_ieq(path_extension(model_path), ".pt")) {
        message =
            "The provided .pt file is not a TorchScript model.\n"
            "This project has no Python runtime dependency, so raw Ultralytics/PyTorch checkpoints cannot be loaded directly.\n\n"
            "Export the model first:\n\n"
            "yolo export model=models/best.pt format=torchscript imgsz=640\n\n"
            "Then run with:\n"
            "--model models/best.torchscript --runtime torchscript";
        if (!error.empty()) {
            message += "\n\nLibTorch error: " + error;
        }
        return message;
    }
    if (!error.empty()) {
        message += ": " + error;
    }
    return message;
}

bool tensor_looks_like_detection(const torch::Tensor &tensor) {
    if (!tensor.defined() || tensor.numel() <= 0 || tensor.dim() < 2) {
        return false;
    }
    const int64_t last = tensor.size(tensor.dim() - 1);
    if (last >= 5 && last <= 4096) {
        return true;
    }
    if (tensor.dim() >= 2) {
        const int64_t attrs = tensor.size(tensor.dim() - 2);
        if (attrs >= 5 && attrs <= 4096) {
            return true;
        }
    }
    return false;
}

bool extract_detection_tensor_recursive(const torch::jit::IValue &value, int depth, torch::Tensor *out) {
    if (!out || depth > 4) {
        return false;
    }
    if (value.isTensor()) {
        torch::Tensor tensor = value.toTensor();
        if (tensor_looks_like_detection(tensor)) {
            *out = tensor;
            return true;
        }
        return false;
    }
    if (value.isTuple()) {
        const auto elements = value.toTuple()->elements();
        for (const torch::jit::IValue &item : elements) {
            if (extract_detection_tensor_recursive(item, depth + 1, out)) {
                return true;
            }
        }
    }
    if (value.isList()) {
        const auto list = value.toList();
        for (size_t i = 0; i < list.size(); ++i) {
            if (extract_detection_tensor_recursive(list.get(i), depth + 1, out)) {
                return true;
            }
        }
    }
    return false;
}

int add_resolved_class_id(InferenceConfig &config, int class_id) {
    if (class_id < 0 || class_id >= config.class_count) {
        vcp_inference_set_last_error("class filter ID is outside the configured label range");
        return -1;
    }
    for (int i = 0; i < config.class_filter_id_count; ++i) {
        if (config.class_filter_ids[i] == class_id) {
            return 0;
        }
    }
    if (config.class_filter_id_count >= VCP_MAX_CLASS_FILTERS) {
        vcp_inference_set_last_error("too many class filters");
        return -1;
    }
    config.class_filter_ids[config.class_filter_id_count++] = class_id;
    return 0;
}

int resolve_class_configuration(InferenceConfig &config) {
    std::vector<std::string> labels = read_labels(config.labels_path);
    if (config.class_count <= 0) {
        if (labels.empty()) {
            vcp_inference_set_last_error("class count is not configured and labels file is empty or unreadable");
            return -1;
        }
        config.class_count = (int)labels.size();
    }

    const int initial_id_count = config.class_filter_id_count;
    int initial_ids[VCP_MAX_CLASS_FILTERS] = {0};
    std::copy(config.class_filter_ids, config.class_filter_ids + initial_id_count, initial_ids);
    config.class_filter_id_count = 0;
    for (int i = 0; i < initial_id_count; ++i) {
        if (add_resolved_class_id(config, initial_ids[i]) != 0) {
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
            vcp_inference_set_last_error("class filter name was not found in labels file: " + std::string(config.class_filter_names[i]));
            return -1;
        }
        if (add_resolved_class_id(config, found) != 0) {
            return -1;
        }
    }
    return 0;
}

double elapsed_event_ms(cudaEvent_t begin, cudaEvent_t end) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, begin, end) != cudaSuccess) {
        return 0.0;
    }
    return (double)ms;
}

class TorchScriptYoloEngine final : public InferenceEngineImpl {
public:
    explicit TorchScriptYoloEngine(const InferenceConfig &cfg) : config_(cfg) {}

    ~TorchScriptYoloEngine() override {
        release_cuda_preprocess_resources();
    }

    int init() {
        char validation[256] = {0};
        if (inference_runtime_validate_model_path(INFERENCE_RUNTIME_TORCHSCRIPT,
                                                  config_.model_path,
                                                  validation,
                                                  sizeof(validation)) != 0) {
            vcp_inference_set_last_error(validation);
            return -1;
        }
        if (resolve_class_configuration(config_) != 0) {
            return -1;
        }

        use_cuda_ = config_.backend_device == BACKEND_DEVICE_CUDA;
        if (use_cuda_ && !torch::cuda::is_available()) {
            if (!config_.allow_host_backend) {
                vcp_inference_set_last_error("TorchScript CUDA backend was requested, but torch::cuda::is_available() is false");
                return -1;
            }
            use_cuda_ = false;
        }
        device_ = use_cuda_ ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

        try {
            module_ = torch::jit::load(config_.model_path, device_);
            module_.eval();
            module_.to(device_, torch::kFloat32);
        } catch (const c10::Error &e) {
            vcp_inference_set_last_error(raw_pt_export_message(config_.model_path, e.what_without_backtrace()));
            return -1;
        }

        input_buffer_.resize((size_t)3 * (size_t)config_.input_height * (size_t)config_.input_width);
        return 0;
    }

    int run(const Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        if (!frame || !result || !timing || frame->format != FRAME_FORMAT_NV12 || !frame->planes[0] || !frame->planes[1]) {
            vcp_inference_set_last_error("TorchScript backend requires CPU NV12 frames");
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
        timing->preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

        torch::Tensor input_cpu = torch::from_blob(input_buffer_.data(),
                                                   {1, 3, config_.input_height, config_.input_width},
                                                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        torch::Tensor input = input_cpu;
        if (use_cuda_) {
            auto upload_start = std::chrono::high_resolution_clock::now();
            input = input_cpu.to(device_, torch::kFloat32, false, true);
            auto upload_end = std::chrono::high_resolution_clock::now();
            timing->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
        } else {
            timing->upload_ms = 0.0;
        }

        return execute_model_and_postprocess(input,
                                             frame->index,
                                             frame->width,
                                             frame->height,
                                             scale,
                                             pad_x,
                                             pad_y,
                                             result,
                                             timing);
    }

    int run_device(const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) override {
        if (!frame || !result || !timing || !cuda_nv12_frame_is_valid(frame)) {
            vcp_inference_set_last_error("invalid CUDA NV12 frame for TorchScript inference");
            return -1;
        }
        if (!use_cuda_) {
            vcp_inference_set_last_error("TorchScript CUDA NV12 frames require --backend-device cuda");
            return -1;
        }

        const float scale = std::min((float)config_.input_width / (float)frame->width,
                                     (float)config_.input_height / (float)frame->height);
        const float resized_w = (float)frame->width * scale;
        const float resized_h = (float)frame->height * scale;
        const float pad_x = ((float)config_.input_width - resized_w) * 0.5f;
        const float pad_y = ((float)config_.input_height - resized_h) * 0.5f;

        const size_t input_bytes = input_tensor_bytes();
        if (ensure_cuda_preprocess_resources(input_bytes) != 0) {
            return -1;
        }

        if (cudaEventRecord(preprocess_start_, preprocess_stream_) != cudaSuccess) {
            vcp_inference_set_last_error("failed to record TorchScript CUDA preprocess start event");
            return -1;
        }
        const cudaError_t preprocess_result = vcp_cuda_preprocess_nv12_to_nchw_fp32(frame->d_y,
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
                                                                                   d_input_,
                                                                                   preprocess_stream_);
        if (preprocess_result != cudaSuccess) {
            vcp_inference_set_last_error(std::string("failed to launch TorchScript CUDA NV12 preprocess kernel: ") +
                                         cudaGetErrorString(preprocess_result));
            return -1;
        }
        if (cudaEventRecord(preprocess_done_, preprocess_stream_) != cudaSuccess ||
            cudaStreamSynchronize(preprocess_stream_) != cudaSuccess) {
            vcp_inference_set_last_error("TorchScript CUDA preprocess synchronization failed");
            return -1;
        }

        timing->upload_ms = 0.0;
        timing->preprocess_ms = elapsed_event_ms(preprocess_start_, preprocess_done_);

        torch::Tensor input = torch::from_blob(d_input_,
                                               {1, 3, config_.input_height, config_.input_width},
                                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        return execute_model_and_postprocess(input,
                                             frame->index,
                                             frame->width,
                                             frame->height,
                                             scale,
                                             pad_x,
                                             pad_y,
                                             result,
                                             timing);
    }

    int get_batch_capability(InferenceBatchCapability *capability) const override {
        if (!capability) {
            vcp_inference_set_last_error("invalid batch capability output");
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
        capability->output_bytes_per_frame = 0;
        std::snprintf(capability->description,
                      sizeof(capability->description),
                      "LibTorch TorchScript single-frame execution on %s",
                      use_cuda_ ? "cuda" : "cpu");
        return 0;
    }

    int set_parallel_contexts(int context_count) override {
        if (context_count != 1) {
            vcp_inference_set_last_error("TorchScript backend currently supports one inference context");
            return -1;
        }
        return 0;
    }

    int run_batch(FrameBatch *batch) override {
        if (!batch || batch->use_cuda_frames || batch->valid_frames < 0 || batch->valid_frames > batch->capacity) {
            vcp_inference_set_last_error("invalid CPU FrameBatch for TorchScript inference");
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
        if (!batch || !batch->use_cuda_frames || batch->valid_frames < 0 || batch->valid_frames > batch->capacity) {
            vcp_inference_set_last_error("invalid CUDA FrameBatch for TorchScript inference");
            return -1;
        }
        for (int i = 0; i < batch->valid_frames; ++i) {
            if (run_device(&batch->cuda_frames[i], &batch->detections[i], &batch->timings[i]) != 0) {
                return -1;
            }
        }
        return 0;
    }

private:
    size_t input_tensor_bytes() const {
        return (size_t)3 * (size_t)config_.input_height * (size_t)config_.input_width * sizeof(float);
    }

    int ensure_cuda_preprocess_resources(size_t input_bytes) {
        if (!preprocess_stream_) {
            if (cudaStreamCreateWithFlags(&preprocess_stream_, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreate(&preprocess_start_) != cudaSuccess ||
                cudaEventCreate(&preprocess_done_) != cudaSuccess) {
                vcp_inference_set_last_error("failed to create TorchScript CUDA preprocess resources");
                release_cuda_preprocess_resources();
                return -1;
            }
        }
        if (d_input_bytes_ < input_bytes) {
            if (d_input_) {
                cudaFree(d_input_);
                d_input_ = nullptr;
                d_input_bytes_ = 0;
            }
            if (cudaMalloc((void **)&d_input_, input_bytes) != cudaSuccess) {
                vcp_inference_set_last_error("failed to allocate TorchScript CUDA input tensor");
                return -1;
            }
            d_input_bytes_ = input_bytes;
        }
        return 0;
    }

    void release_cuda_preprocess_resources() {
        if (d_input_) {
            cudaFree(d_input_);
            d_input_ = nullptr;
            d_input_bytes_ = 0;
        }
        if (preprocess_start_) {
            cudaEventDestroy(preprocess_start_);
            preprocess_start_ = nullptr;
        }
        if (preprocess_done_) {
            cudaEventDestroy(preprocess_done_);
            preprocess_done_ = nullptr;
        }
        if (preprocess_stream_) {
            cudaStreamDestroy(preprocess_stream_);
            preprocess_stream_ = nullptr;
        }
    }

    int execute_model_and_postprocess(const torch::Tensor &input,
                                      int frame_index,
                                      int src_width,
                                      int src_height,
                                      float scale,
                                      float pad_x,
                                      float pad_y,
                                      DetectionResult *result,
                                      FrameTiming *timing) {
        torch::Tensor host_output;
        auto inference_start = std::chrono::high_resolution_clock::now();
        try {
            torch::InferenceMode guard;
            torch::jit::IValue raw = module_.forward({input});
            if (use_cuda_) {
                cudaDeviceSynchronize();
            }
            torch::Tensor output;
            if (!extract_detection_tensor_recursive(raw, 0, &output)) {
                vcp_inference_set_last_error("TorchScript model output does not contain a YOLO detection tensor");
                return -1;
            }
            auto inference_end = std::chrono::high_resolution_clock::now();
            timing->inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

            auto download_start = std::chrono::high_resolution_clock::now();
            host_output = output.to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto download_end = std::chrono::high_resolution_clock::now();
            timing->download_ms = std::chrono::duration<double, std::milli>(download_end - download_start).count();
        } catch (const c10::Error &e) {
            vcp_inference_set_last_error(std::string("TorchScript inference failed: ") + e.what_without_backtrace());
            return -1;
        }

        if (host_output.numel() <= 0) {
            vcp_inference_set_last_error("TorchScript output tensor is empty");
            return -1;
        }

        std::vector<int> dims;
        dims.reserve((size_t)host_output.dim());
        for (int64_t dim : host_output.sizes()) {
            if (dim <= 0 || dim > INT_MAX) {
                vcp_inference_set_last_error("TorchScript output tensor has unsupported dimensions");
                return -1;
            }
            dims.push_back((int)dim);
        }

        auto post_start = std::chrono::high_resolution_clock::now();
        YoloPostprocessConfig post_config{};
        post_config.frame_index = frame_index;
        post_config.src_width = src_width;
        post_config.src_height = src_height;
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
        if (yolo_postprocess(host_output.data_ptr<float>(),
                             (size_t)host_output.numel(),
                             dims.data(),
                             (int)dims.size(),
                             &post_config,
                             result) != 0) {
            vcp_inference_set_last_error("YOLOv5 TorchScript postprocess failed");
            return -1;
        }
        auto post_end = std::chrono::high_resolution_clock::now();
        timing->postprocess_ms = std::chrono::duration<double, std::milli>(post_end - post_start).count();
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

    InferenceConfig config_{};
    torch::jit::script::Module module_;
    torch::Device device_{torch::kCPU};
    bool use_cuda_ = false;
    std::vector<float> input_buffer_;
    cudaStream_t preprocess_stream_ = nullptr;
    cudaEvent_t preprocess_start_ = nullptr;
    cudaEvent_t preprocess_done_ = nullptr;
    float *d_input_ = nullptr;
    size_t d_input_bytes_ = 0;
};

}  // namespace

std::unique_ptr<InferenceEngineImpl> vcp_create_libtorch_yolo_engine(const InferenceConfig &config) {
    std::unique_ptr<TorchScriptYoloEngine> engine(new TorchScriptYoloEngine(config));
    if (engine->init() != 0) {
        return nullptr;
    }
    return engine;
}
