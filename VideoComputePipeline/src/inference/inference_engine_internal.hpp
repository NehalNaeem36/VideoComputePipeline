#ifndef VIDEOCOMPUTEPIPELINE_INFERENCE_ENGINE_INTERNAL_HPP
#define VIDEOCOMPUTEPIPELINE_INFERENCE_ENGINE_INTERNAL_HPP

#include "inference/inference_engine.h"

#include <memory>
#include <string>

class InferenceEngineImpl {
public:
    virtual ~InferenceEngineImpl() = default;
    virtual int run(const Frame *frame, DetectionResult *result, FrameTiming *timing) = 0;
    virtual int run_device(const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) = 0;
    virtual int get_batch_capability(InferenceBatchCapability *capability) const = 0;
    virtual int set_parallel_contexts(int context_count) = 0;
    virtual int get_async_lane_count() const { return 0; }
    virtual int submit_device_async(int lane, const CudaNV12Frame *frame, FrameTiming *timing) {
        (void)lane;
        (void)frame;
        (void)timing;
        return -1;
    }
    virtual int is_lane_ready(int lane, int *ready) {
        (void)lane;
        if (ready) {
            *ready = 0;
        }
        return -1;
    }
    virtual int finish_lane(int lane, DetectionResult *result, FrameTiming *timing) {
        (void)lane;
        (void)result;
        (void)timing;
        return -1;
    }
    virtual int run_batch(FrameBatch *batch) = 0;
    virtual int run_device_batch(FrameBatch *batch) = 0;
};

void vcp_inference_set_last_error(const std::string &message);
void vcp_inference_set_last_error(const char *message);

#ifdef VCP_ENABLE_LIBTORCH
std::unique_ptr<InferenceEngineImpl> vcp_create_libtorch_yolo_engine(const InferenceConfig &config);
#endif

#endif
