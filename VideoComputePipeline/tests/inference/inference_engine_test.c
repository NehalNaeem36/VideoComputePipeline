#include "inference/inference_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    InferenceEngine *engine = NULL;
    InferenceConfig config;
    memset(&config, 0, sizeof(config));
    config.input_width = 640;
    config.input_height = 640;
    config.class_count = 80;
    config.confidence_threshold = 0.25f;
    config.iou_threshold = 0.45f;
    config.use_fp16 = 1;

#ifdef VCP_ENABLE_CUDA_INFERENCE
    strcpy(config.model_path, "models/does_not_exist.engine");
    TEST_ASSERT(inference_engine_create /* module: inference/inference_engine */ (&engine, &config) != 0);
    TEST_ASSERT(engine == NULL);

#ifndef VCP_TEST_TENSORRT_ENGINE
#define VCP_TEST_TENSORRT_ENGINE ""
#endif
    if (strlen(VCP_TEST_TENSORRT_ENGINE) > 0) {
        strcpy(config.model_path, VCP_TEST_TENSORRT_ENGINE);
        if (inference_engine_create /* module: inference/inference_engine */ (&engine, &config) == 0) {
            Frame frame;
            DetectionResult result;
            FrameTiming timing;
            frame_init /* module: core/frame */ (&frame);
            detection_result_init /* module: inference/detection_result */ (&result);
            memset(&timing, 0, sizeof(timing));
            TEST_ASSERT(frame_alloc /* module: core/frame */ (&frame, 640, 480, FRAME_FORMAT_NV12) == 0);
            TEST_ASSERT(detection_result_alloc /* module: inference/detection_result */ (&result, 300) == 0);
            TEST_ASSERT(inference_engine_run_nv12 /* module: inference/inference_engine */ (engine, &frame, &result, &timing) == 0);
            TEST_ASSERT(result.count <= result.capacity);
            detection_result_free /* module: inference/detection_result */ (&result);
            frame_free /* module: core/frame */ (&frame);
            inference_engine_destroy /* module: inference/inference_engine */ (engine);
        }
    }
#else
    TEST_ASSERT(inference_engine_create /* module: inference/inference_engine */ (&engine, &config) != 0);
    TEST_ASSERT(engine == NULL);
    TEST_ASSERT(strstr(inference_engine_last_error /* module: inference/inference_engine */ (), "CUDA/TensorRT") != NULL);
    inference_engine_destroy /* module: inference/inference_engine */ (engine);
#endif
    return 0;
}
