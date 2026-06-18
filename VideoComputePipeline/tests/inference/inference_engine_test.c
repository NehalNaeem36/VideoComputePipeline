#include "inference/inference_engine.h"

#include <stdio.h>
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

    TEST_ASSERT(inference_engine_create /* module: inference/inference_engine */ (&engine, &config) != 0);
    TEST_ASSERT(engine == NULL);
    TEST_ASSERT(strstr(inference_engine_last_error /* module: inference/inference_engine */ (), "CUDA/TensorRT") != NULL);
    inference_engine_destroy /* module: inference/inference_engine */ (engine);
    return 0;
}
