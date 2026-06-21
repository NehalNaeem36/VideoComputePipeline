#include "inference/detection_result.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    DetectionResult result;
    detection_result_init /* module: inference/detection_result */ (&result);

    TEST_ASSERT(detection_result_alloc /* module: inference/detection_result */ (&result, 1) == 0);
    TEST_ASSERT(result.capacity == 1);
    TEST_ASSERT(result.count == 0);

    Detection detection = {0};
    detection.frame_index = 7;
    detection.class_id = 2;
    detection.confidence = 0.9f;
    TEST_ASSERT(detection_result_add /* module: inference/detection_result */ (&result, &detection) == 0);
    TEST_ASSERT(result.count == 1);
    TEST_ASSERT(result.items[0].frame_index == 7);
    TEST_ASSERT(detection_result_add /* module: inference/detection_result */ (&result, &detection) != 0);

    detection_result_clear /* module: inference/detection_result */ (&result);
    TEST_ASSERT(result.count == 0);
    detection_result_free /* module: inference/detection_result */ (&result);
    TEST_ASSERT(result.items == NULL);
    return 0;
}
