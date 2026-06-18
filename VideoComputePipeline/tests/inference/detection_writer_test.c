#include "inference/detection_writer.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    DetectionWriter writer;
    DetectionResult result;
    detection_writer_init /* module: inference/detection_writer */ (&writer);
    detection_result_init /* module: inference/detection_result */ (&result);

    TEST_ASSERT(detection_result_alloc /* module: inference/detection_result */ (&result, 1) == 0);
    Detection detection = {0};
    detection.frame_index = 3;
    detection.timestamp_ms = 120.0;
    detection.class_id = 1;
    detection.confidence = 0.75f;
    detection.x1 = 1.0f;
    detection.y1 = 2.0f;
    detection.x2 = 3.0f;
    detection.y2 = 4.0f;
    TEST_ASSERT(detection_result_add /* module: inference/detection_result */ (&result, &detection) == 0);

    TEST_ASSERT(detection_writer_open /* module: inference/detection_writer */ (&writer, "benchmarks/detection_writer_test.csv", "") == 0);
    TEST_ASSERT(detection_writer_write_frame /* module: inference/detection_writer */ (&writer, &result) == 0);
    TEST_ASSERT(detection_writer_close /* module: inference/detection_writer */ (&writer) == 0);

    detection_result_free /* module: inference/detection_result */ (&result);
    return 0;
}
