#ifndef VIDEOCOMPUTEPIPELINE_INFERENCE_DETECTION_RESULT_H
#define VIDEOCOMPUTEPIPELINE_INFERENCE_DETECTION_RESULT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int frame_index;
    double timestamp_ms;
    int class_id;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
} Detection;

typedef struct {
    Detection *items;
    size_t count;
    size_t capacity;
} DetectionResult;

void detection_result_init(DetectionResult *result);
int detection_result_alloc(DetectionResult *result, size_t capacity);
void detection_result_clear(DetectionResult *result);
int detection_result_add(DetectionResult *result, const Detection *detection);
void detection_result_free(DetectionResult *result);

#ifdef __cplusplus
}
#endif

#endif
