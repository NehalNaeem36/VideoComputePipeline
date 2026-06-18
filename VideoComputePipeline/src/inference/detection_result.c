#include "inference/detection_result.h"

#include <stdlib.h>
#include <string.h>

void detection_result_init(DetectionResult *result) {
    if (!result) {
        return;
    }

    result->items = NULL;
    result->count = 0;
    result->capacity = 0;
}

int detection_result_alloc(DetectionResult *result, size_t capacity) {
    if (!result || capacity == 0) {
        return -1;
    }

    Detection *items = (Detection *)calloc(capacity, sizeof(*items));
    if (!items) {
        return -1;
    }

    detection_result_free /* module: inference/detection_result */ (result);
    result->items = items;
    result->count = 0;
    result->capacity = capacity;
    return 0;
}

void detection_result_clear(DetectionResult *result) {
    if (!result) {
        return;
    }

    result->count = 0;
}

int detection_result_add(DetectionResult *result, const Detection *detection) {
    if (!result || !detection || !result->items || result->count >= result->capacity) {
        return -1;
    }

    result->items[result->count++] = *detection;
    return 0;
}

void detection_result_free(DetectionResult *result) {
    if (!result) {
        return;
    }

    free(result->items);
    detection_result_init /* module: inference/detection_result */ (result);
}
