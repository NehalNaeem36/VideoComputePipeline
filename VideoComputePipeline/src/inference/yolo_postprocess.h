#ifndef VIDEOCOMPUTEPIPELINE_INFERENCE_YOLO_POSTPROCESS_H
#define VIDEOCOMPUTEPIPELINE_INFERENCE_YOLO_POSTPROCESS_H

#include "config.h"
#include "inference/detection_result.h"

#include <stddef.h>

typedef struct {
    int frame_index;
    int src_width;
    int src_height;
    int input_width;
    int input_height;
    int class_count;
    int class_filter_id_count;
    int class_filter_ids[VCP_MAX_CLASS_FILTERS];
    float confidence_threshold;
    float iou_threshold;
    float scale;
    float pad_x;
    float pad_y;
} YoloPostprocessConfig;

int yolo_postprocess(const float *output,
                     size_t element_count,
                     const int *dims,
                     int nb_dims,
                     const YoloPostprocessConfig *config,
                     DetectionResult *result);

#endif
