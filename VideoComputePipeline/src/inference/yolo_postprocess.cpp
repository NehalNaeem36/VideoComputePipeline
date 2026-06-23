/*
 * YOLO postprocess module: decodes raw YOLOv5 TensorRT output, applies
 * confidence filtering and NMS, and maps boxes back to original frame
 * coordinates. The TensorRT engine calls it after output download.
 */
#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <vector>

static float clamp_value(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

static float box_iou(const Detection &a, const Detection &b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float intersection = w * h;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float denom = area_a + area_b - intersection;
    return denom > 0.0f ? intersection / denom : 0.0f;
}

static bool class_filter_contains(const YoloPostprocessConfig *config, int class_id) {
    if (!config || config->class_filter_id_count <= 0) {
        return true;
    }
    for (int i = 0; i < config->class_filter_id_count; ++i) {
        if (config->class_filter_ids[i] == class_id) {
            return true;
        }
    }
    return false;
}

static int parse_output_shape(const int *dims,
                              int nb_dims,
                              int attributes,
                              size_t element_count,
                              int *prediction_count,
                              int *transposed) {
    if (!dims || nb_dims <= 0 || !prediction_count || !transposed || attributes <= 0 || element_count == 0) {
        return -1;
    }

    if (dims[nb_dims - 1] == attributes) {
        *prediction_count = (int)(element_count / (size_t)attributes);
        *transposed = 0;
        return *prediction_count > 0 ? 0 : -1;
    }

    if (nb_dims >= 2 && dims[nb_dims - 2] == attributes) {
        *prediction_count = dims[nb_dims - 1];
        *transposed = 1;
        return *prediction_count > 0 ? 0 : -1;
    }

    return -1;
}

static float read_prediction_value(const float *output, int prediction, int attr, int prediction_count, int attributes, int transposed) {
    if (transposed) {
        return output[attr * prediction_count + prediction];
    }
    return output[prediction * attributes + attr];
}

int yolo_postprocess(const float *output,
                     size_t element_count,
                     const int *dims,
                     int nb_dims,
                     const YoloPostprocessConfig *config,
                     DetectionResult *result) {
    if (!output || !dims || !config || !result || config->class_count <= 0 || config->scale <= 0.0f) {
        return -1;
    }

    detection_result_clear(result);
    const int attributes = 5 + config->class_count;
    int prediction_count = 0;
    int transposed = 0;
    if (parse_output_shape(dims, nb_dims, attributes, element_count, &prediction_count, &transposed) != 0) {
        return -1;
    }

    std::vector<Detection> candidates;
    candidates.reserve((size_t)prediction_count / 8u);
    for (int i = 0; i < prediction_count; ++i) {
        const float objectness = read_prediction_value(output, i, 4, prediction_count, attributes, transposed);
        if (objectness <= 0.0f) {
            continue;
        }

        int best_class = -1;
        float best_score = 0.0f;
        if (config->class_filter_id_count > 0) {
            for (int f = 0; f < config->class_filter_id_count; ++f) {
                const int c = config->class_filter_ids[f];
                if (c < 0 || c >= config->class_count) {
                    continue;
                }
                const float score = read_prediction_value(output, i, 5 + c, prediction_count, attributes, transposed);
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }
        } else {
            for (int c = 0; c < config->class_count; ++c) {
                const float score = read_prediction_value(output, i, 5 + c, prediction_count, attributes, transposed);
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }
        }

        const float confidence = objectness * best_score;
        if (best_class < 0 || !class_filter_contains(config, best_class) || confidence < config->confidence_threshold) {
            continue;
        }

        const float cx = read_prediction_value(output, i, 0, prediction_count, attributes, transposed);
        const float cy = read_prediction_value(output, i, 1, prediction_count, attributes, transposed);
        const float w = read_prediction_value(output, i, 2, prediction_count, attributes, transposed);
        const float h = read_prediction_value(output, i, 3, prediction_count, attributes, transposed);

        Detection detection = {};
        detection.frame_index = config->frame_index;
        detection.class_id = best_class;
        detection.confidence = confidence;
        detection.x1 = ((cx - w * 0.5f) - config->pad_x) / config->scale;
        detection.y1 = ((cy - h * 0.5f) - config->pad_y) / config->scale;
        detection.x2 = ((cx + w * 0.5f) - config->pad_x) / config->scale;
        detection.y2 = ((cy + h * 0.5f) - config->pad_y) / config->scale;
        detection.x1 = clamp_value(detection.x1, 0.0f, (float)config->src_width);
        detection.y1 = clamp_value(detection.y1, 0.0f, (float)config->src_height);
        detection.x2 = clamp_value(detection.x2, 0.0f, (float)config->src_width);
        detection.y2 = clamp_value(detection.y2, 0.0f, (float)config->src_height);
        if (detection.x2 > detection.x1 && detection.y2 > detection.y1) {
            candidates.push_back(detection);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Detection &a, const Detection &b) {
        return a.confidence > b.confidence;
    });

    std::vector<Detection> kept;
    kept.reserve(std::min(candidates.size(), result->capacity));
    for (const Detection &candidate : candidates) {
        bool suppressed = false;
        for (const Detection &accepted : kept) {
            if (candidate.class_id == accepted.class_id && box_iou(candidate, accepted) > config->iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (kept.size() >= result->capacity) {
                break;
            }
        }
    }

    for (const Detection &detection : kept) {
        if (detection_result_add(result, &detection) != 0) {
            return -1;
        }
    }
    return 0;
}
