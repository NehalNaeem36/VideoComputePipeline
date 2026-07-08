#ifndef VIDEOCOMPUTEPIPELINE_INFERENCE_DETECTION_WRITER_H
#define VIDEOCOMPUTEPIPELINE_INFERENCE_DETECTION_WRITER_H

#include "inference/detection_result.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *file;
    char *buffer;
    char **labels;
    size_t label_count;
} DetectionWriter;

void detection_writer_init(DetectionWriter *writer);
int detection_writer_open(DetectionWriter *writer, const char *path, const char *labels_path);
int detection_writer_write_frame(DetectionWriter *writer, const DetectionResult *result);
int detection_writer_close(DetectionWriter *writer);

#ifdef __cplusplus
}
#endif

#endif
