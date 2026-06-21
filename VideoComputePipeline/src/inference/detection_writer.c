#include "inference/detection_writer.h"
#include "utils/file_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *copy_string(const char *src) {
    if (!src) {
        return NULL;
    }

    const size_t len = strlen(src);
    char *dst = (char *)malloc(len + 1u);
    if (!dst) {
        return NULL;
    }

    memcpy(dst, src, len + 1u);
    return dst;
}

static void trim_line(char *line) {
    if (!line) {
        return;
    }

    size_t len = strlen(line);
    while (len > 0 && (line[len - 1u] == '\n' || line[len - 1u] == '\r' || line[len - 1u] == ' ' || line[len - 1u] == '\t')) {
        line[--len] = '\0';
    }
}

static void free_labels(DetectionWriter *writer) {
    if (!writer || !writer->labels) {
        return;
    }

    for (size_t i = 0; i < writer->label_count; ++i) {
        free(writer->labels[i]);
    }
    free(writer->labels);
    writer->labels = NULL;
    writer->label_count = 0;
}

static int load_labels(DetectionWriter *writer, const char *labels_path) {
    if (!writer || !labels_path || labels_path[0] == '\0') {
        return 0;
    }

    FILE *file = fopen(labels_path, "r");
    if (!file) {
        return 0;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        trim_line /* module: inference/detection_writer */ (line);
        if (line[0] == '\0') {
            continue;
        }

        char **new_labels = (char **)realloc(writer->labels, (writer->label_count + 1u) * sizeof(*new_labels));
        if (!new_labels) {
            fclose(file);
            return -1;
        }
        writer->labels = new_labels;
        writer->labels[writer->label_count] = copy_string /* module: inference/detection_writer */ (line);
        if (!writer->labels[writer->label_count]) {
            fclose(file);
            return -1;
        }
        writer->label_count++;
    }

    fclose(file);
    return 0;
}

void detection_writer_init(DetectionWriter *writer) {
    if (!writer) {
        return;
    }

    writer->file = NULL;
    writer->labels = NULL;
    writer->label_count = 0;
}

int detection_writer_open(DetectionWriter *writer, const char *path, const char *labels_path) {
    if (!writer || !path) {
        return -1;
    }

    detection_writer_close /* module: inference/detection_writer */ (writer);
    if (create_parent_directory_if_missing /* module: utils/file_utils */ (path) != 0) {
        return -1;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return -1;
    }

    writer->file = file;
    if (load_labels /* module: inference/detection_writer */ (writer, labels_path) != 0) {
        detection_writer_close /* module: inference/detection_writer */ (writer);
        return -1;
    }

    if (fprintf(file, "frame_index,timestamp_ms,class_id,class_name,confidence,x1,y1,x2,y2\n") < 0) {
        detection_writer_close /* module: inference/detection_writer */ (writer);
        return -1;
    }

    return 0;
}

int detection_writer_write_frame(DetectionWriter *writer, const DetectionResult *result) {
    if (!writer || !writer->file || !result) {
        return -1;
    }

    FILE *file = (FILE *)writer->file;
    for (size_t i = 0; i < result->count; ++i) {
        const Detection *d = &result->items[i];
        const char *class_name = "";
        if (d->class_id >= 0 && (size_t)d->class_id < writer->label_count && writer->labels[(size_t)d->class_id]) {
            class_name = writer->labels[(size_t)d->class_id];
        }

        if (fprintf(file, "%d,%.6f,%d,%s,%.6f,%.3f,%.3f,%.3f,%.3f\n",
                    d->frame_index,
                    d->timestamp_ms,
                    d->class_id,
                    class_name,
                    d->confidence,
                    d->x1,
                    d->y1,
                    d->x2,
                    d->y2) < 0) {
            return -1;
        }
    }

    return 0;
}

int detection_writer_close(DetectionWriter *writer) {
    if (!writer) {
        return 0;
    }

    int result = 0;
    if (writer->file) {
        FILE *file = (FILE *)writer->file;
        writer->file = NULL;
        result = fclose(file) == 0 ? 0 : -1;
    }
    free_labels /* module: inference/detection_writer */ (writer);
    return result;
}
