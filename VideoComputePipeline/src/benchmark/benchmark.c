#include "benchmark/benchmark.h"
#include "utils/file_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void benchmark_init(Benchmark *bench) {
    if (!bench) {
        return;
    }

    bench->items = NULL;
    bench->count = 0;
    bench->capacity = 0;
}

int benchmark_add_frame_result(Benchmark *bench, const FrameTiming *timing) {
    if (!bench || !timing) {
        return -1;
    }

    if (bench->count == bench->capacity) {
        const size_t new_capacity = bench->capacity == 0 ? 256 : bench->capacity * 2;
        FrameTiming *new_items = (FrameTiming *)realloc(bench->items, new_capacity * sizeof(*new_items));
        if (!new_items) {
            return -1;
        }
        bench->items = new_items;
        bench->capacity = new_capacity;
    }

    bench->items[bench->count++] = *timing;
    return 0;
}

int benchmark_write_csv(const Benchmark *bench, const char *path) {
    if (!bench || !path) {
        return -1;
    }

    if (create_parent_directory_if_missing(path) != 0) {
        return -1;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return -1;
    }

    fprintf(file, "frame_index,decode_ms,process_ms,upload_ms,kernel_ms,download_ms,encode_ms,total_ms\n");
    for (size_t i = 0; i < bench->count; ++i) {
        const FrameTiming *t = &bench->items[i];
        fprintf(file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                t->frame_index,
                t->decode_ms,
                t->process_ms,
                t->upload_ms,
                t->kernel_ms,
                t->download_ms,
                t->encode_ms,
                t->total_ms);
    }

    fclose(file);
    return 0;
}

void benchmark_print_summary(const Benchmark *bench) {
    if (!bench || bench->count == 0) {
        printf("Benchmark: no frames recorded\n");
        return;
    }

    double total_ms = 0.0;
    double process_ms = 0.0;
    for (size_t i = 0; i < bench->count; ++i) {
        total_ms += bench->items[i].total_ms;
        process_ms += bench->items[i].process_ms;
    }

    printf("Benchmark summary:\n");
    printf("  frames: %zu\n", bench->count);
    printf("  total_ms: %.3f\n", total_ms);
    printf("  avg_total_ms: %.3f\n", total_ms / (double)bench->count);
    printf("  avg_process_ms: %.3f\n", process_ms / (double)bench->count);
    if (total_ms > 0.0) {
        printf("  processed_fps: %.3f\n", (double)bench->count * 1000.0 / total_ms);
    }
}

void benchmark_free(Benchmark *bench) {
    if (!bench) {
        return;
    }

    free(bench->items);
    benchmark_init(bench);
}
