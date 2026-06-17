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
    bench->total_ms = 0.0;
    bench->process_ms = 0.0;
    bench->csv_file = NULL;
}

static int write_csv_header(FILE *file) {
    return fprintf(file, "frame_index,decode_ms,process_ms,upload_ms,kernel_ms,download_ms,encode_ms,total_ms\n") < 0 ? -1 : 0;
}

static int write_csv_row(FILE *file, const FrameTiming *t) {
    return fprintf(file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                   t->frame_index,
                   t->decode_ms,
                   t->process_ms,
                   t->upload_ms,
                   t->kernel_ms,
                   t->download_ms,
                   t->encode_ms,
                   t->total_ms) < 0 ? -1 : 0;
}

int benchmark_open_csv(Benchmark *bench, const char *path) {
    if (!bench || !path || bench->csv_file) {
        return -1;
    }

    if (create_parent_directory_if_missing /* module: utils/file_utils */ (path) != 0) {
        return -1;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return -1;
    }

    if (write_csv_header /* module: benchmark/benchmark */ (file) != 0) {
        fclose(file);
        return -1;
    }

    bench->csv_file = file;
    return 0;
}

static int append_in_memory(Benchmark *bench, const FrameTiming *timing) {
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

int benchmark_add_frame_result(Benchmark *bench, const FrameTiming *timing) {
    if (!bench || !timing) {
        return -1;
    }

    if (bench->csv_file) {
        if (write_csv_row /* module: benchmark/benchmark */ ((FILE *)bench->csv_file, timing) != 0) {
            return -1;
        }
        bench->count++;
        if (bench->count % 300u == 0u) {
            fflush((FILE *)bench->csv_file);
        }
    } else if (append_in_memory /* module: benchmark/benchmark */ (bench, timing) != 0) {
        return -1;
    }

    bench->total_ms += timing->total_ms;
    bench->process_ms += timing->process_ms;
    return 0;
}

int benchmark_write_csv(const Benchmark *bench, const char *path) {
    if (!bench || !path) {
        return -1;
    }

    if (create_parent_directory_if_missing /* module: utils/file_utils */ (path) != 0) {
        return -1;
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        return -1;
    }

    if (write_csv_header /* module: benchmark/benchmark */ (file) != 0) {
        fclose(file);
        return -1;
    }
    for (size_t i = 0; i < bench->count; ++i) {
        if (write_csv_row /* module: benchmark/benchmark */ (file, &bench->items[i]) != 0) {
            fclose(file);
            return -1;
        }
    }

    fclose(file);
    return 0;
}

int benchmark_close_csv(Benchmark *bench) {
    if (!bench || !bench->csv_file) {
        return 0;
    }

    FILE *file = (FILE *)bench->csv_file;
    bench->csv_file = NULL;
    return fclose(file) == 0 ? 0 : -1;
}

void benchmark_print_summary(const Benchmark *bench) {
    if (!bench || bench->count == 0) {
        printf("Benchmark: no frames recorded\n");
        return;
    }

    printf("Benchmark summary:\n");
    printf("  frames: %zu\n", bench->count);
    printf("  total_ms: %.3f\n", bench->total_ms);
    printf("  avg_total_ms: %.3f\n", bench->total_ms / (double)bench->count);
    printf("  avg_process_ms: %.3f\n", bench->process_ms / (double)bench->count);
    if (bench->total_ms > 0.0) {
        printf("  processed_fps: %.3f\n", (double)bench->count * 1000.0 / bench->total_ms);
    }
}

void benchmark_free(Benchmark *bench) {
    if (!bench) {
        return;
    }

    benchmark_close_csv /* module: benchmark/benchmark */ (bench);
    free(bench->items);
    benchmark_init /* module: benchmark/benchmark */ (bench);
}
