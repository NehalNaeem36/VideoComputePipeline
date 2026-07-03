#include "benchmark/benchmark.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

static int read_first_line(const char *path, char *buffer, size_t size) {
    FILE *file = fopen(path, "r");
    if (!file) {
        return -1;
    }
    const int ok = fgets(buffer, (int)size, file) != NULL;
    fclose(file);
    return ok ? 0 : -1;
}

int main(void) {
    Benchmark bench;
    benchmark_init /* module: benchmark/benchmark */ (&bench);
    FrameTiming timing = {0};
    timing.frame_index = 1;
    timing.total_ms = 3.5;
    benchmark_set_wall_clock_ms /* module: benchmark/benchmark */ (&bench, 2.0);
    TEST_ASSERT(bench.wall_clock_ms == 2.0);
    TEST_ASSERT(benchmark_add_frame_result /* module: benchmark/benchmark */ (&bench, &timing) == 0);
    TEST_ASSERT(bench.count == 1);
    TEST_ASSERT(benchmark_write_csv /* module: benchmark/benchmark */ (&bench, "benchmarks/benchmark_test.csv") == 0);
    TEST_ASSERT(benchmark_open_csv /* module: benchmark/benchmark */ (&bench, "benchmarks/benchmark_stream_test.csv") == 0);
    TEST_ASSERT(benchmark_add_frame_result /* module: benchmark/benchmark */ (&bench, &timing) == 0);
    TEST_ASSERT(benchmark_close_csv /* module: benchmark/benchmark */ (&bench) == 0);

    char header[1024];
    TEST_ASSERT(read_first_line("benchmarks/benchmark_stream_test.csv", header, sizeof(header)) == 0);
    TEST_ASSERT(strcmp(header, "frame_index,decode_ms,upload_ms,kernel_ms,download_ms,process_ms,encode_ms,total_ms\n") == 0);
    TEST_ASSERT(strstr(header, "runtime_backend") == NULL);

    TEST_ASSERT(benchmark_open_csv_with_schema /* module: benchmark/benchmark */ (&bench,
                                                                                 "benchmarks/benchmark_detection_stream_test.csv",
                                                                                 BENCHMARK_SCHEMA_DETECTION) == 0);
    strcpy(timing.runtime_backend, "tensorrt");
    strcpy(timing.backend_device, "cuda");
    strcpy(timing.precision, "fp16");
    timing.detections_count = 2;
    timing.raw_frame_upload_bytes = 0;
    timing.metadata_download_bytes = 4096;
    TEST_ASSERT(benchmark_add_frame_result /* module: benchmark/benchmark */ (&bench, &timing) == 0);
    TEST_ASSERT(benchmark_close_csv /* module: benchmark/benchmark */ (&bench) == 0);
    TEST_ASSERT(read_first_line("benchmarks/benchmark_detection_stream_test.csv", header, sizeof(header)) == 0);
    TEST_ASSERT(strstr(header, "raw_frame_upload_bytes") != NULL);
    TEST_ASSERT(strstr(header, "detections_count") != NULL);
    TEST_ASSERT(strstr(header, "runtime_backend") != NULL);
    TEST_ASSERT(strstr(header, "kernel_ms") == NULL);

    benchmark_print_summary /* module: benchmark/benchmark */ (&bench);
    benchmark_free /* module: benchmark/benchmark */ (&bench);
    return 0;
}
