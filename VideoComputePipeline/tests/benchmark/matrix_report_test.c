#include "benchmark/benchmark.h"
#include "benchmark/matrix_report.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    Benchmark bench;
    benchmark_init /* module: benchmark/benchmark */ (&bench);
    FrameTiming timing = {0};
    timing.frame_index = 0;
    timing.total_ms = 10.0;
    TEST_ASSERT(benchmark_add_frame_result /* module: benchmark/benchmark */ (&bench, &timing) == 0);
    TEST_ASSERT(benchmark_write_csv /* module: benchmark/benchmark */ (&bench, "benchmarks/matrix_report_test.csv") == 0);

    MatrixReportStats stats;
    TEST_ASSERT(matrix_report_read_csv_summary /* module: benchmark/matrix_report */ ("benchmarks/matrix_report_test.csv", &stats) == 0);
    TEST_ASSERT(stats.total_frames == 1);
    matrix_report_print_comparison /* module: benchmark/matrix_report */ (&stats, &stats);
    benchmark_free /* module: benchmark/benchmark */ (&bench);
    return 0;
}
