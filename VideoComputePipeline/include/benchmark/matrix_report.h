#ifndef VIDEOCOMPUTEPIPELINE_BENCHMARK_MATRIX_REPORT_H
#define VIDEOCOMPUTEPIPELINE_BENCHMARK_MATRIX_REPORT_H

#include "benchmark/benchmark.h"

/**
 * Generate CSV report with per-frame timing data
 */
int matrix_report_generate_csv(BenchmarkStats *stats, const char *output_file);

/**
 * Generate markdown summary report
 */
int matrix_report_generate_markdown(BenchmarkStats *stats, const char *output_file);

/**
 * Print comparison matrix to stdout
 */
void matrix_report_print_comparison(BenchmarkStats *cpu_stats, BenchmarkStats *gpu_stats);

#endif // VIDEOCOMPUTEPIPELINE_BENCHMARK_MATRIX_REPORT_H
