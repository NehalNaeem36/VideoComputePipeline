#ifndef VIDEOCOMPUTEPIPELINE_MATRIX_REPORT_H
#define VIDEOCOMPUTEPIPELINE_MATRIX_REPORT_H

typedef struct {
    int total_frames;
    double total_time_ms;
    double average_ms_per_frame;
    double processed_fps;
} MatrixReportStats;

int matrix_report_read_csv_summary(const char *path, MatrixReportStats *stats);
void matrix_report_print_comparison(const MatrixReportStats *cpu, const MatrixReportStats *gpu);

#endif
