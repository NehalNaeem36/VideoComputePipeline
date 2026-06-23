/*
 * Matrix report module: reads benchmark CSV files and summarizes CPU/GPU
 * performance comparisons. It stays downstream of pipeline execution and does
 * not participate in frame processing.
 */
#include "benchmark/matrix_report.h"

#include <stdio.h>
#include <string.h>

int matrix_report_read_csv_summary(const char *path, MatrixReportStats *stats) {
    if (!path || !stats) {
        return -1;
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        return -1;
    }

    memset(stats, 0, sizeof(*stats));

    char line[512];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return -1;
    }

    while (fgets(line, sizeof(line), file)) {
        int frame_index = 0;
        double decode_ms = 0.0;
        double process_ms = 0.0;
        double upload_ms = 0.0;
        double kernel_ms = 0.0;
        double download_ms = 0.0;
        double encode_ms = 0.0;
        double total_ms = 0.0;

        if (sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                   &frame_index,
                   &decode_ms,
                   &process_ms,
                   &upload_ms,
                   &kernel_ms,
                   &download_ms,
                   &encode_ms,
                   &total_ms) == 8) {
            (void)frame_index;
            stats->total_frames++;
            stats->total_time_ms += total_ms;
        }
    }

    fclose(file);

    if (stats->total_frames > 0) {
        stats->average_ms_per_frame = stats->total_time_ms / (double)stats->total_frames;
        if (stats->total_time_ms > 0.0) {
            stats->processed_fps = (double)stats->total_frames * 1000.0 / stats->total_time_ms;
        }
    }

    return 0;
}

void matrix_report_print_comparison(const MatrixReportStats *cpu, const MatrixReportStats *gpu) {
    printf("CPU vs GPU summary:\n");
    printf("Mode,Frames,Total ms,Avg ms/frame,FPS\n");
    if (cpu) {
        printf("CPU,%d,%.3f,%.3f,%.3f\n",
               cpu->total_frames,
               cpu->total_time_ms,
               cpu->average_ms_per_frame,
               cpu->processed_fps);
    }
    if (gpu) {
        printf("GPU,%d,%.3f,%.3f,%.3f\n",
               gpu->total_frames,
               gpu->total_time_ms,
               gpu->average_ms_per_frame,
               gpu->processed_fps);
    }
    if (cpu && gpu && gpu->total_time_ms > 0.0) {
        printf("Speedup: %.3fx\n", cpu->total_time_ms / gpu->total_time_ms);
    }
}
