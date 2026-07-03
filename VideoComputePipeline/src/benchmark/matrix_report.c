/*
 * Matrix report module: reads benchmark CSV files and summarizes CPU/GPU
 * performance comparisons. It stays downstream of pipeline execution and does
 * not participate in frame processing.
 */
#include "benchmark/matrix_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int find_csv_column_index(char *header, const char *name) {
    int index = 0;
    char *token = strtok(header, ",\r\n");
    while (token) {
        if (strcmp(token, name) == 0) {
            return index;
        }
        ++index;
        token = strtok(NULL, ",\r\n");
    }
    return -1;
}

static int parse_csv_double_column(char *line, int column_index, double *out_value) {
    int index = 0;
    char *token = strtok(line, ",\r\n");
    while (token) {
        if (index == column_index) {
            char *end = NULL;
            const double value = strtod(token, &end);
            if (end == token) {
                return -1;
            }
            *out_value = value;
            return 0;
        }
        ++index;
        token = strtok(NULL, ",\r\n");
    }
    return -1;
}

int matrix_report_read_csv_summary(const char *path, MatrixReportStats *stats) {
    if (!path || !stats) {
        return -1;
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        return -1;
    }

    memset(stats, 0, sizeof(*stats));

    char line[2048];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return -1;
    }

    char header[2048];
    strncpy(header, line, sizeof(header) - 1u);
    header[sizeof(header) - 1u] = '\0';
    const int total_ms_column = find_csv_column_index /* module: benchmark/matrix_report */ (header, "total_ms");
    if (total_ms_column < 0) {
        fclose(file);
        return -1;
    }

    while (fgets(line, sizeof(line), file)) {
        double total_ms = 0.0;
        char row[2048];
        strncpy(row, line, sizeof(row) - 1u);
        row[sizeof(row) - 1u] = '\0';

        if (parse_csv_double_column /* module: benchmark/matrix_report */ (row, total_ms_column, &total_ms) == 0) {
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
