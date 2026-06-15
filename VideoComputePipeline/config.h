#ifndef VIDEOCOMPUTEPIPELINE_CONFIG_H
#define VIDEOCOMPUTEPIPELINE_CONFIG_H

#define PROJECT_NAME "VideoComputePipeline"
#define PROJECT_VERSION "0.1.0"

// Default configuration paths
#define DEFAULT_INPUT_FILE "data/input/sample.mp4"
#define DEFAULT_OUTPUT_FILE "data/output/result.mp4"
#define DEFAULT_FILTER_TYPE "grayscale"

// Default settings
#define DEFAULT_USE_GPU 0
#define DEFAULT_NUM_THREADS 4
#define DEFAULT_ENABLE_BENCHMARKS 1

// Buffer sizes
#define FRAME_QUEUE_CAPACITY 10
#define MAX_PATH_LENGTH 512

// Build information
#define BUILD_DATE __DATE__
#define BUILD_TIME __TIME__

#endif // VIDEOCOMPUTEPIPELINE_CONFIG_H
