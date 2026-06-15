#include "pipeline/pipeline_runner.h"
#include <stdlib.h>

PipelineRunner* pipeline_runner_create(PipelineConfig *config) {
    // TODO: Implement runner initialization with video_reader, video_writer, OpenCL
    return NULL;
}

int pipeline_runner_run(PipelineRunner *runner) {
    // TODO: Implement main pipeline loop:
    // 1. Read frame from video_reader
    // 2. Apply filter (CPU or GPU based on config)
    // 3. Write frame to video_writer
    // 4. Record timing if benchmarks enabled
    return -1;
}

void pipeline_runner_stop(PipelineRunner *runner) {
    // TODO: Implement stop logic
}

void pipeline_runner_destroy(PipelineRunner *runner) {
    // TODO: Implement cleanup and resource deallocation
}
