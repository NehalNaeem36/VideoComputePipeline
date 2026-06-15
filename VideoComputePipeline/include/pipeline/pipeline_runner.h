#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_PIPELINE_RUNNER_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_PIPELINE_RUNNER_H

#include "pipeline/pipeline_config.h"

/**
 * Pipeline runner context
 */
typedef struct {
    PipelineConfig *config;
    void *video_reader;
    void *video_writer;
    void *opencl_context;
    void *opencl_program;
    int is_running;
} PipelineRunner;

/**
 * Create and initialize pipeline runner
 */
PipelineRunner* pipeline_runner_create(PipelineConfig *config);

/**
 * Run the entire pipeline
 */
int pipeline_runner_run(PipelineRunner *runner);

/**
 * Stop the pipeline
 */
void pipeline_runner_stop(PipelineRunner *runner);

/**
 * Free pipeline runner resources
 */
void pipeline_runner_destroy(PipelineRunner *runner);

#endif // VIDEOCOMPUTEPIPELINE_PIPELINE_PIPELINE_RUNNER_H
