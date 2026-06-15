#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_PIPELINE_CONFIG_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_PIPELINE_CONFIG_H

/**
 * Pipeline configuration
 */
typedef struct {
    const char *input_file;
    const char *output_file;
    const char *filter_type;  // "grayscale", "blur3x3", "blur5x5", "blur9x9"
    int use_gpu;
    int num_threads;
    int enable_benchmarks;
} PipelineConfig;

/**
 * Create pipeline config with defaults
 */
PipelineConfig* pipeline_config_create(void);

/**
 * Load default configuration
 */
void pipeline_config_load_defaults(PipelineConfig *cfg);

/**
 * Parse command-line arguments
 */
int pipeline_config_parse_args(PipelineConfig *cfg, int argc, char **argv);

/**
 * Validate configuration
 */
int pipeline_config_validate(PipelineConfig *cfg);

/**
 * Print configuration
 */
void pipeline_config_print(PipelineConfig *cfg);

/**
 * Free configuration
 */
void pipeline_config_destroy(PipelineConfig *cfg);

#endif // VIDEOCOMPUTEPIPELINE_PIPELINE_PIPELINE_CONFIG_H
