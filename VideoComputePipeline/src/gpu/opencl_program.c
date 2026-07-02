/*
 * OpenCL program module: loads kernel source files and builds OpenCL programs
 * for GPU filters. It depends on an OpenCL context and keeps kernel compilation
 * details out of the filter and pipeline modules.
 */
#include "gpu/opencl_program.h"
#include "utils/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_text_file(const char *path, size_t *out_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long length = ftell(file);
    if (length < 0) {
        fclose(file);
        return NULL;
    }
    rewind(file);

    char *buffer = (char *)malloc((size_t)length + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }

    const size_t read_count = fread(buffer, 1, (size_t)length, file);
    fclose(file);
    buffer[read_count] = '\0';
    if (out_size) {
        *out_size = read_count;
    }
    return buffer;
}

int opencl_program_build(OpenCLProgram *program, OpenCLContext *ctx, const char *kernel_path) {
    if (!program || !ctx || !ctx->context || !ctx->device || !kernel_path) {
        return -1;
    }

    memset(program, 0, sizeof(*program));

    size_t source_size = 0;
    char *source = read_text_file(kernel_path, &source_size);
    if (!source) {
        log_error /* module: utils/logger */ ("failed to read OpenCL kernel: %s", kernel_path);
        return -1;
    }

    cl_int err = CL_SUCCESS;
    const char *source_ptr = source;
    program->program = clCreateProgramWithSource(ctx->context, 1, &source_ptr, &source_size, &err);
    free(source);
    if (err != CL_SUCCESS || !program->program) {
        return -1;
    }

    err = clBuildProgram(program->program, 1, &ctx->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log_buffer[8192];
        size_t log_size = 0;
        clGetProgramBuildInfo(program->program,
                              ctx->device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(log_buffer) - 1,
                              log_buffer,
                              &log_size);
        log_buffer[log_size < sizeof(log_buffer) ? log_size : sizeof(log_buffer) - 1] = '\0';
        log_error /* module: utils/logger */ ("OpenCL build failed for %s:\n%s", kernel_path, log_buffer);
        opencl_program_release /* module: gpu/opencl_program */ (program);
        return -1;
    }

    return 0;
}

void opencl_program_release(OpenCLProgram *program) {
    if (!program) {
        return;
    }

    if (program->program) {
        clReleaseProgram(program->program);
    }
    memset(program, 0, sizeof(*program));
}
