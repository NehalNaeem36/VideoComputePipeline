/*
 * Hardware video writer stub module: preserves the NVENC writer API when
 * hardware video support is disabled. It lets non-hardware builds link cleanly
 * while returning explicit runtime errors for annotated hardware output.
 */
#include "video/video_hw_writer.h"

#include <string.h>

static const char *g_last_error = "NVENC hardware video backend was not built";

int video_hw_writer_open(VideoHWWriter *writer,
                         const char *output_path,
                         const VideoInfo *input_info,
                         const char *encoder_name,
                         int lossless) {
    (void)output_path;
    (void)input_info;
    (void)encoder_name;
    (void)lossless;
    if (writer) {
        memset(writer, 0, sizeof(*writer));
    }
    g_last_error = "NVENC hardware video backend was not built";
    return -1;
}

int video_hw_writer_write_cuda_nv12(VideoHWWriter *writer, const CudaNV12Frame *frame, FrameTiming *timing) {
    (void)writer;
    (void)frame;
    (void)timing;
    g_last_error = "NVENC hardware video backend was not built";
    return -1;
}

int video_hw_writer_flush(VideoHWWriter *writer) {
    (void)writer;
    return 0;
}

void video_hw_writer_close(VideoHWWriter *writer) {
    if (writer) {
        memset(writer, 0, sizeof(*writer));
    }
}

const char *video_hw_writer_last_error(void) {
    return g_last_error;
}
