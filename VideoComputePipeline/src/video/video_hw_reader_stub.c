#include "video/video_hw_reader.h"

#include <string.h>

static const char *g_last_error = "NVDEC hardware video backend was not built";

int video_hw_reader_open(VideoHWReader *reader, const char *path, int decoder_threads) {
    (void)path;
    (void)decoder_threads;
    if (reader) {
        memset(reader, 0, sizeof(*reader));
    }
    g_last_error = "NVDEC hardware video backend was not built";
    return -1;
}

int video_hw_reader_read_cuda_nv12(VideoHWReader *reader, CudaNV12Frame *out) {
    (void)reader;
    if (out) {
        cuda_nv12_frame_clear /* module: gpu/cuda_frame */ (out);
    }
    g_last_error = "NVDEC hardware video backend was not built";
    return -1;
}

const VideoInfo *video_hw_reader_get_info(const VideoHWReader *reader) {
    return reader ? &reader->info : NULL;
}

void video_hw_reader_release_frame(VideoHWReader *reader, CudaNV12Frame *frame) {
    (void)reader;
    cuda_nv12_frame_clear /* module: gpu/cuda_frame */ (frame);
}

void video_hw_reader_close(VideoHWReader *reader) {
    if (reader) {
        memset(reader, 0, sizeof(*reader));
    }
}

const char *video_hw_reader_last_error(void) {
    return g_last_error;
}
