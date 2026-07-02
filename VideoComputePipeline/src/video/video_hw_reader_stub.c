/*
 * Hardware video reader stub module: preserves the NVDEC reader API when
 * hardware video support is disabled. Detection runners can report a clear
 * backend-not-built error while CPU decode paths remain available.
 */
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

int video_hw_reader_transfer_to_cpu_nv12(VideoHWReader *reader, const CudaNV12Frame *src, Frame *out) {
    (void)reader;
    (void)src;
    (void)out;
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
