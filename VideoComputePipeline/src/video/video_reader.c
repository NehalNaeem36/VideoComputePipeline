#include "video/video_reader.h"
#include <stdlib.h>

// TODO: Include FFmpeg headers when available
// #include <libavformat/avformat.h>
// #include <libavcodec/avcodec.h>

VideoReader* video_reader_open(const char *filename) {
    // TODO: Implement FFmpeg initialization and file opening
    return NULL;
}

Frame* video_reader_read_frame(VideoReader *reader) {
    // TODO: Implement frame decoding
    return NULL;
}

void video_reader_close(VideoReader *reader) {
    // TODO: Implement cleanup and resource deallocation
}

void video_reader_get_properties(VideoReader *reader, uint32_t *width, uint32_t *height, double *fps) {
    // TODO: Implement property accessor
}
