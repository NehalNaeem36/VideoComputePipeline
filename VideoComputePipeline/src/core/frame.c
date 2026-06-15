#include "core/frame.h"
#include <stdlib.h>
#include <string.h>

Frame* frame_create(uint32_t width, uint32_t height, uint64_t frame_number, double timestamp) {
    // TODO: Implement frame allocation
    return NULL;
}

void frame_destroy(Frame *frame) {
    // TODO: Implement frame deallocation
}

uint8_t* frame_get_data(Frame *frame) {
    // TODO: Implement data accessor
    return NULL;
}

void frame_get_dimensions(Frame *frame, uint32_t *width, uint32_t *height) {
    // TODO: Implement dimension accessor
}
