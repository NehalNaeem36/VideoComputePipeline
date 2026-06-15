#ifndef VIDEOCOMPUTEPIPELINE_CORE_FRAME_H
#define VIDEOCOMPUTEPIPELINE_CORE_FRAME_H

#include <stdint.h>
#include <stddef.h>

/**
 * Frame structure for internal RGB24 representation
 */
typedef struct {
    uint8_t *data;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint64_t frame_number;
    double timestamp;
} Frame;

/**
 * Allocate and initialize a new Frame
 */
Frame* frame_create(uint32_t width, uint32_t height, uint64_t frame_number, double timestamp);

/**
 * Free Frame resources
 */
void frame_destroy(Frame *frame);

/**
 * Get pixel data pointer for frame
 */
uint8_t* frame_get_data(Frame *frame);

/**
 * Get frame dimensions
 */
void frame_get_dimensions(Frame *frame, uint32_t *width, uint32_t *height);

#endif // VIDEOCOMPUTEPIPELINE_CORE_FRAME_H
