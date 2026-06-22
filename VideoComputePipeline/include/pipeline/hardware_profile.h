#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_HARDWARE_PROFILE_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_HARDWARE_PROFILE_H

#include <stddef.h>

typedef struct {
    char gpu_name[128];
    int cuda_device_id;
    int compute_major;
    int compute_minor;
    size_t total_vram_bytes;
    size_t free_vram_before_engine_bytes;
    size_t free_vram_after_engine_bytes;
    size_t estimated_engine_bytes;
    int async_engine_count;
    int concurrent_kernels;
    int can_map_host_memory;
    int unified_addressing;
    double h2d_pinned_gbps;
    double d2h_pinned_gbps;
    int cuda_available;
} HardwareProfile;

void hardware_profile_init(HardwareProfile *profile);
int hardware_profile_query_before_engine(HardwareProfile *profile);
int hardware_profile_query_after_engine(HardwareProfile *profile);
int hardware_profile_measure_bandwidth(HardwareProfile *profile);
void hardware_profile_print(const HardwareProfile *profile);

#endif
