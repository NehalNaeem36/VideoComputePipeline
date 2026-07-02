/*
 * Hardware profile module: queries CUDA device capabilities, VRAM availability,
 * and transfer bandwidth for execution planning. Detection planners use this
 * information to choose batch, in-flight, and inference-context settings.
 */
#include "pipeline/hardware_profile.h"

#include <stdio.h>
#include <string.h>

#ifdef VCP_ENABLE_CUDA_INFERENCE
#include <cuda_runtime_api.h>
#endif

void hardware_profile_init(HardwareProfile *profile) {
    if (!profile) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    profile->cuda_device_id = 0;
    strcpy(profile->gpu_name, "unavailable");
}

#ifdef VCP_ENABLE_CUDA_INFERENCE
static int query_cuda_common(HardwareProfile *profile, size_t *free_bytes, size_t *total_bytes) {
    struct cudaDeviceProp props;
    int device = 0;
    if (!profile || cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&props, device) != cudaSuccess ||
        cudaMemGetInfo(free_bytes, total_bytes) != cudaSuccess) {
        return -1;
    }

    profile->cuda_available = 1;
    profile->cuda_device_id = device;
    snprintf(profile->gpu_name, sizeof(profile->gpu_name), "%s", props.name);
    profile->compute_major = props.major;
    profile->compute_minor = props.minor;
    profile->async_engine_count = props.asyncEngineCount;
    profile->concurrent_kernels = props.concurrentKernels;
    profile->can_map_host_memory = props.canMapHostMemory;
    profile->unified_addressing = props.unifiedAddressing;
    profile->total_vram_bytes = *total_bytes;
    return 0;
}
#endif

int hardware_profile_query_before_engine(HardwareProfile *profile) {
#ifdef VCP_ENABLE_CUDA_INFERENCE
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (query_cuda_common(profile, &free_bytes, &total_bytes) != 0) {
        return -1;
    }
    profile->free_vram_before_engine_bytes = free_bytes;
    return 0;
#else
    (void)profile;
    return -1;
#endif
}

int hardware_profile_query_after_engine(HardwareProfile *profile) {
#ifdef VCP_ENABLE_CUDA_INFERENCE
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (query_cuda_common(profile, &free_bytes, &total_bytes) != 0) {
        return -1;
    }
    profile->free_vram_after_engine_bytes = free_bytes;
    if (profile->free_vram_before_engine_bytes > free_bytes) {
        profile->estimated_engine_bytes = profile->free_vram_before_engine_bytes - free_bytes;
    }
    return 0;
#else
    (void)profile;
    return -1;
#endif
}

int hardware_profile_measure_bandwidth(HardwareProfile *profile) {
#ifdef VCP_ENABLE_CUDA_INFERENCE
    static const size_t sizes[] = {16u * 1024u * 1024u, 64u * 1024u * 1024u, 128u * 1024u * 1024u};
    double h2d_sum = 0.0;
    double d2h_sum = 0.0;
    int measurements = 0;

    if (!profile || !profile->cuda_available) {
        return -1;
    }

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        const size_t bytes = sizes[i];
        void *host = NULL;
        void *device = NULL;
        cudaEvent_t start = NULL;
        cudaEvent_t end = NULL;
        cudaStream_t stream = NULL;
        float h2d_ms = 0.0f;
        float d2h_ms = 0.0f;

        if (cudaHostAlloc(&host, bytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaMalloc(&device, bytes) != cudaSuccess ||
            cudaStreamCreate(&stream) != cudaSuccess ||
            cudaEventCreate(&start) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) {
            if (start) cudaEventDestroy(start);
            if (end) cudaEventDestroy(end);
            if (stream) cudaStreamDestroy(stream);
            if (device) cudaFree(device);
            if (host) cudaFreeHost(host);
            continue;
        }

        cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        cudaEventRecord(start, stream);
        cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&h2d_ms, start, end);

        cudaEventRecord(start, stream);
        cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&d2h_ms, start, end);

        if (h2d_ms > 0.0f && d2h_ms > 0.0f) {
            h2d_sum += ((double)bytes / 1000000000.0) / ((double)h2d_ms / 1000.0);
            d2h_sum += ((double)bytes / 1000000000.0) / ((double)d2h_ms / 1000.0);
            measurements++;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(end);
        cudaStreamDestroy(stream);
        cudaFree(device);
        cudaFreeHost(host);
    }

    if (measurements == 0) {
        return -1;
    }
    profile->h2d_pinned_gbps = h2d_sum / (double)measurements;
    profile->d2h_pinned_gbps = d2h_sum / (double)measurements;
    return 0;
#else
    (void)profile;
    return -1;
#endif
}

static double mib(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

void hardware_profile_print(const HardwareProfile *profile) {
    if (!profile || !profile->cuda_available) {
        printf("Hardware profile: CUDA unavailable in this build or on this machine\n");
        return;
    }

    printf("Hardware profile:\n");
    printf("  gpu_name: %s\n", profile->gpu_name);
    printf("  cuda_device_id: %d\n", profile->cuda_device_id);
    printf("  compute_capability: %d.%d\n", profile->compute_major, profile->compute_minor);
    printf("  total_vram_mb: %.3f\n", mib(profile->total_vram_bytes));
    printf("  free_vram_before_engine_mb: %.3f\n", mib(profile->free_vram_before_engine_bytes));
    printf("  free_vram_after_engine_mb: %.3f\n", mib(profile->free_vram_after_engine_bytes));
    printf("  estimated_engine_mb: %.3f\n", mib(profile->estimated_engine_bytes));
    printf("  async_engine_count: %d\n", profile->async_engine_count);
    printf("  concurrent_kernels: %d\n", profile->concurrent_kernels);
    printf("  can_map_host_memory: %d\n", profile->can_map_host_memory);
    printf("  unified_addressing: %d\n", profile->unified_addressing);
    printf("  h2d_pinned_gbps: %.3f\n", profile->h2d_pinned_gbps);
    printf("  d2h_pinned_gbps: %.3f\n", profile->d2h_pinned_gbps);
}
