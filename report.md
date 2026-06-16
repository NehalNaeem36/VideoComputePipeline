# VideoComputePipeline Report

## Current Workflow

VideoComputePipeline is a C11 video-processing benchmark using FFmpeg libraries and OpenCL.

Application flow:

```text
MP4 input
  -> FFmpeg decoder
  -> RGB24 Frame
  -> raw frame pool / raw queue
  -> CPU or OpenCL processor
  -> processed frame pool / processed queue
  -> ordered FFmpeg writer
  -> MP4 output
```

Each decoded frame receives a global frame ID before it enters the pipeline. Processor workers may finish out of order, so the encoder stage writes frames in ascending frame ID order.

## Threading Model

```text
decoder stage: 1 app thread
processor stage: N CPU workers, or 1 GPU OpenCL worker
encoder stage: 1 app thread
```

FFmpeg also uses internal threads:

```text
--decoder-threads N
--encoder-threads N
```

GPU mode currently uses one OpenCL processor worker because the GPU filter context owns one OpenCL context and command queue.

## Frame Format

Internal frames currently use RGB24:

```text
1 pixel = 3 bytes
stride = width * 3
```

For 4K:

```text
3840 * 2160 * 3 = 24,883,200 bytes = about 24.9 MB/frame
```

GPU mode currently transfers:

```text
CPU RGB24 -> OpenCL upload -> kernel -> CPU RGB24 download -> FFmpeg encode
```

This makes upload/download and CPU memory bandwidth major costs.

## Filters

CPU and GPU support:

```text
grayscale
blur3x3
blur5x5
blur9x9
blur13x13
```

Blur filters use box blur with edge clamping.

## Long-Run Memory Management

The pipeline now bounds full-frame CPU memory with reusable frame pools.

Current low-memory 4K profile:

```text
raw pool: 3 frames
processed pool: 3 frames
frame size: 24,883,200 bytes
pooled frame memory: about 149 MB
```

Frame ownership:

```text
decoder acquires raw frame
decoder fills raw frame
raw queue owns raw frame
processor uploads/processes
GPU mode returns raw CPU frame immediately after blocking upload
processor acquires processed frame
processed queue owns processed frame
encoder writes processed frame
encoder returns processed frame to pool
```

This prevents heap growth from scaling with video length.

Benchmark output is streamed to CSV as frames are encoded and flushed periodically. The benchmark module keeps only summary totals in memory for CLI runs.

## Memory Profiles

```text
--memory-profile auto
--memory-profile low
--memory-profile balanced
--memory-profile manual
```

Behavior:

```text
auto      Default, conservative for lossless output
low       Minimal in-flight frames and reduced FFmpeg thread pressure
balanced  Moderate settings for CPU mode
manual    Respect user-provided queue/thread settings
```

`--memory-budget-mb N` can request a frame-pool memory budget. If the profile is not `manual`, the runner may fall back to lower-memory settings and warn.

## Output Quality

Normal output is compressed H.264/MPEG-4.

Lossless mode:

```text
libx264 + --lossless -> libx264rgb, RGB lossless, QP/CRF 0
h264_nvenc + --lossless -> NVENC constant-QP lossless settings, YUV420P
```

NVENC 4:4:4 lossless was tested but caused OpenCL/NVENC resource pressure during 4K blur processing on the RTX 3050 Laptop GPU. The stable NVENC lossless path uses `YUV420P`.

## Observed Performance

Recent 5-minute-limited run:

```text
frames: 18000
total_ms: 765334.823
avg_total_ms: 42.519
avg_process_ms: 13.347
processed_fps: 23.519
```

The output was about 12 minutes because that source/output was 25 FPS:

```text
18000 frames / 25 FPS = 720 seconds = 12 minutes
```

For a 25 FPS video, the first 5 minutes is:

```text
5 * 60 * 25 = 7500 frames
```

Estimated runtime for 7500 frames using the measured result:

```text
765.335 sec * (7500 / 18000) = about 319 sec = 5 min 19 sec
```

Projected runtime for a 36 min 6 sec video at the same measured throughput and same settings:

```text
36 min 6 sec = 2166 sec
25 FPS input = 54150 frames
54150 / 23.519 processed FPS = about 2302 sec = 38 min 22 sec
```

This projection assumes stable thermals, disk throughput, filter, encoder, and resolution.

## Current Bottlenecks

```text
decode
encode
GPU upload
GPU download
CPU memory bandwidth
disk space for lossless output
```

OpenCL kernel time is only one part of total frame time. The RGB24 round trip is expensive.

## Recommended Next Optimizations

1. Add bitrate/quality CLI controls for non-lossless output.
2. Add wall-clock whole-run throughput reporting.
3. Use NV12/YUV internally where possible to reduce frame size.
4. Explore FFmpeg hardware decode through NVDEC.
5. Move toward GPU-resident frames:

```text
NVDEC decode -> GPU processing -> NVENC encode
```

That would reduce CPU/GPU round trips, but requires a larger FFmpeg hardware-frames integration.
