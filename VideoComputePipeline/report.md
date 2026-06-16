# VideoComputePipeline Report

## Current Workflow

VideoComputePipeline is a C11 video-processing benchmark that uses FFmpeg libraries for video I/O and OpenCL for GPU filters. It does not call the FFmpeg command-line tool.

Current high-level flow:

```text
MP4 input
  -> FFmpeg decoder
  -> RGB24 Frame
  -> raw FIFO queue
  -> CPU/OpenCL filter processor
  -> processed frame queue
  -> ordered FFmpeg writer
  -> MP4 output
```

Each decoded frame receives a global `frame.index` before it enters the raw frame queue. Processor workers may finish frames out of order, so the writer buffers processed frames and writes them in ascending frame ID order.

## Threading Model

The pipeline has three app-level stages:

```text
decoder stage -> processor stage -> encoder stage
```

Current worker behavior:

```text
decoder stage: 1 app thread
processor stage: N CPU workers, or 1 GPU OpenCL worker
encoder stage: 1 app thread
```

FFmpeg also uses internal worker threads:

```text
--decoder-threads N
--encoder-threads N
```

For GPU mode, `--processor-workers` is accepted but the pipeline uses one OpenCL processor worker internally. This is intentional because the current GPU implementation owns one OpenCL context and command queue.

## Frame Format

Internal frames currently use RGB24:

```text
1 pixel = 3 bytes
stride = width * 3
```

For 4K input:

```text
3840 * 2160 * 3 = about 24.9 MB per frame
```

GPU mode currently transfers each frame like this:

```text
CPU RGB24 -> GPU upload -> OpenCL kernel -> CPU RGB24 download
```

This means upload/download bandwidth is a major part of total frame time.

## Output Quality

The writer preserves the decoded input width, height, and frame rate for the output video.

Normal output is compressed H.264/MPEG-4 video. Lossless output is enabled with:

```text
--lossless
```

Current lossless behavior:

```text
libx264 + --lossless -> libx264rgb, RGB24, QP/CRF 0
h264_nvenc + --lossless -> NVENC lossless constant QP, YUV420P
```

This avoids the default `YUV420P` chroma subsampling path for lossless x264 output. NVENC lossless output still requires an RGB-to-YUV conversion in the current pipeline. A 4:4:4 NVENC lossless path was tested, but it caused OpenCL/NVENC resource failures during 4K blur processing on the RTX 3050 Laptop GPU, so the stable NVENC lossless path uses `YUV420P`.

MP4 output is still encoded video. Literal raw uncompressed video is not the intended output format for this project.

## Implemented Filters

CPU and GPU modes support:

```text
grayscale
blur3x3
blur5x5
blur9x9
blur13x13
```

Blur filters use box blur with edge clamping. Edge clamping repeats the nearest valid edge pixel when a kernel reads outside the image boundary.

## Benchmark CSV

Each benchmark CSV contains:

```text
frame_index,decode_ms,process_ms,upload_ms,kernel_ms,download_ms,encode_ms,total_ms
```

For CPU mode:

```text
upload_ms = 0
kernel_ms = 0
download_ms = 0
```

For GPU mode:

```text
process_ms includes upload + kernel + download
```

`total_ms` is the per-frame measured pipeline stage total. In threaded mode, per-frame stage timings can be higher than sequential timings because decode, process, encode, memory copies, and FFmpeg internal threads run concurrently and compete for CPU and memory bandwidth.

For threaded runs, the most important metric is overall throughput:

```text
processed FPS = total frames / total wall-clock time
```

## Current Tuned 4K GPU Settings

The best observed full-quality 4K `blur5x5` settings so far are lower-contention settings:

```powershell
.\build-win\bin\VideoComputePipeline.exe `
  --input data\input\15592600_3840_2160_60fps.mp4 `
  --output data\output\threaded_gpu_blur5x5.mp4 `
  --benchmark benchmarks\threaded_gpu_blur5x5.csv `
  --encoder h264_nvenc `
  --mode gpu `
  --filter blur5x5 `
  --frame-slots 3 `
  --decoder-threads 2 `
  --encoder-threads 2 `
  --processor-workers 1
```

These settings reduce contention between:

```text
FFmpeg decode
OpenCL upload/download
GPU kernel execution
FFmpeg encode
CPU memory bandwidth
```

Observed result:

```text
average total frame time consistently below 60 ms/frame for full 4K blur5x5
```

That is a good result for the current RGB24 architecture:

```text
60 ms/frame = about 16.7 FPS
```

This is not real-time 4K60, but it is a meaningful full-pipeline result because it includes decode, GPU transfer, OpenCL processing, download, encode, and benchmark recording.

## Bottlenecks

Current bottlenecks are:

```text
decode
encode
GPU upload
GPU download
CPU memory bandwidth
```

The OpenCL blur kernel itself is not the only cost. For 4K RGB24 frames, each GPU-processed frame requires about:

```text
~24.9 MB upload
~24.9 MB download
```

That is roughly 50 MB of transfer per frame, before decode and encode costs.

## Long-Run Memory Management

The pipeline now bounds full-frame CPU memory with reusable frame pools:

```text
raw frame pool -> raw packet queue -> processor -> processed frame pool -> processed packet queue -> encoder
```

The decoder acquires RGB24 buffers from the raw pool. The processor returns raw buffers when it no longer needs them. In GPU mode, the raw CPU frame is returned immediately after the blocking OpenCL upload completes. The encoder returns processed frames to the processed pool immediately after writing each frame.

For 4K RGB24:

```text
one frame = about 24.9 MB
low profile = 3 raw frames + 3 processed frames = about 149 MB of pooled frame data
```

Benchmark rows are streamed to CSV during encoding and flushed periodically. The benchmark module keeps only summary totals in memory for CLI pipeline runs.

Memory profiles:

```text
--memory-profile auto
--memory-profile low
--memory-profile balanced
--memory-profile manual
```

`manual` preserves user-provided queue and thread settings. Other profiles may reduce frame slots or worker counts to keep heap usage bounded for long 1080p-to-4K videos.

## Recommended Next Optimizations

Practical next steps:

```text
1. Add configurable bitrate, e.g. --bitrate-mbps 50
2. Consider NVDEC hardware decoding
3. Reduce RGB24 round-trips by using NV12/YUV420P internally
4. Explore keeping frames GPU-side longer
5. Add wall-clock whole-run throughput reporting
```

NVENC encoding is available through:

```text
--encoder h264_nvenc
```

The biggest remaining architectural improvement would be:

```text
NVDEC decode -> GPU processing -> NVENC encode
```

That would reduce CPU/GPU memory copies, but it requires a larger FFmpeg hardware-frames integration.
