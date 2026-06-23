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

## Multi-Backend Detection Runtime Direction

The detection pipeline is being refactored so the video path remains shared while only the inference runtime is selected by model/runtime:

```text
NVDEC or CPU NV12 decode
  -> CUDA preprocess
  -> selected runtime backend
  -> YOLOv5 model adapter
  -> CUDA overlay when requested
  -> NVENC or CSV-only output
```

Runtime selection flags:

```text
--runtime auto|tensorrt|onnxruntime|torchscript
--backend-device cuda|cpu
--allow-host-backend
--list-backends
--model-info
--model-type auto|yolov5
```

Auto runtime selection is extension based: `.engine`/`.plan` -> TensorRT, `.onnx` -> ONNX Runtime, and `.pt`/`.ts`/`.torchscript` -> TorchScript. TensorRT remains the default CUDA runtime. ONNX Runtime and TorchScript are optional builds that require the matching ONNX Runtime GPU or LibTorch CUDA packages at configure time.

## CUDA/TensorRT Detection Workflow

The detection milestone is separate from the filter-and-encode pipeline:

```text
MP4 input
  -> FFmpeg decoder
  -> NV12 Frame
  -> CUDA upload
  -> fused NV12 to NCHW preprocess
  -> TensorRT YOLO inference
  -> CPU YOLO decode + NMS
  -> detections CSV
  -> benchmark CSV
```

Detection mode does not initialize the video writer, processed frame pool, processed queue, encoder thread, or ordered pending frame list. It is currently sequential for correctness and simpler memory ownership.

TensorRT 11 engine generation uses:

```cmd
python tools\build_yolov5s_tensorrt.py
```

The script writes:

```text
models\yolov5s_trt11.engine
```

TensorRT 11 no longer accepts the old `trtexec --fp16` flag in this setup. The runtime accepts FP32 or FP16 engine input tensors and selects the matching CUDA preprocess output type.

Detection benchmark fields:

```text
decode_ms       FFmpeg decode and NV12 conversion
upload_ms       CPU NV12 to CUDA buffer transfer
preprocess_ms   fused CUDA resize/colorspace/normalize/NCHW
inference_ms    TensorRT enqueueV3
download_ms     TensorRT output copy to host
postprocess_ms  YOLO decode, confidence filtering, NMS
total_ms        decode + upload + preprocess + inference + download + postprocess
```

Wall-clock FPS is measured separately from summed per-frame latency.

## Auto-Tuned Batch Execution Plan

Detection now has a batch-aware execution-plan layer. The current behavior remains the default special case:

```text
batch_size = 1
inflight_batches = 1
valid_frames = 1
```

The planner combines:

- actual input video dimensions, FPS, frame format, and estimated frame bytes
- CUDA total/free VRAM before and after TensorRT engine creation
- conservative VRAM reserve and budget ratio
- TensorRT input/output byte estimates and batch capability
- measured pinned H2D/D2H bandwidth when auto-tune/profile mode is requested
- whether the run is CPU-decoded or GPU-resident NVDEC/NVENC

GPU-resident detection keeps raw frames on the GPU:

```text
NVDEC -> TensorRT -> postprocess metadata -> CUDA overlay -> NVENC
```

For that mode, the selected plan reports:

```text
frames_per_upload_batch = 0
frames_per_download_batch = 0
```

CPU-decoded detection requires raw upload, so the planner reports upload batch size equal to the selected frame batch size. Full raw-frame download remains a fallback/debug concept, not the preferred throughput path.

Benchmark CSV rows append execution-plan metadata:

```text
batch_size
inflight_batches
total_active_frames
frames_per_upload_batch
frames_per_download_batch
vram_budget_mb
estimated_batch_mb
```

Use `--auto-tune` for planned selection or `--profile-hardware` to print CUDA/VRAM/bandwidth information without running a full detection pipeline.

The static batch-1 optimization path uses parallel staged execution rather than true TensorRT batch enqueue. When `--parallel-inference auto|on` selects more than one context, each active inference slot owns a separate TensorRT execution context and CUDA stream. This lets the pipeline overlap uploads, preprocess, enqueue, output copy, postprocess, overlay, and encode across in-flight frames while preserving frame-id ordering.

Relevant controls:

```text
--pipeline-overlap auto|on|off
--parallel-inference auto|on|off
--inference-contexts auto|N
```

The planner records the selected execution mode and context count in benchmark CSV rows.

## Experimental GPU-Resident Annotated Detection Path

The next hardware path keeps raw video frames on the GPU:

```text
MP4 input
  -> FFmpeg demux
  -> NVDEC CUDA/NV12 frame
  -> CUDA NV12 to TensorRT NCHW preprocess
  -> TensorRT YOLO inference
  -> CPU YOLO postprocess/NMS on compact TensorRT output
  -> CUDA overlay draws white boxes on the NV12 Y plane
  -> NVENC encodes the annotated CUDA frame
  -> FFmpeg mux writes MP4/MKV packets
  -> detections CSV and benchmark CSV
```

The existing CPU-decoded detection path remains the default:

```text
--task detect --decoder cpu
```

The experimental path is selected with:

```text
--task detect --decoder nvdec --draw-boxes --encoder h264_nvenc
```

The hardware path is gated behind:

```text
ENABLE_CUDA_INFERENCE=ON
ENABLE_HW_VIDEO=ON
```

If `--decoder nvdec` fails and fallback is enabled, the runner falls back to the existing CPU NV12 detection path. Use `--no-decoder-fallback` to fail instead.

Benchmark CSV now appends:

```text
overlay_ms
mux_write_ms
```

For NVDEC runs:

```text
upload_ms = 0
download_ms = compact TensorRT output copy only
wall_clock_fps = main throughput metric
latency_equivalent_fps = per-frame summed timing equivalent
```
