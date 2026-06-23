# VideoComputePipeline

VideoComputePipeline is a modular C11 CPU/GPU video-processing benchmark.

It uses FFmpeg libraries for MP4 decode/encode, OpenCL for GPU filters, and optional CUDA/TensorRT for YOLO detection. It does not call the FFmpeg CLI from the application.

Project directory:

```text
VideoComputePipeline/
```

## Implemented

- FFmpeg video reader: MP4 decode to internal RGB24 `Frame`
- FFmpeg video writer: RGB24 frame encode to MP4
- CPU filters: grayscale, 3x3 blur, 5x5 blur, 9x9 blur, 13x13 blur
- OpenCL GPU filters for the same filter set
- Threaded pipeline with ordered output by global frame ID
- NVENC encoder option through FFmpeg: `h264_nvenc`
- Lossless output mode
- Long-run memory profile with bounded reusable frame pools
- Streaming benchmark CSV output
- Matrix report CLI for CPU/GPU CSV comparison
- Detection-only CUDA/TensorRT YOLO path: MP4 -> NV12 -> detections CSV
- Experimental GPU-resident detection path: NVDEC -> TensorRT -> CUDA box overlay -> NVENC
- Batch-aware detection execution plan with optional video/GPU auto-tuning
- Tests for modules

## Build

From PowerShell:

```powershell
cd E:\wAI\first_task\VideoComputePipeline
cmake --build build-win -j 4
```

Run tests:

```powershell
ctest --test-dir build-win --output-on-failure
```

If configuring from scratch in MSYS2 UCRT64:

```bash
cd /e/wAI/first_task/VideoComputePipeline
cmake -S . -B build-win -G "MinGW Makefiles"
cmake --build build-win -j 4
```

CUDA/TensorRT detection builds require MSVC, CUDA Toolkit, TensorRT, and MSVC-compatible FFmpeg development files:

```cmd
cd /d E:\wAI\first_task\VideoComputePipeline
rmdir /s /q build-msvc

cmake -S . -B build-msvc -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DENABLE_CUDA_INFERENCE=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DOPENCL_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" ^
  -DTENSORRT_ROOT="D:\TensorRT\TensorRT-Enterprise-11.1.0.106-Windows-amd64-cuda-13.3-Release-external\TensorRT-11.1.0.106" ^
  -DFFMPEG_ROOT="C:\vcpkg\installed\x64-windows"

cmake --build build-msvc -j 8
```

Experimental NVDEC/NVENC annotated detection additionally enables hardware video:

```cmd
cmake -S . -B build-msvc -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DENABLE_CUDA_INFERENCE=ON ^
  -DENABLE_HW_VIDEO=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DOPENCL_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" ^
  -DTENSORRT_ROOT="D:\TensorRT\TensorRT-Enterprise-11.1.0.106-Windows-amd64-cuda-13.3-Release-external\TensorRT-11.1.0.106" ^
  -DFFMPEG_ROOT="C:\vcpkg\installed\x64-windows"

cmake --build build-msvc -j 8
```

## Run Examples

Short GPU smoke test:

```powershell
.\build-win\bin\VideoComputePipeline.exe `
  --input data\input\15592600_3840_2160_60fps.mp4 `
  --output data\output\gpu_blur13x13_smoke.mp4 `
  --benchmark benchmarks\gpu_blur13x13_smoke.csv `
  --encoder h264_nvenc `
  --mode gpu `
  --filter blur13x13 `
  --max-frames 300 `
  --memory-profile low
```

Lossless NVENC GPU run:

```powershell
.\build-win\bin\VideoComputePipeline.exe `
  --input data\input\15592600_3840_2160_60fps.mp4 `
  --output data\output\lossless_nvenc_blur13x13.mp4 `
  --benchmark benchmarks\lossless_nvenc_blur13x13.csv `
  --encoder h264_nvenc `
  --lossless `
  --mode gpu `
  --filter blur13x13
```

RGB lossless output:

```powershell
.\build-win\bin\VideoComputePipeline.exe `
  --input data\input\15592600_3840_2160_60fps.mp4 `
  --output data\output\lossless_x264rgb_blur13x13.mp4 `
  --benchmark benchmarks\lossless_x264rgb_blur13x13.csv `
  --encoder libx264 `
  --lossless `
  --mode gpu `
  --filter blur13x13 `
  --memory-profile low
```

Matrix report:

```powershell
.\build-win\bin\VideoComputePipeline.exe `
  --matrix-report benchmarks\cpu_run.csv benchmarks\gpu_run.csv
```

Build a TensorRT 11 YOLOv5s engine:

```cmd
cd /d E:\wAI\first_task\VideoComputePipeline
python tools\build_yolov5s_tensorrt.py
```

Detection smoke test:

```powershell
.\build-msvc\bin\VideoComputePipeline.exe `
  --task detect `
  --input data\input\sample.mp4 `
  --model models\yolov5s_trt11.engine `
  --labels models\coco.names `
  --detections benchmarks\detections.csv `
  --benchmark benchmarks\detection_benchmark.csv `
  --confidence 0.25 `
  --iou-threshold 0.45 `
  --classes person,car `
  --input-size 640 `
  --max-frames 300
```

Experimental annotated detection without full-frame CPU transfers:

```powershell
.\build-msvc\bin\VideoComputePipeline.exe `
  --task detect `
  --decoder nvdec `
  --draw-boxes `
  --input data\input\sample.mp4 `
  --output data\output\annotated.mkv `
  --encoder h264_nvenc `
  --model models\yolov5s_trt11.engine `
  --labels models\coco.names `
  --detections benchmarks\detections_hw.csv `
  --benchmark benchmarks\detection_hw.csv `
  --confidence 0.25 `
  --iou-threshold 0.45 `
  --input-size 640 `
  --max-frames 300
```

## CLI

```text
--input path
--output path
--benchmark path
--task filter|detect
--decoder cpu|nvdec
--decoder-fallback cpu|none
--model path
--labels path
--detections path
--draw-boxes
--box-thickness N
--box-confidence value
--confidence value
--iou-threshold value
--classes person,car
--class-ids 0,2
--input-size N
--runtime auto|tensorrt|onnxruntime|torchscript
--backend-device cuda|cpu
--allow-host-backend
--list-backends
--model-info
--model-type auto|yolov5
--inference-backend tensorrt
--precision fp16|fp32
--batch-size auto|N
--inflight-batches auto|N
--auto-tune
--target-fps N
--vram-budget-ratio R
--vram-reserve-mb N
--profile-hardware
--pipeline-overlap auto|on|off
--parallel-inference auto|on|off
--inference-contexts auto|N
--encoder libx264|libx264rgb|h264_nvenc|mpeg4
--mode cpu|gpu
--filter grayscale|blur3x3|blur5x5|blur9x9|blur13x13
--max-frames N
--frame-slots N
--decoder-threads N
--encoder-threads N
--processor-workers N
--lossless
--lossy
--memory-profile auto|low|balanced|manual
--memory-budget-mb N
--matrix-report cpu.csv gpu.csv
--no-benchmark
--no-decoder-fallback
--help
--version
```

`--runtime auto` selects by model extension:
- `.engine` and `.plan` use TensorRT.
- `.onnx` selects ONNX Runtime.
- `.pt`, `.ts`, and `.torchscript` select TorchScript.

TensorRT is the default compiled runtime for CUDA builds. ONNX Runtime and TorchScript are optional: install ONNX Runtime GPU or LibTorch CUDA, then configure with `-DENABLE_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=...` or `-DENABLE_LIBTORCH=ON -DLIBTORCH_ROOT=...`. `--inference-backend tensorrt` remains a backward-compatible alias for `--runtime tensorrt`.

`--max-frames` is frame-count based, not time based. For a 25 FPS video:

```text
5 minutes = 5 * 60 * 25 = 7500 frames
```

For a 60 FPS video:

```text
5 minutes = 5 * 60 * 60 = 18000 frames
```

## Memory Management

The threaded pipeline uses bounded reusable frame pools:

```text
raw frame pool -> raw queue -> processor -> processed frame pool -> processed queue -> encoder
```

GPU mode returns the CPU input frame to the raw pool immediately after the blocking OpenCL upload finishes. The encoder returns processed frames to the processed pool immediately after writing.

Memory profiles:

```text
auto      Default. Uses conservative settings for lossless output.
low       Minimizes in-flight RGB24 frames.
balanced  Reduces CPU worker pressure.
manual    Uses the exact thread and queue values passed on the CLI.
```

Benchmark CSV rows are streamed during encoding and flushed periodically, so long runs do not keep all timing rows in heap memory.

## Detection Auto-Tuning

Detection mode now builds an execution plan before the frame loop. The default plan preserves the existing single-frame behavior:

```text
batch_size = 1
inflight_batches = 1
valid_frames = 1
```

Use `--auto-tune` to let the planner choose batch and in-flight settings from the actual video metadata, CUDA memory state, TensorRT engine capability, and measured pinned-copy bandwidth:

```powershell
.\build-msvc\bin\VideoComputePipeline.exe `
  --task detect `
  --decoder nvdec `
  --draw-boxes `
  --input data\input\sample.mp4 `
  --output data\output\annotated.mkv `
  --encoder h264_nvenc `
  --model models\yolov5s_trt11.engine `
  --labels models\coco.names `
  --detections benchmarks\detections_hw.csv `
  --benchmark benchmarks\detection_hw.csv `
  --auto-tune `
  --target-fps 60 `
  --input-size 640 `
  --max-frames 300
```

The VRAM policy is conservative by default. It uses the smaller of `total_vram * 0.375` and free VRAM after TensorRT engine load minus a reserve. This avoids treating display/OS/other GPU memory as available pipeline working memory.

For a quick CUDA environment check:

```powershell
.\build-msvc\bin\VideoComputePipeline.exe --task detect --profile-hardware
```

Batching is currently integrated through `FrameBatch` and execution-plan metadata. If the TensorRT engine is static batch-1, the planner keeps true inference at batch size 1 or loops per frame inside the batch abstraction. Engines with larger/dynamic batch support can be enabled through the batch inference API without creating a separate pipeline.

For static batch-1 engines, the overlap path can still use multiple TensorRT execution contexts when supported:

```powershell
.\build-msvc\bin\VideoComputePipeline.exe `
  --task detect `
  --decoder nvdec `
  --draw-boxes `
  --auto-tune `
  --pipeline-overlap on `
  --parallel-inference auto `
  --input data\input\sample.mp4 `
  --output data\output\overlap_annotated.mkv `
  --encoder h264_nvenc `
  --model models\yolov5s_trt11.engine `
  --labels models\coco.names `
  --detections benchmarks\nvdec_overlap_detections.csv `
  --benchmark benchmarks\nvdec_overlap_benchmark.csv `
  --max-frames 300
```

This is parallel single-frame inference, not one TensorRT batch enqueue. Each active inference slot owns its own CUDA stream and TensorRT execution context.

## Output Quality

`--lossless` uses lossless encoder settings where supported.

- `--encoder libx264 --lossless` switches internally to `libx264rgb` for RGB lossless output.
- `--encoder h264_nvenc --lossless` uses NVENC constant-QP lossless settings with `YUV420P`, which is the stable lower-resource path for OpenCL plus NVENC on the RTX 3050 Laptop GPU.

MP4 output is still encoded video, not literal raw uncompressed video.

## Layout

```text
VideoComputePipeline/include/core        VideoComputePipeline/src/core        VideoComputePipeline/tests/core
VideoComputePipeline/include/video       VideoComputePipeline/src/video       VideoComputePipeline/tests/video
VideoComputePipeline/include/cpu         VideoComputePipeline/src/cpu         VideoComputePipeline/tests/cpu
VideoComputePipeline/include/gpu         VideoComputePipeline/src/gpu         VideoComputePipeline/tests/gpu
VideoComputePipeline/include/pipeline    VideoComputePipeline/src/pipeline    VideoComputePipeline/tests/pipeline
VideoComputePipeline/include/benchmark   VideoComputePipeline/src/benchmark   VideoComputePipeline/tests/benchmark
VideoComputePipeline/include/utils       VideoComputePipeline/src/utils       VideoComputePipeline/tests/utils
VideoComputePipeline/kernels
VideoComputePipeline/data/input
VideoComputePipeline/data/output
VideoComputePipeline/benchmarks
```

FFmpeg code is isolated in video modules. OpenCL code is isolated in GPU modules. Frame memory is isolated in core/pipeline frame modules. Timing and benchmark output are isolated in benchmark modules.

Detection mode is CSV-only by default: it decodes NV12 frames, runs the selected inference runtime, writes `detections.csv`, and records detection timing fields in the benchmark CSV. Annotated video output is enabled only when `--draw-boxes` and `--output` are provided, with `--output-format mkv` recommended for hardware-video smoke tests.
