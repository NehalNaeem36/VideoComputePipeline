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
  --input-size 640 `
  --max-frames 300
```

## CLI

```text
--input path
--output path
--benchmark path
--task filter|detect
--model path
--labels path
--detections path
--confidence value
--iou-threshold value
--input-size N
--inference-backend tensorrt
--precision fp16|fp32
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
--help
--version
```

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

Detection mode skips video encoding and does not create an output MP4. It decodes NV12 frames, runs TensorRT inference, writes `detections.csv`, and records detection timing fields in the benchmark CSV.
