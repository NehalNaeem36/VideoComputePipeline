# VideoComputePipeline

VideoComputePipeline is a modular C11 CPU/GPU video-processing benchmark.

It uses FFmpeg libraries for MP4 decode/encode, CUDA for GPU filters, and optional CUDA/TensorRT or ONNX Runtime for YOLO detection. It does not call the FFmpeg CLI from the application.

Project directory:

```text
VideoComputePipeline/
```

## Implemented

- FFmpeg video reader: MP4 decode to internal RGB24 `Frame`
- FFmpeg video writer: RGB24 frame encode to MP4
- CPU filters: grayscale, 3x3 blur, 5x5 blur, 9x9 blur, 13x13 blur
- CUDA GPU filters for the same filter set
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

Detection topology is controlled independently:

- `--decoder cpu|nvdec` chooses only where frames are decoded.
- `--backend-device cuda|cpu` chooses only where inference runs.
- `--encoder` matters only when `--draw-boxes` creates annotated video.
- CSV-only detection does not require an encoder or output video.

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

## Windows CUDA 12 Stack

CUDA/TensorRT/NVDEC/NVENC detection builds require Visual Studio 2022 x64, CUDA 12.9, cuDNN 9.23.2 for CUDA 12, TensorRT 11.1.0.106 for CUDA 12, and MSVC-compatible FFmpeg development files.

Verified local roots:

```text
CUDA Toolkit:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9

cuDNN:
D:\cuDNN\cudnn-windows-x86_64-9.23.2.1_cuda12-archive

TensorRT:
D:\TensorRT\TensorRT-11.1.0.106

ONNX Runtime GPU:
E:\wAI\third_party\onnxruntime\Microsoft.ML.OnnxRuntime.Gpu.Windows.1.26.0
```

Required runtime PATH entries when DLLs are not copied beside the executable:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin
D:\cuDNN\cudnn-windows-x86_64-9.23.2.1_cuda12-archive\bin\x64
D:\TensorRT\TensorRT-11.1.0.106\bin
E:\wAI\third_party\onnxruntime\Microsoft.ML.OnnxRuntime.Gpu.Windows.1.26.0\runtimes\win-x64\native
```

Verify the stack:

```powershell
cd E:\wAI\first_task\VideoComputePipeline
.\scripts\verify_cuda12_stack.ps1

where.exe nvcc
nvcc --version
where.exe cudart64_*.dll
where.exe cudnn*.dll
where.exe trtexec
where.exe nvinfer*.dll
where.exe onnxruntime.dll
where.exe onnxruntime_providers_cuda.dll
echo $env:CUDA_PATH
echo $env:CUDNN_ROOT
echo $env:TENSORRT_ROOT
echo $env:ONNXRUNTIME_ROOT
```

TensorRT-only CUDA 12 configure:

```powershell
cd E:\wAI\first_task\VideoComputePipeline

cmake -S . -B build-win-cuda12 `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" `
  -DTENSORRT_ROOT="D:\TensorRT\TensorRT-11.1.0.106" `
  -DCUDNN_ROOT="D:\cuDNN\cudnn-windows-x86_64-9.23.2.1_cuda12-archive" `
  -DENABLE_CUDA_INFERENCE=ON `
  -DENABLE_HW_VIDEO=ON `
  -DENABLE_ONNXRUNTIME=OFF `
  -DENABLE_LIBTORCH=OFF

cmake --build build-win-cuda12 --config Release
```

ONNX Runtime-enabled CUDA 12 configure:

```powershell
cmake -S . -B build-win-cuda12-onnx `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" `
  -DTENSORRT_ROOT="D:\TensorRT\TensorRT-11.1.0.106" `
  -DCUDNN_ROOT="D:\cuDNN\cudnn-windows-x86_64-9.23.2.1_cuda12-archive" `
  -DONNXRUNTIME_ROOT="E:\wAI\third_party\onnxruntime\Microsoft.ML.OnnxRuntime.Gpu.Windows.1.26.0" `
  -DENABLE_CUDA_INFERENCE=ON `
  -DENABLE_HW_VIDEO=ON `
  -DENABLE_ONNXRUNTIME=ON `
  -DENABLE_LIBTORCH=OFF

cmake --build build-win-cuda12-onnx --config Release
```

Shortcut script:

```powershell
.\scripts\configure_cuda12.ps1
.\scripts\configure_cuda12.ps1 -EnableOnnxRuntime
```

Do not mix CUDA 13 and CUDA 12 paths in the same build directory. Do not use MinGW/MSYS2 for CUDA/TensorRT/NVDEC/NVENC builds. Avoid relying on `C:\Windows\System32\onnxruntime.dll`; the intended ONNX Runtime DLLs are under `ONNXRUNTIME_ROOT\runtimes\win-x64\native`.

Existing TensorRT `.engine` files built with the old CUDA/TensorRT stack should be treated as stale and rebuilt after moving to TensorRT 11.1 / CUDA 12.

## Linux Docker Port

The project can also be built and run as a Linux container. This is separate
from the Windows `.exe`: Docker builds a Linux binary and runs it with Linux
CUDA, FFmpeg, cuDNN, and ONNX Runtime libraries.

Current validated Docker paths:

```text
CUDA filter path:
  FFmpeg software decode -> CUDA filter kernels -> FFmpeg encode

ONNX detection CSV path:
  NVDEC decode -> ONNX Runtime CUDA inference -> detections CSV

ONNX annotated path:
  NVDEC decode -> ONNX Runtime CUDA inference -> CUDA overlay -> NVENC MKV
```

The Docker files live in:

```text
VideoComputePipeline/docker/
```

### Docker Storage

CUDA Docker images are large. On Windows, Docker Desktop stores image layers in
its WSL disk image. Move Docker Desktop's disk image location off `C:` if space
is tight:

```text
Docker Desktop -> Settings -> Resources -> Advanced -> Disk image location
```

For example:

```text
E:\DockerDesktop
```

Check Docker disk usage:

```powershell
docker system df
```

Clean unused images/build cache:

```powershell
docker system prune -a --volumes
```

### Build Images

CPU-only Linux build:

```powershell
cd E:\wAI\first_task
docker build -t videocompute:cpu -f VideoComputePipeline\docker\Dockerfile.cpu .
docker run --rm videocompute:cpu --help
```

CUDA image with ONNX Runtime and hardware-video support:

```powershell
cd E:\wAI\first_task

docker build -t videocompute:cuda `
  -f VideoComputePipeline\docker\Dockerfile.cuda `
  --build-arg ENABLE_CUDA_INFERENCE=ON `
  --build-arg ENABLE_ONNXRUNTIME=ON `
  --build-arg ENABLE_TENSORRT=OFF `
  --build-arg ENABLE_HW_VIDEO=ON `
  --build-arg ENABLE_LIBTORCH=OFF `
  .
```

The current CUDA Docker image installs:

```text
CUDA base image: nvidia/cuda:12.9.1-devel-ubuntu24.04
cuDNN: cudnn9-cuda-12 from apt
ONNX Runtime GPU: 1.22.0 extracted to /opt/onnxruntime-gpu
FFmpeg development libraries from Ubuntu apt
```

TensorRT is intentionally disabled in the command above. Linux TensorRT support
requires Linux TensorRT packages inside the image and Linux-built `.engine`
files. Windows TensorRT engines are not portable to Linux.

### Enter The CUDA Container

For NVDEC/NVENC, the container must receive NVIDIA video driver capabilities.
`--gpus all` alone is not always enough.

Use:

```powershell
cd E:\wAI\first_task

docker run --rm -it --gpus all `
  --entrypoint bash `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -v ${PWD}\VideoComputePipeline\data:/workspace/VideoComputePipeline/data `
  -v ${PWD}\VideoComputePipeline\models:/workspace/VideoComputePipeline/models `
  -v ${PWD}\VideoComputePipeline\benchmarks:/workspace/VideoComputePipeline/benchmarks `
  videocompute:cuda
```

Inside the container, verify backends and NVIDIA video support:

```bash
cd /workspace/VideoComputePipeline

LD_LIBRARY_PATH=/opt/onnxruntime-gpu/lib:${LD_LIBRARY_PATH} \
/opt/videocompute/bin/VideoComputePipeline --list-backends

ldconfig -p | grep nvcuvid
nvidia-smi
```

If `ldconfig -p | grep nvcuvid` returns nothing, NVDEC cannot run in that
container session. Restart the container with:

```powershell
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

### Docker CUDA Filter Run

From inside the container:

```bash
cd /workspace/VideoComputePipeline
mkdir -p data/output benchmarks

/opt/videocompute/bin/VideoComputePipeline \
  --task filter \
  --mode gpu \
  --filter grayscale \
  --input data/input/people_short.mp4 \
  --output data/output/docker_cuda_grayscale.mp4 \
  --benchmark benchmarks/docker_cuda_grayscale.csv \
  --max-frames 120 \
  --ffmpeg-log-level error
```

The output is visible on Windows at:

```text
E:\wAI\first_task\VideoComputePipeline\data\output\docker_cuda_grayscale.mp4
```

### Docker ONNX CUDA Detection

CSV-only detection, using NVDEC and ONNX Runtime CUDA:

```bash
cd /workspace/VideoComputePipeline
mkdir -p /tmp/vcp_out

LD_LIBRARY_PATH=/opt/onnxruntime-gpu/lib:${LD_LIBRARY_PATH} \
/opt/videocompute/bin/VideoComputePipeline \
  --task detect \
  --decoder nvdec \
  --decoder-fallback none \
  --runtime onnxruntime \
  --backend-device cuda \
  --input data/input/D01_20260623234212.mp4 \
  --model models/yolov5s.onnx \
  --labels models/coco.names \
  --detections /tmp/vcp_out/detections.csv \
  --benchmark /tmp/vcp_out/benchmark.csv \
  --confidence 0.40 \
  --iou-threshold 0.45 \
  --input-size 640 \
  --class-ids 0 \
  --max-frames 1000 \
  --ffmpeg-log-level error
```

Use `/tmp/vcp_out` for performance tests because it is container-local storage.
Writing every CSV row directly to a Windows bind mount can be much slower. To
keep files after the run, copy them to a mounted folder before exiting:

```bash
cp /tmp/vcp_out/detections.csv data/output/detections.csv
cp /tmp/vcp_out/benchmark.csv benchmarks/docker_onnx_benchmark.csv
```

### Docker Annotated Video

Annotated output uses the GPU-resident path:

```text
NVDEC -> ONNX Runtime CUDA -> CUDA overlay -> NVENC
```

Run:

```bash
cd /workspace/VideoComputePipeline
mkdir -p /tmp/vcp_out data/output

LD_LIBRARY_PATH=/opt/onnxruntime-gpu/lib:${LD_LIBRARY_PATH} \
/opt/videocompute/bin/VideoComputePipeline \
  --task detect \
  --decoder nvdec \
  --decoder-fallback none \
  --runtime onnxruntime \
  --backend-device cuda \
  --input data/input/D01_20260623234212.mp4 \
  --model models/best.onnx \
  --labels models/cement.names \
  --detections /tmp/vcp_out/detections.csv \
  --benchmark /tmp/vcp_out/benchmark_annotated.csv \
  --confidence 0.40 \
  --iou-threshold 0.45 \
  --input-size 640 \
  --class-ids 0 \
  --max-frames 1000 \
  --draw-boxes \
  --output data/output/annotated.mkv \
  --output-format mkv \
  --encoder h264_nvenc \
  --ffmpeg-log-level error \
  --progress-interval 50
```

The video appears on Windows at:

```text
E:\wAI\first_task\VideoComputePipeline\data\output\annotated.mkv
```

For model/label matching:

```text
models/yolov5s.onnx -> models/coco.names
models/best.onnx    -> models/cement.names or models/cement_bag.names
```

Mismatched model and label files can cause YOLO postprocess failures because the
postprocessor uses the label count to interpret the output tensor.

### Docker Timing Notes

`wall_clock_fps` measures real throughput from the running pipeline.

`latency_equivalent_fps` is derived from summed per-frame stage timings:

```text
1000 / avg_total_ms
```

For pipelined hardware detection, wall-clock throughput can be higher than the
latency-equivalent FPS because decode, inference, output, and recycle work can
overlap. If wall-clock is much lower than stage timings suggest, first check
whether detections/benchmark CSVs are being written to Windows bind-mounted
folders instead of `/tmp`.

### Linux UI And Docker

There are two ways to use the UI on Linux.

Recommended:

```text
Run VideoComputePipelineUI on the Linux host.
Have the UI launch docker run commands for the containerized pipeline.
```

This avoids OpenGL/SDL display forwarding inside Docker and keeps the UI native
to the desktop.

Possible but more fragile:

```text
Run VideoComputePipelineUI inside Docker and forward the Linux display.
```

On an X11 Linux desktop:

```bash
xhost +local:docker

docker run --rm -it --gpus all \
  --entrypoint bash \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD/VideoComputePipelineUI:/workspace/VideoComputePipelineUI" \
  -v "$PWD/VideoComputePipeline:/workspace/VideoComputePipeline" \
  videocompute:cuda
```

Inside the container:

```bash
apt-get update
apt-get install -y --no-install-recommends libsdl2-dev libgl1-mesa-dev pkg-config cmake ninja-build
rm -rf /var/lib/apt/lists/*

cmake -S /workspace/VideoComputePipelineUI -B /tmp/vcp-ui-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vcp-ui-build -j
/tmp/vcp-ui-build/Release/VideoComputePipelineUI
```

When finished:

```bash
xhost -local:docker
```

Wayland desktops often expose XWayland through `$DISPLAY`; if not, use the host
UI approach instead of forcing GUI-in-container.

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
.\build-win-cuda12\Release\VideoComputePipeline.exe `
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
.\build-win-cuda12\Release\VideoComputePipeline.exe `
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

GPU mode returns the CPU input frame to the raw pool immediately after the CUDA upload finishes. The encoder returns processed frames to the processed pool immediately after writing.

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
.\build-win-cuda12\Release\VideoComputePipeline.exe `
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
.\build-win-cuda12\Release\VideoComputePipeline.exe --task detect --profile-hardware
```

Batching is currently integrated through `FrameBatch` and execution-plan metadata. If the TensorRT engine is static batch-1, the planner keeps true inference at batch size 1 or loops per frame inside the batch abstraction. Engines with larger/dynamic batch support can be enabled through the batch inference API without creating a separate pipeline.

For static batch-1 engines, the overlap path can still use multiple TensorRT execution contexts when supported:

```powershell
.\build-win-cuda12\Release\VideoComputePipeline.exe `
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
- `--encoder h264_nvenc --lossless` uses NVENC constant-QP lossless settings with `YUV420P`, which is the stable lower-resource path for CUDA filters plus NVENC on the RTX 3050 Laptop GPU.

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
VideoComputePipeline/data/input
VideoComputePipeline/data/output
VideoComputePipeline/benchmarks
```

FFmpeg code is isolated in video modules. CUDA GPU code is isolated in GPU and inference modules. Frame memory is isolated in core/pipeline frame modules. Timing and benchmark output are isolated in benchmark modules.

Detection mode is CSV-only by default: it decodes NV12 frames, runs the selected inference runtime, writes `detections.csv`, and records detection timing fields in the benchmark CSV. Annotated video output is enabled only when `--draw-boxes` and `--output` are provided, with `--output-format mkv` recommended for hardware-video smoke tests.
