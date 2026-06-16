# VideoComputePipeline

VideoComputePipeline is a modular C11 CPU/GPU video-processing benchmark.

It reads MP4 input with FFmpeg libraries, decodes frames one at a time into an internal RGB24 `Frame`, processes each frame with CPU filters or OpenCL kernels, writes an output MP4, and records per-frame benchmark timings.

No FFmpeg CLI commands are used.

## Implemented

- Frame memory and metadata management
- FFmpeg video reader: MP4 decode to RGB24 frames
- FFmpeg video writer: RGB24 frames to MP4
- CPU filters: grayscale, 3x3 blur, 5x5 blur, 9x9 blur
- OpenCL context, program build, kernels, and GPU filters
- Threaded pipeline: decoder stage -> raw FIFO queue -> processor worker(s) -> processed queue -> ordered writer stage
- Benchmark CSV output and summary printing
- Matrix report CSV summary helper
- Frame slot and bounded frame queue support
- Tests for all modules

## Build On Windows MSYS2 UCRT64

Open the **MSYS2 UCRT64** terminal:

```bash
cd /path/to/VideoComputePipeline
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
```

If using Unix Makefiles in MSYS2:

```bash
cmake -S . -B build -G "Unix Makefiles"
cmake --build build
```

If CMake cannot find FFmpeg or OpenCL automatically, pass roots explicitly:

```bash
cmake -S . -B build -G "MinGW Makefiles" -DFFMPEG_ROOT=C:/ffmpeg -DOPENCL_ROOT=C:/OpenCL-SDK
cmake --build build
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

## Run

CPU grayscale:

```bash
./build/bin/VideoComputePipeline.exe \
  --input data/input/15592600_3840_2160_60fps.mp4 \
  --output data/output/cpu_grayscale.mp4 \
  --benchmark benchmarks/cpu_grayscale.csv \
  --mode cpu \
  --filter grayscale
```

GPU 5x5 blur:

```bash
./build/bin/VideoComputePipeline.exe \
  --input data/input/15592600_3840_2160_60fps.mp4 \
  --output data/output/gpu_blur5x5.mp4 \
  --benchmark benchmarks/gpu_blur5x5.csv \
  --mode gpu \
  --filter blur5x5
```

Threaded GPU run with explicit worker settings:

```bash
./build/bin/VideoComputePipeline.exe \
  --input data/input/15592600_3840_2160_60fps.mp4 \
  --output data/output/gpu_threaded_grayscale.mp4 \
  --benchmark benchmarks/gpu_threaded_grayscale.csv \
  --encoder h264_nvenc \
  --mode gpu \
  --filter grayscale \
  --frame-slots 8 \
  --decoder-threads 4 \
  --encoder-threads 4 \
  --processor-workers 4
```

GPU mode uses one OpenCL processor worker internally even if a higher `--processor-workers` value is requested, because the current GPU implementation owns one OpenCL context and command queue. CPU mode uses `--processor-workers` worker threads.

Recommended lower-contention GPU settings for full 4K blur benchmarks:

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

These settings reduce contention between FFmpeg decode/encode, CPU memory bandwidth, and OpenCL upload/download. On the 4K high-quality input video, consistent average total frame times below 60 ms/frame for `blur5x5` are a good result for the current RGB24 pipeline. That is roughly 16+ processed frames per second end-to-end, including decode, GPU upload, kernel execution, download, encode, and benchmark recording.

Lossless output mode:

```powershell
.\build-win\bin\VideoComputePipeline.exe `
  --input data\input\15592600_3840_2160_60fps.mp4 `
  --output data\output\lossless_gpu_blur5x5.mp4 `
  --benchmark benchmarks\lossless_gpu_blur5x5.csv `
  --encoder h264_nvenc `
  --lossless `
  --mode gpu `
  --filter blur5x5 `
  --frame-slots 3 `
  --decoder-threads 2 `
  --encoder-threads 2 `
  --processor-workers 1
```

`--lossless` preserves the decoded input frame dimensions and frame rate in the output and uses lossless encoder settings where supported. With `--encoder libx264`, the writer switches to `libx264rgb` so RGB24 frames are not converted to `YUV420P`. With `--encoder h264_nvenc`, the writer uses NVENC lossless constant-QP settings with `YUV420P`; this is the stable lower-resource path for OpenCL processing plus NVENC on the RTX 3050 Laptop GPU. MP4 output is still an encoded video stream, not literal raw uncompressed video.

For a quick smoke run:

```bash
./build/bin/VideoComputePipeline.exe --max-frames 30
```

## CLI

```text
--input path
--output path
--benchmark path
--encoder libx264|libx264rgb|h264_nvenc|mpeg4
--mode cpu|gpu
--filter grayscale|blur3x3|blur5x5|blur9x9|blur13x13
--max-frames N
--frame-slots N
--decoder-threads N
--encoder-threads N
--processor-workers N
--lossless
--no-benchmark
--help
--version
```

`--max-frames 0` means process the full input.

Frame IDs are assigned by the decoder stage before frames are placed into the raw FIFO queue. Processor workers may finish out of order, so the writer stage buffers processed frames and writes them in ascending frame ID order.

## Benchmark CSV

CSV columns:

```text
frame_index,decode_ms,process_ms,upload_ms,kernel_ms,download_ms,encode_ms,total_ms
```

For CPU mode, upload/kernel/download columns remain zero. For GPU mode, process time includes upload, kernel execution, and download.

`--encoder h264_nvenc` uses NVIDIA NVENC for H.264 output when the installed FFmpeg build exposes that encoder. `--encoder libx264` uses CPU x264, `--encoder libx264rgb` uses RGB x264, and `--encoder mpeg4` uses FFmpeg's MPEG-4 fallback encoder. Use `--encoder libx264 --lossless` for RGB lossless output, or `--encoder h264_nvenc --lossless` for faster GPU-assisted lossless-quantized H.264 output.

## Module Layout

```text
include/core        src/core        tests/core
include/video       src/video       tests/video
include/cpu         src/cpu         tests/cpu
include/gpu         src/gpu         tests/gpu
include/pipeline    src/pipeline    tests/pipeline
include/benchmark   src/benchmark   tests/benchmark
include/utils       src/utils       tests/utils
kernels
data/input
data/output
benchmarks
```

FFmpeg code is isolated in video modules. OpenCL code is isolated in GPU modules. Frame memory is isolated in the frame module. Timing is isolated in benchmark/timer modules.
