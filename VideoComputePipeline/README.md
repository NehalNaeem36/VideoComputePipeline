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
- Sequential pipeline: decode -> process -> encode -> benchmark
- Benchmark CSV output and summary printing
- Matrix report CSV summary helper
- Frame slot and bounded frame queue support for later threaded pipeline work
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

For a quick smoke run:

```bash
./build/bin/VideoComputePipeline.exe --max-frames 30
```

## CLI

```text
--input path
--output path
--benchmark path
--mode cpu|gpu
--filter grayscale|blur3x3|blur5x5|blur9x9
--max-frames N
--no-benchmark
--help
--version
```

`--max-frames 0` means process the full input.

## Benchmark CSV

CSV columns:

```text
frame_index,decode_ms,process_ms,upload_ms,kernel_ms,download_ms,encode_ms,total_ms
```

For CPU mode, upload/kernel/download columns remain zero. For GPU mode, process time includes upload, kernel execution, and download.

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
