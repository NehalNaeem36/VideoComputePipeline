# VideoComputePipeline

VideoComputePipeline is a modular C11 CPU/GPU video-processing benchmark project.

This repository is currently at the first implementation milestone. The executable builds, prints configuration, and runs a pipeline skeleton that says:

```text
pipeline not implemented yet
```

FFmpeg, OpenCL, CPU filters, benchmark CSV output, matrix reports, frame queues, and threaded execution are not implemented in this milestone.

## Implemented In Milestone 1

- Project skeleton and directory layout
- `CMakeLists.txt`
- `README.md`
- `.gitignore`
- `config.h`
- `main.c`
- `frame` module
- `timer` module
- `logger` module
- `pipeline_config` module
- `pipeline_runner` skeleton

## Directory Layout

```text
include/core
include/video
include/cpu
include/gpu
include/pipeline
include/benchmark
include/utils
src/core
src/video
src/cpu
src/gpu
src/pipeline
src/benchmark
src/utils
tests/core
tests/video
tests/cpu
tests/gpu
tests/pipeline
tests/benchmark
tests/utils
kernels
data/input
data/output
benchmarks
```

## Build On Windows MSYS2 UCRT64

Open the **MSYS2 UCRT64** terminal from the Start menu, then run:

```bash
cd /path/to/VideoComputePipeline
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
```

If your MSYS2 setup uses Unix Makefiles instead:

```bash
cmake -S . -B build -G "Unix Makefiles"
cmake --build build
```

Run:

```bash
./build/bin/VideoComputePipeline.exe
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

## Current CLI

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

These options are parsed and printed, but no video processing is performed yet.

## Later Milestones

Future work will add:

- FFmpeg video reader and writer using `libavformat`, `libavcodec`, `libavutil`, and `libswscale`
- CPU grayscale and box blur filters
- Benchmark CSV output
- OpenCL context/program setup
- OpenCL kernels and GPU filters
- Sequential CPU and GPU pipelines
- Matrix reports
- Frame slots and frame queues for threaded pipeline support

The final project will use FFmpeg libraries directly and will not call the FFmpeg CLI.
