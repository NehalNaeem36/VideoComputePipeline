# VideoComputePipeline

A modular CPU/GPU video processing pipeline designed for real-time video frame processing with performance benchmarking capabilities.

## Project Purpose

VideoComputePipeline is a high-performance video processing framework that enables:
- **Frame-by-frame video processing** using FFmpeg for decoding/encoding
- **Dual-mode processing**: CPU filters and GPU (OpenCL) accelerated kernels
- **Real-time benchmarking**: Comprehensive timing analysis for CPU vs GPU comparison
- **Modular architecture**: Clean separation of concerns for easy extension and maintenance

## Module Layout

### Core Module (`include/core/`, `src/core/`)
- **frame.h/c**: Internal RGB24 Frame structure and memory management

### Video Module (`include/video/`, `src/video/`)
- **video_reader.h/c**: FFmpeg-based video file decoding and frame extraction
- **video_writer.h/c**: FFmpeg-based video file encoding and output

### CPU Module (`include/cpu/`, `src/cpu/`)
- **cpu_filters.h/c**: CPU implementations of image filters (grayscale, blur3x3, blur5x5, blur9x9)

### GPU Module (`include/gpu/`, `src/gpu/`)
- **opencl_context.h/c**: OpenCL platform, device, and context initialization
- **opencl_program.h/c**: OpenCL program compilation and kernel management
- **gpu_filters.h/c**: GPU-accelerated filter execution

### Pipeline Module (`include/pipeline/`, `src/pipeline/`)
- **frame_slot.h/c**: Single frame slot for synchronized processing
- **frame_queue.h/c**: Thread-safe queue for frame buffering
- **pipeline_config.h/c**: Configuration management and command-line argument parsing
- **pipeline_runner.h/c**: Main pipeline orchestrator coordinating all components

### Benchmark Module (`include/benchmark/`, `src/benchmark/`)
- **timer.h/c**: High-resolution timing for performance measurement
- **benchmark.h/c**: Per-frame and aggregate statistics collection
- **matrix_report.h/c**: CSV and markdown export for benchmark results

### Utils Module (`include/utils/`, `src/utils/`)
- **logger.h/c**: Structured logging with multiple verbosity levels
- **file_utils.h/c**: File system utilities (path building, existence checks, etc.)

### OpenCL Kernels (`kernels/`)
- **grayscale.cl**: Grayscale conversion kernel
- **blur3x3.cl**: 3x3 box blur kernel
- **blur5x5.cl**: 5x5 box blur kernel
- **blur9x9.cl**: 9x9 box blur kernel

## Directory Structure

```
VideoComputePipeline/
├── include/          # Public header files organized by module
├── src/              # Implementation files organized by module
├── tests/            # Test files (one per module component)
├── kernels/          # OpenCL kernel source files
├── data/
│   ├── input/        # Input video files
│   └── output/       # Processed output videos
├── benchmarks/       # Generated benchmark reports (CSV, markdown)
├── main.c            # Entry point
├── config.h          # Global configuration constants
├── CMakeLists.txt    # Build configuration
├── README.md         # This file
└── .gitignore        # Git ignore rules
```

## Build Instructions (Placeholder)

### Prerequisites
- CMake >= 3.10
- C11 compatible compiler
- FFmpeg development libraries (libavformat, libavcodec, libavutil)
- OpenCL development libraries (optional, for GPU support)

### Build Steps
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Run Tests
```bash
cmake --build . --target test
```

### Run Pipeline
```bash
./bin/VideoComputePipeline -i data/input/video.mp4 -o data/output/result.mp4 -f grayscale
```

## Planned Benchmark Outputs

### Per-Frame Timing
- Frame number
- CPU processing time (ms)
- GPU processing time (ms)
- I/O overhead (reading/writing)

### Aggregate Statistics
- Total processing time (CPU vs GPU)
- Average frame processing time
- Throughput (frames per second)
- Acceleration ratio (CPU time / GPU time)

### Export Formats
- CSV: Detailed per-frame timing matrix
- Markdown: Summary tables and comparison analysis

## CPU vs GPU Comparison

The pipeline generates comprehensive benchmarks comparing:
- **CPU Performance**: Single-threaded and multi-threaded baseline
- **GPU Performance**: OpenCL kernel execution with data transfer overhead
- **Acceleration Factor**: Measured speedup of GPU over CPU
- **Efficiency**: Performance per watt (framework prepared, actual measurement depends on hardware)

## Design Principles

1. **Strict 3-file module structure**: Every module has .h, .c, and _test.c files
2. **Clear separation of concerns**: Each module has a single responsibility
3. **FFmpeg library integration**: Native library usage, not CLI commands
4. **OpenCL for GPU computing**: Direct GPU kernel execution
5. **Comprehensive benchmarking**: Frame-level timing with detailed statistics
6. **Extensibility**: Easy to add new filters or processing stages

## Next Steps

1. Implement core module (Frame structure and memory management)
2. Implement video I/O module (FFmpeg integration)
3. Implement CPU and GPU filter modules
4. Implement pipeline orchestration and frame queuing
5. Implement benchmarking and reporting
6. Integration testing and performance validation

## License

To be determined

## Contributing

Development guidelines TBD
