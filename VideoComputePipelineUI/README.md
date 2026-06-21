# VideoComputePipelineUI

Windows-first Dear ImGui launcher for `VideoComputePipeline.exe`.

The UI is a standalone C++17 project. It does not link FFmpeg, CUDA, TensorRT, OpenCL, or any pipeline internals. It builds command lines from the current pipeline flags, launches the pipeline as a subprocess, captures stdout/stderr asynchronously, and shows partial progress from the pipeline logs.

## Build

From a Visual Studio Developer Prompt with vcpkg SDL2 installed:

```cmd
cd /d E:\wAI\first_task\VideoComputePipelineUI
cmake -S . -B build-win -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build-win -j 8
```

Expected executable:

```text
build-win\Release\VideoComputePipelineUI.exe
```

## Usage

Run the UI from `E:\wAI\first_task\VideoComputePipelineUI` so the default relative paths resolve:

```text
..\VideoComputePipeline\build-win\bin\VideoComputePipeline.exe
..\VideoComputePipeline
```

If your pipeline binary is in `build-msvc` or `build-msvc-hw`, edit the Pipeline exe field in the Run Config tab.

## Presets

- People Detection - CSV Only: CPU decode, TensorRT detection, detections CSV, no output video.
- People Detection - Annotated Video: NVDEC decode, TensorRT detection, CUDA boxes, NVENC MKV output.
- Fast GPU Annotated Video: same as annotated video, with FP16 selected.
- Safe CPU Detection: CPU decode, FP32, small max-frame smoke default.
- NVDEC/NVENC Stress Test: hardware decode/encode, no CPU decoder fallback.
- Custom: selected automatically when you edit fields manually.

## Tabs

- Run Config: presets, paths, model/labels, decoder/encoder, thresholds, command preview, Run/Stop.
- Monitor: process status, elapsed time, parsed frame count, FPS, output size when available.
- Logs: live stdout/stderr, filtering, highlighting for errors/warnings/TensorRT lines, save/clear.
- Help: short explanations for pipeline modes and common failures.

## Notes

The pipeline currently emits human-readable progress logs. The UI parses those lines and is ready for future `PROGRESS key=value` output if the pipeline adds it later.
