# VideoComputePipelineUI

Windows-first Dear ImGui launcher for `VideoComputePipeline.exe`.

The UI is a standalone C++17 project. It does not link FFmpeg, CUDA, TensorRT, ONNX Runtime, or any pipeline internals. It builds command lines from the current pipeline flags, launches the pipeline as a subprocess, captures stdout/stderr asynchronously, and shows partial progress from the pipeline logs.

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
..\VideoComputePipeline\build-win-cuda12\Release\VideoComputePipeline.exe
..\VideoComputePipeline
```

Use the CUDA 12 pipeline executable for TensorRT, ONNX Runtime, NVDEC, CUDA overlay, and NVENC:

```text
..\VideoComputePipeline\build-win-cuda12\Release\VideoComputePipeline.exe
```

The pipeline build is expected to copy the vcpkg FFmpeg DLLs beside the executable. The UI checks for:

```text
avcodec-62.dll
avformat-62.dll
avutil-60.dll
swscale-9.dll
swresample-6.dll
avdevice-62.dll
avfilter-11.dll
```

This keeps launches independent of global `PATH` and avoids accidentally loading MSYS2 FFmpeg DLLs.

## Presets

- People Detection - CSV Only: CPU decode, ONNX Runtime detection, detections CSV, no output video.
- People Detection - Staged Annotated Video: NVDEC decode, ONNX Runtime detection, CUDA boxes, NVENC MKV output.
- Fast Staged GPU Annotated Video: TensorRT hardware path with schedule batch 2, three in-flight batches, pipeline overlap on, and two inference lanes.
- Safe CPU Detection: CPU decode, FP32, small max-frame smoke default.
- NVDEC/NVENC Stress Test: TensorRT hardware decode/encode, no CPU decoder fallback, staged overlap on, and two requested inference lanes.
- Custom: selected automatically when you edit fields manually.

Detection presets load the labels file into a searchable class list. Leaving the class selection empty detects every class. Selecting one or more labels makes the UI emit `--class-ids`, so CSV output and annotated boxes contain only the selected classes.

## Tabs

- Run Config: presets, paths, runtime, model/labels, class selection, decoder/encoder, thresholds, command preview, Run/Stop.
- Monitor: process status, elapsed time, parsed frame count, FPS, staged topology, schedule/backend batch, in-flight batch pool, inference workers, and output size when available.
- Logs: live stdout/stderr, filtering, highlighting for errors/warnings/TensorRT lines, save/clear.
- Help: short explanations for pipeline modes and common failures.

## Notes

The pipeline currently emits human-readable progress logs. The UI parses those lines and is ready for future `PROGRESS key=value` output if the pipeline adds it later.

Hardware detection uses a staged NVDEC -> inference -> output pipeline. `Schedule batch` is the number of frames per reusable `FrameBatch`. `In-flight batches` is the number of active reusable batch objects, so decode can fill batch N+2 while inference processes N+1 and output/NVENC writes N. `Backend batch` is separate: it is the number of frames the selected runtime can infer in one model call. Static batch-1 models can still benefit from schedule batches and multiple inference workers.

Decoder, inference device, and encoder are independent controls. `Decoder` chooses CPU software decode or NVDEC frame source. `Inference device` chooses CPU or CUDA model execution. `Encoder` is used only when annotated video output is enabled; CSV-only detection does not need an output video writer.

TensorRT uses `.engine` or `.plan` models. ONNX Runtime uses `.onnx` models. TorchScript appears in the runtime selector only for forward compatibility; it requires a pipeline build with `ENABLE_LIBTORCH=ON`.
