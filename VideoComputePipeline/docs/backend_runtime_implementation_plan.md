# Backend Runtime Implementation Plan

## Summary

The detection pipeline must remain one shared GPU-resident batched pipeline. The replaceable component is the inference runtime stage, selected by `--runtime`, while decode, CUDA preprocess, model adapter postprocess, overlay, encode, detection CSV, and benchmark writing remain shared.

Supported runtime targets:

- TensorRT for `.engine` and `.plan`
- ONNX Runtime CUDA for `.onnx`
- LibTorch TorchScript CUDA for `.pt`, `.ts`, and `.torchscript`

`.pt` means TorchScript only. Python checkpoints are not loaded or converted at runtime.

## Architecture

- Add shared runtime descriptors in `include/inference/inference_types.h`.
- Add backend discovery and runtime/model validation in `include/inference/backend_registry.h`.
- Keep the existing `inference_engine_*` API as a compatibility wrapper while the runner migrates to backend-native calls.
- Refactor TensorRT behind the new backend model first, preserving current behavior.
- Add a `ModelAdapter` layer for YOLOv5 so runtime backends only execute tensors and never own YOLO decode/NMS logic.
- Keep `FrameBatch`, `VideoProfile`, `HardwareProfile`, and `PipelineExecutionPlan` as the planning inputs for all runtimes.

## Implementation Stages

1. Add runtime/device/model CLI and backend registry diagnostics.
2. Validate model extension against selected runtime before loading video or CUDA resources.
3. Preserve TensorRT `.engine` behavior through the compatibility wrapper.
4. Add shared GPU tensor batch descriptors and move TensorRT I/O toward that representation.
5. Introduce YOLOv5 model adapter and shared CUDA postprocess/NMS.
6. Add optional ONNX Runtime CUDA backend with GPU I/O binding.
7. Add optional LibTorch TorchScript CUDA backend with CUDA tensors and safe ownership.
8. Extend benchmark and model/backend diagnostics for fair backend comparison.

## Validation Requirements

- Existing TensorRT detection and filter modes must keep working.
- TensorRT-only builds must not require ONNX Runtime or LibTorch.
- `--list-backends` must report disabled optional backends clearly.
- `--runtime auto` must select by extension.
- Invalid runtime/model combinations must fail clearly.
- NVDEC/NVENC fast path must report zero raw-frame upload/download.
- Benchmarks must append backend/runtime/model fields without removing existing columns.

