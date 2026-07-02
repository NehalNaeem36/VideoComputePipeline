param(
    [switch]$EnableOnnxRuntime,
    [string]$BuildDir = "",
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    [string]$CudnnRoot = "D:\cuDNN\cudnn-windows-x86_64-9.23.2.1_cuda12-archive",
    [string]$TensorRtRoot = "D:\TensorRT\TensorRT-11.1.0.106",
    [string]$OnnxRuntimeRoot = "E:\wAI\third_party\onnxruntime\Microsoft.ML.OnnxRuntime.Gpu.Windows.1.26.0",
    [string]$FfmpegRoot = "C:\vcpkg\installed\x64-windows"
)

$ErrorActionPreference = "Stop"

if (-not $BuildDir) {
    $BuildDir = if ($EnableOnnxRuntime) { "build-win-cuda12-onnx" } else { "build-win-cuda12" }
}

$onnxValue = if ($EnableOnnxRuntime) { "ON" } else { "OFF" }

$args = @(
    "-S", ".",
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCUDAToolkit_ROOT=$CudaRoot",
    "-DTENSORRT_ROOT=$TensorRtRoot",
    "-DCUDNN_ROOT=$CudnnRoot",
    "-DFFMPEG_ROOT=$FfmpegRoot",
    "-DENABLE_CUDA_INFERENCE=ON",
    "-DENABLE_HW_VIDEO=ON",
    "-DENABLE_ONNXRUNTIME=$onnxValue",
    "-DENABLE_LIBTORCH=OFF"
)

if ($EnableOnnxRuntime) {
    $args += "-DONNXRUNTIME_ROOT=$OnnxRuntimeRoot"
}

Write-Host "Configuring VideoComputePipeline CUDA 12 build..."
Write-Host "  BuildDir: $BuildDir"
Write-Host "  ONNX Runtime: $onnxValue"
Write-Host ""

& cmake @args
