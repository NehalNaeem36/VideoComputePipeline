param(
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    [string]$CudnnRoot = "D:\cuDNN\cudnn-windows-x86_64-9.23.2.1_cuda12-archive",
    [string]$TensorRtRoot = "D:\TensorRT\TensorRT-11.1.0.106",
    [string]$OnnxRuntimeRoot = "E:\wAI\third_party\onnxruntime\Microsoft.ML.OnnxRuntime.Gpu.Windows.1.26.0"
)

$ErrorActionPreference = "Continue"
$failed = 0

function Test-RequiredPath {
    param([string]$Label, [string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Write-Host "[PASS] ${Label}: $Path"
    } else {
        Write-Host "[FAIL] ${Label}: $Path"
        $script:failed = 1
    }
}

function Test-CommandVisible {
    param([string]$Label, [string]$Command)
    $resolved = Get-Command $Command -ErrorAction SilentlyContinue
    if ($resolved) {
        Write-Host "[PASS] ${Label}: $($resolved.Source)"
    } else {
        Write-Host "[FAIL] ${Label}: $Command not found in PATH"
        $script:failed = 1
    }
}

function Test-WhereVisible {
    param([string]$Label, [string]$Pattern)
    $matches = & where.exe $Pattern 2>$null
    if ($LASTEXITCODE -eq 0 -and $matches) {
        Write-Host "[PASS] ${Label}:"
        $matches | ForEach-Object { Write-Host "       $_" }
        return $matches
    }
    Write-Host "[FAIL] ${Label}: $Pattern not found in PATH"
    $script:failed = 1
    return @()
}

Write-Host "Verifying Windows CUDA 12 stack..."
Write-Host ""

Test-RequiredPath "CUDA root" $CudaRoot
Test-RequiredPath "CUDA nvcc" (Join-Path $CudaRoot "bin\nvcc.exe")
Test-RequiredPath "CUDA runtime" (Join-Path $CudaRoot "bin\cudart64_12.dll")

Test-RequiredPath "cuDNN root" $CudnnRoot
Test-RequiredPath "cuDNN include" (Join-Path $CudnnRoot "include")
Test-RequiredPath "cuDNN lib" (Join-Path $CudnnRoot "lib\x64")
Test-RequiredPath "cuDNN runtime" (Join-Path $CudnnRoot "bin\x64\cudnn64_9.dll")

Test-RequiredPath "TensorRT root" $TensorRtRoot
Test-RequiredPath "TensorRT NvInfer.h" (Join-Path $TensorRtRoot "include\NvInfer.h")
Test-RequiredPath "TensorRT NvOnnxParser.h" (Join-Path $TensorRtRoot "include\NvOnnxParser.h")
Test-RequiredPath "TensorRT nvinfer_11.lib" (Join-Path $TensorRtRoot "lib\nvinfer_11.lib")
Test-RequiredPath "TensorRT nvinfer_11.dll" (Join-Path $TensorRtRoot "bin\nvinfer_11.dll")
Test-RequiredPath "TensorRT nvinfer_plugin_11.dll" (Join-Path $TensorRtRoot "bin\nvinfer_plugin_11.dll")
Test-RequiredPath "TensorRT nvonnxparser_11.dll" (Join-Path $TensorRtRoot "bin\nvonnxparser_11.dll")

$ortNative = Join-Path $OnnxRuntimeRoot "runtimes\win-x64\native"
Test-RequiredPath "ONNX Runtime root" $OnnxRuntimeRoot
Test-RequiredPath "ONNX Runtime include" (Join-Path $OnnxRuntimeRoot "buildTransitive\native\include\onnxruntime_cxx_api.h")
Test-RequiredPath "ONNX Runtime lib" (Join-Path $ortNative "onnxruntime.lib")
Test-RequiredPath "ONNX Runtime DLL" (Join-Path $ortNative "onnxruntime.dll")
Test-RequiredPath "ONNX Runtime CUDA provider" (Join-Path $ortNative "onnxruntime_providers_cuda.dll")
Test-RequiredPath "ONNX Runtime shared provider" (Join-Path $ortNative "onnxruntime_providers_shared.dll")

Write-Host ""
Test-CommandVisible "nvcc" "nvcc.exe"
Test-CommandVisible "trtexec" "trtexec.exe"
Test-WhereVisible "cudart64_12.dll in PATH" "cudart64_12.dll" | Out-Null
Test-WhereVisible "cuDNN DLLs in PATH" "cudnn*.dll" | Out-Null
Test-WhereVisible "TensorRT DLLs in PATH" "nvinfer*.dll" | Out-Null
$onnxMatches = Test-WhereVisible "ONNX Runtime DLL in PATH" "onnxruntime.dll"
Test-WhereVisible "ONNX Runtime CUDA provider in PATH" "onnxruntime_providers_cuda.dll" | Out-Null

if ($onnxMatches) {
    $firstOnnx = ($onnxMatches | Select-Object -First 1)
    if ($firstOnnx -like "C:\Windows\System32\*") {
        Write-Host "[WARN] System32 onnxruntime.dll resolves before the intended ONNX Runtime root."
        Write-Host "       Put $ortNative before System32-sensitive app launch paths or rely on post-build DLL copy."
    }
}

Write-Host ""
Write-Host "Environment variables:"
Write-Host "  CUDA_PATH=$env:CUDA_PATH"
Write-Host "  CUDNN_ROOT=$env:CUDNN_ROOT"
Write-Host "  TENSORRT_ROOT=$env:TENSORRT_ROOT"
Write-Host "  ONNXRUNTIME_ROOT=$env:ONNXRUNTIME_ROOT"

if ($failed -ne 0) {
    Write-Host ""
    Write-Host "CUDA 12 stack verification failed."
    exit 1
}

Write-Host ""
Write-Host "CUDA 12 stack verification passed."
