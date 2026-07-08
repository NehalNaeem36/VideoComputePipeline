# Docker Builds

This folder contains Linux container builds for `VideoComputePipeline`.

The containers build Linux binaries. They do not package or run the Windows
`.exe`.

## Images

CPU compile/test image:

```powershell
cd E:\wAI\first_task
docker build -t videocompute:cpu -f VideoComputePipeline\docker\Dockerfile.cpu .
docker run --rm videocompute:cpu --help
```

CUDA image with ONNX Runtime GPU and hardware-video support:

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

The CUDA image installs cuDNN 9 for CUDA 12 through apt and extracts ONNX
Runtime GPU 1.22.0 to:

```text
/opt/onnxruntime-gpu
```

## Enter CUDA Container

NVDEC/NVENC require NVIDIA video driver capabilities:

```powershell
docker run --rm -it --gpus all `
  --entrypoint bash `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -v ${PWD}\VideoComputePipeline\data:/workspace/VideoComputePipeline/data `
  -v ${PWD}\VideoComputePipeline\models:/workspace/VideoComputePipeline/models `
  -v ${PWD}\VideoComputePipeline\benchmarks:/workspace/VideoComputePipeline/benchmarks `
  videocompute:cuda
```

Inside:

```bash
cd /workspace/VideoComputePipeline
LD_LIBRARY_PATH=/opt/onnxruntime-gpu/lib:${LD_LIBRARY_PATH} \
/opt/videocompute/bin/VideoComputePipeline --list-backends

ldconfig -p | grep nvcuvid
nvidia-smi
```

If `nvcuvid` is missing, restart the container with:

```powershell
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

## ONNX Runtime CUDA Detection

```bash
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

Use `/tmp/vcp_out` for benchmark runs. It avoids slow Windows bind-mounted CSV
writes. Copy results back before exiting:

```bash
cp /tmp/vcp_out/detections.csv data/output/detections.csv
cp /tmp/vcp_out/benchmark.csv benchmarks/docker_onnx_benchmark.csv
```

## Annotated Video

```bash
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
  --ffmpeg-log-level error
```

The annotated video appears on the host at:

```text
VideoComputePipeline/data/output/annotated.mkv
```

## Linux UI

Recommended architecture:

```text
Run VideoComputePipelineUI on the Linux host.
Have it launch docker run commands for the backend pipeline.
```

Running the UI inside Docker is possible with X11 forwarding, but it is more
fragile because SDL/OpenGL need display access:

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

Inside:

```bash
apt-get update
apt-get install -y --no-install-recommends libsdl2-dev libgl1-mesa-dev pkg-config cmake ninja-build
rm -rf /var/lib/apt/lists/*

cmake -S /workspace/VideoComputePipelineUI -B /tmp/vcp-ui-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vcp-ui-build -j
/tmp/vcp-ui-build/Release/VideoComputePipelineUI
```

Then revoke X access:

```bash
xhost -local:docker
```
