# VideoComputePipeline Architecture Diagrams

Branch documented: `feature/cuda-yolo-nv12-inference`

This document is written as codebase documentation for a mostly C/CUDA/C++ modular project. The diagrams are therefore **module, data-structure, memory-flow, and sequence diagrams**, rather than pure object-oriented class diagrams.

Grounding note: these diagrams are based on the branch README and public source files visible on GitHub, especially `pipeline_runner.c`, `pipeline_config.h`, `frame.h`, `cuda_frame.h`, `inference_engine.h`, `video_reader.h`, `video_writer.h`, `video_hw_reader.h`, `video_hw_writer.h`, `cuda_overlay.h`, `benchmark.h`, and `CMakeLists.txt`.

---

## 1. Project-Level Component Diagram

```mermaid
flowchart TB
    CLI["main.c / CLI arguments"] --> Config["PipelineConfig\ninclude/pipeline/pipeline_config.h\nsrc/pipeline/pipeline_config.c"]
    Config --> Runner["Pipeline Runner\nsrc/pipeline/pipeline_runner.c"]

    Runner --> FilterPath["Filter task\nPIPELINE_TASK_FILTER"]
    Runner --> DetectPath["Detection task\nPIPELINE_TASK_DETECT"]

    subgraph CoreFrame["Core frame and memory structures"]
        Frame["Frame\nRGB24 / GRAY8 / NV12\ninclude/core/frame.h"]
        CudaFrame["CudaNV12Frame\nGPU NV12 frame wrapper\ninclude/gpu/cuda_frame.h"]
        Pool["FramePool / FrameQueue / FrameSlot\nsrc/pipeline/frame_pool.c\nsrc/pipeline/frame_queue.c\nsrc/pipeline/frame_slot.c"]
    end

    subgraph VideoCPU["CPU video I/O"]
        Reader["VideoReader\nFFmpeg CPU decode\ninclude/video/video_reader.h"]
        Writer["VideoWriter\nFFmpeg CPU frame encode\ninclude/video/video_writer.h"]
    end

    subgraph VideoHW["Hardware video I/O"]
        HWReader["VideoHWReader\nNVDEC/CUDA frame decode\ninclude/video/video_hw_reader.h"]
        HWWriter["VideoHWWriter\nNVENC/CUDA frame encode\ninclude/video/video_hw_writer.h"]
    end

    subgraph Processing["Processing engines"]
        CPUFilters["CPU filters\nsrc/cpu/cpu_filters.c"]
        OpenCLFilters["OpenCL filters\nsrc/gpu/gpu_filters.c\nkernels/*.cl"]
        TRT["InferenceEngine\nTensorRT backend or stub\ninclude/inference/inference_engine.h"]
        Overlay["CUDA Overlay\nNV12 box drawing\ninclude/gpu/cuda_overlay.h"]
    end

    subgraph Output["Output and reporting"]
        DetectionResult["DetectionResult\ninclude/inference/detection_result.h"]
        DetectionWriter["DetectionWriter\nCSV output\ninclude/inference/detection_writer.h"]
        Benchmark["Benchmark / FrameTiming\ninclude/benchmark/benchmark.h"]
        MatrixReport["Matrix report\nsrc/benchmark/matrix_report.c"]
    end

    FilterPath --> Reader
    FilterPath --> Pool
    FilterPath --> CPUFilters
    FilterPath --> OpenCLFilters
    FilterPath --> Writer
    FilterPath --> Benchmark

    DetectPath --> Reader
    DetectPath --> HWReader
    DetectPath --> TRT
    DetectPath --> Overlay
    DetectPath --> HWWriter
    DetectPath --> DetectionResult
    DetectPath --> DetectionWriter
    DetectPath --> Benchmark

    Reader --> Frame
    Writer --> Frame
    HWReader --> CudaFrame
    HWWriter --> CudaFrame
    TRT --> DetectionResult
    Overlay --> CudaFrame
```

### Explanation

The codebase has two main runtime tasks:

- `PIPELINE_TASK_FILTER`: the original benchmark pipeline for CPU/OpenCL filters and encoded video output.
- `PIPELINE_TASK_DETECT`: the TensorRT detection path, with both CPU-decoded NV12 fallback and an experimental NVDEC/CUDA/NVENC hardware path.

The most important architectural split is between **CPU frame flow** using `Frame` and **GPU-resident hardware video flow** using `CudaNV12Frame`.

---

## 2. Build Option / Feature Flag Diagram

```mermaid
flowchart TD
    Start["CMake configure"] --> Base["Base C11 project\nFFmpeg + OpenCL + tests"]

    Base --> CUDAChoice{"ENABLE_CUDA_INFERENCE?"}

    CUDAChoice -- OFF --> StubInference["Use inference_engine_stub.c\nNo CUDA/TensorRT backend"]
    StubInference --> HWOffForced["Hardware video must be OFF"]

    CUDAChoice -- ON --> EnableCXXCUDA["Enable CXX + CUDA language\nC++17 / CUDA17"]
    EnableCXXCUDA --> FindCUDA["find_package(CUDAToolkit REQUIRED)"]
    FindCUDA --> FindTRT["Find TensorRT headers/libs\nNvInfer.h / nvinfer"]
    FindTRT --> RealInference["Build inference_engine_tensorrt.cpp\nyolo_postprocess.cpp\ncuda_preprocess.cu"]

    RealInference --> HWChoice{"ENABLE_HW_VIDEO?"}
    HWChoice -- OFF --> HWStubs["Use cuda_overlay_stub.c\nvideo_hw_reader_stub.c\nvideo_hw_writer_stub.c"]
    HWChoice -- ON --> RealHW["Build cuda_overlay.cu\nvideo_hw_reader.cpp\nvideo_hw_writer.cpp"]

    Base --> FFmpeg["Find FFmpeg\nlibavformat / libavcodec / libavutil / libswscale"]
    Base --> OpenCL["Find OpenCL"]

    RealInference --> LinkCUDA["Link CUDA::cudart + TensorRT"]
    RealHW --> DefineHW["Define VCP_ENABLE_HW_VIDEO=1"]
    RealInference --> DefineCUDA["Define VCP_ENABLE_CUDA_INFERENCE=1"]

    StubInference --> App["VideoComputePipeline executable"]
    HWStubs --> App
    RealHW --> App
    DefineCUDA --> App
    DefineHW --> App
```

### Explanation

`ENABLE_CUDA_INFERENCE` controls whether the real TensorRT/CUDA inference backend is built. `ENABLE_HW_VIDEO` controls whether the real NVDEC/NVENC hardware-video files are built. Hardware video requires CUDA inference to be enabled. If hardware video is disabled, stub implementations are built for overlay, hardware reader, and hardware writer.

---

## 3. CLI Argument to PipelineConfig Diagram

```mermaid
flowchart TB
    Args["argv from main.c"] --> Parse["pipeline_config_parse_args"]
    Default["pipeline_config_default"] --> Config["PipelineConfig"]
    Parse --> Config

    subgraph TaskFields["Task selection"]
        Task["task\nPIPELINE_TASK_FILTER or PIPELINE_TASK_DETECT"]
        Mode["mode\nPROCESS_CPU or PROCESS_GPU"]
        Filter["filter\ngrayscale / blur3x3 / blur5x5 / blur9x9 / blur13x13"]
    end

    subgraph IOFields["Input/output paths"]
        Input["input_path"]
        Output["output_path"]
        BenchmarkPath["benchmark_path"]
        DetectionsPath["detections_path"]
    end

    subgraph DetectionFields["Detection fields"]
        Model["model_path"]
        Labels["labels_path"]
        Backend["inference_backend"]
        Precision["inference_precision"]
        Conf["confidence_threshold"]
        IOU["iou_threshold"]
        InputSize["inference_input_size"]
        MaxDet["max_detections_per_frame"]
        ClassCount["detection_class_count"]
    end

    subgraph HardwareFields["Hardware video / overlay fields"]
        Decoder["decoder_mode\nVIDEO_DECODER_CPU or VIDEO_DECODER_NVDEC"]
        Fallback["decoder_fallback\nDECODER_FALLBACK_CPU or NONE"]
        Draw["draw_boxes"]
        Thick["box_thickness"]
        BoxConf["box_confidence"]
        OutFmt["output_format\nAUTO / MP4 / MKV"]
    end

    subgraph RuntimeFields["Runtime and memory fields"]
        MaxFrames["max_frames"]
        FrameSlots["frame_slots"]
        DecThreads["decoder_threads"]
        EncThreads["encoder_threads"]
        Workers["processor_workers"]
        MemProfile["memory_profile"]
        MemBudget["memory_budget_mb"]
        Encoder["encoder_name"]
        Lossless["lossless_output"]
        BenchEnable["enable_benchmark"]
    end

    Config --> TaskFields
    Config --> IOFields
    Config --> DetectionFields
    Config --> HardwareFields
    Config --> RuntimeFields

    Config --> Runner["run_pipeline(config)"]
    Runner --> Branch{"config->task"}
    Branch -- FILTER --> FilterRun["run_filter_pipeline(config)"]
    Branch -- DETECT --> DetectRun["run_detection_pipeline(config)"]
```

### Explanation

`PipelineConfig` is the central control object. It carries both the old filter-pipeline controls and the newer detection/hardware-video controls. The detection path depends on model, labels, detection CSV, thresholds, input size, decoder mode, draw-box settings, and optional output video settings.

---

## 4. Source-Level Pipeline Selection Diagram

```mermaid
flowchart TD
    Run["run_pipeline(config)"] --> TaskCheck{"config->task"}

    TaskCheck -- "PIPELINE_TASK_FILTER" --> Filter["run_filter_pipeline(config)"]
    TaskCheck -- "PIPELINE_TASK_DETECT" --> Detect["run_detection_pipeline(config)"]

    Detect --> DecoderCheck{"config->decoder_mode"}
    DecoderCheck -- "VIDEO_DECODER_CPU" --> DetectCPU["run_detection_pipeline_cpu(config)"]
    DecoderCheck -- "VIDEO_DECODER_NVDEC" --> DetectNVDEC["run_detection_pipeline_nvdec(config)"]

    DetectCPU --> CPUFlow["CPU FFmpeg decode as NV12\nTensorRT inference\nDetection CSV\nBenchmark CSV"]
    DetectNVDEC --> HWFlow["NVDEC CUDA/NV12 decode\nTensorRT device inference\nDetection CSV\nOptional CUDA overlay + NVENC output\nBenchmark CSV"]
```

### Explanation

The source declares separate static functions for the filter pipeline, detection dispatcher, CPU detection pipeline, and NVDEC detection pipeline. This is a clean split: the new hardware path does not have to disturb the old CPU/OpenCL filter path.

---

## 5. CPU/OpenCL Filter Pipeline Data Flow

```mermaid
flowchart LR
    Input["Input video file"] --> Reader["VideoReader\nvideo_reader_open_with_threads"]
    Reader --> Decode["FFmpeg CPU decode"]
    Decode --> RGBFrame["Frame\nformat usually RGB24"]

    RGBFrame --> RawPool["Raw frame pool"]
    RawPool --> RawQueue["Raw queue"]
    RawQueue --> Processor{"config->mode"}

    Processor -- "PROCESS_CPU" --> CPUFilter["CPU filter\ngrayscale / blur"]
    Processor -- "PROCESS_GPU" --> OpenCLUpload["OpenCL upload"]
    OpenCLUpload --> OpenCLKernel["OpenCL filter kernel"]
    OpenCLKernel --> OpenCLDownload["OpenCL download"]

    CPUFilter --> ProcessedFrame["Processed Frame"]
    OpenCLDownload --> ProcessedFrame

    ProcessedFrame --> ProcessedPool["Processed frame pool"]
    ProcessedPool --> ProcessedQueue["Processed queue"]
    ProcessedQueue --> OrderedWriter["Ordered encoder stage"]
    OrderedWriter --> Writer["VideoWriter\nvideo_writer_write_frame"]
    Writer --> Encoded["Encoded packets"]
    Encoded --> Output["Output MP4/MKV"]

    OrderedWriter --> Timing["FrameTiming"]
    Timing --> Benchmark["Benchmark CSV"]
```

### Explanation

The original filter path uses CPU-side `Frame` objects and bounded pools/queues. GPU mode here means the frame is uploaded to OpenCL for filtering and downloaded back to CPU memory before the regular FFmpeg writer encodes it. This is separate from the newer CUDA/NVDEC/NVENC hardware path.

---

## 6. CPU-Decoded TensorRT Detection Flow

```mermaid
sequenceDiagram
    participant Runner as run_detection_pipeline_cpu
    participant Reader as VideoReader
    participant Frame as Frame CPU NV12
    participant Engine as InferenceEngine TensorRT
    participant Result as DetectionResult
    participant DetCSV as DetectionWriter
    participant Bench as Benchmark

    Runner->>Reader: video_reader_open_with_threads(input_path, decoder_threads)
    Runner->>DetCSV: detection_writer_open(detections_path, labels_path)
    Runner->>Bench: benchmark_open_csv(benchmark_path)
    Runner->>Engine: inference_engine_create(InferenceConfig)

    loop until EOF or max_frames
        Runner->>Reader: video_reader_read_frame_as(..., FRAME_FORMAT_NV12)
        Reader-->>Frame: CPU NV12 Frame
        Runner->>Result: detection_result_clear()
        Runner->>Engine: inference_engine_run_nv12(Frame, DetectionResult, FrameTiming)
        Engine-->>Result: detection boxes/classes/confidences
        Runner->>Result: assign timestamp_ms
        Runner->>DetCSV: detection_writer_write_frame(Result)
        Runner->>Bench: benchmark_add_frame_result(FrameTiming)
    end

    Runner->>Bench: benchmark_set_wall_clock_ms()
    Runner->>Bench: benchmark_close_csv()
    Runner->>Engine: inference_engine_destroy()
    Runner->>Reader: video_reader_close()
```

### Explanation

This is the safer fallback detection path. It decodes to a CPU `Frame` in NV12 format, then calls `inference_engine_run_nv12`. The TensorRT backend may perform CUDA upload and preprocessing internally. It writes detections and timings, but this path does **not** encode annotated video.

---

## 7. NVDEC + TensorRT + CUDA Overlay + NVENC Hardware Flow

```mermaid
sequenceDiagram
    participant Runner as run_detection_pipeline_nvdec
    participant HWReader as VideoHWReader
    participant CudaFrame as CudaNV12Frame
    participant Engine as InferenceEngine
    participant Result as DetectionResult
    participant DetCSV as DetectionWriter
    participant Overlay as CUDA Overlay
    participant HWWriter as VideoHWWriter
    participant Bench as Benchmark

    Runner->>HWReader: video_hw_reader_open(input_path, decoder_threads)
    Runner->>DetCSV: detection_writer_open(detections_path, labels_path)
    Runner->>Bench: benchmark_open_csv(benchmark_path)
    Runner->>Engine: inference_engine_create(InferenceConfig)

    alt draw_boxes enabled
        Runner->>HWWriter: video_hw_writer_open(output_path, input_info, encoder_name, lossless)
    end

    loop until EOF or max_frames
        Runner->>HWReader: video_hw_reader_read_cuda_nv12(&frame)
        HWReader-->>CudaFrame: GPU-resident NV12 frame
        Runner->>Result: detection_result_clear()
        Runner->>Engine: inference_engine_run_cuda_nv12(CudaNV12Frame, DetectionResult, FrameTiming)
        Engine-->>Result: detections
        Runner->>Result: assign timestamp_ms
        Runner->>DetCSV: detection_writer_write_frame(Result)

        alt draw_boxes enabled
            Runner->>Overlay: cuda_overlay_draw_nv12_boxes(CudaNV12Frame, DetectionResult, thickness, box_confidence, class_filter=-1)
            Overlay-->>CudaFrame: annotated GPU NV12 frame
            Runner->>HWWriter: video_hw_writer_write_cuda_nv12(CudaNV12Frame, FrameTiming)
            HWWriter-->>Runner: encoded packet timing in encode_ms / mux_write_ms
        end

        Runner->>Bench: benchmark_add_frame_result(FrameTiming)
        Runner->>HWReader: video_hw_reader_release_frame(&frame)
    end

    alt writer opened
        Runner->>HWWriter: video_hw_writer_flush()
        Runner->>HWWriter: video_hw_writer_close()
    end

    Runner->>Bench: benchmark_set_wall_clock_ms()
    Runner->>Bench: benchmark_close_csv()
    Runner->>Engine: inference_engine_destroy()
    Runner->>HWReader: video_hw_reader_close()
```

### Explanation

This is the high-performance path. A CUDA/NV12 frame is decoded by the hardware reader, passed directly into `inference_engine_run_cuda_nv12`, optionally annotated by `cuda_overlay_draw_nv12_boxes`, and optionally encoded by `video_hw_writer_write_cuda_nv12`. The goal of this path is to avoid full-frame CPU/GPU copies.

---

## 8. CPU Memory vs GPU Memory Movement Diagram

```mermaid
flowchart TB
    subgraph CPU["CPU RAM / Host side"]
        FilePackets["Compressed input packets"]
        Control["Pipeline control\nconfig, loops, timestamps"]
        DetMeta["Detection metadata\nclass, confidence, boxes"]
        CSV["detections.csv\nbenchmark.csv"]
        EncodedPackets["Compressed output packets"]
    end

    subgraph GPU["GPU VRAM / CUDA side"]
        NVDECSurface["NVDEC decoded CUDA surface"]
        CudaNV12["CudaNV12Frame\nd_y + d_uv"]
        Tensor["TensorRT input tensor\nNCHW"]
        TRTExec["TensorRT engine execution"]
        OverlayFrame["Annotated NV12 CUDA frame"]
        NVENCFrame["NVENC input frame"]
    end

    FilePackets --> NVDECSurface
    NVDECSurface --> CudaNV12
    CudaNV12 --> Tensor
    Tensor --> TRTExec
    TRTExec --> DetMeta
    DetMeta --> CSV
    DetMeta --> OverlayFrame
    CudaNV12 --> OverlayFrame
    OverlayFrame --> NVENCFrame
    NVENCFrame --> EncodedPackets
    EncodedPackets --> OutputFile["Output MKV/MP4"]
    Control --> GPU

    classDef host fill:#eef,stroke:#335
    classDef device fill:#efe,stroke:#363
```

### Explanation

The final hardware path should keep raw decoded frames in GPU memory. The CPU still controls the pipeline, writes CSV, and handles compressed file I/O, but should not receive full raw 4K frames unless a fallback/debug path is used.

---

## 9. Key Data Structures Diagram

```mermaid
classDiagram
    class PipelineConfig {
        char input_path[]
        char output_path[]
        char benchmark_path[]
        char detections_path[]
        char model_path[]
        char labels_path[]
        char inference_backend[]
        char inference_precision[]
        char encoder_name[]
        PipelineTask task
        ProcessMode mode
        FilterType filter
        VideoDecoderMode decoder_mode
        DecoderFallbackMode decoder_fallback
        OutputFormat output_format
        int draw_boxes
        int box_thickness
        float box_confidence
        float confidence_threshold
        float iou_threshold
        int inference_input_size
        int max_frames
    }

    class Frame {
        int index
        int width
        int height
        int channels
        FrameFormat format
        size_t stride
        size_t size
        uint8_t* data
        uint8_t* planes[4]
        size_t linesize[4]
    }

    class CudaNV12Frame {
        int index
        int width
        int height
        int64_t pts
        int64_t dts
        double timestamp_ms
        uint8_t* d_y
        uint8_t* d_uv
        size_t y_pitch
        size_t uv_pitch
        void* cuda_stream
        void* av_frame
        void* hw_frames_ctx
        int owns_av_frame
        int owns_cuda_memory
    }

    class InferenceConfig {
        char model_path[]
        char labels_path[]
        int input_width
        int input_height
        int class_count
        float confidence_threshold
        float iou_threshold
        int use_fp16
    }

    class FrameTiming {
        int frame_index
        double decode_ms
        double process_ms
        double upload_ms
        double kernel_ms
        double download_ms
        double encode_ms
        double total_ms
        double preprocess_ms
        double inference_ms
        double postprocess_ms
        double overlay_ms
        double mux_write_ms
    }

    class Benchmark {
        FrameTiming* items
        size_t count
        size_t capacity
        double wall_clock_ms
        void* csv_file
    }

    PipelineConfig --> InferenceConfig : creates for detection
    FrameTiming --> Benchmark : accumulated into
    Frame --> "CPU VideoReader/VideoWriter" : used by
    CudaNV12Frame --> "VideoHWReader/VideoHWWriter" : used by
    CudaNV12Frame --> "InferenceEngine/CUDA Overlay" : consumed by
```

### Explanation

`Frame` is the CPU-side frame representation. `CudaNV12Frame` is the GPU-resident frame representation for the hardware path. `PipelineConfig` selects the task, decoder, encoder, detection thresholds, and overlay behavior. `FrameTiming` records per-frame timing, and `Benchmark` stores/writes timing data.

---

## 10. CPU Frame Format Diagram

```mermaid
flowchart LR
    Frame["Frame"] --> Fields["index, width, height, channels, format"]
    Frame --> Memory["data pointer\nplanes[4]\nlinesize[4]"]
    Frame --> Formats{"FrameFormat"}

    Formats --> RGB["FRAME_FORMAT_RGB24\nused by filter pipeline writer"]
    Formats --> Gray["FRAME_FORMAT_GRAY8\nused by grayscale filter outputs"]
    Formats --> NV12["FRAME_FORMAT_NV12\nused by CPU-decoded detection"]

    NV12 --> YPlane["planes[0] Y"]
    NV12 --> UVPlane["planes[1] interleaved UV"]
```

### Explanation

The branch added/uses `FRAME_FORMAT_NV12`, which lets the CPU detection fallback read decoded video in NV12 form before TensorRT preprocessing. The regular filter output path still works through CPU-side frames and the regular `VideoWriter`.

---

## 11. CUDA NV12 Frame Ownership Diagram

```mermaid
flowchart TD
    Init["cuda_nv12_frame_init"] --> Empty["Empty CudaNV12Frame"]
    HWRead["video_hw_reader_read_cuda_nv12"] --> Fill["Fill index, size, pts/dts, timestamp, d_y, d_uv, pitches, av_frame, hw_frames_ctx"]
    Fill --> Ownership{"Ownership flags"}

    Ownership --> OwnAV["owns_av_frame\nreader/release must unref/free AVFrame if owned"]
    Ownership --> OwnCuda["owns_cuda_memory\nclear/release may free CUDA allocation if owned"]
    Ownership --> Borrowed["If not owned, pointer lifetime tied to FFmpeg hardware frame"]

    Fill --> Use1["inference_engine_run_cuda_nv12 reads frame"]
    Use1 --> Use2["cuda_overlay_draw_nv12_boxes may modify frame"]
    Use2 --> Use3["video_hw_writer_write_cuda_nv12 consumes frame"]
    Use3 --> Release["video_hw_reader_release_frame"]
    Release --> Clear["cuda_nv12_frame_clear"]
```

### Explanation

The hardware path depends on clear frame lifetime rules. The decoded CUDA frame must remain valid until inference, overlay, and encoding are done. In the current sequential hardware loop, the frame is released after detection writing, optional overlay, optional NVENC encoding, and benchmark logging.

---

## 12. Inference Engine API Diagram

```mermaid
flowchart TB
    Config["InferenceConfig"] --> Create["inference_engine_create"]
    Create --> Engine["InferenceEngine opaque handle"]

    Engine --> CPUAPI["inference_engine_run_nv12\ninput: const Frame* CPU NV12"]
    Engine --> CUDAAPI["inference_engine_run_cuda_nv12\ninput: const CudaNV12Frame* GPU NV12"]

    CPUAPI --> Upload["Upload CPU NV12 to CUDA\nrecord upload_ms"]
    Upload --> Preprocess1["CUDA preprocess NV12 -> TensorRT input"]
    CUDAAPI --> Preprocess2["CUDA preprocess GPU NV12 -> TensorRT input\nno full-frame upload"]

    Preprocess1 --> TRT["TensorRT enqueue/inference"]
    Preprocess2 --> TRT
    TRT --> RawOutput["Raw YOLO output tensor"]
    RawOutput --> Post["YOLO postprocess / NMS"]
    Post --> Result["DetectionResult"]
    Post --> Timing["preprocess_ms / inference_ms / postprocess_ms / download_ms"]

    Engine --> Destroy["inference_engine_destroy"]
```

### Explanation

The public API exposes two inference paths: a CPU-NV12 path and a CUDA-NV12 path. The CUDA path is the important one for NVDEC integration because it avoids re-uploading the full frame.

---

## 13. Detection CSV Flow Diagram

```mermaid
flowchart LR
    Result["DetectionResult"] --> Items["Detection items\nframe index, class id/name, confidence, bbox"]
    Timestamp["timestamp_ms assigned by pipeline runner"] --> Items
    Labels["labels_path\ncoco.names / custom.names"] --> Writer["DetectionWriter"]
    Items --> Writer
    Writer --> CSV["detections.csv"]

    CSV --> Columns["Typical columns:\nframe_index\ntimestamp_ms\nclass_id\nclass_name\nconfidence\nx1,y1,x2,y2"]
```

### Explanation

Both detection paths write detection metadata through `DetectionWriter`. The hardware path still writes CSV even when annotated video is disabled.

---

## 14. Benchmark Timing Flow Diagram

```mermaid
flowchart TB
    FrameLoop["Per-frame pipeline loop"] --> Timing["FrameTiming"]

    subgraph TimingFields["FrameTiming fields"]
        Decode["decode_ms"]
        Upload["upload_ms"]
        Kernel["kernel_ms"]
        Preprocess["preprocess_ms"]
        Infer["inference_ms"]
        Post["postprocess_ms"]
        Download["download_ms"]
        Overlay["overlay_ms"]
        Encode["encode_ms"]
        Mux["mux_write_ms"]
        Process["process_ms"]
        Total["total_ms"]
    end

    Timing --> Decode
    Timing --> Upload
    Timing --> Kernel
    Timing --> Preprocess
    Timing --> Infer
    Timing --> Post
    Timing --> Download
    Timing --> Overlay
    Timing --> Encode
    Timing --> Mux
    Timing --> Process
    Timing --> Total

    Timing --> Add["benchmark_add_frame_result"]
    Add --> Bench["Benchmark accumulator"]
    Bench --> CSV["benchmark.csv\nstreamed rows"]
    Bench --> Summary["benchmark_print_summary\nbenchmark_print_detection_summary"]
    Wall["wall_timer"] --> Bench
```

### Explanation

The branch extends timing beyond the original decode/process/encode timing. Detection and hardware-video timing fields include `preprocess_ms`, `inference_ms`, `postprocess_ms`, `overlay_ms`, and `mux_write_ms`. In the NVDEC path, upload/download should be zero or not used unless internal implementation requires small transfers.

---

## 15. Hardware Detection Frame Loop Diagram

```mermaid
flowchart TD
    StartLoop["start frame loop"] --> Limit{"max_frames reached?"}
    Limit -- yes --> Finish["finish loop"]
    Limit -- no --> DecodeTimer["timer_start(decode_timer)"]
    DecodeTimer --> ReadHW["video_hw_reader_read_cuda_nv12(&reader, &frame)"]
    ReadHW --> ReadResult{"read_result"}

    ReadResult -- "0 EOF" --> Finish
    ReadResult -- "<0 error" --> Error["log_error and cleanup"]
    ReadResult -- ">0 frame" --> SetID["frame.index = frame_id\ntiming.frame_index = frame_id"]

    SetID --> ClearResult["detection_result_clear"]
    ClearResult --> Infer["inference_engine_run_cuda_nv12(engine, &frame, &detection_result, &timing)"]
    Infer --> InferOK{"inference OK?"}
    InferOK -- no --> ReleaseErr["video_hw_reader_release_frame\ncleanup"]
    InferOK -- yes --> Timestamp["assign timestamp_ms to detections"]

    Timestamp --> WriteDet["detection_writer_write_frame"]
    WriteDet --> DrawCheck{"draw_boxes?"}

    DrawCheck -- no --> TimingCalc["process_ms = preprocess + inference + download + postprocess + overlay\ntotal_ms = decode + process + encode + mux"]
    DrawCheck -- yes --> Overlay["cuda_overlay_draw_nv12_boxes"]
    Overlay --> OverlayOK{"overlay OK?"}
    OverlayOK -- no --> ReleaseErr
    OverlayOK -- yes --> Encode["video_hw_writer_write_cuda_nv12"]
    Encode --> EncodeOK{"encode OK?"}
    EncodeOK -- no --> ReleaseErr
    EncodeOK -- yes --> TimingCalc

    TimingCalc --> BenchCheck{"enable_benchmark?"}
    BenchCheck -- yes --> BenchAdd["benchmark_add_frame_result"]
    BenchCheck -- no --> Release["video_hw_reader_release_frame"]
    BenchAdd --> Release
    Release --> Inc["frame_id++"]
    Inc --> Progress{"progress interval?"}
    Progress -- yes --> LogFPS["log wall_clock_fps"]
    Progress -- no --> Limit
    LogFPS --> Limit

    Finish --> Flush{"writer_opened?"}
    Flush -- yes --> WriterFlush["video_hw_writer_flush"]
    Flush -- no --> CloseBench["benchmark close/summary"]
    WriterFlush --> CloseBench
    CloseBench --> Cleanup["destroy engine, free result, close reader/writer"]
```

### Explanation

This diagram expands the exact hardware detection loop. The code reads a CUDA/NV12 frame, runs device inference, writes detections, optionally draws boxes and encodes, updates benchmark timing, then releases the hardware frame.

---

## 16. CUDA Overlay Flow Diagram

```mermaid
flowchart LR
    Frame["CudaNV12Frame\nd_y / d_uv / pitches"] --> OverlayAPI["cuda_overlay_draw_nv12_boxes"]
    Dets["DetectionResult"] --> Filter["filter detections\nconfidence >= min_confidence\nclass_filter or all classes"]
    Filter --> Clamp["clamp x1,y1,x2,y2 to frame bounds"]
    Clamp --> Kernel["CUDA kernel draws box borders"]
    OverlayAPI --> Kernel
    Kernel --> YPlane["Modify Y plane\nwhite box first implementation"]
    Kernel --> Timing["overlay_ms"]
    YPlane --> Annotated["Annotated GPU NV12 frame"]
    Annotated --> HWWriter["video_hw_writer_write_cuda_nv12"]
```

### Explanation

The overlay API receives a GPU-resident NV12 frame and detection results. The initial practical approach is to draw simple visible boxes by modifying the Y plane. Color overlays can be added later by modifying UV as well.

---

## 17. Video Reader / Writer Module Diagram

```mermaid
flowchart TB
    subgraph CPUReaderWriter["CPU FFmpeg path"]
        VR["VideoReader"] --> VROpen["video_reader_open_with_threads"]
        VR --> VRRead["video_reader_read_frame / video_reader_read_frame_as"]
        VRRead --> CPUFrame["Frame"]
        CPUFrame --> VWWrite["video_writer_write_frame"]
        VW["VideoWriter"] --> VWOpen["video_writer_open_with_options"]
        VW --> VWWrite
        VW --> VWFlush["video_writer_flush"]
        VW --> VWClose["video_writer_close"]
    end

    subgraph HWReaderWriter["CUDA hardware video path"]
        HWR["VideoHWReader"] --> HWROpen["video_hw_reader_open"]
        HWR --> HWRRead["video_hw_reader_read_cuda_nv12"]
        HWRRead --> GPUFrame["CudaNV12Frame"]
        GPUFrame --> HWWWrite["video_hw_writer_write_cuda_nv12"]
        HWW["VideoHWWriter"] --> HWWOpen["video_hw_writer_open"]
        HWW --> HWWWrite
        HWW --> HWWFlush["video_hw_writer_flush"]
        HWW --> HWWClose["video_hw_writer_close"]
    end
```

### Explanation

The repo now contains separate CPU and hardware video reader/writer APIs. The CPU path uses `Frame`. The hardware path uses `CudaNV12Frame`.

---

## 18. Fallback and Capability Matrix

```mermaid
flowchart TD
    DetectTask["--task detect"] --> Decoder{"--decoder"}

    Decoder -- "cpu" --> CPUDet["CPU decoder detection path"]
    Decoder -- "nvdec" --> HWTry["Try NVDEC hardware path"]

    HWTry --> HWBuild{"Built with ENABLE_HW_VIDEO=ON?"}
    HWBuild -- no --> HWStubFail["hardware-video stub returns error"]
    HWBuild -- yes --> HWRuntime{"NVDEC/NVENC available at runtime?"}

    HWRuntime -- yes --> HWPath["NVDEC -> TensorRT -> optional overlay -> optional NVENC"]
    HWRuntime -- no --> Fallback{"decoder_fallback"}

    Fallback -- "cpu" --> CPUDet
    Fallback -- "none" --> Fail["return error"]

    CPUDet --> CSVOnly["detections.csv + benchmark.csv"]
    HWPath --> Draw{"--draw-boxes and --output?"}
    Draw -- yes --> Annotated["annotated video + detections.csv + benchmark.csv"]
    Draw -- no --> HWCSVOnly["GPU decode/inference CSV-only"]
```

### Explanation

The intended behavior is to keep CPU detection as fallback while hardware-video support remains optional. The exact runtime fallback behavior should be verified during local testing, but the config includes both decoder mode and decoder fallback fields.

---

## 19. Output Modes Diagram

```mermaid
flowchart LR
    Task["Task"] --> FilterTask["filter"]
    Task --> DetectTask["detect"]

    FilterTask --> FilterOutput["Encoded filtered video\nvia VideoWriter"]
    FilterTask --> FilterBench["Benchmark CSV"]

    DetectTask --> CPUDetect["CPU decoder detection"]
    CPUDetect --> CPUDetCSV["detections.csv"]
    CPUDetect --> CPUDetBench["benchmark.csv"]
    CPUDetect -. no annotated video .-> NoVid["No output video in CPU detection path"]

    DetectTask --> HWDetect["NVDEC detection"]
    HWDetect --> HWCSV["detections.csv"]
    HWDetect --> HWBench["benchmark.csv"]
    HWDetect --> DrawBoxes{"draw_boxes?"}
    DrawBoxes -- yes --> HWVideo["Annotated MKV/MP4\nvia VideoHWWriter/NVENC"]
    DrawBoxes -- no --> HWNoVideo["CSV-only hardware detection"]
```

### Explanation

The README currently contains a small wording contradiction: it documents experimental annotated detection with an output file, while also saying detection mode skips video encoding. The factual distinction is: CPU/CSV-only detection skips video encoding; hardware detection can produce annotated video when `--draw-boxes` and output writer options are used.

---

## 20. Suggested README Architecture Section Diagram

```mermaid
flowchart TB
    README["README Architecture Section"] --> Overview["Project overview"]
    README --> Build["Build matrix"]
    README --> Pipelines["Pipeline diagrams"]
    README --> Commands["Run commands"]
    README --> Notes["Known limitations"]

    Build --> BaseBuild["Base: FFmpeg + OpenCL"]
    Build --> CUDABuild["CUDA/TensorRT: ENABLE_CUDA_INFERENCE=ON"]
    Build --> HWBuild["Hardware video: ENABLE_HW_VIDEO=ON"]

    Pipelines --> FilterDiagram["CPU/OpenCL filter"]
    Pipelines --> CPUDetectDiagram["CPU-decoded detection"]
    Pipelines --> HWDetectDiagram["NVDEC -> TensorRT -> overlay -> NVENC"]

    Notes --> MP4Note["MP4 may not be convenient for live viewing while writing"]
    Notes --> MKVNote["MKV is better for stream-style observation"]
    Notes --> HWNote["Hardware path is experimental and requires matching CUDA/TensorRT/FFmpeg/NVIDIA support"]
```

### Explanation

For repo documentation, these diagrams should probably live in `docs/architecture_diagrams.md`, with a shorter overview copied into `README.md`.

---

## 21. File-to-Responsibility Map

```mermaid
flowchart LR
    subgraph ConfigFiles["Configuration"]
        PCfgH["include/pipeline/pipeline_config.h"]
        PCfgC["src/pipeline/pipeline_config.c"]
    end

    subgraph PipelineFiles["Pipeline orchestration"]
        Runner["src/pipeline/pipeline_runner.c"]
        Pool["src/pipeline/frame_pool.c"]
        Queue["src/pipeline/frame_queue.c"]
        Slot["src/pipeline/frame_slot.c"]
    end

    subgraph CoreFiles["Frame memory"]
        FrameH["include/core/frame.h"]
        FrameC["src/core/frame.c"]
        CudaFrameH["include/gpu/cuda_frame.h"]
        CudaFrameC["src/gpu/cuda_frame.c"]
    end

    subgraph VideoFiles["Video I/O"]
        ReaderH["include/video/video_reader.h"]
        ReaderC["src/video/video_reader.c"]
        WriterH["include/video/video_writer.h"]
        WriterC["src/video/video_writer.c"]
        HWReaderH["include/video/video_hw_reader.h"]
        HWReaderC["src/video/video_hw_reader.cpp"]
        HWWriterH["include/video/video_hw_writer.h"]
        HWWriterC["src/video/video_hw_writer.cpp"]
    end

    subgraph ProcessingFiles["Processing"]
        CPUF["src/cpu/cpu_filters.c"]
        GPF["src/gpu/gpu_filters.c"]
        OCL["src/gpu/opencl_context.c / opencl_program.c"]
        InferH["include/inference/inference_engine.h"]
        InferCPP["src/inference/inference_engine_tensorrt.cpp"]
        PreCU["src/inference/cuda_preprocess.cu"]
        PostCPP["src/inference/yolo_postprocess.cpp"]
        OverlayH["include/gpu/cuda_overlay.h"]
        OverlayCU["src/gpu/cuda_overlay.cu"]
    end

    subgraph OutputFiles["CSV and benchmark"]
        DetRH["include/inference/detection_result.h"]
        DetRC["src/inference/detection_result.c"]
        DetWH["include/inference/detection_writer.h"]
        DetWC["src/inference/detection_writer.c"]
        BenchH["include/benchmark/benchmark.h"]
        BenchC["src/benchmark/benchmark.c"]
    end

    PCfgH --> Runner
    PCfgC --> Runner
    Runner --> CoreFiles
    Runner --> VideoFiles
    Runner --> ProcessingFiles
    Runner --> OutputFiles
```

### Explanation

This map is useful for maintenance and code review. It shows which modules are responsible for configuration, orchestration, frame memory, video I/O, filtering/inference/overlay, and reporting.

---

## 22. Recommended Integration View

```mermaid
flowchart TD
    Main["main branch\noriginal filter benchmark"] --> Merge["merge feature/cuda-yolo-nv12-inference"]
    Feature["feature branch\nCUDA/TensorRT + NVDEC/NVENC detection"] --> Merge

    Merge --> Tests["Run test matrix"]

    Tests --> Build1["Base build\nENABLE_CUDA_INFERENCE=OFF"]
    Tests --> Build2["CUDA build\nENABLE_CUDA_INFERENCE=ON"]
    Tests --> Build3["Hardware build\nENABLE_CUDA_INFERENCE=ON\nENABLE_HW_VIDEO=ON"]

    Tests --> Smoke1["CPU filter smoke test"]
    Tests --> Smoke2["OpenCL GPU filter smoke test"]
    Tests --> Smoke3["CPU detection CSV-only smoke test"]
    Tests --> Smoke4["NVDEC + CUDA overlay + NVENC annotated smoke test"]

    Smoke1 --> Ready["Merge ready"]
    Smoke2 --> Ready
    Smoke3 --> Ready
    Smoke4 --> Ready
```

### Explanation

The branch is large enough that merging should be treated as a feature release. The most important thing is to preserve the old CPU/OpenCL filter path while making CUDA/TensorRT and hardware video optional.

---

# Notes and Known Documentation Issue

The README documents both:

1. `Detection-only CUDA/TensorRT YOLO path: MP4 -> NV12 -> detections CSV`
2. `Experimental GPU-resident detection path: NVDEC -> TensorRT -> CUDA box overlay -> NVENC`

Near the bottom, it also says detection mode skips video encoding. That should be clarified as:

> CSV-only detection mode skips video encoding. Hardware annotated detection can create output video when `--decoder nvdec`, `--draw-boxes`, `--output`, and an NVENC encoder are used.

---
