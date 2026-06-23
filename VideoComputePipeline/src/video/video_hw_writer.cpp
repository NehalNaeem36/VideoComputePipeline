/*
 * Hardware video writer module: sends GPU-resident NV12 frames to FFmpeg/NVENC
 * and muxes annotated output video. It is used only by the hardware detection
 * path after CUDA overlay has modified the frame in place.
 */
#include "video/video_hw_writer.h"

#include "utils/file_utils.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}

#include <cstring>
#include <ctime>
#include <string>

namespace {

thread_local std::string g_last_error = "no error";

void set_av_error(const char *message, int error_code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(error_code, buffer, sizeof(buffer));
    g_last_error = std::string(message) + ": " + buffer;
}

AVRational fps_to_time_base(double fps) {
    if (fps <= 0.0) {
        fps = 30.0;
    }
    AVRational rate = av_d2q(fps, 100000);
    return av_inv_q(rate);
}

int write_available_packets(VideoHWWriter *writer, FrameTiming *timing) {
    AVFormatContext *format_ctx = (AVFormatContext *)writer->format_ctx;
    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    AVStream *stream = (AVStream *)writer->stream;
    AVPacket *packet = (AVPacket *)writer->packet;

    for (;;) {
        const int receive_result = avcodec_receive_packet(codec_ctx, packet);
        if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
            return 0;
        }
        if (receive_result < 0) {
            set_av_error("NVENC receive_packet failed", receive_result);
            return -1;
        }

        av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
        packet->stream_index = stream->index;
        const clock_t start = clock();
        const int write_result = av_interleaved_write_frame(format_ctx, packet);
        const clock_t end = clock();
        if (timing) {
            timing->mux_write_ms += ((double)(end - start) * 1000.0) / (double)CLOCKS_PER_SEC;
        }
        if (write_result < 0) {
            av_packet_unref(packet);
            set_av_error("hardware mux write failed", write_result);
            return -1;
        }
        writer->written_packets++;
        av_packet_unref(packet);
    }
}

}  // namespace

extern "C" int video_hw_writer_open(VideoHWWriter *writer,
                                     const char *output_path,
                                     const VideoInfo *input_info,
                                     const char *encoder_name,
                                     int lossless) {
    if (!writer || !output_path || !input_info || input_info->width <= 0 || input_info->height <= 0) {
        g_last_error = "invalid hardware writer open arguments";
        return -1;
    }
    if (!encoder_name || encoder_name[0] == '\0') {
        encoder_name = "h264_nvenc";
    }
    if (std::strcmp(encoder_name, "h264_nvenc") != 0 && std::strcmp(encoder_name, "hevc_nvenc") != 0) {
        g_last_error = "hardware writer supports h264_nvenc or hevc_nvenc";
        return -1;
    }
    if (create_parent_directory_if_missing /* module: utils/file_utils */ (output_path) != 0) {
        g_last_error = "failed to create output directory";
        return -1;
    }

    std::memset(writer, 0, sizeof(*writer));
    std::snprintf(writer->encoder_name, sizeof(writer->encoder_name), "%s", encoder_name);

    AVFormatContext *format_ctx = nullptr;
    int result = avformat_alloc_output_context2(&format_ctx, nullptr, nullptr, output_path);
    if (result < 0 || !format_ctx) {
        set_av_error("failed to allocate hardware output context", result);
        return -1;
    }

    const AVCodec *encoder = avcodec_find_encoder_by_name(encoder_name);
    if (!encoder) {
        avformat_free_context(format_ctx);
        g_last_error = "requested NVENC encoder is not available";
        return -1;
    }

    AVStream *stream = avformat_new_stream(format_ctx, nullptr);
    AVCodecContext *codec_ctx = avcodec_alloc_context3(encoder);
    AVPacket *packet = av_packet_alloc();
    AVBufferRef *hw_device_ctx = nullptr;
    AVBufferRef *hw_frames_ctx = nullptr;
    if (!stream || !codec_ctx || !packet) {
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        g_last_error = "failed to allocate hardware writer objects";
        return -1;
    }

    result = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT);
    if (result < 0) {
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        set_av_error("failed to create CUDA encode device", result);
        return -1;
    }

    hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ctx) {
        av_buffer_unref(&hw_device_ctx);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        g_last_error = "failed to allocate CUDA encode frame context";
        return -1;
    }
    {
        AVHWFramesContext *frames_ctx = (AVHWFramesContext *)hw_frames_ctx->data;
        frames_ctx->format = AV_PIX_FMT_CUDA;
        frames_ctx->sw_format = AV_PIX_FMT_NV12;
        frames_ctx->width = input_info->width;
        frames_ctx->height = input_info->height;
        frames_ctx->initial_pool_size = 8;
    }
    result = av_hwframe_ctx_init(hw_frames_ctx);
    if (result < 0) {
        av_buffer_unref(&hw_frames_ctx);
        av_buffer_unref(&hw_device_ctx);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        set_av_error("failed to initialize CUDA encode frame context", result);
        return -1;
    }

    codec_ctx->codec_id = encoder->id;
    codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx->width = input_info->width;
    codec_ctx->height = input_info->height;
    codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
    codec_ctx->sw_pix_fmt = AV_PIX_FMT_NV12;
    codec_ctx->time_base = fps_to_time_base(input_info->fps);
    codec_ctx->framerate = av_inv_q(codec_ctx->time_base);
    codec_ctx->bit_rate = lossless ? 1000000000LL : 12000000LL;
    codec_ctx->gop_size = 30;
    codec_ctx->max_b_frames = 0;
    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    codec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ctx);
    if (!codec_ctx->hw_device_ctx || !codec_ctx->hw_frames_ctx) {
        av_buffer_unref(&hw_frames_ctx);
        av_buffer_unref(&hw_device_ctx);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        g_last_error = "failed to reference CUDA encode contexts";
        return -1;
    }
    if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    av_opt_set(codec_ctx->priv_data, "preset", "p1", 0);
    av_opt_set(codec_ctx->priv_data, "tune", lossless ? "lossless" : "ull", 0);
    av_opt_set(codec_ctx->priv_data, "rc", lossless ? "constqp" : "cbr", 0);
    av_opt_set(codec_ctx->priv_data, "zerolatency", "1", 0);
    av_opt_set(codec_ctx->priv_data, "delay", "0", 0);
    av_opt_set(codec_ctx->priv_data, "bf", "0", 0);
    if (lossless) {
        av_opt_set(codec_ctx->priv_data, "qp", "0", 0);
    }

    result = avcodec_open2(codec_ctx, encoder, nullptr);
    if (result < 0 || avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0) {
        av_buffer_unref(&hw_frames_ctx);
        av_buffer_unref(&hw_device_ctx);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        set_av_error("failed to open NVENC encoder", result);
        return -1;
    }

    stream->time_base = codec_ctx->time_base;
    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        result = avio_open(&format_ctx->pb, output_path, AVIO_FLAG_WRITE);
        if (result < 0) {
            av_buffer_unref(&hw_frames_ctx);
            av_buffer_unref(&hw_device_ctx);
            av_packet_free(&packet);
            avcodec_free_context(&codec_ctx);
            avformat_free_context(format_ctx);
            set_av_error("failed to open hardware output file", result);
            return -1;
        }
    }

    result = avformat_write_header(format_ctx, nullptr);
    if (result < 0) {
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx->pb);
        }
        av_buffer_unref(&hw_frames_ctx);
        av_buffer_unref(&hw_device_ctx);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        set_av_error("failed to write hardware output header", result);
        return -1;
    }

    writer->format_ctx = format_ctx;
    writer->codec_ctx = codec_ctx;
    writer->stream = stream;
    writer->packet = packet;
    writer->hw_device_ctx = hw_device_ctx;
    writer->hw_frames_ctx = hw_frames_ctx;
    writer->width = input_info->width;
    writer->height = input_info->height;
    writer->fps = input_info->fps;
    writer->is_open = 1;
    return 0;
}

extern "C" int video_hw_writer_write_cuda_nv12(VideoHWWriter *writer, const CudaNV12Frame *frame, FrameTiming *timing) {
    if (!writer || !writer->is_open || !frame || !frame->av_frame) {
        g_last_error = "invalid hardware writer frame";
        return -1;
    }

    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    AVFrame *src = (AVFrame *)frame->av_frame;
    AVFrame *send_frame = av_frame_alloc();
    if (!send_frame) {
        av_frame_free(&send_frame);
        g_last_error = "failed to allocate CUDA frame for encode";
        return -1;
    }

    int result = 0;
    if (src->hw_frames_ctx && codec_ctx->hw_frames_ctx && src->hw_frames_ctx->data == codec_ctx->hw_frames_ctx->data) {
        result = av_frame_ref(send_frame, src);
    } else {
        send_frame->format = AV_PIX_FMT_CUDA;
        send_frame->width = codec_ctx->width;
        send_frame->height = codec_ctx->height;
        result = av_hwframe_get_buffer(codec_ctx->hw_frames_ctx, send_frame, 0);
        if (result >= 0) {
            result = av_hwframe_transfer_data(send_frame, src, 0);
        }
    }
    if (result < 0) {
        av_frame_free(&send_frame);
        set_av_error("failed to prepare CUDA frame for encode", result);
        return -1;
    }
    send_frame->pts = writer->next_pts++;

    const clock_t start = clock();
    result = avcodec_send_frame(codec_ctx, send_frame);
    av_frame_free(&send_frame);
    if (result < 0) {
        set_av_error("NVENC send_frame failed", result);
        return -1;
    }
    writer->submitted_frames++;
    result = write_available_packets(writer, timing);
    const clock_t end = clock();
    if (timing) {
        timing->encode_ms += ((double)(end - start) * 1000.0) / (double)CLOCKS_PER_SEC;
    }
    return result;
}

extern "C" int video_hw_writer_flush(VideoHWWriter *writer) {
    if (!writer || !writer->is_open) {
        return 0;
    }
    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    const int result = avcodec_send_frame(codec_ctx, nullptr);
    if (result < 0 && result != AVERROR_EOF) {
        set_av_error("NVENC flush failed", result);
        return -1;
    }
    if (write_available_packets(writer, nullptr) != 0) {
        return -1;
    }
    if (writer->submitted_frames > 0 && writer->written_packets == 0) {
        g_last_error = "NVENC flush completed but no encoded packets were written";
        return -1;
    }
    return 0;
}

extern "C" void video_hw_writer_close(VideoHWWriter *writer) {
    if (!writer) {
        return;
    }
    AVFormatContext *format_ctx = (AVFormatContext *)writer->format_ctx;
    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    AVPacket *packet = (AVPacket *)writer->packet;
    AVBufferRef *hw_frames_ctx = (AVBufferRef *)writer->hw_frames_ctx;
    AVBufferRef *hw_device_ctx = (AVBufferRef *)writer->hw_device_ctx;
    if (writer->is_open && format_ctx) {
        av_write_trailer(format_ctx);
    }
    if (format_ctx && !(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&format_ctx->pb);
    }
    av_buffer_unref(&hw_frames_ctx);
    av_buffer_unref(&hw_device_ctx);
    av_packet_free(&packet);
    avcodec_free_context(&codec_ctx);
    avformat_free_context(format_ctx);
    std::memset(writer, 0, sizeof(*writer));
}

extern "C" const char *video_hw_writer_last_error(void) {
    return g_last_error.c_str();
}
