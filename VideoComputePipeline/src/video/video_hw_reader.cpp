#include "video/video_hw_reader.h"

#include "utils/logger.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
}

#include <cstring>
#include <string>

namespace {

thread_local std::string g_last_error = "no error";

double rational_to_double(AVRational r) {
    return r.den == 0 ? 0.0 : (double)r.num / (double)r.den;
}

void set_av_error(const char *message, int error_code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(error_code, buffer, sizeof(buffer));
    g_last_error = std::string(message) + ": " + buffer;
}

enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    (void)ctx;
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_CUDA) {
            return *p;
        }
    }
    return AV_PIX_FMT_NONE;
}

bool decoder_supports_cuda(const AVCodec *decoder) {
    for (int i = 0;; ++i) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
        if (!config) {
            return false;
        }
        if ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) &&
            config->device_type == AV_HWDEVICE_TYPE_CUDA &&
            config->pix_fmt == AV_PIX_FMT_CUDA) {
            return true;
        }
    }
}

int fill_cuda_frame(VideoHWReader *reader, AVFrame *decoded, CudaNV12Frame *out) {
    if (!reader || !decoded || !out || decoded->format != AV_PIX_FMT_CUDA) {
        g_last_error = "decoded frame is not AV_PIX_FMT_CUDA";
        return -1;
    }

    AVFrame *ref = av_frame_alloc();
    if (!ref || av_frame_ref(ref, decoded) < 0) {
        av_frame_free(&ref);
        g_last_error = "failed to reference decoded CUDA frame";
        return -1;
    }

    cuda_nv12_frame_clear(out);
    out->index = reader->next_frame_index++;
    out->width = decoded->width;
    out->height = decoded->height;
    out->pts = decoded->pts;
    out->dts = decoded->pkt_dts;
    out->timestamp_ms = decoded->pts != AV_NOPTS_VALUE
                            ? rational_to_double(((AVStream *)reader->stream)->time_base) * (double)decoded->pts * 1000.0
                            : 0.0;
    out->d_y = decoded->data[0];
    out->d_uv = decoded->data[1] ? decoded->data[1] : decoded->data[0] + (size_t)decoded->linesize[0] * (size_t)decoded->height;
    out->y_pitch = decoded->linesize[0] > 0 ? (size_t)decoded->linesize[0] : (size_t)decoded->width;
    out->uv_pitch = decoded->linesize[1] > 0 ? (size_t)decoded->linesize[1] : out->y_pitch;
    out->av_frame = ref;
    out->hw_frames_ctx = decoded->hw_frames_ctx;
    out->owns_av_frame = 1;
    out->owns_cuda_memory = 0;
    return cuda_nv12_frame_is_valid(out) ? 1 : -1;
}

int receive_hw_frame(VideoHWReader *reader, CudaNV12Frame *out) {
    AVCodecContext *codec_ctx = (AVCodecContext *)reader->codec_ctx;
    AVFrame *decoded = (AVFrame *)reader->decoded_frame;
    const int receive_result = avcodec_receive_frame(codec_ctx, decoded);
    if (receive_result == AVERROR(EAGAIN)) {
        return 0;
    }
    if (receive_result == AVERROR_EOF) {
        return 2;
    }
    if (receive_result < 0) {
        set_av_error("NVDEC receive_frame failed", receive_result);
        return -1;
    }

    const int fill_result = fill_cuda_frame(reader, decoded, out);
    av_frame_unref(decoded);
    return fill_result;
}

}  // namespace

extern "C" int video_hw_reader_open(VideoHWReader *reader, const char *path, int decoder_threads) {
    if (!reader || !path) {
        g_last_error = "invalid hardware reader open arguments";
        return -1;
    }

    std::memset(reader, 0, sizeof(*reader));
    reader->video_stream_index = -1;

    AVFormatContext *format_ctx = nullptr;
    int result = avformat_open_input(&format_ctx, path, nullptr, nullptr);
    if (result < 0) {
        set_av_error("failed to open input", result);
        return -1;
    }
    result = avformat_find_stream_info(format_ctx, nullptr);
    if (result < 0) {
        avformat_close_input(&format_ctx);
        set_av_error("failed to find stream info", result);
        return -1;
    }

    const int stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
        avformat_close_input(&format_ctx);
        g_last_error = "failed to find video stream";
        return -1;
    }

    AVStream *stream = format_ctx->streams[stream_index];
    const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder || !decoder_supports_cuda(decoder)) {
        avformat_close_input(&format_ctx);
        g_last_error = "selected decoder does not expose CUDA/NVDEC hardware frames";
        return -1;
    }

    AVBufferRef *hw_device_ctx = nullptr;
    result = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT);
    if (result < 0) {
        avformat_close_input(&format_ctx);
        set_av_error("failed to create CUDA hardware device", result);
        return -1;
    }

    AVCodecContext *codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx) {
        av_buffer_unref(&hw_device_ctx);
        avformat_close_input(&format_ctx);
        g_last_error = "failed to allocate decoder context";
        return -1;
    }

    result = avcodec_parameters_to_context(codec_ctx, stream->codecpar);
    if (result < 0) {
        avcodec_free_context(&codec_ctx);
        av_buffer_unref(&hw_device_ctx);
        avformat_close_input(&format_ctx);
        set_av_error("failed to copy codec parameters", result);
        return -1;
    }

    codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    codec_ctx->get_format = get_hw_format;
    if (decoder_threads > 0) {
        codec_ctx->thread_count = decoder_threads;
        codec_ctx->thread_type = FF_THREAD_FRAME;
    }

    result = avcodec_open2(codec_ctx, decoder, nullptr);
    if (result < 0) {
        avcodec_free_context(&codec_ctx);
        av_buffer_unref(&hw_device_ctx);
        avformat_close_input(&format_ctx);
        set_av_error("failed to open NVDEC decoder", result);
        return -1;
    }

    AVPacket *packet = av_packet_alloc();
    AVFrame *decoded = av_frame_alloc();
    if (!packet || !decoded) {
        av_packet_free(&packet);
        av_frame_free(&decoded);
        avcodec_free_context(&codec_ctx);
        av_buffer_unref(&hw_device_ctx);
        avformat_close_input(&format_ctx);
        g_last_error = "failed to allocate decode packet/frame";
        return -1;
    }

    reader->format_ctx = format_ctx;
    reader->codec_ctx = codec_ctx;
    reader->stream = stream;
    reader->packet = packet;
    reader->decoded_frame = decoded;
    reader->hw_device_ctx = hw_device_ctx;
    reader->video_stream_index = stream_index;
    reader->info.width = codec_ctx->width;
    reader->info.height = codec_ctx->height;
    reader->info.fps = rational_to_double(av_guess_frame_rate(format_ctx, stream, nullptr));
    if (reader->info.fps <= 0.0) {
        reader->info.fps = rational_to_double(stream->avg_frame_rate);
    }
    if (reader->info.fps <= 0.0) {
        reader->info.fps = 30.0;
    }
    reader->info.total_frames = stream->nb_frames;
    reader->info.duration_sec = format_ctx->duration > 0 ? (double)format_ctx->duration / (double)AV_TIME_BASE : 0.0;
    return 0;
}

extern "C" int video_hw_reader_read_cuda_nv12(VideoHWReader *reader, CudaNV12Frame *out) {
    if (!reader || !out || !reader->codec_ctx) {
        g_last_error = "invalid hardware reader read arguments";
        return -1;
    }

    int result = receive_hw_frame(reader, out);
    if (result == 1) return 1;
    if (result == 2) return 0;
    if (result < 0) return -1;

    AVFormatContext *format_ctx = (AVFormatContext *)reader->format_ctx;
    AVCodecContext *codec_ctx = (AVCodecContext *)reader->codec_ctx;
    AVPacket *packet = (AVPacket *)reader->packet;

    for (;;) {
        const int read_result = av_read_frame(format_ctx, packet);
        if (read_result < 0) {
            if (read_result != AVERROR_EOF) {
                set_av_error("av_read_frame failed", read_result);
                return -1;
            }
            break;
        }

        if (packet->stream_index == reader->video_stream_index) {
            const int send_result = avcodec_send_packet(codec_ctx, packet);
            av_packet_unref(packet);
            if (send_result < 0) {
                set_av_error("NVDEC send_packet failed", send_result);
                return -1;
            }
            result = receive_hw_frame(reader, out);
            if (result == 1) return 1;
            if (result == 2) return 0;
            if (result < 0) return -1;
        } else {
            av_packet_unref(packet);
        }
    }

    if (!reader->decoder_flushed) {
        reader->decoder_flushed = 1;
        const int flush_result = avcodec_send_packet(codec_ctx, nullptr);
        if (flush_result < 0 && flush_result != AVERROR_EOF) {
            set_av_error("NVDEC flush failed", flush_result);
            return -1;
        }
    }

    result = receive_hw_frame(reader, out);
    if (result == 1) return 1;
    if (result < 0) return -1;
    return 0;
}

extern "C" const VideoInfo *video_hw_reader_get_info(const VideoHWReader *reader) {
    return reader ? &reader->info : nullptr;
}

extern "C" void video_hw_reader_release_frame(VideoHWReader *reader, CudaNV12Frame *frame) {
    (void)reader;
    if (!frame) {
        return;
    }
    AVFrame *av_frame = (AVFrame *)frame->av_frame;
    av_frame_free(&av_frame);
    cuda_nv12_frame_clear(frame);
}

extern "C" void video_hw_reader_close(VideoHWReader *reader) {
    if (!reader) {
        return;
    }
    AVPacket *packet = (AVPacket *)reader->packet;
    AVFrame *decoded = (AVFrame *)reader->decoded_frame;
    AVCodecContext *codec_ctx = (AVCodecContext *)reader->codec_ctx;
    AVFormatContext *format_ctx = (AVFormatContext *)reader->format_ctx;
    AVBufferRef *hw_device_ctx = (AVBufferRef *)reader->hw_device_ctx;
    av_packet_free(&packet);
    av_frame_free(&decoded);
    avcodec_free_context(&codec_ctx);
    av_buffer_unref(&hw_device_ctx);
    avformat_close_input(&format_ctx);
    std::memset(reader, 0, sizeof(*reader));
}

extern "C" const char *video_hw_reader_last_error(void) {
    return g_last_error.c_str();
}
