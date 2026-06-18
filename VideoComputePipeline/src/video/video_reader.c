#include "video/video_reader.h"
#include "utils/logger.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>

#include <string.h>

static double rational_to_double(AVRational r) {
    return r.den == 0 ? 0.0 : (double)r.num / (double)r.den;
}

static void log_ffmpeg_error(const char *message, int error_code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = { 0 };
    av_strerror(error_code, buffer, sizeof(buffer));
    log_error /* module: utils/logger */ ("%s: %s", message, buffer);
}
/*typedef struct {
    void *format_ctx;
    void *codec_ctx;
    void *stream;
    void *packet;
    void *decoded_frame;
    void *sws_ctx;
    int video_stream_index;
    int next_frame_index;
    int decoder_flushed;
    VideoInfo info;
} VideoReader;*/
int video_reader_open_with_threads(VideoReader *reader, const char *input_path, int decoder_threads) {
    if (!reader || !input_path) {
        return -1;
    }

    memset(reader, 0, sizeof(*reader));
    reader->video_stream_index = -1;

    AVFormatContext *format_ctx = NULL;
    if (avformat_open_input(&format_ctx, input_path, NULL, NULL) < 0) {
        log_error /* module: utils/logger */ ("failed to open input: %s", input_path);
        return -1;
    }

    if (avformat_find_stream_info(format_ctx, NULL) < 0) {
        avformat_close_input(&format_ctx);
        return -1;
    }

    const int stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (stream_index < 0) {
        avformat_close_input(&format_ctx);
        return -1;
    }

    AVStream *stream = format_ctx->streams[stream_index];
    const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder) {
        avformat_close_input(&format_ctx);
        return -1;
    }

    AVCodecContext *codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx) {
        avformat_close_input(&format_ctx);
        return -1;
    }

    if (avcodec_parameters_to_context(codec_ctx, stream->codecpar) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return -1;
    }

    if (decoder_threads > 0) {
        codec_ctx->thread_count = decoder_threads;
        codec_ctx->thread_type = FF_THREAD_FRAME;
    }

    if (avcodec_open2(codec_ctx, decoder, NULL) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return -1;
    }

    AVPacket *packet = av_packet_alloc();
    AVFrame *decoded = av_frame_alloc();
    if (!packet || !decoded) {
        av_packet_free(&packet);
        av_frame_free(&decoded);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return -1;
    }

    struct SwsContext *sws = sws_getContext(codec_ctx->width,
                                           codec_ctx->height,
                                           codec_ctx->pix_fmt,
                                           codec_ctx->width,
                                           codec_ctx->height,
                                           AV_PIX_FMT_RGB24,
                                           SWS_BILINEAR,
                                           NULL,
                                           NULL,
                                           NULL);
    if (!sws) {
        av_packet_free(&packet);
        av_frame_free(&decoded);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return -1;
    }

    reader->format_ctx = format_ctx;
    reader->codec_ctx = codec_ctx;
    reader->stream = stream;
    reader->packet = packet;
    reader->decoded_frame = decoded;
    reader->sws_ctx = sws;
    reader->video_stream_index = stream_index;
    reader->next_frame_index = 0;
    reader->decoder_flushed = 0;
    reader->info.width = codec_ctx->width;
    reader->info.height = codec_ctx->height;
    reader->info.fps = rational_to_double /* module: video/video_reader */ (av_guess_frame_rate(format_ctx, stream, NULL));
    if (reader->info.fps <= 0.0) {
        reader->info.fps = rational_to_double /* module: video/video_reader */ (stream->avg_frame_rate);
    }
    if (reader->info.fps <= 0.0) {
        reader->info.fps = 30.0;
    }
    reader->info.total_frames = stream->nb_frames;
    reader->info.duration_sec = format_ctx->duration > 0 ? (double)format_ctx->duration / (double)AV_TIME_BASE : 0.0;

    return 0;
}

static enum AVPixelFormat frame_format_to_av_format(FrameFormat format) {
    switch (format) {
        case FRAME_FORMAT_RGB24:
            return AV_PIX_FMT_RGB24;
        case FRAME_FORMAT_NV12:
            return AV_PIX_FMT_NV12;
        case FRAME_FORMAT_GRAY8:
            return AV_PIX_FMT_GRAY8;
        default:
            return AV_PIX_FMT_NONE;
    }
}

static const char *frame_format_name(FrameFormat format) {
    switch (format) {
        case FRAME_FORMAT_RGB24:
            return "RGB24";
        case FRAME_FORMAT_NV12:
            return "NV12";
        case FRAME_FORMAT_GRAY8:
            return "GRAY8";
        default:
            return "unknown";
    }
}

static int receive_frame_as(VideoReader *reader, Frame *out_frame, FrameFormat format) {
    AVCodecContext *codec_ctx = (AVCodecContext *)reader->codec_ctx;
    AVFrame *decoded = (AVFrame *)reader->decoded_frame;
    struct SwsContext *sws = (struct SwsContext *)reader->sws_ctx;
    const enum AVPixelFormat dst_format = frame_format_to_av_format /* module: video/video_reader */ (format);
    if (dst_format == AV_PIX_FMT_NONE) {
        return -1;
    }

    const int receive_result = avcodec_receive_frame(codec_ctx, decoded);
    if (receive_result == AVERROR(EAGAIN)) {
        return 0;
    }
    if (receive_result == AVERROR_EOF) {
        return 2;
    }
    if (receive_result < 0) {
        log_ffmpeg_error /* module: video/video_reader */ ("decoder receive_frame failed", receive_result);
        return -1;
    }

    if (!frame_is_valid /* module: core/frame */ (out_frame) ||
        out_frame->width != codec_ctx->width ||
        out_frame->height != codec_ctx->height ||
        out_frame->format != format) {
        frame_free /* module: core/frame */ (out_frame);
        if (frame_alloc /* module: core/frame */ (out_frame, codec_ctx->width, codec_ctx->height, format) != 0) {
            log_error /* module: utils/logger */ ("failed to allocate decoded %s frame: %dx%d", frame_format_name /* module: video/video_reader */ (format), codec_ctx->width, codec_ctx->height);
            av_frame_unref(decoded);
            return -1;
        }
    }

    out_frame->index = reader->next_frame_index++;
    sws = sws_getCachedContext(sws,
                               codec_ctx->width,
                               codec_ctx->height,
                               codec_ctx->pix_fmt,
                               codec_ctx->width,
                               codec_ctx->height,
                               dst_format,
                               SWS_BILINEAR,
                               NULL,
                               NULL,
                               NULL);
    if (!sws) {
        av_frame_unref(decoded);
        return -1;
    }
    reader->sws_ctx = sws;

    uint8_t *dst_data[4] = { out_frame->planes[0], out_frame->planes[1], out_frame->planes[2], out_frame->planes[3] };
    int dst_linesize[4] = {
        (int)out_frame->linesize[0],
        (int)out_frame->linesize[1],
        (int)out_frame->linesize[2],
        (int)out_frame->linesize[3]
    };
    sws_scale(sws,
              (const uint8_t * const *)decoded->data,
              decoded->linesize,
              0,
              codec_ctx->height,
              dst_data,
              dst_linesize);
    av_frame_unref(decoded);
    return 1;
}

int video_reader_read_frame_as(VideoReader *reader, Frame *out_frame, FrameFormat format) {
    if (!reader || !out_frame || !reader->codec_ctx) {
        return -1;
    }

    int result = receive_frame_as /* module: video/video_reader */ (reader, out_frame, format);
    if (result == 1) {
        return 1;
    }
    if (result == 2) {
        return 0;
    }
    if (result < 0) {
        return -1;
    }

    AVFormatContext *format_ctx = (AVFormatContext *)reader->format_ctx;
    AVCodecContext *codec_ctx = (AVCodecContext *)reader->codec_ctx;
    AVPacket *packet = (AVPacket *)reader->packet;

    for (;;) {
        const int read_result = av_read_frame(format_ctx, packet);
        if (read_result < 0) {
            if (read_result != AVERROR_EOF) {
                log_ffmpeg_error /* module: video/video_reader */ ("av_read_frame failed", read_result);
                return -1;
            }
            break;
        }

        if (packet->stream_index == reader->video_stream_index) {
            for (;;) {
                const int send_result = avcodec_send_packet(codec_ctx, packet);
                if (send_result == AVERROR(EAGAIN)) {
                    result = receive_frame_as /* module: video/video_reader */ (reader, out_frame, format);
                    if (result == 1) {
                        av_packet_unref(packet);
                        return 1;
                    }
                    if (result == 2) {
                        av_packet_unref(packet);
                        return 0;
                    }
                    if (result < 0) {
                        av_packet_unref(packet);
                        return -1;
                    }
                    av_packet_unref(packet);
                    log_error /* module: utils/logger */ ("decoder requested frame draining but no frame was available");
                    return -1;
                }
                av_packet_unref(packet);
                if (send_result < 0) {
                    log_ffmpeg_error /* module: video/video_reader */ ("decoder send_packet failed", send_result);
                    return -1;
                }
                break;
            }

            result = receive_frame_as /* module: video/video_reader */ (reader, out_frame, format);
            if (result == 1) {
                return 1;
            }
            if (result == 2) {
                return 0;
            }
            if (result < 0) {
                return -1;
            }
        } else {
            av_packet_unref(packet);
        }
    }

    if (!reader->decoder_flushed) {
        reader->decoder_flushed = 1;
        const int flush_result = avcodec_send_packet(codec_ctx, NULL);
        if (flush_result < 0 && flush_result != AVERROR_EOF) {
            log_ffmpeg_error /* module: video/video_reader */ ("decoder flush failed", flush_result);
            return -1;
        }
    }

    result = receive_frame_as /* module: video/video_reader */ (reader, out_frame, format);
    if (result == 1) {
        return 1;
    }
    if (result < 0) {
        return -1;
    }
    return 0;
}

int video_reader_read_frame(VideoReader *reader, Frame *out_frame) {
    return video_reader_read_frame_as /* module: video/video_reader */ (reader, out_frame, FRAME_FORMAT_RGB24);
}

void video_reader_close(VideoReader *reader) {
    if (!reader) {
        return;
    }

    AVPacket *packet = (AVPacket *)reader->packet;
    AVFrame *decoded = (AVFrame *)reader->decoded_frame;
    AVCodecContext *codec_ctx = (AVCodecContext *)reader->codec_ctx;
    AVFormatContext *format_ctx = (AVFormatContext *)reader->format_ctx;
    struct SwsContext *sws = (struct SwsContext *)reader->sws_ctx;

    sws_freeContext(sws);
    av_packet_free(&packet);
    av_frame_free(&decoded);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
    memset(reader, 0, sizeof(*reader));
}

const VideoInfo *video_reader_get_info(const VideoReader *reader) {
    return reader ? &reader->info : NULL;
}
