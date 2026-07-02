/*
 * Video writer module: encodes CPU RGB24 frames to output video through FFmpeg.
 * The filter pipeline uses it after CPU/CUDA processing; hardware detection
 * uses the separate NVENC GPU-frame writer instead.
 */
#include "video/video_writer.h"
#include "utils/file_utils.h"
#include "utils/logger.h"

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>

#include <stdint.h>
#include <string.h>

static int is_h264_encoder(const char *name) {
    return name &&
           (strcmp(name, "libx264") == 0 ||
            strcmp(name, "libx264rgb") == 0 ||
            strcmp(name, "h264_nvenc") == 0);
}

static int is_rgb_encoder(const char *name) {
    return name && strcmp(name, "libx264rgb") == 0;
}

static int is_nvenc_encoder(const char *name) {
    return name && strcmp(name, "h264_nvenc") == 0;
}

static AVRational fps_to_time_base(double fps) {
    if (fps <= 0.0) {
        fps = 30.0;
    }

    AVRational rate = av_d2q(fps, 100000);
    return av_inv_q(rate);
}

static int write_available_packets(VideoWriter *writer) {
    AVFormatContext *format_ctx = (AVFormatContext *)writer->format_ctx;
    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    AVStream *stream = (AVStream *)writer->stream;
    AVPacket *packet = (AVPacket *)writer->packet;

    for (;;) {
        const int result = avcodec_receive_packet(codec_ctx, packet);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
            return 0;
        }
        if (result < 0) {
            return -1;
        }

        av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
        packet->stream_index = stream->index;
        const int write_result = av_interleaved_write_frame(format_ctx, packet);
        av_packet_unref(packet);
        if (write_result < 0) {
            return -1;
        }
    }
}

int video_writer_open_with_options(VideoWriter *writer, const char *output_path, int width, int height, double fps, int encoder_threads, const char *encoder_name, int lossless) {
    if (!writer || !output_path || width <= 0 || height <= 0) {
        return -1;
    }

    if (!encoder_name || encoder_name[0] == '\0') {
        encoder_name = "libx264";
    }
    if (lossless && strcmp(encoder_name, "libx264") == 0) {
        encoder_name = "libx264rgb";
    }
    if (lossless && strcmp(encoder_name, "mpeg4") == 0) {
        log_error /* module: utils/logger */ ("mpeg4 output is not supported for lossless mode; use libx264 or h264_nvenc");
        return -1;
    }
    if (lossless && is_nvenc_encoder /* module: video/video_writer */ (encoder_name)) {
        log_warn /* module: utils/logger */ ("h264_nvenc lossless uses YUV420P to avoid CUDA filter/NVENC 4:4:4 resource pressure; use --encoder libx264 for RGB lossless output");
    }

    memset(writer, 0, sizeof(*writer));
    if (create_parent_directory_if_missing /* module: utils/file_utils */ (output_path) != 0) {
        return -1;
    }

    AVFormatContext *format_ctx = NULL;
    if (avformat_alloc_output_context2(&format_ctx, NULL, NULL, output_path) < 0 || !format_ctx) {
        return -1;
    }

    const AVCodec *encoder = avcodec_find_encoder_by_name(encoder_name);
    if (!encoder && strcmp(encoder_name, "libx264") == 0) {
        log_warn /* module: utils/logger */ ("libx264 encoder is not available; falling back to mpeg4 for CPU-frame output");
        encoder_name = "mpeg4";
        encoder = avcodec_find_encoder_by_name(encoder_name);
        if (!encoder) {
            encoder = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
        }
    }
    if (!encoder) {
        log_error /* module: utils/logger */ ("requested encoder is not available: %s", encoder_name);
        avformat_free_context(format_ctx);
        return -1;
    }

    if (strcmp(encoder_name, "mpeg4") == 0 && encoder->id != AV_CODEC_ID_MPEG4) {
        encoder = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
    }
    if (!encoder) {
        avformat_free_context(format_ctx);
        return -1;
    }

    AVStream *stream = avformat_new_stream(format_ctx, NULL);
    AVCodecContext *codec_ctx = avcodec_alloc_context3(encoder);
    AVPacket *packet = av_packet_alloc();
    AVFrame *rgb = av_frame_alloc();
    AVFrame *yuv = av_frame_alloc();
    if (!stream || !codec_ctx || !packet || !rgb || !yuv) {
        av_packet_free(&packet);
        av_frame_free(&rgb);
        av_frame_free(&yuv);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        return -1;
    }

    codec_ctx->codec_id = encoder->id;
    codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx->width = width;
    codec_ctx->height = height;
    enum AVPixelFormat output_pix_fmt = AV_PIX_FMT_YUV420P;
    if (is_rgb_encoder /* module: video/video_writer */ (encoder_name)) {
        output_pix_fmt = AV_PIX_FMT_RGB24;
    } else if (lossless && is_nvenc_encoder /* module: video/video_writer */ (encoder_name)) {
        output_pix_fmt = AV_PIX_FMT_YUV420P;
    }

    const int nvenc_lossless = lossless && is_nvenc_encoder /* module: video/video_writer */ (encoder_name);

    codec_ctx->pix_fmt = output_pix_fmt;
    codec_ctx->time_base = fps_to_time_base /* module: video/video_writer */ (fps);
    codec_ctx->framerate = av_inv_q(codec_ctx->time_base);
    codec_ctx->bit_rate = nvenc_lossless ? 1000000000LL : (lossless ? 0 : 4000000);
    if (nvenc_lossless) {
        codec_ctx->rc_max_rate = 1000000000LL;
        codec_ctx->rc_buffer_size = 1000000000;
    }
    codec_ctx->gop_size = 30;
    codec_ctx->max_b_frames = 0;
    codec_ctx->thread_count = encoder_threads > 0 ? encoder_threads : 4;
    codec_ctx->thread_type = FF_THREAD_SLICE;

    if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (is_nvenc_encoder /* module: video/video_writer */ (encoder_name)) {
        av_opt_set(codec_ctx->priv_data, "preset", "p1", 0);
        av_opt_set(codec_ctx->priv_data, "tune", lossless ? "lossless" : "ull", 0);
        av_opt_set(codec_ctx->priv_data, "rc", lossless ? "constqp" : "cbr", 0);
        av_opt_set(codec_ctx->priv_data, "zerolatency", "1", 0);
        av_opt_set(codec_ctx->priv_data, "delay", "0", 0);
        av_opt_set(codec_ctx->priv_data, "bf", "0", 0);
        if (lossless) {
            av_opt_set(codec_ctx->priv_data, "qp", "0", 0);
            av_opt_set(codec_ctx->priv_data, "init_qpI", "0", 0);
            av_opt_set(codec_ctx->priv_data, "init_qpP", "0", 0);
            av_opt_set(codec_ctx->priv_data, "init_qpB", "0", 0);
            av_opt_set(codec_ctx->priv_data, "rgb_mode", "yuv444", 0);
            av_opt_set(codec_ctx->priv_data, "surfaces", "4", 0);
        }
    } else if (is_h264_encoder /* module: video/video_writer */ (encoder_name) || encoder->id == AV_CODEC_ID_H264) {
        av_opt_set(codec_ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(codec_ctx->priv_data, "tune", "zerolatency", 0);
        av_opt_set(codec_ctx->priv_data, "rc-lookahead", "0", 0);
        av_opt_set(codec_ctx->priv_data, "sync-lookahead", "0", 0);
        av_opt_set(codec_ctx->priv_data, "bframes", "0", 0);
        if (lossless) {
            av_opt_set(codec_ctx->priv_data, "qp", "0", 0);
            av_opt_set(codec_ctx->priv_data, "crf", "0", 0);
        }
    }

    if (avcodec_open2(codec_ctx, encoder, NULL) < 0 ||
        avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0) {
        av_packet_free(&packet);
        av_frame_free(&rgb);
        av_frame_free(&yuv);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        return -1;
    }

    stream->time_base = codec_ctx->time_base;

    yuv->format = codec_ctx->pix_fmt;
    yuv->width = width;
    yuv->height = height;
    if (av_frame_get_buffer(yuv, 32) < 0) {
        av_packet_free(&packet);
        av_frame_free(&rgb);
        av_frame_free(&yuv);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        return -1;
    }

    struct SwsContext *sws = NULL;
    if (codec_ctx->pix_fmt != AV_PIX_FMT_RGB24) {
        sws = sws_getContext(width,
                             height,
                             AV_PIX_FMT_RGB24,
                             width,
                             height,
                             codec_ctx->pix_fmt,
                             SWS_BILINEAR,
                             NULL,
                             NULL,
                             NULL);
        if (!sws) {
            av_packet_free(&packet);
            av_frame_free(&rgb);
            av_frame_free(&yuv);
            avcodec_free_context(&codec_ctx);
            avformat_free_context(format_ctx);
            return -1;
        }
    }

    if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&format_ctx->pb, output_path, AVIO_FLAG_WRITE) < 0) {
            sws_freeContext(sws);
            av_packet_free(&packet);
            av_frame_free(&rgb);
            av_frame_free(&yuv);
            avcodec_free_context(&codec_ctx);
            avformat_free_context(format_ctx);
            return -1;
        }
    }

    if (avformat_write_header(format_ctx, NULL) < 0) {
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx->pb);
        }
        sws_freeContext(sws);
        av_packet_free(&packet);
        av_frame_free(&rgb);
        av_frame_free(&yuv);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        return -1;
    }

    writer->format_ctx = format_ctx;
    writer->codec_ctx = codec_ctx;
    writer->stream = stream;
    writer->packet = packet;
    writer->rgb_frame = rgb;
    writer->yuv_frame = yuv;
    writer->sws_ctx = sws;
    writer->width = width;
    writer->height = height;
    writer->fps = fps;
    writer->next_pts = 0;
    writer->is_open = 1;
    snprintf(writer->encoder_name, sizeof(writer->encoder_name), "%s", encoder_name);
    return 0;
}

int video_writer_write_frame(VideoWriter *writer, const Frame *frame) {
    if (!writer || !writer->is_open || !frame_is_valid /* module: core/frame */ (frame) || frame->format != FRAME_FORMAT_RGB24) {
        return -1;
    }

    if (frame->width != writer->width || frame->height != writer->height) {
        return -1;
    }

    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    AVFrame *yuv = (AVFrame *)writer->yuv_frame;
    struct SwsContext *sws = (struct SwsContext *)writer->sws_ctx;

    if (av_frame_make_writable(yuv) < 0) {
        return -1;
    }

    const uint8_t *src_data[4] = { frame->data, NULL, NULL, NULL };
    int src_linesize[4] = { (int)frame->stride, 0, 0, 0 };
    if (codec_ctx->pix_fmt == AV_PIX_FMT_RGB24) {
        const int row_bytes = writer->width * 3;
        for (int y = 0; y < writer->height; ++y) {
            memcpy(yuv->data[0] + y * yuv->linesize[0], frame->data + (size_t)y * frame->stride, (size_t)row_bytes);
        }
    } else {
        sws_scale(sws, src_data, src_linesize, 0, writer->height, yuv->data, yuv->linesize);
    }
    yuv->pts = writer->next_pts++;

    if (avcodec_send_frame(codec_ctx, yuv) < 0) {
        return -1;
    }

    return write_available_packets(writer);
}

int video_writer_flush(VideoWriter *writer) {
    if (!writer || !writer->is_open) {
        return -1;
    }

    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    if (avcodec_send_frame(codec_ctx, NULL) < 0) {
        return -1;
    }

    return write_available_packets(writer);
}

void video_writer_close(VideoWriter *writer) {
    if (!writer) {
        return;
    }

    AVFormatContext *format_ctx = (AVFormatContext *)writer->format_ctx;
    AVCodecContext *codec_ctx = (AVCodecContext *)writer->codec_ctx;
    AVPacket *packet = (AVPacket *)writer->packet;
    AVFrame *rgb = (AVFrame *)writer->rgb_frame;
    AVFrame *yuv = (AVFrame *)writer->yuv_frame;
    struct SwsContext *sws = (struct SwsContext *)writer->sws_ctx;

    if (writer->is_open && format_ctx) {
        av_write_trailer(format_ctx);
    }
    if (format_ctx && !(format_ctx->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&format_ctx->pb);
    }

    sws_freeContext(sws);
    av_packet_free(&packet);
    av_frame_free(&rgb);
    av_frame_free(&yuv);
    avcodec_free_context(&codec_ctx);
    avformat_free_context(format_ctx);
    memset(writer, 0, sizeof(*writer));
}
