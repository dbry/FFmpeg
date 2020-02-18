/*
 * Direct Stream Transfer (DST) decoder
 * Copyright (c) 2014 Peter Ross <pross@xvid.org>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Direct Stream Transfer (DST) decoder
 * ISO/IEC 14496-3 Part 3 Subpart 10: Technical description of lossless coding of oversampled audio
 */

#include "libavutil/avassert.h"
#include "libavutil/intreadwrite.h"
#include "internal.h"
#include "get_bits.h"
#include "avcodec.h"
#include "thread.h"
#include "golomb.h"
#include "mathops.h"
#include "dsd.h"

#define DST_MAX_CHANNELS 6
#define DST_MAX_ELEMENTS (2 * DST_MAX_CHANNELS)

#define DSD_FS44(sample_rate) (sample_rate * 8LL / 44100)

#define DST_SAMPLES_PER_FRAME(sample_rate) (588 * DSD_FS44(sample_rate))

static const int8_t fsets_code_pred_coeff[3][3] = {
    {  -8 },
    { -16,  8 },
    {  -9, -5, 6 },
};

static const int8_t probs_code_pred_coeff[3][3] = {
    {  -8 },
    { -16,  8 },
    { -24, 24, -8 },
};

typedef struct ArithCoder {
    unsigned int a;
    unsigned int c;
    int overread;
} ArithCoder;

typedef struct Table {
    unsigned int elements;
    unsigned int length[DST_MAX_ELEMENTS];
    int coeff[DST_MAX_ELEMENTS][128];
} Table;

typedef struct DSTContext {
    AVClass *class;

    GetBitContext gb;
    ArithCoder ac;
    Table fsets, probs;
    DECLARE_ALIGNED(16, uint8_t, status)[DST_MAX_CHANNELS][16];
    DECLARE_ALIGNED(16, int16_t, filter)[DST_MAX_ELEMENTS][16][256];
    ThreadFrame curr_frame, prev_frame;
    DSDContext *dsdctx;
} DSTContext;

static av_cold int decode_init(AVCodecContext *avctx)
{
    DSTContext *s = avctx->priv_data;
    int i;

    if (avctx->channels > DST_MAX_CHANNELS) {
        avpriv_request_sample(avctx, "Channel count %d", avctx->channels);
        return AVERROR_PATCHWELCOME;
    }

    avctx->internal->allocate_progress = 1;
    avctx->sample_fmt = AV_SAMPLE_FMT_FLT;

    s->curr_frame.f = av_frame_alloc();
    s->prev_frame.f = av_frame_alloc();
    if (!s->curr_frame.f || !s->prev_frame.f)
        return AVERROR(ENOMEM);

    // the DSD to PCM context is shared (and used serially) between all decoding threads

    s->dsdctx = av_calloc(DST_MAX_CHANNELS, sizeof (DSDContext));

    for (i = 0; i < avctx->channels; i++)
        memset(s->dsdctx[i].buf, 0x69, sizeof(s->dsdctx[i].buf));

    ff_init_dsd_data();

    return 0;
}

#if HAVE_THREADS
static int init_thread_copy(AVCodecContext *avctx)
{
    DSTContext *s = avctx->priv_data;

    s->curr_frame.f = av_frame_alloc();
    s->prev_frame.f = av_frame_alloc();

    return 0;
}

static int update_thread_context(AVCodecContext *dst, const AVCodecContext *src)
{
    DSTContext *fsrc = src->priv_data;
    DSTContext *fdst = dst->priv_data;
    int ret;

    if (dst == src)
        return 0;

    ff_thread_release_buffer(dst, &fdst->curr_frame);
    if (fsrc->curr_frame.f->data[0]) {
        if ((ret = ff_thread_ref_frame(&fdst->curr_frame, &fsrc->curr_frame)) < 0)
            return ret;
    }

    return 0;
}
#endif

static av_cold int decode_end(AVCodecContext *avctx)
{
    DSTContext *s = avctx->priv_data;

    ff_thread_release_buffer(avctx, &s->curr_frame);
    av_frame_free(&s->curr_frame.f);

    ff_thread_release_buffer(avctx, &s->prev_frame);
    av_frame_free(&s->prev_frame.f);

    if (!avctx->internal->is_copy)
        av_freep(&s->dsdctx);

    return 0;
}

static int read_map(GetBitContext *gb, Table *t, unsigned int map[DST_MAX_CHANNELS], int channels)
{
    int ch;
    t->elements = 1;
    map[0] = 0;
    if (!get_bits1(gb)) {
        for (ch = 1; ch < channels; ch++) {
            int bits = av_log2(t->elements) + 1;
            map[ch] = get_bits(gb, bits);
            if (map[ch] == t->elements) {
                t->elements++;
                if (t->elements >= DST_MAX_ELEMENTS)
                    return AVERROR_INVALIDDATA;
            } else if (map[ch] > t->elements) {
                return AVERROR_INVALIDDATA;
            }
        }
    } else {
        memset(map, 0, sizeof(*map) * DST_MAX_CHANNELS);
    }
    return 0;
}

static av_always_inline int get_sr_golomb_dst(GetBitContext *gb, unsigned int k)
{
    int v = get_ur_golomb_jpegls(gb, k, get_bits_left(gb), 0);
    if (v && get_bits1(gb))
        v = -v;
    return v;
}

static void read_uncoded_coeff(GetBitContext *gb, int *dst, unsigned int elements,
                               int coeff_bits, int is_signed, int offset)
{
    int i;

    for (i = 0; i < elements; i++) {
        dst[i] = (is_signed ? get_sbits(gb, coeff_bits) : get_bits(gb, coeff_bits)) + offset;
    }
}

static int read_table(GetBitContext *gb, Table *t, const int8_t code_pred_coeff[3][3],
                      int length_bits, int coeff_bits, int is_signed, int offset)
{
    unsigned int i, j, k;
    for (i = 0; i < t->elements; i++) {
        t->length[i] = get_bits(gb, length_bits) + 1;
        if (!get_bits1(gb)) {
            read_uncoded_coeff(gb, t->coeff[i], t->length[i], coeff_bits, is_signed, offset);
        } else {
            int method = get_bits(gb, 2), lsb_size;
            if (method == 3)
                return AVERROR_INVALIDDATA;

            read_uncoded_coeff(gb, t->coeff[i], method + 1, coeff_bits, is_signed, offset);

            lsb_size  = get_bits(gb, 3);
            for (j = method + 1; j < t->length[i]; j++) {
                int c, x = 0;
                for (k = 0; k < method + 1; k++)
                    x += code_pred_coeff[method][k] * t->coeff[i][j - k - 1];
                c = get_sr_golomb_dst(gb, lsb_size);
                if (x >= 0)
                    c -= (x + 4) / 8;
                else
                    c += (-x + 3) / 8;
                if (!is_signed) {
                    if (c < offset || c >= offset + (1<<coeff_bits))
                        return AVERROR_INVALIDDATA;
                }
                t->coeff[i][j] = c;
            }
        }
    }
    return 0;
}

static void ac_init(ArithCoder *ac, GetBitContext *gb)
{
    ac->a = 4095;
    ac->c = get_bits(gb, 12);
    ac->overread = 0;
}

static av_always_inline void ac_get(ArithCoder *ac, GetBitContext *gb, int p, int *e)
{
    unsigned int k = (ac->a >> 8) | ((ac->a >> 7) & 1);
    unsigned int q = k * p;
    unsigned int a_q = ac->a - q;

    *e = ac->c < a_q;
    if (*e) {
        ac->a  = a_q;
    } else {
        ac->a  = q;
        ac->c -= a_q;
    }

    if (ac->a < 2048) {
        int n = 11 - av_log2(ac->a);
        ac->a <<= n;
        if (get_bits_left(gb) < n)
            ac->overread ++;
        ac->c = (ac->c << n) | get_bits(gb, n);
    }
}

static uint8_t prob_dst_x_bit(int c)
{
    return (ff_reverse[c & 127] >> 1) + 1;
}

static void build_filter(int16_t table[DST_MAX_ELEMENTS][16][256], const Table *fsets)
{
    int i, j, k, l;

    for (i = 0; i < fsets->elements; i++) {
        int length = fsets->length[i];

        for (j = 0; j < 16; j++) {
            int total = av_clip(length - j * 8, 0, 8);

            for (k = 0; k < 256; k++) {
                int v = 0;

                for (l = 0; l < total; l++)
                    v += (((k >> l) & 1) * 2 - 1) * fsets->coeff[i][j * 8 + l];
                table[i][j][k] = v;
            }
        }
    }
}

static void decode_flush(AVCodecContext *avctx)
{
    DSTContext *s = avctx->priv_data;

    if (!avctx->internal->is_copy) {
        for (int i = 0; i < avctx->channels; i++)
            memset(s->dsdctx[i].buf, 0x69, sizeof(s->dsdctx[i].buf));
    }
}

static int decode_frame(AVCodecContext *avctx, void *data,
                        int *got_frame_ptr, AVPacket *avpkt)
{
    unsigned samples_per_frame = DST_SAMPLES_PER_FRAME(avctx->sample_rate);
    unsigned map_ch_to_felem[DST_MAX_CHANNELS];
    unsigned map_ch_to_pelem[DST_MAX_CHANNELS];
    unsigned i, ch, same_map, dst_x_bit;
    unsigned half_prob[DST_MAX_CHANNELS];
    const int channels = avctx->channels;
    DSTContext *s = avctx->priv_data;
    GetBitContext *gb = &s->gb;
    ArithCoder *ac = &s->ac;
    AVFrame *frame;
    uint8_t *dsd;
    float *pcm;
    int ret;

    if (avpkt->size <= 1)
        return AVERROR_INVALIDDATA;

    ff_thread_release_buffer(avctx, &s->prev_frame);
    FFSWAP(ThreadFrame, s->curr_frame, s->prev_frame);
    frame = s->curr_frame.f;

    frame->nb_samples = samples_per_frame / 8;
    if ((ret = ff_thread_get_buffer(avctx, &s->curr_frame, AV_GET_BUFFER_FLAG_REF)) < 0)
        return ret;
    dsd = frame->data[0];
    pcm = (float *)frame->data[0];

    ff_thread_finish_setup(avctx);

    if ((ret = init_get_bits8(gb, avpkt->data, avpkt->size)) < 0)
        goto error;

    if (!get_bits1(gb)) {
        skip_bits1(gb);
        if (get_bits(gb, 6)) {
            ret = AVERROR_INVALIDDATA;
            goto error;
        }
        memcpy(frame->data[0], avpkt->data + 1, FFMIN(avpkt->size - 1, frame->nb_samples * channels));
        goto dsd;
    }

    /* Segmentation (10.4, 10.5, 10.6) */

    if (!get_bits1(gb)) {
        avpriv_request_sample(avctx, "Not Same Segmentation");
        ret = AVERROR_PATCHWELCOME;
        goto error;
    }

    if (!get_bits1(gb)) {
        avpriv_request_sample(avctx, "Not Same Segmentation For All Channels");
        ret = AVERROR_PATCHWELCOME;
        goto error;
    }

    if (!get_bits1(gb)) {
        avpriv_request_sample(avctx, "Not End Of Channel Segmentation");
        ret = AVERROR_PATCHWELCOME;
        goto error;
    }

    /* Mapping (10.7, 10.8, 10.9) */

    same_map = get_bits1(gb);

    if ((ret = read_map(gb, &s->fsets, map_ch_to_felem, channels)) < 0)
        goto error;

    if (same_map) {
        s->probs.elements = s->fsets.elements;
        memcpy(map_ch_to_pelem, map_ch_to_felem, sizeof(map_ch_to_felem));
    } else {
        avpriv_request_sample(avctx, "Not Same Mapping");
        if ((ret = read_map(gb, &s->probs, map_ch_to_pelem, channels)) < 0)
            goto error;
    }

    /* Half Probability (10.10) */

    for (ch = 0; ch < channels; ch++)
        half_prob[ch] = get_bits1(gb);

    /* Filter Coef Sets (10.12) */

    ret = read_table(gb, &s->fsets, fsets_code_pred_coeff, 7, 9, 1, 0);
    if (ret < 0)
        goto error;

    /* Probability Tables (10.13) */

    ret = read_table(gb, &s->probs, probs_code_pred_coeff, 6, 7, 0, 1);
    if (ret < 0)
        goto error;

    /* Arithmetic Coded Data (10.11) */

    if (get_bits1(gb)) {
        ret = AVERROR_INVALIDDATA;
        goto error;
    }
    ac_init(ac, gb);

    build_filter(s->filter, &s->fsets);

    memset(s->status, 0xAA, sizeof(s->status));
    memset(dsd, 0, frame->nb_samples * 4 * channels);

    ac_get(ac, gb, prob_dst_x_bit(s->fsets.coeff[0][0]), &dst_x_bit);

    for (i = 0; i < samples_per_frame; i++) {
        for (ch = 0; ch < channels; ch++) {
            const unsigned felem = map_ch_to_felem[ch];
            int16_t (*filter)[256] = s->filter[felem];
            uint8_t *status = s->status[ch];
            int prob, residual, v;

#define F(x) filter[(x)][status[(x)]]
            const int16_t predict = F( 0) + F( 1) + F( 2) + F( 3) +
                                    F( 4) + F( 5) + F( 6) + F( 7) +
                                    F( 8) + F( 9) + F(10) + F(11) +
                                    F(12) + F(13) + F(14) + F(15);
#undef F

            if (!half_prob[ch] || i >= s->fsets.length[felem]) {
                unsigned pelem = map_ch_to_pelem[ch];
                unsigned index = FFABS(predict) >> 3;
                prob = s->probs.coeff[pelem][FFMIN(index, s->probs.length[pelem] - 1)];
            } else {
                prob = 128;
            }

            if (ac->overread > 16) {
                ret = AVERROR_INVALIDDATA;
                goto error;
            }

            ac_get(ac, gb, prob, &residual);
            v = ((predict >> 15) ^ residual) & 1;
            dsd[((i >> 3) * channels + ch) << 2] |= v << (7 - (i & 0x7 ));

            AV_WL64A(status + 8, (AV_RL64A(status + 8) << 1) | ((AV_RL64A(status) >> 63) & 1));
            AV_WL64A(status, (AV_RL64A(status) << 1) | v);
        }
    }

dsd:
    // we must wait for the previous frame to be complete before doing the DSD2PCM
    ff_thread_await_progress(&s->prev_frame, INT_MAX, 0);
    ff_thread_release_buffer(avctx, &s->prev_frame);

    for (i = 0; i < channels; i++) {
        ff_dsd2pcm_translate(&s->dsdctx[i], frame->nb_samples, 0,
                             frame->data[0] + i * 4,
                             channels * 4, pcm + i, channels);
    }

    // report that the DSD2PCM is done for this frame
    ff_thread_report_progress(&s->curr_frame, INT_MAX, 0);

    if ((ret = av_frame_ref(data, s->curr_frame.f)) < 0)
        return ret;

    *got_frame_ptr = 1;

    return avpkt->size;

error:
    ff_thread_await_progress(&s->prev_frame, INT_MAX, 0);
    ff_thread_release_buffer(avctx, &s->prev_frame);
    ff_thread_report_progress(&s->curr_frame, INT_MAX, 0);
    return ret;
}

AVCodec ff_dst_decoder = {
    .name           = "dst",
    .long_name      = NULL_IF_CONFIG_SMALL("DST (Digital Stream Transfer)"),
    .type           = AVMEDIA_TYPE_AUDIO,
    .id             = AV_CODEC_ID_DST,
    .priv_data_size = sizeof(DSTContext),
    .init           = decode_init,
    .close          = decode_end,
    .decode         = decode_frame,
    .flush          = decode_flush,
    .init_thread_copy = ONLY_IF_THREADS_ENABLED(init_thread_copy),
    .update_thread_context = ONLY_IF_THREADS_ENABLED(update_thread_context),
    .capabilities   = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_FRAME_THREADS,
    .sample_fmts    = (const enum AVSampleFormat[]) { AV_SAMPLE_FMT_FLT,
                                                      AV_SAMPLE_FMT_NONE },
};
