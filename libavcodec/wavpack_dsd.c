/*
 * WavPack lossless DSD audio decoder
 * Copyright (c) 2006,2011 Konstantin Shishkov
 * Copyright (c) 2019 David Bryant
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

#include "libavutil/channel_layout.h"

#include "avcodec.h"
#include "bytestream.h"
#include "internal.h"
#include "wavpack.h"
#include "dsd.h"

/**
 * @file
 * WavPack lossless DSD audio decoder
 */

#define DSD_BYTE_READY(low,high) (!(((low) ^ (high)) & 0xff000000))

#define PTABLE_BITS 8
#define PTABLE_BINS (1<<PTABLE_BITS)
#define PTABLE_MASK (PTABLE_BINS-1)

#define UP   0x010000fe
#define DOWN 0x00010000
#define DECAY 8

#define PRECISION 20
#define VALUE_ONE (1 << PRECISION)
#define PRECISION_USE 12

#define RATE_S 20

#define MAX_HISTORY_BITS    5
#define MAX_HISTORY_BINS    (1 << MAX_HISTORY_BITS)

typedef struct WavpackFrameContext {
    AVCodecContext *avctx;
    int stereo, stereo_in;
    uint32_t CRC;
    int samples;
    GetByteContext dsd_gb;
    int ptable [PTABLE_BINS];
    int16_t summed_probabilities [MAX_HISTORY_BINS] [256];
    unsigned char probabilities [MAX_HISTORY_BINS] [256];
    unsigned char *value_lookup [MAX_HISTORY_BINS];
    DSDContext dsdctx[2];
} WavpackFrameContext;

#define WV_MAX_FRAME_DECODERS 14

typedef struct WavpackContext {
    AVCodecContext *avctx;

    WavpackFrameContext *fdec[WV_MAX_FRAME_DECODERS];
    int fdec_num;

    int block;
    int samples;
    int ch_offset;
} WavpackContext;

static inline int wv_check_crc(WavpackFrameContext *s, uint32_t crc)
{
    if (crc != s->CRC) {
        av_log(s->avctx, AV_LOG_ERROR, "CRC error\n");
        return AVERROR_INVALIDDATA;
    }

    return 0;
}

static void init_ptable (int *table, int rate_i, int rate_s)
{
    int value = 0x808000, rate = rate_i << 8, c, i;

    for (c = (rate + 128) >> 8; c--;)
        value += (DOWN - value) >> DECAY;

    for (i = 0; i < PTABLE_BINS/2; ++i) {
        table [i] = value;
        table [PTABLE_BINS-1-i] = 0x100ffff - value;

        if (value > 0x010000) {
            rate += (rate * rate_s + 128) >> 8;

            for (c = (rate + 64) >> 7; c--;)
                value += (DOWN - value) >> DECAY;
        }
    }
}

typedef struct {
    int32_t value, filter0, filter1, filter2, filter3, filter4, filter5, filter6, factor, byte;
} DSDfilters;

static int wv_unpack_dsd_high(WavpackFrameContext *s, void *dst_l, void *dst_r)
{
    uint8_t *dsd_l                  = dst_l;
    uint8_t *dsd_r                  = dst_r;
    uint32_t crc                    = 0xFFFFFFFF;
    int total_samples = s->samples, stereo = dst_r ? 1 : 0;
    DSDfilters filters [2], *sp = filters;
    int channel, rate_i, rate_s, i;
    uint32_t low, high, value;

    if (bytestream2_get_bytes_left(&s->dsd_gb) < (stereo ? 20 : 13))
        return AVERROR_INVALIDDATA;

    rate_i = bytestream2_get_byte(&s->dsd_gb);
    rate_s = bytestream2_get_byte(&s->dsd_gb);

    if (rate_s != RATE_S)
        return AVERROR_INVALIDDATA;

    init_ptable (s->ptable, rate_i, rate_s);

    for (channel = 0; channel < stereo + 1; ++channel) {
        DSDfilters *sp = filters + channel;

        sp->filter1 = bytestream2_get_byte(&s->dsd_gb) << (PRECISION - 8);
        sp->filter2 = bytestream2_get_byte(&s->dsd_gb) << (PRECISION - 8);
        sp->filter3 = bytestream2_get_byte(&s->dsd_gb) << (PRECISION - 8);
        sp->filter4 = bytestream2_get_byte(&s->dsd_gb) << (PRECISION - 8);
        sp->filter5 = bytestream2_get_byte(&s->dsd_gb) << (PRECISION - 8);
        sp->filter6 = 0;
        sp->factor = bytestream2_get_byte(&s->dsd_gb) & 0xff;
        sp->factor |= (bytestream2_get_byte(&s->dsd_gb) << 8) & 0xff00;
        sp->factor = (sp->factor << 16) >> 16;
    }

    high = 0xffffffff;
    low = 0x0;

    for (i = 4; i--;)
        value = (value << 8) | bytestream2_get_byte(&s->dsd_gb);

    memset (dst_l, 0x69, total_samples * 4);

    if (stereo)
        memset (dst_r, 0x69, total_samples * 4);

    while (total_samples--) {
        int bitcount = 8;

        sp [0].value = sp [0].filter1 - sp [0].filter5 + ((sp [0].filter6 * sp [0].factor) >> 2);

        if (stereo)
            sp [1].value = sp [1].filter1 - sp [1].filter5 + ((sp [1].filter6 * sp [1].factor) >> 2);

        while (bitcount--) {
            int32_t *pp = s->ptable + ((sp [0].value >> (PRECISION - PRECISION_USE)) & PTABLE_MASK);
            uint32_t split = low + ((high - low) >> 8) * (*pp >> 16);

            if (value <= split) {
                high = split;
                *pp += (UP - *pp) >> DECAY;
                sp [0].filter0 = -1;
            }
            else {
                low = split + 1;
                *pp += (DOWN - *pp) >> DECAY;
                sp [0].filter0 = 0;
            }

            while (DSD_BYTE_READY (high, low) && bytestream2_get_bytes_left(&s->dsd_gb)) {
                value = (value << 8) | bytestream2_get_byte(&s->dsd_gb);
                high = (high << 8) | 0xff;
                low <<= 8;
            }

            sp [0].value += sp [0].filter6 << 3;
            sp [0].byte = (sp [0].byte << 1) | (sp [0].filter0 & 1);
            sp [0].factor += (((sp [0].value ^ sp [0].filter0) >> 31) | 1) & ((sp [0].value ^ (sp [0].value - (sp [0].filter6 << 4))) >> 31);
            sp [0].filter1 += ((sp [0].filter0 & VALUE_ONE) - sp [0].filter1) >> 6;
            sp [0].filter2 += ((sp [0].filter0 & VALUE_ONE) - sp [0].filter2) >> 4;
            sp [0].filter3 += (sp [0].filter2 - sp [0].filter3) >> 4;
            sp [0].filter4 += (sp [0].filter3 - sp [0].filter4) >> 4;
            sp [0].value = (sp [0].filter4 - sp [0].filter5) >> 4;
            sp [0].filter5 += sp [0].value;
            sp [0].filter6 += (sp [0].value - sp [0].filter6) >> 3;
            sp [0].value = sp [0].filter1 - sp [0].filter5 + ((sp [0].filter6 * sp [0].factor) >> 2);

            if (!stereo)
                continue;

            pp = s->ptable + ((sp [1].value >> (PRECISION - PRECISION_USE)) & PTABLE_MASK);
            split = low + ((high - low) >> 8) * (*pp >> 16);

            if (value <= split) {
                high = split;
                *pp += (UP - *pp) >> DECAY;
                sp [1].filter0 = -1;
            }
            else {
                low = split + 1;
                *pp += (DOWN - *pp) >> DECAY;
                sp [1].filter0 = 0;
            }

            while (DSD_BYTE_READY (high, low) && bytestream2_get_bytes_left(&s->dsd_gb)) {
                value = (value << 8) | bytestream2_get_byte(&s->dsd_gb);
                high = (high << 8) | 0xff;
                low <<= 8;
            }

            sp [1].value += sp [1].filter6 << 3;
            sp [1].byte = (sp [1].byte << 1) | (sp [1].filter0 & 1);
            sp [1].factor += (((sp [1].value ^ sp [1].filter0) >> 31) | 1) & ((sp [1].value ^ (sp [1].value - (sp [1].filter6 << 4))) >> 31);
            sp [1].filter1 += ((sp [1].filter0 & VALUE_ONE) - sp [1].filter1) >> 6;
            sp [1].filter2 += ((sp [1].filter0 & VALUE_ONE) - sp [1].filter2) >> 4;
            sp [1].filter3 += (sp [1].filter2 - sp [1].filter3) >> 4;
            sp [1].filter4 += (sp [1].filter3 - sp [1].filter4) >> 4;
            sp [1].value = (sp [1].filter4 - sp [1].filter5) >> 4;
            sp [1].filter5 += sp [1].value;
            sp [1].filter6 += (sp [1].value - sp [1].filter6) >> 3;
            sp [1].value = sp [1].filter1 - sp [1].filter5 + ((sp [1].filter6 * sp [1].factor) >> 2);
        }

        crc += (crc << 1) + (*dsd_l = sp [0].byte & 0xff);
        sp [0].factor -= (sp [0].factor + 512) >> 10;
        dsd_l += 4;

        if (stereo) {
            crc += (crc << 1) + (*dsd_r = filters [1].byte & 0xff);
            filters [1].factor -= (filters [1].factor + 512) >> 10;
            dsd_r += 4;
        }
    }

    if ((s->avctx->err_recognition & AV_EF_CRCCHECK) && wv_check_crc(s, crc))
        return AVERROR_INVALIDDATA;

    return 0;
}

static int wv_unpack_dsd_fast(WavpackFrameContext *s, void *dst_l, void *dst_r)
{
    uint8_t *dsd_l                  = dst_l;
    uint8_t *dsd_r                  = dst_r;
    unsigned char history_bits, max_probability;
    int total_summed_probabilities  = 0;
    int total_samples               = s->samples;
    int history_bins, p0, p1, chan;
    uint32_t crc                    = 0xFFFFFFFF;
    uint32_t low, high, value;
    int ret = 0, i;

    if (!bytestream2_get_bytes_left(&s->dsd_gb))
        return AVERROR_INVALIDDATA;

    history_bits = bytestream2_get_byte(&s->dsd_gb);

    if (!bytestream2_get_bytes_left(&s->dsd_gb) || history_bits > MAX_HISTORY_BITS)
        return AVERROR_INVALIDDATA;

    history_bins = 1 << history_bits;
    max_probability = bytestream2_get_byte(&s->dsd_gb);

    if (max_probability < 0xff) {
        unsigned char *outptr = (unsigned char *) s->probabilities;
        unsigned char *outend = outptr + sizeof (*s->probabilities) * history_bins;

        while (outptr < outend && bytestream2_get_bytes_left(&s->dsd_gb)) {
            int code = bytestream2_get_byte(&s->dsd_gb);

            if (code > max_probability) {
                int zcount = code - max_probability;

                while (outptr < outend && zcount--)
                    *outptr++ = 0;
            }
            else if (code)
                *outptr++ = code;
            else
                break;
        }

        if (outptr < outend || (bytestream2_get_bytes_left(&s->dsd_gb) && bytestream2_get_byte(&s->dsd_gb)))
            return AVERROR_INVALIDDATA;
    }
    else if (bytestream2_get_bytes_left(&s->dsd_gb) > (int) sizeof (*s->probabilities) * history_bins)
        bytestream2_get_buffer(&s->dsd_gb, (uint8_t *) s->probabilities, sizeof (*s->probabilities) * history_bins);
    else
        return AVERROR_INVALIDDATA;

    for (p0 = 0; p0 < history_bins; ++p0) {
        int32_t sum_values;
        unsigned char *vp;

        for (sum_values = i = 0; i < 256; ++i)
            s->summed_probabilities [p0] [i] = sum_values += s->probabilities [p0] [i];

        if (sum_values) {
            total_summed_probabilities += sum_values;
            vp = s->value_lookup [p0] = av_malloc (sum_values);

            for (i = 0; i < 256; i++) {
                int c = s->probabilities [p0] [i];

                while (c--)
                    *vp++ = i;
            }
        }
    }

    if (bytestream2_get_bytes_left(&s->dsd_gb) < 4 || total_summed_probabilities > history_bins * 1280) {
        ret = AVERROR_INVALIDDATA;
        goto done;
    }

    for (i = 4; i--;)
        value = (value << 8) | bytestream2_get_byte(&s->dsd_gb);

    chan = p0 = p1 = 0;
    low = 0; high = 0xffffffff;

    memset (dst_l, 0x69, total_samples * 4);

    if (dst_r) {
        memset (dst_r, 0x69, total_samples * 4);
        total_samples *= 2;
    }

    while (total_samples--) {
        int mult, index, code;

        if (!s->summed_probabilities [p0] [255]) {
            ret = AVERROR_INVALIDDATA;
            goto done;
        }

        mult = (high - low) / s->summed_probabilities [p0] [255];

        if (!mult) {
            if (bytestream2_get_bytes_left(&s->dsd_gb) >= 4)
                for (i = 4; i--;)
                    value = (value << 8) | bytestream2_get_byte(&s->dsd_gb);

            low = 0;
            high = 0xffffffff;
            mult = high / s->summed_probabilities [p0] [255];

            if (!mult) {
                ret = AVERROR_INVALIDDATA;
                goto done;
            }
        }

        index = (value - low) / mult;

        if (index >= s->summed_probabilities [p0] [255]) {
            ret = AVERROR_INVALIDDATA;
            goto done;
        }

        if (!dst_r) {
            if ((*dsd_l = code = s->value_lookup [p0] [index]))
                low += s->summed_probabilities [p0] [code-1] * mult;

            dsd_l += 4;
        }
        else {
            if ((code = s->value_lookup [p0] [index]))
                low += s->summed_probabilities [p0] [code-1] * mult;

            if (chan) {
                *dsd_r = code;
                dsd_r += 4;
            }
            else {
                *dsd_l = code;
                dsd_l += 4;
            }

            chan ^= 1;
        }

        high = low + s->probabilities [p0] [code] * mult - 1;
        crc += (crc << 1) + code;

        if (!dst_r)
            p0 = code & (history_bins-1);
        else {
            p0 = p1;
            p1 = code & (history_bins-1);
        }

        while (DSD_BYTE_READY (high, low) && bytestream2_get_bytes_left(&s->dsd_gb)) {
            value = (value << 8) | bytestream2_get_byte(&s->dsd_gb);
            high = (high << 8) | 0xff;
            low <<= 8;
        }
    }

done:
    for (p0 = 0; p0 < history_bins; ++p0)
        if (s->value_lookup [p0]) {
            av_free (s->value_lookup [p0]);
            s->value_lookup [p0] = NULL;
        }

    if (ret < 0)
        return ret;

    if ((s->avctx->err_recognition & AV_EF_CRCCHECK) && wv_check_crc(s, crc))
        ret = AVERROR_INVALIDDATA;

    return ret;
}

static int wv_unpack_dsd_copy(WavpackFrameContext *s, void *dst_l, void *dst_r)
{
    uint8_t *dsd_l              = dst_l;
    uint8_t *dsd_r              = dst_r;
    int total_samples           = s->samples;
    uint32_t crc                = 0xFFFFFFFF;

    if (bytestream2_get_bytes_left (&s->dsd_gb) != total_samples * (dst_r ? 2 : 1))
        return AVERROR_INVALIDDATA;

    memset (dst_l, 0x69, total_samples * 4);

    if (dst_r)
        memset (dst_r, 0x69, total_samples * 4);

    while (total_samples--) {
        crc += (crc << 1) + (*dsd_l = bytestream2_get_byte(&s->dsd_gb));
        dsd_l += 4;

        if (dst_r) {
            crc += (crc << 1) + (*dsd_r = bytestream2_get_byte(&s->dsd_gb));
            dsd_r += 4;
        }
    }

    if ((s->avctx->err_recognition & AV_EF_CRCCHECK) && wv_check_crc(s, crc))
        return AVERROR_INVALIDDATA;

    return 0;
}

static av_cold int wv_alloc_frame_context(WavpackContext *c)
{
    if (c->fdec_num == WV_MAX_FRAME_DECODERS)
        return -1;

    c->fdec[c->fdec_num] = av_mallocz(sizeof(**c->fdec));
    if (!c->fdec[c->fdec_num])
        return -1;
    c->fdec_num++;
    c->fdec[c->fdec_num - 1]->avctx = c->avctx;
    memset(c->fdec[c->fdec_num - 1]->dsdctx[0].buf, 0x69, sizeof(c->fdec[c->fdec_num - 1]->dsdctx[0].buf));
    memset(c->fdec[c->fdec_num - 1]->dsdctx[1].buf, 0x69, sizeof(c->fdec[c->fdec_num - 1]->dsdctx[1].buf));

    return 0;
}

static av_cold int wavpack_decode_init(AVCodecContext *avctx)
{
    WavpackContext *s = avctx->priv_data;

    s->avctx = avctx;

    s->fdec_num = 0;

    ff_init_dsd_data();

    return 0;
}

static av_cold int wavpack_decode_end(AVCodecContext *avctx)
{
    WavpackContext *s = avctx->priv_data;

    s->fdec_num = 0;

    return 0;
}

static int wavpack_decode_block(AVCodecContext *avctx, int block_no,
                                AVFrame *frame, const uint8_t *buf, int buf_size)
{
    WavpackContext *wc = avctx->priv_data;
    WavpackFrameContext *s;
    GetByteContext gb;
    void *samples_l = NULL, *samples_r = NULL;
    int ret;
    int got_dsd = 0;
    int id, size, ssize;
    int chan = 0, chmask = 0, sample_rate = 0, rate_x = 1, dsd_mode = 0;
    int frame_flags, multiblock;

    if (block_no >= wc->fdec_num && wv_alloc_frame_context(wc) < 0) {
        av_log(avctx, AV_LOG_ERROR, "Error creating frame decode context\n");
        return AVERROR_INVALIDDATA;
    }

    s = wc->fdec[block_no];
    if (!s) {
        av_log(avctx, AV_LOG_ERROR, "Context for block %d is not present\n",
               block_no);
        return AVERROR_INVALIDDATA;
    }

    bytestream2_init(&gb, buf, buf_size);

    s->samples = bytestream2_get_le32(&gb);
    if (s->samples != wc->samples) {
        av_log(avctx, AV_LOG_ERROR, "Mismatching number of samples in "
               "a sequence: %d and %d\n", wc->samples, s->samples);
        return AVERROR_INVALIDDATA;
    }
    frame_flags = bytestream2_get_le32(&gb);
    multiblock     = (frame_flags & WV_SINGLE_BLOCK) != WV_SINGLE_BLOCK;

    s->stereo         = !(frame_flags & WV_MONO);
    s->stereo_in      =  (frame_flags & WV_FALSE_STEREO) ? 0 : s->stereo;
    s->CRC            = bytestream2_get_le32(&gb);

    // parse metadata blocks
    while (bytestream2_get_bytes_left(&gb)) {
        id   = bytestream2_get_byte(&gb);
        size = bytestream2_get_byte(&gb);
        if (id & WP_IDF_LONG) {
            size |= (bytestream2_get_byte(&gb)) << 8;
            size |= (bytestream2_get_byte(&gb)) << 16;
        }
        size <<= 1; // size is specified in words
        ssize  = size;
        if (id & WP_IDF_ODD)
            size--;
        if (size < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "Got incorrect block %02X with size %i\n", id, size);
            break;
        }
        if (bytestream2_get_bytes_left(&gb) < ssize) {
            av_log(avctx, AV_LOG_ERROR,
                   "Block size %i is out of bounds\n", size);
            break;
        }
        switch (id & WP_IDF_MASK) {
        case WP_ID_DSD_DATA:
            if (size < 2) {
                av_log(avctx, AV_LOG_ERROR, "Invalid DSD_DATA, size = %i\n",
                       size);
                bytestream2_skip(&gb, ssize);
                continue;
            }
            rate_x = 1 << bytestream2_get_byte(&gb);
            dsd_mode = bytestream2_get_byte(&gb);
            if (dsd_mode && dsd_mode != 1 && dsd_mode != 3) {
                av_log(avctx, AV_LOG_ERROR, "Invalid DSD encoding mode: %d\n", dsd_mode);
                return AVERROR_INVALIDDATA;
            }
            bytestream2_init(&s->dsd_gb, gb.buffer, size-2);
            bytestream2_skip(&gb, size-2);
            got_dsd      = 1;
            break;
        case WP_ID_CHANINFO:
            if (size <= 1) {
                av_log(avctx, AV_LOG_ERROR,
                       "Insufficient channel information\n");
                return AVERROR_INVALIDDATA;
            }
            chan = bytestream2_get_byte(&gb);
            switch (size - 2) {
            case 0:
                chmask = bytestream2_get_byte(&gb);
                break;
            case 1:
                chmask = bytestream2_get_le16(&gb);
                break;
            case 2:
                chmask = bytestream2_get_le24(&gb);
                break;
            case 3:
                chmask = bytestream2_get_le32(&gb);
                break;
            case 4:
                size = bytestream2_get_byte(&gb);
                chan  |= (bytestream2_get_byte(&gb) & 0xF) << 8;
                chan  += 1;
                if (avctx->channels != chan)
                    av_log(avctx, AV_LOG_WARNING, "%i channels signalled"
                           " instead of %i.\n", chan, avctx->channels);
                chmask = bytestream2_get_le24(&gb);
                break;
            case 5:
                size = bytestream2_get_byte(&gb);
                chan  |= (bytestream2_get_byte(&gb) & 0xF) << 8;
                chan  += 1;
                if (avctx->channels != chan)
                    av_log(avctx, AV_LOG_WARNING, "%i channels signalled"
                           " instead of %i.\n", chan, avctx->channels);
                chmask = bytestream2_get_le32(&gb);
                break;
            default:
                av_log(avctx, AV_LOG_ERROR, "Invalid channel info size %d\n",
                       size);
                chan   = avctx->channels;
                chmask = avctx->channel_layout;
            }
            break;
        case WP_ID_SAMPLE_RATE:
            if (size != 3) {
                av_log(avctx, AV_LOG_ERROR, "Invalid custom sample rate.\n");
                return AVERROR_INVALIDDATA;
            }
            sample_rate = bytestream2_get_le24(&gb);
            break;
        default:
            bytestream2_skip(&gb, size);
        }
        if (id & WP_IDF_ODD)
            bytestream2_skip(&gb, 1);
    }

    if (!got_dsd) {
        av_log(avctx, AV_LOG_ERROR, "Packed samples not found\n");
        return AVERROR_INVALIDDATA;
    }

    if (!wc->ch_offset) {
        int sr = (frame_flags >> 23) & 0xf;
        if (sr == 0xf) {
            if (!sample_rate) {
                av_log(avctx, AV_LOG_ERROR, "Custom sample rate missing.\n");
                return AVERROR_INVALIDDATA;
            }
            avctx->sample_rate = sample_rate * rate_x;
        } else
            avctx->sample_rate = wv_rates[sr] * rate_x;

        if (multiblock) {
            if (chan)
                avctx->channels = chan;
            if (chmask)
                avctx->channel_layout = chmask;
        } else {
            avctx->channels       = s->stereo ? 2 : 1;
            avctx->channel_layout = s->stereo ? AV_CH_LAYOUT_STEREO :
                                                AV_CH_LAYOUT_MONO;
        }

        /* get output buffer */
        frame->nb_samples = s->samples + 1;
        if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)
            return ret;
        frame->nb_samples = s->samples;
    }

    if (wc->ch_offset + s->stereo >= avctx->channels) {
        av_log(avctx, AV_LOG_WARNING, "Too many channels coded in a packet.\n");
        return ((avctx->err_recognition & AV_EF_EXPLODE) || !wc->ch_offset) ? AVERROR_INVALIDDATA : 0;
    }

    samples_l = frame->extended_data[wc->ch_offset];
    if (s->stereo)
        samples_r = frame->extended_data[wc->ch_offset + 1];

    wc->ch_offset += 1 + s->stereo;

    if (s->stereo_in) {
        if (dsd_mode == 3)
            ret = wv_unpack_dsd_high(s, samples_l, samples_r);
        else if (dsd_mode == 1)
            ret = wv_unpack_dsd_fast(s, samples_l, samples_r);
        else
            ret = wv_unpack_dsd_copy(s, samples_l, samples_r);
    }
    else {
        if (dsd_mode == 3)
            ret = wv_unpack_dsd_high(s, samples_l, NULL);
        else if (dsd_mode == 1)
            ret = wv_unpack_dsd_fast(s, samples_l, NULL);
        else
            ret = wv_unpack_dsd_copy(s, samples_l, NULL);

        if (s->stereo)
            memcpy(samples_r, samples_l, 4 * s->samples);
    }

    ff_dsd2pcm_translate (&s->dsdctx [0], s->samples, 0, samples_l, 4, samples_l, 1);

    if (s->stereo)
        ff_dsd2pcm_translate (&s->dsdctx [1], s->samples, 0, samples_r, 4, samples_r, 1);

    return ret;
}

static int wavpack_decode_frame(AVCodecContext *avctx, void *data,
                                int *got_frame_ptr, AVPacket *avpkt)
{
    WavpackContext *s  = avctx->priv_data;
    const uint8_t *buf = avpkt->data;
    int buf_size       = avpkt->size;
    AVFrame *frame     = data;
    int frame_size, ret, frame_flags;

    if (avpkt->size <= WV_HEADER_SIZE)
        return AVERROR_INVALIDDATA;

    s->block     = 0;
    s->ch_offset = 0;

    /* determine number of samples */
    s->samples  = AV_RL32(buf + 20);
    frame_flags = AV_RL32(buf + 24);
    if (s->samples <= 0 || s->samples > WV_MAX_SAMPLES) {
        av_log(avctx, AV_LOG_ERROR, "Invalid number of samples: %d\n",
               s->samples);
        return AVERROR_INVALIDDATA;
    }

    if (!(frame_flags & WV_DSD_DATA)) {
        av_log(avctx, AV_LOG_ERROR, "Encountered a non-DSD frame\n");
        return AVERROR_INVALIDDATA;
    }

    avctx->sample_fmt = AV_SAMPLE_FMT_FLTP;

    while (buf_size > 0) {
        if (buf_size <= WV_HEADER_SIZE)
            break;
        frame_size = AV_RL32(buf + 4) - 12;
        buf       += 20;
        buf_size  -= 20;
        if (frame_size <= 0 || frame_size > buf_size) {
            av_log(avctx, AV_LOG_ERROR,
                   "Block %d has invalid size (size %d vs. %d bytes left)\n",
                   s->block, frame_size, buf_size);
            return AVERROR_INVALIDDATA;
        }
        if ((ret = wavpack_decode_block(avctx, s->block,
                                        frame, buf, frame_size)) < 0) {
            return ret;
        }
        s->block++;
        buf      += frame_size;
        buf_size -= frame_size;
    }

    if (s->ch_offset != avctx->channels) {
        av_log(avctx, AV_LOG_ERROR, "Not enough channels coded in a packet.\n");
        return AVERROR_INVALIDDATA;
    }

    *got_frame_ptr = 1;

    return avpkt->size;
}

AVCodec ff_wavpack_dsd_decoder = {
    .name           = "wavpack_dsd",
    .long_name      = NULL_IF_CONFIG_SMALL("WavPack DSD"),
    .type           = AVMEDIA_TYPE_AUDIO,
    .id             = AV_CODEC_ID_WAVPACK_DSD,
    .priv_data_size = sizeof(WavpackContext),
    .init           = wavpack_decode_init,
    .close          = wavpack_decode_end,
    .decode         = wavpack_decode_frame,
    .capabilities   = AV_CODEC_CAP_DR1,
};
