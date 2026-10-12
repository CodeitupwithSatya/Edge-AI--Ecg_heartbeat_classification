#include "ecg_pipeline.hpp"
#pragma once
#include "ecg_pipeline.hpp"
#include "main.h"
#include "network.h"          // X-CUBE-AI generated
#include "network_data.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

// ---------- constants to set ----------
static constexpr float   INPUT_SCALE = 0.006491835f;  // <- paste qp scale printed after the final training
static constexpr int32_t INPUT_ZP    = -91;           // <- paste qp zero-point
static constexpr int     OVERSAMPLE  = 8;             // 1 kHz -> 125 Hz
static constexpr int     N_SAMPLES   = 187;
static constexpr int     TAIL_MARGIN = 8;             // samples kept after the next R-peak (tune)
static constexpr int     MIN_RR      = 38;            // 0.3 s  (>200 bpm rejected)
static constexpr int     REFRACTORY  = 25;            // 0.2 s
static constexpr int     RING        = 512;           // ~4 s of 125 Hz samples (power of 2)
static constexpr float   HP_ALPHA    = 0.9755f;       // ~0.5 Hz high-pass at 125 Hz
// --------------------------------------

static const char CLS[5] = {'N', 'S', 'V', 'F', 'Q'};

// 1) ISR side: boxcar decimation (adds anti-alias averaging) --> 125 Hz sample queue
static volatile uint16_t q_buf[64];
static volatile uint8_t  q_head = 0, q_tail = 0;
static uint32_t acc = 0; static uint8_t acc_n = 0;

extern "C" void ecg_on_adc_sample(uint16_t s) {
    acc += s;
    if (++acc_n == OVERSAMPLE) {
        q_buf[q_head & 63] = (uint16_t)(acc / OVERSAMPLE);
        q_head++; acc = 0; acc_n = 0;
    }
}

// 2) Filtering + R-peak detection state
static float    ring[RING];
static uint32_t n_idx = 0;                  // absolute 125 Hz sample counter
static float    hp_y = 0, hp_xprev = 0;
static float    peak_est = 0;               // running estimate of R amplitude
static float    s1 = 0, s2 = 0;             // previous two filtered samples
static int64_t  last_peak = -1000, prev_peak = -1;   // absolute indices
static int64_t  emit_at = -1, emit_start = -1, emit_end = -1;

// 3) Neural network
static ai_handle network = AI_HANDLE_NULL;
AI_ALIGNED(32) static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_buffer *ai_in, *ai_out;

extern "C" void ecg_init() {
    ai_network_params params = AI_NETWORK_PARAMS_INIT(
        AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        AI_NETWORK_DATA_ACTIVATIONS(activations));
    ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    ai_network_init(network, &params);
    ai_in  = ai_network_inputs_get(network, nullptr);
    ai_out = ai_network_outputs_get(network, nullptr);

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;   // cycle counter for latency
    DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline float at(int64_t abs_idx) { return ring[abs_idx & (RING - 1)]; }

static void classify(int64_t start, int64_t end) {   // window [start, end)
    int len = (int)(end - start);
    if (len < MIN_RR) return;
    if (len > N_SAMPLES) len = N_SAMPLES;

    float mn = 1e9f, mx = -1e9f;
    for (int i = 0; i < len; i++) { float v = at(start + i); mn = std::min(mn, v); mx = std::max(mx, v); }
    float rng = mx - mn; if (rng < 1e-6f) return;

    int8_t* in = (int8_t*)ai_in[0].data;
    for (int i = 0; i < N_SAMPLES; i++) {
        float x = (i < len) ? (at(start + i) - mn) / rng : 0.0f;      // 0..1, zero padded like the dataset
        int32_t q = (int32_t)lroundf(x / INPUT_SCALE) + INPUT_ZP;
        in[i] = (int8_t)std::min<int32_t>(127, std::max<int32_t>(-128, q));
    }

    uint32_t t0 = DWT->CYCCNT;
    ai_network_run(network, &ai_in[0], &ai_out[0]);
    uint32_t cyc = DWT->CYCCNT - t0;

    const int8_t* o = (const int8_t*)ai_out[0].data;
    int best = 0; for (int k = 1; k < 5; k++) if (o[k] > o[best]) best = k;
    printf("beat class %c  logits [%d %d %d %d %d]  %lu us\r\n", CLS[best],
           o[0], o[1], o[2], o[3], o[4], (unsigned long)(cyc / (SystemCoreClock / 1000000)));
}

static void process(float raw) {
    // high-pass (baseline wander removal): y[n] = a*(y[n-1] + x[n] - x[n-1])
    hp_y = HP_ALPHA * (hp_y + raw - hp_xprev); hp_xprev = raw;
    float f = hp_y;
    ring[n_idx & (RING - 1)] = f;

    // local-maximum R detector on s1 (previous sample), adaptive threshold
    if (n_idx >= 2) {
        float thr = (peak_est > 0) ? 0.5f * peak_est : 1e9f;
        bool is_peak = (s1 > s2) && (s1 >= f) && (s1 > thr) && ((int64_t)n_idx - 1 - last_peak > REFRACTORY);
        if (peak_est == 0 && n_idx > 250 ) {            // bootstrap: learn amplitude from first ~2 s
            float m = 0; for (int i = 0; i < 250; i++) m = std::max(m, at((int64_t)n_idx - i));
            peak_est = m;
        }
        if (is_peak) {
            int64_t p = (int64_t)n_idx - 1;
            peak_est = 0.875f * peak_est + 0.125f * s1;
            if (last_peak >= 0 && p - last_peak <= (N_SAMPLES - TAIL_MARGIN) && p - last_peak >= MIN_RR) {
                emit_start = last_peak; emit_end = p + TAIL_MARGIN; emit_at = emit_end;   // wait for the tail
            }
            prev_peak = last_peak; last_peak = p;
        }
    }
    s2 = s1; s1 = f; n_idx++;

    if (emit_at >= 0 && (int64_t)n_idx >= emit_at) {      // tail now available
        classify(emit_start, emit_end);
        emit_at = -1;
    }
}

extern "C" void ecg_poll() {
    while (q_tail != q_head) {
        uint16_t s = q_buf[q_tail & 63]; q_tail++;
        process((float)s / 4095.0f);
    }
}
