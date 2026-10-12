# Edge AI Challenge: technical note

**Model:** 4-layer strided Conv1D (Conv1D+BN+ReLU, stride 2, no pooling) -> static Reshape -> Dense(5), int8 logits output.
7165 params, ~110k MACs, 13.6 KiB file. Ops: CONV_2D, RESHAPE, FULLY_CONNECTED (checker: PASS).

**Preprocessing:** none outside the graph; the first conv learns filtering. Input quantised with the model's own
scale/zero-point; calibration set mixes clean and augmented beats of all 5 classes.

**Class imbalance:** sqrt-inverse-frequency class weights, label smoothing 0.05, fresh augmentation each epoch,
macro-F1 tracked on clean and augmented validation.

**Augmentation (from train.csv only):** amplitude scaling, DC offset, sinusoidal baseline wander, Gaussian noise,
random re-normalisation and clipping. No time shifts (beats start at the R-peak; length is a class cue).

**Optimisation:** strided convs instead of pooling (fewer MACs, no MAX_POOL op); Flatten-style head to keep timing
information (GAP was worse); plain ReLU; no softmax; static reshape (avoids SHAPE/PACK/STRIDED_SLICE ops);
knowledge distillation from a larger float teacher trained on train.csv only; SWA over last 8 epochs;
best of 2 seeds; per-class logit bias offsets tuned on a held-out half of validation and folded into the Dense bias.

**Results (int8):** val macro-F1 clean 0.9032, augmented 0.8832. Hidden score is expected to be lower
(no patient split available).

**STM32 sensor code (bonus):** stm32/ecg_pipeline.{hpp,cpp}: AD8232 -> ADC1 at 1 kHz triggered by TIM6, decimated to 125 Hz,
baseline high-pass, adaptive R-peak detection, R-to-R beat windowing, per-beat min-max, zero padding to 187 samples,
int8 quantisation with the model's scale/zero-point, X-CUBE-AI inference and DWT-based latency printout.