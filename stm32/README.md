# ECG Heartbeat Classification on a Cortex-M7 (Edge AI Challenge)

🏆 **First place** in the Edge AI Challenge, organized by the **Electronics and IoT Club, IIT (ISM) Dhanbad**.

A 7k-parameter, fully int8 1-D CNN that classifies single ECG heartbeats into 5 classes
(N, S, V, F, Q), designed to fit an STM32H7 (32 KiB weights, 16 KiB activations), plus the
STM32 C++ pipeline that turns a raw AD8232 ECG signal into the exact input the model expects.

Team: <LAL HARPIC> · Members: <Satyaroop Chattopadhyay> · Event: Edge AI Challenge, Electronics and IoT Club, IIT (ISM) Dhanbad (2-hour team event)
## 1. Problem overview

A wearable ECG patch must flag abnormal heartbeats on the device itself, with no cloud and almost no memory.

- **Input:** one pre-segmented heartbeat, 187 samples at 125 Hz, scaled to 0-1 (shorter beats zero-padded at the end).
- **Output:** one of 5 classes: N (normal), S (supraventricular ectopic), V (ventricular ectopic), F (fusion), Q (unclassifiable/paced).
- **Data:** only the organizer-supplied `train.csv` (87,554 beats, ~83% N, <1% F). Hidden test data adds sensor noise, baseline wander and amplitude changes.
- **Format:** exactly one fully int8 TFLite model, input `(1,187,1)` int8, output `(1,5)` int8, preprocessing must live inside the graph.
- **Hard limits:** weights <= 32 KiB, activation RAM <= 16 KiB, a restricted operator list.
- **Scoring:** macro-F1 (45), latency (20), size/RAM (15), deployment (10), documentation (10), with an accuracy gate before speed and size count. Latency and size are scored relative to the best team, so after the gate, small and fast beats big and slightly more accurate.

The task files (`train.csv`, `edge_helpers.py`, `check_submission.py`, problem statement) are provided by the
organizers and are **not** included in this repository.

## 2. Our approach in one paragraph

We treated the contest as an accuracy-versus-cost trade-off. EDA showed that beat length and where the beat
ends carry strong class information and that beats are per-beat min-max normalized and start near an R-peak,
so we used a strided Conv1D stack with a position-preserving head, avoided time-shift augmentation, and trained
with class weights and heavy noise/baseline/amplitude augmentation. We then swept model sizes, picked the
cost-effective knee, and added accuracy that costs nothing on the chip (knowledge distillation, SWA, bias folding).
The model uses only `CONV_2D`, `RESHAPE` and `FULLY_CONNECTED`.

## 3. Repository structure

```
.
├── README.md            this file
├── model.tflite         final fully-int8 model (13.6 KiB file, 7,165 parameters)
├── train.ipynb          Kaggle notebook: EDA, augmentation, model sweep, distillation, SWA, bias tuning, int8 export, checks
├── validation.txt       int8 validation macro-F1, per-class F1 and file size (written by the official helper)
├── note.md              one-page technical note submitted to the organizers
├── .gitignore
└── stm32/
    ├── ecg_pipeline.hpp   C interface: ecg_init(), ecg_on_adc_sample(), ecg_poll()
    └── ecg_pipeline.cpp   ECG acquisition and preprocessing pipeline + X-CUBE-AI inference call
```

### What each file does

| File | Role |
|---|---|
| `model.tflite` | The deliverable. Quantization parameters are stored inside the model (input scale 0.00649, zero-point -91). Passes `check_submission.py` (RESULT: PASS). |
| `train.ipynb` | Full pipeline, runnable top to bottom on Kaggle. Contains the EDA, the augmentation function, the model builder, class-weighted training, the teacher and student distillation, SWA, the size sweep, per-class bias tuning, int8 export through a batch-1 model, and evaluation of the *int8* model on clean and augmented validation sets. |
| `validation.txt` | Output of `evaluate_tflite` from the organizers' helper: int8 macro-F1, per-class F1, file size. |
| `note.md` | The short technical note (architecture, preprocessing, imbalance handling, optimization decisions). |
| `stm32/ecg_pipeline.hpp` | Three-function interface so the pipeline can be called from CubeMX-generated `main.c`. |
| `stm32/ecg_pipeline.cpp` | Sensor-side pipeline (see section 7). |

## 4. Final model

```
Input (187,1)
 -> Conv1D 8  k7 s2 + BN + ReLU   -> (94, 8)
 -> Conv1D 16 k5 s2 + BN + ReLU   -> (47,16)
 -> Conv1D 24 k5 s2 + BN + ReLU   -> (24,24)
 -> Conv1D 32 k3 s2 + BN + ReLU   -> (12,32)
 -> Reshape (384) -> Dropout -> Dense(5)   (int8 logits, no softmax)
```

7,165 parameters, about 110k MACs, 13.6 KiB file. Largest activation tensor is 94x8 bytes, far below the 16 KiB limit.

## 5. What we tried and what we learned

All numbers are int8 macro-F1 on a 15% stratified validation split. "Aug" is the same split with noise, baseline wander and amplitude changes applied, to mimic the hidden data.

### 5.1 Model size sweep (strided Conv1D + Flatten-style head)

| Model | Params | MACs | Clean | Aug |
|---|---|---|---|---|
| tiny (6-12-16-24) | 4.2k | 59k | 0.875 | 0.855 |
| **base (8-16-24-32)** | **7.2k** | **111k** | **0.902** | **0.885** |
| wide (12-24-32-48) | 13.3k | 226k | 0.910 | 0.887 |

### 5.2 Larger, higher-resolution alternative (stride-1 convs + MaxPool)

| Model | Params | MACs | Clean | Aug |
|---|---|---|---|---|
| global average pooling head | 20.8k | 584k | 0.909 | 0.881 |
| Flatten head | 24.0k | 587k | 0.914 | 0.898 |

Takeaways: the Flatten-style head beats global average pooling at equal cost, and the 5x larger model gained only about +0.01 clean F1 while adding MaxPool
(an operator outside the converter's safe set) and about 5x latency. We therefore chose **base**. A further, still larger variant (~980k MACs) was
rejected on cost without being trained. Differences below ~0.02 are within noise: class F has only 96 validation beats.

### 5.3 Free-at-inference improvements

| Technique | Result |
|---|---|
| Knowledge distillation from a larger float teacher (trained on `train.csv` only) | +0.009 aug F1 vs the plain student, zero inference cost |
| SWA over the last 8 epochs | removes bias from picking the best epoch on validation |
| Best of 2 seeds | more stable final model |
| Per-class logit bias tuned on half of validation, checked on the other half, shrunk by 0.5, folded into the Dense bias | kept only if held-out F1 improved |

### 5.4 Engineering issues we hit and fixed

| Issue | Fix |
|---|---|
| Checker rejected `PACK`, `SHAPE`, `STRIDED_SLICE` | Keras `Flatten` with a dynamic batch size produced them. We export a batch-1 copy of the model and use a static `Reshape`. |
| `RELU6` is outside the converter's known-safe operator set | Switched to plain ReLU. |
| Default representative set subsampled to 500 beats, thinning the rare classes | Pass a stratified clean + augmented calibration set with `n_rep=len(rep_x)`. |
| `label_smoothing` argument unsupported in our Keras version | Custom smoothed cross-entropy loss. |

### 5.5 Final result

| | Clean val | Augmented val |
|---|---|---|
| Final int8 model | **0.903** | **0.883** |

S and F are the hardest classes (mostly confused with N). The hidden-set score is expected to be lower: `train.csv` has no patient IDs, so a random split is optimistic.

## 6. What makes this solution different

- **Data-driven design.** Augmentation and architecture follow the EDA: no time shifts (beats start at the R-peak, length is a class cue), padding kept clean, re-normalization and clipping hedged because the hidden scaling is unknown, and a head that keeps timing information.
- **Cost-aware model selection.** We measured MACs against F1 and the converter's operator set, instead of maximizing accuracy alone.
- **Zero-cost accuracy.** Distillation, SWA and bias folding improve results without adding a single operator or byte.
- **End-to-end view.** The repository also contains the code that produces the model's input on real hardware (next section), not just the model.

## 7. STM32 C++ pipeline (`stm32/`): its role

The model alone expects a perfectly prepared beat: 187 samples, 0-1 scaled per beat, zero-padded, int8 quantized with the model's own scale and zero-point. A real wearable only has a continuous analog signal from an ECG front-end. `ecg_pipeline.cpp` is the bridge between the two, and it makes the on-device input match the distribution the model was trained on. It is **sensor-side code only**; it is not part of the scored model file.

**Data flow**

```
AD8232 analog out
  -> ADC1 @ 1 kHz (triggered by TIM6 TRGO, end-of-conversion interrupt)
  -> ISR: boxcar-average 8 samples -> 125 Hz stream (matches the dataset rate, adds anti-alias averaging)
  -> main loop: single-pole high-pass (~0.5 Hz) removes baseline wander
  -> adaptive-threshold R-peak detector (running amplitude estimate, 200 ms refractory period)
  -> beat window = R-peak to next R-peak + small tail margin (mimics the dataset's segmentation)
  -> per-beat min-max to 0-1, zero-pad to 187 samples
  -> quantize: q = round(x / 0.00649) + (-91), clamped to int8   (values read from our model)
  -> X-CUBE-AI: ai_network_run()
  -> argmax of int8 logits -> class label (N/S/V/F/Q) + latency in microseconds via the DWT cycle counter
```

**Interface (`ecg_pipeline.hpp`)**

| Function | Called from | Purpose |
|---|---|---|
| `ecg_init()` | `main()` after peripheral init | Initialises the X-CUBE-AI network and the cycle counter |
| `ecg_on_adc_sample(uint16_t)` | `HAL_ADC_ConvCpltCallback` (1 kHz) | Decimates to 125 Hz and queues samples |
| `ecg_poll()` | main loop | Filters, detects R-peaks, builds beats, runs inference, prints the result |

**CubeMX setup:** TIM6 at 1 kHz with TRGO on update; ADC1 12-bit with TIM6 TRGO as external trigger and its interrupt enabled; X-CUBE-AI with the network named `network`; `printf` over UART for output.

**Honest limitations**
- This code has **not been compiled or run on hardware**. It is a reference implementation written against the standard HAL and X-CUBE-AI API patterns; generated names may differ slightly per tool version.
- The beat-window convention (R-peak to next R-peak plus a tail margin, `TAIL_MARGIN` in the source) is inferred from our EDA of `train.csv`, not from the organizers' documentation, and may need tuning.
- If the model is retrained, update `INPUT_SCALE` and `INPUT_ZP` at the top of `ecg_pipeline.cpp` to the new model's printed input quantization parameters.
- Mains (50 Hz) interference is not notched; a short moving average or notch can be added in `process()`.

## 8. Reproducing the results

1. Get `train.csv`, `edge_helpers.py` and `check_submission.py` from the organizers.
2. Create a Kaggle notebook, attach `train.csv` as an input dataset, and place `edge_helpers.py` and `check_submission.py` in the working directory (the notebook copies them from `/kaggle/input` when present).
3. Import or paste `train.ipynb` and run it top to bottom (GPU recommended; the teacher is the slowest step).
4. Verify with `python check_submission.py --model model.tflite --check-only` (expect `RESULT: PASS`).

Requirements: Python 3.10+, TensorFlow 2.x/Keras, scikit-learn, NumPy, pandas.

## 9. Acknowledgements and licence

The challenge, dataset and helper scripts were provided by the Electronics and IoT Club, IIT (ISM) Dhanbad.