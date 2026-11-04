# Cost statistics of the thesis models

18 models: ten trainings in the version selected by the mean of the UCF/MIVIA Protocol Means at seed 42 (OVAD, guard 30 s), and eight E3-S sequential combinations. The version is fixed across seeds. E3-SV and E3-SO are excluded (BRIEF chapter 4, rules 3 and 7). Only deployed inference forms are reported (rule 10).
Seed 42 is reported first; variability is separate. Per-seed values follow the order 42, 67, 104 where available.

## 1. Selected checkpoint over the four epochs

Global step = steps inherited by the reported version along its chain (each warm start or pre-adaptation receipt names the step of its parent) + the step the model selection kept in it (validation AUROC every 920 steps). One epoch = 5524 steps. A continuation inherits the LAST step of its parent; E3-S includes its one-epoch source.

| Model | Chain (seed 42, epochs) | Epochs (version / chain) | Seeds | Local step | Global step | Chain steps | Val AUROC | Adapter params | Head params |
|---|---|---|---|---|---|---|---|---|---|
| Linear probe | e2c 1 → e3m 2 → e3m4 1 | 3 / 4 | 3 | 9 200 · 11 040 · 10 120 | 14 724 · 16 564 · 15 644 | 22 096 | 0.7924 · 0.7887 · 0.7872 | 0 | 3 073 |
| Attentive probe | e2d 1 → e3m 2 → e3m4 1 | 1 / 4 | 3 | 4 600 · 3 680 · 2 760 | 4 600 · 3 680 · 2 760 | 22 096 | 0.9370 · 0.9150 · 0.9288 | 0 | 49 339 393 |
| LoRA r=32 | e3r 1 → e3m 2 → e3m4 1 | 1 / 4 | 3 | 5 520 · 4 600 · 3 680 | 5 520 · 4 600 · 3 680 | 22 096 | 0.9466 · 0.9484 · 0.9499 | 14 155 776 | 49 339 393 |
| DoRA r=32 | e3r 1 → e3m 2 → e3m4 1 | 4 / 4 | 3 | 5 520 · 920 · 1 840 | 22 092 · 17 492 · 18 412 | 22 096 | 0.9489 · 0.9453 · 0.9486 | 14 376 960 | 49 339 393 |
| VPT-deep n=8 | e3p 1 → e3m 2 → e3m4 1 | 4 / 4 | 3 | 5 520 · 2 760 · 4 600 | 22 092 · 19 332 · 21 172 | 22 096 | 0.9273 · 0.9261 · 0.9305 | 196 608 | 49 339 393 |
| VPT-shallow n=8 | e3p 1 → e3m 2 → e3m4 1 | 1 / 4 | 3 | 5 520 · 5 520 · 5 520 | 5 520 · 5 520 · 5 520 | 22 096 | 0.9320 · 0.9215 · 0.9303 | 8 192 | 49 339 393 |
| LoRA r=32 + VPT-shallow n=8 | e3c 3 → e3c4 1 | 4 / 4 | 3 | 920 · 4 600 · 920 | 17 492 · 21 172 · 17 492 | 22 096 | 0.9483 · 0.9485 · 0.9456 | 14 163 968 | 49 339 393 |
| LoRA r=32 + VPT-deep n=8 | e3c 3 → e3c4 1 | 4 / 4 | 3 | 3 680 · 4 600 · 4 600 | 20 252 · 21 172 · 21 172 | 22 096 | 0.9423 · 0.9498 · 0.9425 | 14 352 384 | 49 339 393 |
| DoRA r=32 + VPT-shallow n=8 | e3c 3 → e3c4 1 | 4 / 4 | 3 | 3 680 · 3 680 · 3 680 | 20 252 · 20 252 · 20 252 | 22 096 | 0.9416 · 0.9493 · 0.9534 | 14 385 152 | 49 339 393 |
| DoRA r=32 + VPT-deep n=8 | e3c 3 → e3c4 1 | 4 / 4 | 3 | 920 · 920 · 920 | 17 492 · 17 492 · 17 492 | 22 096 | 0.9515 · 0.9357 · 0.9480 | 14 573 568 | 49 339 393 |
| VPT-shallow n=8 on fused LoRA r=32 | e3r 1 → e3s 3 | 4 / 4 | 3 | 12 880 · 13 800 · 13 800 | 18 404 · 19 324 · 19 324 | 22 096 | 0.9404 · 0.9466 · 0.9381 | 8 192 | 49 339 393 |
| VPT-deep n=8 on fused LoRA r=32 | e3r 1 → e3s 3 | 4 / 4 | 3 | 15 640 · 12 880 · 3 680 | 21 164 · 18 404 · 9 204 | 22 096 | 0.9407 · 0.9477 · 0.9372 | 196 608 | 49 339 393 |
| VPT-shallow n=8 on fused DoRA r=32 | e3r 1 → e3s 3 | 4 / 4 | 3 | 11 960 · 11 960 · 13 800 | 17 484 · 17 484 · 19 324 | 22 096 | 0.9420 · 0.9466 · 0.9388 | 8 192 | 49 339 393 |
| VPT-deep n=8 on fused DoRA r=32 | e3r 1 → e3s 3 | 4 / 4 | 3 | 14 720 · 11 040 · 11 040 | 20 244 · 16 564 · 16 564 | 22 096 | 0.9401 · 0.9423 · 0.9368 | 196 608 | 49 339 393 |
| LoRA r=32 on frozen VPT-shallow n=8 | e3p 1 → e3s 3 | 4 / 4 | 3 | 12 880 · 10 120 · 14 720 | 18 404 · 15 644 · 20 244 | 22 096 | 0.9560 · 0.9480 · 0.9449 | 14 163 968 | 49 331 201 |
| LoRA r=32 on frozen VPT-deep n=8 | e3p 1 → e3s 3 | 4 / 4 | 3 | 9 200 · 16 560 · 11 960 | 14 724 · 22 084 · 17 484 | 22 096 | 0.9431 · 0.9464 · 0.9466 | 14 352 384 | 49 142 785 |
| DoRA r=32 on frozen VPT-shallow n=8 | e3p 1 → e3s 3 | 4 / 4 | 3 | 14 720 · 9 200 · 13 800 | 20 244 · 14 724 · 19 324 | 22 096 | 0.9511 · 0.9470 · 0.9518 | 14 385 152 | 49 331 201 |
| DoRA r=32 on frozen VPT-deep n=8 | e3p 1 → e3s 3 | 4 / 4 | 3 | 11 960 · 16 560 · 12 880 | 17 484 · 22 084 · 18 404 | 22 096 | 0.9457 · 0.9443 · 0.9399 | 14 573 568 | 49 142 785 |

## 2. Training cost

Seconds per step: median of the intervals between checkpoints without a validation in the SLURM logs, pooled over the runs of the same architecture; a validation pass is the median excess of the intervals containing one. Hours = steps × s/step + validations × s/validation, on ONE GPU type (never converted). *4 epochs* = the whole chain; *to checkpoint* = the chain up to the selected step. *Consumed* = elapsed time of the real seed-42 training jobs (resumes, migrations and final evaluation included, A100 + H100).

| Model | s/step A100 | s/step H100 | validation min A100 | validation min H100 | 4 ep. h A100 | 4 ep. h H100 | to ckpt h A100 s42 | to ckpt h H100 s42 | to ckpt h A100 mean ± sd | to ckpt h H100 mean ± sd | consumed GPU-h (A100 + H100) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear probe | 2.00 | 1.40 | 1.9 | 1.6 | 13.1 | 9.2 | 8.7 | 6.1 | 9.3 ± 0.5 | 6.5 ± 0.4 | 0.0 + 2.4 |
| Attentive probe | — | 1.53 | — | 2.0 | — | 10.2 | — | 2.1 | — | 1.7 ± 0.4 | 0.0 + 2.9 |
| LoRA r=32 | 8.84 | 5.17 | 27.4 | 16.1 | 65.2 | 38.2 | 16.3 | 9.5 | 13.6 ± 2.7 | 7.9 ± 1.6 | 19.9 + 0.0 |
| DoRA r=32 | 11.48 | 6.81 | 34.1 | 20.3 | 84.1 | 49.9 | 84.1 | 49.9 | 73.6 ± 9.3 | 43.7 ± 5.5 | 20.9 + 1.9 |
| VPT-deep n=8 | 6.19 | 3.49 | 18.1 | 10.2 | 45.2 | 25.5 | 45.2 | 25.5 | 42.7 ± 2.9 | 24.1 ± 1.6 | 12.2 + 0.0 |
| VPT-shallow n=8 | 6.07 | 3.42 | 17.6 | 10.0 | 44.3 | 25.0 | 11.1 | 6.3 | 11.1 ± 0.0 | 6.3 ± 0.0 | 12.1 + 0.0 |
| LoRA r=32 + VPT-shallow n=8 | — | 5.13 | — | 15.2 | — | 37.6 | — | 29.7 | — | 31.8 ± 3.6 | 0.0 + 0.0 |
| LoRA r=32 + VPT-deep n=8 | — | 5.17 | — | 15.3 | — | 37.9 | — | 34.7 | — | 35.8 ± 0.9 | 0.0 + 0.0 |
| DoRA r=32 + VPT-shallow n=8 | — | 6.74 | — | 19.4 | — | 49.1 | — | 45.0 | — | 45.0 ± 0.0 | 0.0 + 0.0 |
| DoRA r=32 + VPT-deep n=8 | — | 6.79 | — | 19.5 | — | 49.5 | — | 39.2 | — | 39.2 ± 0.0 | 0.0 + 0.0 |
| VPT-shallow n=8 on fused LoRA r=32 | — | — | — | — | — | — | — | — | — | — | 19.9 + 0.0 |
| VPT-deep n=8 on fused LoRA r=32 | — | — | — | — | — | — | — | — | — | — | 19.9 + 0.0 |
| VPT-shallow n=8 on fused DoRA r=32 | — | — | — | — | — | — | — | — | — | — | 20.9 + 1.9 |
| VPT-deep n=8 on fused DoRA r=32 | — | — | — | — | — | — | — | — | — | — | 20.9 + 1.9 |
| LoRA r=32 on frozen VPT-shallow n=8 | — | — | — | — | — | — | — | — | — | — | 12.1 + 0.0 |
| LoRA r=32 on frozen VPT-deep n=8 | — | — | — | — | — | — | — | — | — | — | 12.2 + 0.0 |
| DoRA r=32 on frozen VPT-shallow n=8 | — | — | — | — | — | — | — | — | — | — | 12.1 + 0.0 |
| DoRA r=32 on frozen VPT-deep n=8 | — | — | — | — | — | — | — | — | — | — | 12.2 + 0.0 |

s/step and validation are those of the model's own (last) stage; a sequential model's chain also contains its one-epoch source, costed with the source architecture.

## 3. Memory

Training: peak GPU memory of the real training jobs (`gres/gpumem` of SLURM accounting, NVML: allocator cache and CUDA context included; not accounted on every node, hence the job count), micro-batch 2 in bf16, and host RSS (data-loader workers included). Inference: peak memory allocated by PyTorch in E5b, batch 1, deployed form.

| Model | train GPU peak GiB | jobs (with gpumem / all) | train host RSS GiB | inference GiB (deployed) |
|---|---|---|---|---|
| Linear probe | 2.85 | 2/3 | 39.4 | 1.631 |
| Attentive probe | — | 0/3 | 44.0 | 1.783 |
| LoRA r=32 | 27.58 | 5/8 | 59.4 | 1.783 |
| DoRA r=32 | 43.84 | 3/12 | 59.9 | 1.783 |
| VPT-deep n=8 | 18.32 | 8/14 | 51.3 | 1.784 |
| VPT-shallow n=8 | 18.32 | 4/6 | 51.4 | 1.783 |
| LoRA r=32 + VPT-shallow n=8 | — | 0/4 | 59.6 | 1.783 |
| LoRA r=32 + VPT-deep n=8 | — | 0/4 | 60.5 | 1.783 |
| DoRA r=32 + VPT-shallow n=8 | — | 0/4 | 59.6 | 1.782 |
| DoRA r=32 + VPT-deep n=8 | — | 0/4 | 59.1 | 1.783 |
| VPT-shallow n=8 on fused LoRA r=32 | — | 0/0 | — | 1.783 |
| VPT-deep n=8 on fused LoRA r=32 | — | 0/0 | — | 1.783 |
| VPT-shallow n=8 on fused DoRA r=32 | — | 0/0 | — | 1.782 |
| VPT-deep n=8 on fused DoRA r=32 | — | 0/0 | — | 1.783 |
| LoRA r=32 on frozen VPT-shallow n=8 | — | 0/0 | — | 1.783 |
| LoRA r=32 on frozen VPT-deep n=8 | — | 0/0 | — | 1.783 |
| DoRA r=32 on frozen VPT-shallow n=8 | — | 0/0 | — | 1.782 |
| DoRA r=32 on frozen VPT-deep n=8 | — | 0/0 | — | 1.783 |

## 4. Response time and detection delay

`alarm delay = grid delay + compute + decoding`. Compute: E5b, batch 1, bf16, deployed form (reparam merged; partially merged for the combinations, ADR-064). Decoding: 40.8 ms/window (E0d). Grid delay: median over the 276 test events, intersection rule (ADR-061), seed 42 followed by mean ± sd over the available seeds. Real-time factor = hop (4 s) / response.

| Model | merge | compute ms A100 | compute ms H100 | response ms A100 | response ms H100 | grid median s42 s | grid p90 s42 s | alarm s42 A100 s | alarm s42 H100 s | grid median mean ± sd s | grid p90 mean ± sd s | alarm A100 mean ± sd s | alarm H100 mean ± sd s | RTF A100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear probe | nessuna | 121.2 | 71.7 | 162.0 | 112.5 | 2.600 | 3.967 | 2.762 | 2.712 | 2.539 ± 0.054 | 3.944 ± 0.019 | 2.701 ± 0.054 | 2.651 ± 0.054 | 24.7 |
| Attentive probe | nessuna | 131.8 | 77.5 | 172.6 | 118.3 | 2.967 | 5.887 | 3.139 | 3.085 | 2.967 ± 0.000 | 5.693 ± 0.370 | 3.139 ± 0.000 | 3.085 ± 0.000 | 23.2 |
| LoRA r=32 | totale | 132.0 | 77.5 | 172.8 | 118.3 | 2.900 | 5.967 | 3.073 | 3.018 | 2.844 ± 0.126 | 5.811 ± 0.537 | 3.017 ± 0.126 | 2.963 ± 0.126 | 23.1 |
| DoRA r=32 | totale | 131.9 | 77.8 | 172.7 | 118.6 | 2.883 | 6.240 | 3.056 | 3.002 | 2.717 ± 0.144 | 5.547 ± 0.601 | 2.889 ± 0.144 | 2.835 ± 0.144 | 23.2 |
| VPT-deep n=8 | nessuna | 125.5 | 72.9 | 166.3 | 113.7 | 2.750 | 5.333 | 2.916 | 2.864 | 2.761 ± 0.134 | 5.196 ± 0.122 | 2.927 ± 0.134 | 2.875 ± 0.134 | 24.1 |
| VPT-shallow n=8 | nessuna | 124.5 | 72.3 | 165.3 | 113.1 | 2.933 | 5.360 | 3.099 | 3.046 | 2.878 ± 0.096 | 5.317 ± 0.067 | 3.043 ± 0.096 | 2.991 ± 0.096 | 24.2 |
| LoRA r=32 + VPT-shallow n=8 | parziale | 124.2 | 73.6 | 165.0 | 114.4 | 2.800 | 5.547 | 2.965 | 2.914 | 2.828 ± 0.095 | 5.696 ± 0.235 | 2.993 ± 0.095 | 2.942 ± 0.095 | 24.2 |
| LoRA r=32 + VPT-deep n=8 | parziale | 125.8 | 74.1 | 166.6 | 114.9 | 2.883 | 6.060 | 3.050 | 2.998 | 2.789 ± 0.082 | 5.710 ± 0.414 | 2.955 ± 0.082 | 2.904 ± 0.082 | 24.0 |
| DoRA r=32 + VPT-shallow n=8 | parziale | 125.2 | 73.4 | 166.0 | 114.2 | 2.767 | 5.293 | 2.933 | 2.881 | 2.833 ± 0.088 | 5.707 ± 0.360 | 2.999 ± 0.088 | 2.948 ± 0.088 | 24.1 |
| DoRA r=32 + VPT-deep n=8 | parziale | 126.2 | 74.2 | 167.0 | 115.0 | 2.967 | 6.190 | 3.134 | 3.082 | 2.889 ± 0.107 | 5.739 ± 0.477 | 3.056 ± 0.107 | 3.004 ± 0.107 | 24.0 |
| VPT-shallow n=8 on fused LoRA r=32 | parziale | 124.2 | 73.6 | 165.0 | 114.4 | 2.700 | 5.100 | 2.865 | 2.814 | 2.833 ± 0.153 | 5.554 ± 0.697 | 2.998 ± 0.153 | 2.948 ± 0.153 | 24.2 |
| VPT-deep n=8 on fused LoRA r=32 | parziale | 125.8 | 74.1 | 166.6 | 114.9 | 2.933 | 6.130 | 3.100 | 3.048 | 2.889 ± 0.077 | 5.841 ± 0.299 | 3.055 ± 0.077 | 3.004 ± 0.077 | 24.0 |
| VPT-shallow n=8 on fused DoRA r=32 | parziale | 125.2 | 73.4 | 166.0 | 114.2 | 3.000 | 6.367 | 3.166 | 3.114 | 2.822 ± 0.158 | 5.658 ± 0.614 | 2.988 ± 0.158 | 2.936 ± 0.158 | 24.1 |
| VPT-deep n=8 on fused DoRA r=32 | parziale | 126.2 | 74.2 | 167.0 | 115.0 | 2.933 | 6.213 | 3.100 | 3.048 | 2.767 ± 0.153 | 5.829 ± 0.416 | 2.934 ± 0.153 | 2.882 ± 0.153 | 24.0 |
| LoRA r=32 on frozen VPT-shallow n=8 | parziale | 124.2 | 73.6 | 165.0 | 114.4 | 2.950 | 6.267 | 3.115 | 3.064 | 2.828 ± 0.107 | 5.629 ± 0.591 | 2.993 ± 0.107 | 2.942 ± 0.107 | 24.2 |
| LoRA r=32 on frozen VPT-deep n=8 | parziale | 125.8 | 74.1 | 166.6 | 114.9 | 2.733 | 5.930 | 2.900 | 2.848 | 2.844 ± 0.139 | 5.879 ± 0.522 | 3.011 ± 0.139 | 2.959 ± 0.139 | 24.0 |
| DoRA r=32 on frozen VPT-shallow n=8 | parziale | 125.2 | 73.4 | 166.0 | 114.2 | 2.800 | 5.207 | 2.966 | 2.914 | 2.806 ± 0.125 | 5.547 ± 0.683 | 2.972 ± 0.125 | 2.920 ± 0.125 | 24.1 |
| DoRA r=32 on frozen VPT-deep n=8 | parziale | 126.2 | 74.2 | 167.0 | 115.0 | 2.833 | 5.300 | 3.000 | 2.948 | 2.833 ± 0.033 | 5.393 ± 0.134 | 3.000 ± 0.033 | 2.948 ± 0.033 | 24.0 |

## 5. Provenance and caveats

* **A100**: the combination profile `e5b4_combo_gpuq` and the reference `e5b3_efficienza_gpuq` share dora-r32-fusa, frozen, lora-r32-fusa, vptdeep-n8, vptshallow-n8; maximum latency deviation 0.40%.
* **H100**: the combination profile `e5b4_combo_aiq` and the reference `e5b3_efficienza_aiq` share dora-r32-fusa, frozen, lora-r32-fusa, vptdeep-n8, vptshallow-n8; maximum latency deviation 1.97%.
* The step cost of an architecture is the median over all its runs; per-run medians agree within 0.5%, except the two one-epoch probes at seed 42 (31 August, H100): 1.74 and 1.40 s/step against 1.52 and 1.17 of every later run (`per_run` in the JSON).
* The reported version is read from OVAD guard-30 selezione_versioni.json: mean of the UCF/MIVIA Protocol Means at seed 42, on the TEST sets. Checkpoint selection within each version remains validation AUROC. The same version is used for every seed.
* Seed availability is listed per model. Training costs are pooled per architecture and GPU.
* A100 and H100 hours are two measurements, never a conversion: where a GPU never ran an architecture long enough (5 intervals), its cell is empty.
* Training GPU memory is NVML (allocator cache and context included) and comes from the nodes that account `gres/gpumem`; inference memory is the PyTorch allocated peak. They are not the same quantity.
* The response time is a composition of measures taken on different machines (model on A100/H100, decoder on an H100 node) without scheduling or contention.
