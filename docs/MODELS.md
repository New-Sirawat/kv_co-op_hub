# Target Model Families & Version Pinning (WP1-1)

> **Work Package**: `[WP1-1] Research and specify target model families & version pinning (16GB budget)`  
> **Parent Issue**: [`khemingkapat/kv_co-op_hub#1`](https://github.com/khemingkapat/kv_co-op_hub/issues/1)  
> **Code Issue**: [`khemingkapat/kv_bench#2`](https://github.com/khemingkapat/kv_bench/issues/2)  
> **Target Hardware**: Slurm HPC Node (RTX 4090 capped at 16 GB VRAM) & Local Desktop (RTX 4060 8 GB VRAM)

---

## 1. Executive Summary & Design Rationale

The objective of **WP1-1** is to evaluate, select, and pin open-source model checkpoints for the **KV-Cache Co-Optimization** benchmark testbed.

To evaluate KV-cache management across structural (compression/quantization), spatial (CPU offloading), and temporal (transfer scheduling) dimensions, models must adhere to four strict principles:
1. **Unmodified Weights (FP16/BF16):** Model weights cannot be pre-quantized (no AWQ/GPTQ/GGUF), preserving full KV attention dynamics.
2. **Headroom for KV Scaling:** Static model weights must occupy $\le 8\text{ GB}$, leaving at least $8\text{ GB}$ of dedicated VRAM headroom on the 16 GB hardware cap to sweep long contexts ($512 \rightarrow 32,768$ tokens) and simulate concurrent SME traffic (10–15 sessions).
3. **Pure Dense vs. Architectural Hybrid Separation:** 
   * **Pure Dense GQA Models** serve as the core benchmark workhorses. In these models, 100% of the layers produce uniform global attention ($72\text{--}144\text{ KiB/token}$), exposing the raw hardware memory wall.
   * **Hybrid Models (Linear Attention / Sliding Window)** serve as comparative case studies to evaluate how system-level co-optimization interacts with models that already incorporate architectural KV suppression.
4. **Immutable Version Pinning:** Every checkpoint is locked to an exact Git commit SHA on Hugging Face to ensure strict academic reproducibility.

---

## 2. Pinned Model Registry

All models use native Bfloat16 precision and support long-context windows ($\ge 32,768$ tokens).

| Category / Role | Model Name | Hugging Face Repo ID | Pinned Git Commit SHA | Arch Class | Layers ($L$) | Attn Heads ($Q$ / $KV$) | Head Dim ($D$) | KV Bytes / Token | Native Max Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Dense (16GB)** | **Qwen 3 4B Instruct (2507)** | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | `Qwen3ForCausalLM` | 36 | 32 / 8 (4:1) | 128 | **$144\text{ KiB}$** | 262,144 |
| **Primary Dense Peer (16GB)** | **Phi-4 mini Instruct** | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | `Phi3ForCausalLM` | 32 | 24 / 8 (3:1) | 128 | **$128\text{ KiB}$** | 131,072 |
| **Comparative Dense (16GB)** | **Ministral 3 3B Base (2512)** | `mistralai/Ministral-3-3B-Base-2512` | `6f9c4b12a95b139af68670a6713616b757923735` | `Mistral3` (text-only) | 26 | 32 / 8 (4:1) | 128 | **$104\text{ KiB}$** | 262,144 |
| **Efficiency Dense (16GB)** | **SmolLM3 3B** | `HuggingFaceTB/SmolLM3-3B` | `a07cc9a04f16550a088caea529712d1d335b0ac1` | `SmolLM3ForCausalLM` | 36 | 16 / 4 (4:1) | 128 | **$72\text{ KiB}$** | 65,536 |
| **Local 8GB Adaptation** | **Qwen 3 1.7B** | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `Qwen3ForCausalLM` | 28 | 16 / 8 (2:1) | 128 | **$112\text{ KiB}$** | 40,960 |
| **Comparative Hybrid (16GB)** | **Qwen 3.5 4B** | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `Qwen3_5ForConditionalGen` | 32 ($24\text{ GDN} + 8\text{ Attn}$) | 16 / 4 (4:1) | 256 | **$32\text{ KiB}$** (+ GDN) | 262,144 |
| **Local 8GB Hybrid** | **Qwen 3.5 2B** | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | `Qwen3_5ForConditionalGen` | 24 ($18\text{ GDN} + 6\text{ Attn}$) | 8 / 2 (4:1) | 256 | **$12\text{ KiB}$** (+ GDN) | 262,144 |
| **Comparative SWA (16GB)** | **Gemma 4 E4B Instruct** | `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | `Gemma4` (Sliding+Shared) | 42 | 8 / 2 | 256 | Heterogeneous | 131,072 |
| **Literature Reference** | **Qwen 2.5 3B Instruct** | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `Qwen2ForCausalLM` | 36 | 16 / 2 (8:1) | 128 | **$36\text{ KiB}$** | 32,768 |

> **Multimodal Flag Requirement:**  
> Checkpoints that bundle vision backbones (`Qwen 3.5`, `Ministral 3`, `Gemma 4`) must be executed with `--language-model-only` to avoid reserving $0.8\text{--}1.5\text{ GB}$ of VRAM for vision encoders during text-only KV benchmarking.

---

## 3. KV-Cache Memory Footprint Breakdown

### Mathematical Formula
For Grouped-Query Attention (GQA) in BF16 ($2\text{ bytes}$ per value, storing both Keys and Values):
$$\text{KV Bytes per Token} = 2 \times 2 \times L_{\text{full\_attn}} \times H_{KV} \times D_{\text{head}} = 4 \times L_{\text{full\_attn}} \times H_{KV} \times D_{\text{head}}$$
$$\text{Total KV Cache (GB)} = \frac{\text{KV Bytes per Token} \times \text{Sequence Length} \times \text{Concurrent Sessions}}{1024^3}$$

### Empirical Allocation Matrix (16GB & 8GB Budgets)

| Model | BF16 Weights | KV Bytes / Token | 4K Context (1x) | 32K Context (1x) | 10 Users @ 4K | 10 Users @ 32K | Headroom on 16GB Cap (Weights Subtracted) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3 4B Instruct** | $\approx 8.06\text{ GB}$ | **$144\text{ KiB}$** | $0.56\text{ GB}$ | **$4.50\text{ GB}$** | $5.63\text{ GB}$ | $45.0\text{ GB}$ *(OOM)* | **$\approx 7.94\text{ GB}$** (Fits 1 user @ 32K; OOMs at multi-user) |
| **Phi-4 mini Instruct** | $\approx 7.65\text{ GB}$ | **$128\text{ KiB}$** | $0.50\text{ GB}$ | **$4.00\text{ GB}$** | $5.00\text{ GB}$ | $40.0\text{ GB}$ *(OOM)* | **$\approx 8.35\text{ GB}$** (Fits 1 user @ 32K; OOMs at multi-user) |
| **Ministral 3 3B Base** | $\approx 6.80\text{ GB}$ | **$104\text{ KiB}$** | $0.41\text{ GB}$ | **$3.25\text{ GB}$** | $4.06\text{ GB}$ | $32.5\text{ GB}$ *(OOM)* | **$\approx 9.20\text{ GB}$** (High headroom for concurrent sweeps) |
| **SmolLM3 3B** | $\approx 6.00\text{ GB}$ | **$72\text{ KiB}$** | $0.28\text{ GB}$ | **$2.25\text{ GB}$** | $2.81\text{ GB}$ | $22.5\text{ GB}$ *(OOM)* | **$\approx 10.00\text{ GB}$** |
| **Qwen 3.5 4B (Hybrid)** | $\approx 8.10\text{ GB}$ | **$32\text{ KiB}$** + GDN | $0.13\text{ GB}$ | **$1.00\text{ GB}$** | $1.25\text{ GB}$ | $10.0\text{ GB}$ *(OOM)* | **$\approx 7.90\text{ GB}$** (Architectural reduction active) |
| **Qwen 3 1.7B (Local 8GB)**| $\approx 3.45\text{ GB}$ | **$112\text{ KiB}$** | $0.44\text{ GB}$ | **$3.50\text{ GB}$** | $4.38\text{ GB}$ *(OOM)* | $35.0\text{ GB}$ *(OOM)* | **$\approx 4.55\text{ GB}$ on 8GB cap** (1x 32K uses $6.95\text{ GB}$) |
| **Qwen 2.5 3B (Lit. Ref)** | $\approx 6.10\text{ GB}$ | **$36\text{ KiB}$** | $0.14\text{ GB}$ | **$1.13\text{ GB}$** | $1.41\text{ GB}$ | $11.3\text{ GB}$ *(OOM)* | **$\approx 9.90\text{ GB}$** |

---

## 4. Software Stack & Engine Compatibility

To support modern architectures while eliminating legacy monkey patches in `kv_bench`:

* **Inference Engine:** `vllm >= 0.20.0` (V1 Engine Architecture).
* **Tiered Cache Engine:** `lmcache >= 0.5.0` with `LMCacheMPConnector` for decoupled inter-process cache transfers.
* **Kernel Extensions:** `flash-linear-attention` and `causal-conv1d` for hybrid Gated DeltaNet layers.
* **Version Parity:** All models in the benchmark run against the identical engine version, PyTorch version ($\ge 2.6.0$), and CUDA toolchain (12.2) to prevent confounding software optimizations from compromising empirical validity.
