# Chapter 2 Blueprint: Related Works & Theoretical Foundations

> **KMUTT Baseline Mapping:** Chapter 2 — *Related Works* (`Progress_Sem1_ENG_Example2.pdf`, Sections `2.1`–`2.7`)  
> **Primary Repo Sources:** `docs/literature_survey/` (20 completed technical reviews: 3 `foundations/`, 11 `techniques/`, 6 `benchmarks/`), `docs/MODELS.md`, `docs/METRICS_AND_AXES.md`, `docs/WORKFLOW_SPEC.md`, `results/manifests/run_manifest_baseline_*.json`  
> **Blueprint Role:** Templating and agent handoff specification — **not** final thesis prose.

---

## Chapter-Level Scope & Guardrails

- **Literature Coverage:** Ground Chapter 2 strictly in the **20 completed implementation-grade reviews** on `main` in `docs/literature_survey/`. Any unreviewed foundational/technique papers mentioned for completeness (FlashAttention, PagedAttention/vLLM, Jiang 2026 KV Survey, SnapKV, PyramidKV, KIVI, PG-19/WikiText-2) MUST be explicitly tagged `[NEEDS_VERIFICATION]` until their review files are completed or primary citations are directly verified.
- **Scope Enforcement (`.agents/AGENTS.md`):**
  - Frame all theoretical limits around **consumer GPU VRAM ceilings** (16 GB cluster cap on RTX 4090; 8 GB local cap on RTX 4060) and **controlled unmodified BF16/FP16 base weights** (no pre-quantized GGUF/AWQ base models).
  - Emphasize the **Two Primary Quality Pillars** (Micro-Level Logit Divergence + Macro-Level LongMemEval) and explain theoretically (via *Illusion of Equivalence* and *LoCoMo* / *SEAL*) why Perplexity ($\Delta\text{PPL}$) is strictly supplementary.
  - Keep **WP-4 co-optimization policy architecture** framed as an evidence-driven synthesis target (`[OPEN_DECISION]`), referencing WP-3 diagnostic findings only under `[DRAFT/PR#21]`.

---

## Section 2.1: Background

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`2.1 Background`):** Establishes the domain context of local/on-premise LLM deployment for SMEs and individual developers, framing why long-context autoregressive inference hits a hard consumer-GPU memory wall before compute saturation.
- **Committee Expectation:** Show clear domain motivation linking real-world adoption (local AI assistants, multi-session chat, code/document review) to the specific hardware bottleneck investigated in this Senior Project.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Modern LLM inference is bifurcated into compute-bound **prefill** ($O(N^2)$ GEMM) and memory-bandwidth-bound **decode** ($O(N)$ GEMV, arithmetic intensity $\approx 1\text{ FLOP/byte}$). While KV-caching eliminates quadratic recomputation, its linear memory growth collides with consumer GPU VRAM ceilings (8–16 GB) long before compute saturates, causing immediate OOM crashes under 32K contexts or 10–15 concurrent SME sessions.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf` & `Slide_Content_and_Executive_Summary.md`):**
  - *Retained:* Consumer hardware floor motivation (OpenClaw on Slide 02 [PDF p. 3] of `KV-Cache Co-optimization.pdf`; Ollama, LM Studio, and 64K warnings in `Slide_Content_and_Executive_Summary.md` Slide 03; RTX 4060 8 GB / 16 GB consumer envelope).
  - *Refined:* Replaced generic 7B–8B memory examples with exact, empirically pinned 3B–4B BF16 footprint calculations from `docs/MODELS.md` (e.g., `Qwen3-4B-Instruct-2507` at $144\text{ KiB/tok} \rightarrow 4.50\text{ GB}$ for 1×32K and $45.0\text{ GB}$ OOM for 10×32K on a 16 GB cap).

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Source / Artifact | Exact Location / Key | Usage in Section 2.1 | Readiness |
| :--- | :--- | :--- | :---: |
| NVIDIA Inference Optimization Review | `docs/literature_survey/foundations/nvidia_inference_optimization.md` | Prefill vs. decode roofline, arithmetic intensity ($\approx 1\text{ FLOP/byte}$), RTX 4060 ($272\text{ GB/s}$) vs. RTX 4090 ($1008\text{ GB/s}$) bandwidth comparison | `[READY]` |
| Pinned Model Footprint Matrix | `docs/MODELS.md` (§1, §3) | Concrete BF16 weight + KV allocation headroom on 16 GB & 8 GB caps | `[READY]` |
| Proposal Defense Deck & Summary | `deliverables/s1_proposal_defense/KV-Cache Co-optimization.pdf` (Slide 02, PDF p. 3) & `Slide_Content_and_Executive_Summary.md` (Slides 02–03) | Real-world SME / local gateway motivation (OpenClaw in defense deck; Ollama, LM Studio, 64K warnings in executive summary) | `[READY]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  1. Rise of private, on-premise LLM serving on consumer GPUs (8–16 GB VRAM).
  2. Two-phase autoregressive execution: Prefill (compute-bound GEMM) vs. Decode (memory-bandwidth-bound GEMV).
  3. The Consumer Hardware Floor: why standard 3B–4B BF16 models exhaust VRAM under 32K single-stream or multi-session SME concurrency.
- **Figure Spec — Fig. 2.1 (Memory Wall Illustration):** Stacked bar chart comparing static BF16 weights vs. KV-cache footprint for `Qwen3-4B-Instruct-2507` (4K×1, 32K×1, 4K×10, 32K×10) against horizontal 8 GB and 16 GB VRAM cap lines. *Data source:* `docs/MODELS.md` §3 `[READY]`.
- **Open Questions:**
  - `[OPEN_DECISION]` Should Section 2.1 cite commercial gateway documentation (e.g., Ollama / OpenClaw 64K warnings from Proposal Slide 03) as formal web footnotes, or keep the motivation purely hardware-analytical?

---

## Section 2.2: Theory and Core Concepts

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`2.2 Theory and Core Concepts`):** Presents the mathematical equations, architectural mechanics, and formal definitions underlying all subsequent design choices and evaluation metrics.
- **Committee Expectation:** Rigorous equations with consistent notation (matching Front Matter List of Symbols) and direct linkage to the project's 3-axis framing and quality pillars.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Six theoretical pillars govern KV-cache co-optimization on consumer GPUs: (1) linear KV-cache scaling under GQA, (2) architectural KV suppression in Hybrid/SWA models vs. Pure Dense GQA, (3) FP16/BF16 numerical non-associativity and GQA error broadcasting, (4) the three orthogonal optimization axes (`STRUCTURAL`, `SPATIAL`, `TEMPORAL`), (5) micro-level logit divergence (Top-1 agreement & KL divergence) vs. macro-level multi-session memory evaluation, and (6) why token-level perplexity masks sparse fact loss.
- **Delta from Proposal Defense:**
  - *Added:* **Numerical Non-Associativity & GQA Error Broadcasting** (from Chodavarapu 2026, *The Illusion of Equivalence*) — justifies why compressed caches MUST be evaluated against an **uncompressed cached BF16 baseline**, never cache-free recomputation, and why BF16's 7-bit mantissa amplifies drift across GQA groups.
  - *Added:* **Pure Dense GQA vs. Hybrid/SWA Taxonomy** (`Qwen3-4B` global attention vs. `Qwen3.5-4B` Gated DeltaNet + full attention vs. `Gemma-4-E4B-it` sliding window).
  - *Refined:* **Quality Theory** — formally demotes $\Delta\text{PPL}$ to a supplementary diagnostic based on *Illusion of Equivalence*, *LoCoMo*, and *SEAL* findings.

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Concept | Exact Source Path / Reference | Key Formulas / Facts | Readiness |
| :--- | :--- | :--- | :---: |
| KV-Cache Footprint & GQA | `docs/MODELS.md` (§3); `docs/literature_survey/foundations/nvidia_inference_optimization.md` | $\text{KV Bytes/tok} = 4 \cdot L_{\text{full\_attn}} \cdot H_{KV} \cdot D_{\text{head}}$ (BF16); total GB scaling with $S \times B$ | `[READY]` |
| FP16/BF16 Non-Associativity & GQA Broadcasting | `docs/literature_survey/foundations/illusion_of_equivalence.md` | Chodavarapu (2026, `arXiv:2604.15409v1`): $(a+b)+c \neq a+(b+c)$; $\epsilon_{\text{accum}} \approx 0.036\%$; FP32 falsification ($0.0\%$ flip rate vs. $1.03\text{--}1.15\%$ FP16); GQA sharing ratio $R = H_Q / H_{KV}$ broadcasts $\delta K, \delta V$ | `[READY]` |
| Effective Context vs. Claimed Context | `docs/literature_survey/foundations/longeval.md`; `docs/literature_survey/benchmarks/ruler.md` | LMSYS LongEval (2023) & RULER (COLM 2024): claimed context window $\neq$ effective retrieval context | `[READY]` |
| 3-Axis Co-Optimization Taxonomy | `docs/METRICS_AND_AXES.md` (§1, §3) | `STRUCTURAL` (only lossy axis, $Q_{\text{err}} \ge 0$), `SPATIAL` (lossless placement, $Q_{\text{err}} = 0$), `TEMPORAL` (lossless scheduling/reuse, $Q_{\text{err}} = 0$) | `[READY]` |
| Micro Logit Divergence & Macro LongMemEval | `docs/METRICS_AND_AXES.md` (§3); `docs/literature_survey/benchmarks/longmemeval.md` | Top-1 agreement, $\text{KL}(P_{\text{base}} \parallel P_{\text{struct}})$, 5 LongMemEval capabilities; why $\Delta\text{PPL}$ averages out sparse needle loss | `[READY]` |
| K-V Quantization Asymmetry & Distortion | `docs/literature_survey/techniques/turboquant.md`; `docs/literature_survey/techniques/headq.md` | Keys drive softmax attention routing (exponentially sensitive to error); Values act as linear weighted sums (more tolerant to low-bit quantization) | `[READY]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  - `2.2.1` Autoregressive KV-Cache Linear Scaling & Memory-Bandwidth Wall
  - `2.2.2` Attention Architectures: Pure Dense GQA vs. Hybrid Linear / Sliding Window Attention (SWA)
  - `2.2.3` Floating-Point Non-Associativity, BF16 Precision Drift & GQA Error Broadcasting
  - `2.2.4` The Three Orthogonal Optimization Axes (`STRUCTURAL`, `SPATIAL`, `TEMPORAL`)
  - `2.2.5` Quality Fidelity Theory: Micro Logit Divergence, Macro Conversational Memory, and Perplexity Blind Spots
- **Table Spec — Table 2.1 (3-Axis Formal Taxonomy):** Columns: `Axis`, `Control Dimension`, `Canonical Mechanisms`, `Primary Bottleneck Addressed`, `Theoretical Quality Distortion (Q_err)`. *Source:* `docs/METRICS_AND_AXES.md` §1 & §3 `[READY]`.
- **Open Questions:**
  - `[OPEN_DECISION]` Should we include the formal Shannon-$\phi$ plane error decomposition from `turboquant.md` and Fisher quadratic key-distortion metric from `headq.md` in `2.2.5`, or defer their detailed math to Chapter 4 (`04_algorithms_and_implementation.md`)?

---

## Section 2.3: Development Tools, Languages, and Technologies

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`2.3 Development Tools, Languages, and Technologies`):** Inventories the software stack, inference engines, HPC cluster environment, and repository infrastructure used to build and validate the system.
- **Committee Expectation:** Concise justification of *why* each tool/framework was chosen and exact version/hardware specifications for reproducibility.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Our engineering stack intentionally separates **production serving baselines** (vLLM V1) from **white-box KV-cache algorithm prototyping** (PyTorch Native SDPA + Hugging Face Transformers), unified under a Two-Tier Multi-Repo architecture (`kv_co-op_hub` + `kv_bench` + GitHub Project #9) and executed on KMUTT CPE's Slurm HPC cluster (`gpu4090` capped to 16 GB) alongside local 8 GB RTX 4060 validation.
- **Delta from Proposal Defense:**
  - *Major Pivot:* Proposal assumed vLLM + custom C++/CUDA kernels for all baseline and candidate evaluations. In Semester 1 (`WP2-5`, `WP3-1-3`), we introduced a **Dual-Backend Strategy**: vLLM V1 (`0.30.0`) for production throughput/concurrency baselines, and **Unified PyTorch Native SDPA** (`PyTorch 2.6.0` + `transformers`) for white-box implementation of all 11 WP-3 techniques and post-hoc layer retention diagnostics (`WP1-4-1`), avoiding opaque vLLM C++ paged-attention monkey-patching during algorithmic evaluation.
  - *Deferred to S2:* `lmcache >= 0.5.0` (`LMCacheMPConnector`) and custom CUDA/Triton kernels remain planned for Semester 2 (`WP-6`–`WP-7`).

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Component | Exact Repo Source & Keys | Verified Spec / Version | Readiness |
| :--- | :--- | :--- | :---: |
| Production Serving Engine | `results/manifests/run_manifest_baseline_16gb.json` (`execution_environment`) | `vllm 0.30.0`, `PyTorch 2.13.0+cu130`, `CUDA 13.0`, NVIDIA Driver `580.178.04`, node `gpu01.slurm.cpe.kmutt.ac.th` (RTX 4090, 16.0 GiB allocator cap) | `[READY]` |
| White-Box Prototyping Engine | `results/manifests/run_manifest_baseline_pytorch_16gb.json` (`execution_environment`) | PyTorch Native SDPA (`PyTorch 2.6.0`, `CUDA 12.2`), Hugging Face Transformers, node `gpu06.slurm.cpe.kmutt.ac.th` (RTX 4090, 16.0 GiB allocator cap) | `[READY]` |
| Target Stack Spec (Docs) | `docs/MODELS.md` (§4) | Lists `vllm >= 0.20.0`, `lmcache >= 0.5.0`, `flash-linear-attention`, `causal-conv1d`, `PyTorch >= 2.6.0`, `CUDA 12.2` | `[NEEDS_VERIFICATION]` |
| Multi-Repo & CI Infrastructure | `docs/WORKFLOW_SPEC.md`; `.github/workflows/hub-ci.yml` | Two-Tier Architecture: `kv_co-op_hub` (Purpose/Data Plane) + `kv_bench` (Execution Plane) + GitHub Project #9; automated SHA256 manifest CI validation | `[READY]` |
| Local Consumer Target (`WP1-5`) | `docs/MODELS.md` (header & §3); `WORK_PACKAGES.md` (`WP-1`) | Local Desktop NVIDIA RTX 4060 (8 GB VRAM cap) — decide whether `WP1-5` runs before S1 defense or is formally deferred to S2 | `[OPEN_DECISION]` / `[NEEDS_VERIFICATION]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  1. Hardware & HPC Execution Environment (KMUTT CPE Slurm `gpu4090` 16 GB allocator cap vs. local RTX 4060 8 GB cap).
  2. Dual-Backend Inference Stack (vLLM V1 vs. PyTorch Native SDPA + Hugging Face Transformers; planned `lmcache` & kernel extensions for S2).
  3. Reproducibility & Engineering Plane (`kv_co-op_hub`, `kv_bench`, GitHub Project #9, SHA256 provenance manifests, Hub CI).
- **Table Spec — Table 2.2 (Development Stack & Environment Matrix):** Columns: `Layer`, `Tool / Framework`, `Verified Version / Hardware`, `Role in Project`, `Phase Status`.
- **Open Questions:**
  - `[NEEDS_VERIFICATION]` Reconcile the software version mismatch between `docs/MODELS.md` §4 (`CUDA 12.2`, `PyTorch >= 2.6.0`, `vllm >= 0.20.0`) and `run_manifest_baseline_16gb.json` (`CUDA 13.0`, `PyTorch 2.13.0+cu130`, `vllm 0.30.0`) — confirm whether two distinct Slurm Conda/Apptainer environments (`vllm` env vs. `pytorch` env) are intentionally maintained and document both explicitly.

---

## Section 2.4: Related Researches

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`2.4 Related Researches`):** Critically surveys prior academic work, grouped logically by research theme, comparing mechanisms, empirical trade-offs, and relevance to our problem.
- **Committee Expectation:** Comprehensive, non-superficial literature synthesis (not isolated paragraph summaries) culminating in a structured comparative table showing where prior art succeeds and fails.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Prior KV-cache research clusters into three technical families—Structural Eviction/Retention, Structural Quantization/Compression, and Spatial Offloading/Temporal Prefetching—evaluated across an evolving ecosystem of long-context and conversational memory benchmarks. While each method advances one or two axes, none resolves the full consumer-GPU trilemma across unmodified BF16 GQA models.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`, `Slide_Content_and_Executive_Summary.md`, & `docs/proposal.md`):**
  - *Accurate Proposal Baseline:* `KV-Cache Co-optimization.pdf` Slide 05 (PDF p. 6) already surveyed **7 techniques** (H2O, StreamingLLM, QServe, FlexGen, InfiniGen, ShadowKV, ArkVale; note that KIVI and DeepSpeed appeared only in the earlier `Slide_Content_and_Executive_Summary.md` Slide 05), Slide 14 (PDF p. 16) already cited HeadQ, LongMemEval, RULER, and LongEval, and `docs/proposal.md` §2.4–§2.5 already listed all 11 techniques.
  - *True Semester 1 Delta:* Upgraded from high-level slide/proposal mentions to **20 complete, implementation-grade literature reviews** in `docs/literature_survey/` (3 foundations, **11 techniques** empirically reproduced and benchmarked in WP-3, and **6 benchmarks**), adding **LoCoMo (ACL 2024)** plus deep mathematical and systems dissections of **EpiCache (ICML 2026)**, **CacheGen (SIGCOMM 2024)**, **HeadQ (2026)**, **TurboQuant (2026)**, **The Illusion of Equivalence (2026)**, **Memory NIAH (2024)**, **SEAL (2025)**, and **RealTalk (2025)**.

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)

#### 2.4.1 Structural Eviction & Retention
| Paper / System | Venue & Year | Primary / Secondary Axis | Verified Review File | Readiness |
| :--- | :--- | :--- | :--- | :---: |
| **StreamingLLM** (Xiao et al.) | ICLR 2024 (Oral) | Structural (Attention sinks + sliding window) / None | `docs/literature_survey/techniques/streamingllm.md` | `[READY]` |
| **H2O** (Zhang et al.) | NeurIPS 2023 | Structural (Cumulative attention heavy-hitter eviction) / Spatial (FlexGen integration) | `docs/literature_survey/techniques/h2o.md` | `[READY]` |
| **EpiCache** (Kim et al., Apple) | ICML 2026 | Structural (Episodic block eviction + sensitivity layer budgeting) / Spatial (Host episodic routing) | `docs/literature_survey/techniques/epicache.md` | `[READY]` |

#### 2.4.2 Structural Quantization & Compression
| Paper / System | Venue & Year | Primary / Secondary Axis | Verified Review File | Readiness |
| :--- | :--- | :--- | :--- | :---: |
| **QServe / QoQ** (Lin et al., MIT) | MLSys 2025 | Structural (W4A8KV4, SmoothAttention, per-head KV4) / Kernel Fusion | `docs/literature_survey/techniques/qserve.md` | `[READY]` |
| **CacheGen** (Liu et al.) | ACM SIGCOMM 2024 | Structural (Delta residual + layer-varying quant + arithmetic codec) / Temporal & Spatial streaming | `docs/literature_survey/techniques/cachegen.md` | `[READY]` |
| **HeadQ** (Ruiz Williams) | arXiv 2026 (`2605.03562v2`) | Structural (2-bit Key quantizer-corrector + rank-4 side-code; Fisher distortion) / K-V precision asymmetry | `docs/literature_survey/techniques/headq.md` | `[READY]` |
| **TurboQuant** (D'Alberto, AMD) | arXiv 2026 (`2605.08114v1`) | Structural (Walsh-Hadamard rotation + K-V asymmetric quant) / Statistical quality measures | `docs/literature_survey/techniques/turboquant.md` | `[READY]` |

#### 2.4.3 Spatial Offloading & Temporal Prefetching
| Paper / System | Venue & Year | Primary / Secondary Axis | Verified Review File | Readiness |
| :--- | :--- | :--- | :--- | :---: |
| **FlexGen** (Sheng et al.) | ICML 2023 | Spatial (GPU/CPU/NVMe tiering + LP schedule) / Structural (4-bit group-wise quant) | `docs/literature_survey/techniques/flexgen.md` | `[READY]` |
| **InfiniGen** (Lee et al., SNU) | USENIX OSDI 2024 | Spatial (Host CPU offload + selective PCIe prefetch) / Temporal (1-layer-ahead speculative rehearsal) | `docs/literature_survey/techniques/infinigen.md` | `[READY]` |
| **ShadowKV** (Sun et al., ByteDance/CMU) | arXiv 2024/2025 (`2410.21465v3`) | Spatial (Value CPU offload + low-rank Key on GPU) / Structural (SVD pre-RoPE Keys + landmarks) | `docs/literature_survey/techniques/shadowkv.md` | `[READY]` |
| **ArkVale** (Chen et al.) | NeurIPS 2024 | Structural (Page-level eviction + bounding-box digest) / Spatial (Async CPU backup + PCIe recall) *(Note: `docs/literature_survey/README.md` tags Spatial+Temporal while `arkvale.md` & `arkvale_hardware_pytorch_16gb.json` tag `["STRUCTURAL", "SPATIAL"]` — reconcile before final prose)* | `docs/literature_survey/techniques/arkvale.md` | `[READY]` / `[NEEDS_VERIFICATION]` |

#### 2.4.4 Long-Context & Conversational Memory Benchmarks
| Benchmark | Venue & Year | Primary Evaluation Focus | Verified Review File | Readiness |
| :--- | :--- | :--- | :--- | :---: |
| **LongMemEval** (Wu et al.) | ICLR 2025 | Multi-session interactive chat memory across 5 core capabilities (adopted as our Primary Pillar 2) | `docs/literature_survey/benchmarks/longmemeval.md` | `[READY]` |
| **LoCoMo** (Maharana et al.) | ACL 2024 (`2402.17753v1`) | Very long-term conversational QA, event timeline, multimodal memory | `docs/literature_survey/benchmarks/locomo.md` | `[READY]` |
| **RULER** (Hsieh et al., NVIDIA) | COLM 2024 (`2404.06654v3`) | Synthetic retrieval, multi-hop variable tracing, aggregation, QA; *effective context size* | `docs/literature_survey/benchmarks/ruler.md` | `[READY]` |
| **Memory NIAH** (Nelson et al., IBM) | arXiv 2024 (`2407.01437v2`) | Positional 2D retrieval diagnostics on memory-managed LLMs | `docs/literature_survey/benchmarks/memory_niah.md` | `[READY]` |
| **SEAL** (Lee et al., POSTECH) | arXiv 2025 (`2501.15225v1`) | Retrieval-head vs. distractor-head specialization; proves uniform head eviction harms recall | `docs/literature_survey/benchmarks/seal.md` | `[READY]` |
| **RealTalk** (Lee et al., USC/Snap) | arXiv 2025 (`2502.13270v1`) | 21-day real-world messaging corpus & memory probing | `docs/literature_survey/benchmarks/realtalk.md` | `[READY]` |

#### 2.4.5 Planned / Unreviewed Contextual Literature (Optional Additions)
| Paper / System | Planned File in `docs/literature_survey/` | Role if Cited | Readiness |
| :--- | :--- | :--- | :---: |
| FlashAttention (1/2/3), PagedAttention/vLLM (SOSP 2023), KV Survey (Jiang 2026) | `foundations/flashattention.md`, `foundations/pagedattention_vllm.md`, `foundations/kv_survey_jiang2026.md` | Systems baseline context (IO-aware tiling, virtual KV paging, 3-axis trilemma framing) | `[NEEDS_VERIFICATION]` |
| SnapKV (2024), PyramidKV (2024), KIVI (ICML 2024) | `techniques/snapkv.md`, `techniques/pyramidkv.md`, `techniques/kivi.md` | Adjacent structural baselines not yet formally reviewed in `docs/literature_survey/` | `[NEEDS_VERIFICATION]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  - `2.4.1` Structural Eviction & Retention (StreamingLLM, H2O, EpiCache)
  - `2.4.2` Structural Quantization & Compression (QServe, CacheGen, HeadQ, TurboQuant)
  - `2.4.3` Spatial Offloading & Temporal Prefetching (FlexGen, InfiniGen, ShadowKV, ArkVale)
  - `2.4.4` Long-Context & Conversational Memory Benchmarks (LongMemEval, LoCoMo, RULER, Memory NIAH, SEAL, RealTalk)
  - `2.4.5` Comparative Synthesis of Related Works
- **Table Spec — Table 2.3 (Master 11-Technique Comparison Matrix):** Columns: `Technique (Year/Venue)`, `Primary Axis`, `Secondary Axis`, `Core Mechanism`, `Memory Reduction Strategy`, `Quality Risk / Failure Mode`, `Consumer GPU Bottleneck`. *Source:* All 11 files in `docs/literature_survey/techniques/` `[READY]`.
- **Table Spec — Table 2.4 (Benchmark Comparison Matrix):** Columns: `Benchmark (Year/Venue)`, `Context Regime`, `Evaluation Target`, `Why Adopted / Not Adopted in kv_bench`. *Source:* All 6 files in `docs/literature_survey/benchmarks/` `[READY]`.
- **Open Questions:**
  - `[OPEN_DECISION]` Should the team complete short reviews for **PagedAttention (vLLM)** and **FlashAttention** before the Dec 1 advisor draft so those foundational citations are backed by verified notes in `docs/literature_survey/foundations/`, or cite them directly from their published SOSP/NeurIPS PDFs?

---

## Section 2.5: Challenges and Limitations in Existing KV-Cache Systems

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`2.5 Challenges and Limitations`):** Distills the gaps surfaced in Section 2.4 into concrete, unresolved engineering and scientific limitations that justify this project's contribution.
- **Committee Expectation:** Move beyond "Method X is slow" to articulate *structural* root causes (the 3-Axis Trilemma, hardware constraints, architectural shifts like GQA).

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Existing systems suffer from a **3-Axis Trilemma** on 8–16 GB consumer GPUs: (1) pure structural eviction causes irreversible amnesia in multi-session QA, (2) pure low-bit quantization still scales linearly with sequence length and suffers from K-V asymmetry and GQA error broadcasting, and (3) pure spatial offloading stalls decode on PCIe Gen4 bandwidth ($\approx 16\text{--}32\text{ GB/s}$ vs. $272\text{--}1008\text{ GB/s}$ GPU VRAM [GDDR6/GDDR6X]). Furthermore, most prior methods were validated on datacenter GPUs (A100/H100 80 GB) or legacy MHA models rather than consumer-capped GQA checkpoints, and relied on perplexity ($\Delta\text{PPL}$) rather than factual memory retention.
- **Delta from Proposal Defense:**
  - *Deepened:* Added **head/layer heterogeneity** (SEAL, EpiCache, HeadQ) and **GQA error broadcasting** (*Illusion of Equivalence*) as newly surfaced failure modes not present in the Proposal Defense deck.
  - *Grounded:* Explicitly distinguishes algorithmic limitations from software/kernel overhead on consumer hardware (previewing WP-3 findings `[DRAFT/PR#21]`).

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Limitation Theme | Verified Source Paths | Key Evidence | Readiness |
| :--- | :--- | :--- | :---: |
| Irreversible Fact Loss under Eviction | `docs/literature_survey/techniques/streamingllm.md`, `h2o.md`, `benchmarks/seal.md` | Window/HH eviction drops middle-context needles; uniform head eviction destroys retrieval heads | `[READY]` |
| Linear Growth & K-V Asymmetry in Quantization | `docs/literature_survey/techniques/qserve.md`, `headq.md`, `turboquant.md`, `foundations/illusion_of_equivalence.md` | Quantization shrinks bytes/tok by 2–8× but remains $O(N)$; Keys are far more distortion-sensitive than Values; GQA broadcasts errors across $R$ query heads | `[READY]` |
| PCIe Bandwidth Wall in Offloading | `docs/literature_survey/techniques/flexgen.md`, `infinigen.md`, `shadowkv.md`, `arkvale.md` | PCIe transfer latency dominates decode TPOT unless overlapped with compute or filtered via sparse recall | `[READY]` |
| Evaluation Blind Spots ($\Delta\text{PPL}$ & Datacenter Bias) | `docs/METRICS_AND_AXES.md` (§3); `docs/literature_survey/benchmarks/longmemeval.md`, `ruler.md` | Perplexity masks needle loss; prior benchmarks evaluated on 80 GB A100/H100 rather than 8–16 GB consumer caps | `[READY]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  1. The 3-Axis Trilemma on 8–16 GB Consumer GPUs (Memory Capacity vs. Latency/PCIe Wall vs. Conversational Quality).
  2. Architectural Mismatch: Datacenter MHA Assumptions vs. Consumer GQA & Hybrid Models.
  3. Evaluation Gap: Why Perplexity-Only and Black-Box NIAH Fail to Diagnose Multi-Session Amnesia.
- **Figure Spec — Fig. 2.2 (The 3-Axis KV-Cache Trilemma Triangle):** Conceptual diagram placing `STRUCTURAL` (Eviction/Quantization), `SPATIAL` (Host Offload), and `TEMPORAL` (Prefetch/Reuse) at the three vertices, plotting where each of the 11 surveyed techniques sits and highlighting the unaddressed consumer-GPU sweet spot in the center. *Source:* `docs/METRICS_AND_AXES.md` §1 & all 11 files in `docs/literature_survey/techniques/` `[READY]`.
- **Open Questions:**
  - None blocking; ensure tone stays objective and cites exact failure modes from the completed reviews.

---

## Section 2.6: Design Rationale: Addressing Limitations

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`2.6 Design Rationale`):** Maps each limitation in Section 2.5 directly to a principled design requirement for our system, forming the logical bridge to Chapter 3.
- **Committee Expectation:** Explicit 1-to-1 traceability from literature gaps $\rightarrow$ design principles $\rightarrow$ measurable requirements.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Resolving the consumer-GPU trilemma requires two tightly coupled deliverables: (1) a **white-box, apples-to-apples 3-axis evaluation harness** (`kv_bench`) with controlled BF16 models, strict 16 GB/8 GB caps, and dual-pillar quality + post-hoc layer retention diagnostics; and (2) an **evidence-driven 3-axis co-optimization policy** (`WP-4`) that coordinates lossless spatial/temporal tiering with sensitivity-aware structural compression rather than relying on any single isolated axis.
- **Delta from Proposal Defense:**
  - *Refined:* Shifts from claiming a pre-decided "3-layer heuristic" solves everything, to a **two-stage scientific rationale**: first isolate true algorithmic strengths/weaknesses under controlled white-box conditions (`WP-1`–`WP-3`), then synthesize a falsifiable co-optimization policy (`WP-4`) directly from those empirical findings.

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Design Principle | Source Path | Traceability | Readiness |
| :--- | :--- | :--- | :---: |
| Controlled BF16 Models + 16 GB/8 GB Caps | `docs/MODELS.md` (§1) | Eliminates weight-quantization confounders; isolates KV-cache behavior | `[READY]` |
| Dual Quality Pillars + Post-Hoc Diagnostics | `docs/METRICS_AND_AXES.md` (§3); `results/manifests/run_manifest_baseline_pytorch_16gb.json` | Micro logit fidelity + 150q LongMemEval semantic judging + causal attribution | `[READY]` |
| Cross-Axis Co-Optimization Opportunity | `docs/METRICS_AND_AXES.md` (§1); `docs/literature_survey/techniques/` (`arkvale.md`, `shadowkv.md`, `epicache.md`, `headq.md`, `turboquant.md`) | Combines recallable/offloaded tiering with asymmetric K-V compression and layer/head sensitivity | `[READY]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  1. Requirement 1 — Controlled Hardware & Model Envelopes (eliminating confounders).
  2. Requirement 2 — White-Box Dual-Pillar & Causal Diagnostics (exposing silent amnesia).
  3. Requirement 3 — Decoupled 3-Axis Orchestration (why hybrid structural + spatial + temporal coordination is necessary).
- **Table Spec — Table 2.5 (Gap-to-Design-Rationale Traceability Matrix):** Columns: `Identified Limitation (Sec. 2.5)`, `Design Principle`, `Concrete Realization in This Project`, `Target Work Package`.
- **Open Questions:**
  - None blocking.

---

## Section 2.7: Preliminary Design & Evolution from Proposal Defense

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`2.7 Preliminary Design`):** Closes Chapter 2 by summarizing how the literature survey shaped our preliminary architectural direction before Chapter 3 details the full system methodology.
- **Committee Expectation:** Transparent reflection on how initial proposal assumptions matured into a more rigorous, empirically grounded design during Semester 1.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Between Proposal Defense and the end of Semester 1, our literature survey and early WP-1/WP-3 experiments drove four major design evolutions: (1) pinning 9 exact BF16 models split into Dense GQA vs. Hybrid/SWA, (2) adopting a Dual-Backend harness (vLLM + PyTorch Native SDPA), (3) upgrading quality evaluation to 150q offline-judged LongMemEval + post-hoc layer retention, and (4) **re-evaluating the Proposal's fixed 3-layer heuristic** in favor of an evidence-driven WP-4 co-optimization policy informed by our 11-method WP-3 sweep.
- **Delta from Proposal Defense:** Explicitly contrasts the Proposal's conceptual 3-Layer Policy (`KV-Cache Co-optimization.pdf` backup Slide 07 [PDF p. 17], where L3 = `TRANSFER SCHEDULER`; and `Slide_Content_and_Executive_Summary.md` Slide 08, with CPU dequantization detailed on Slide 10) against the newly surfaced design space (e.g., asymmetric K-V precision from HeadQ/TurboQuant, recallable page digests from ArkVale, low-rank GPU Keys + CPU Values from ShadowKV, sensitivity-aware layer budgets from EpiCache).

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Evolution Dimension | Proposal Baseline | Current Verified Source | Readiness |
| :--- | :--- | :--- | :---: |
| Conceptual 3-Layer Policy | `deliverables/s1_proposal_defense/KV-Cache Co-optimization.pdf` (backup Slide 07, PDF p. 17) & `Slide_Content_and_Executive_Summary.md` (Slides 08 & 10) | Held as `[OPEN_DECISION]` for WP-4 synthesis; see WP-3 interpretation in PR #21 (`analysis/wp3/*`) | `[DRAFT/PR#21]` / `[OPEN_DECISION]` |
| Survey-to-Implementation Coverage | 7 surveyed / 5 planned baselines | 20 completed reviews (`docs/literature_survey/`); 11 techniques benchmarked (`results/runs/wp3/`) | `[READY]` |
| Delta Summary Table | `deliverables/s1_progress_report/README.md` (§4) | Canonical high-level diff table across all 6 dimensions | `[READY]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  1. Summary of Literature-Driven Architectural Pivots since Proposal Defense.
  2. Candidate Building Blocks Surfaced from the 11 Surveyed Techniques (Eviction/Recall, K-V Asymmetric Compression, Speculative/Async Offload).
  3. Explicit Transition to Chapter 3 (Methodology & System Design) and Open WP-4 Design Questions.
- **Open Questions (DO NOT COMMIT BLINDLY — Require Team & Advisor Sign-Off in WP-4):**
  - `[OPEN_DECISION]` **WP-4 Co-Optimization Composition:** Should our Semester 2 co-optimized policy combine (a) **asymmetric K-V quantization** (e.g., higher-bit Keys + lower-bit Values inspired by TurboQuant/HeadQ) with **recallable CPU offloading** (ArkVale/ShadowKV), or (b) **sensitivity-aware layer/episodic budgeting** (EpiCache) with **asynchronous PCIe prefetching** (InfiniGen)?
  - `[OPEN_DECISION]` **Runtime Integration Target for S2 (`WP-6`):** Should the WP-4 policy be implemented as a custom `LMCache` / vLLM V1 connector, or first validated as a fused PyTorch/Triton prototype before vLLM V1 integration?
