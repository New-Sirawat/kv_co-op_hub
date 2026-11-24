# Chapter 4 Blueprint — Algorithms and Implementation

> **KMUTT Report Mapping:** Chapter 4 (`4.1`–`4.11` in `Progress_Sem1_ENG_Example2.pdf`, adapted to 5 cohesive subsections for our systems + algorithmic benchmark & co-design study)  
> **Document Role:** Templating, Drafting & Agent Handoff Specification (**not** final polished thesis prose)  
> **Branch Baseline:** `docs/s1-progress-report-blueprint` (branched from `main`; PR #21 unmerged)  
> **Target Milestone:** First advisor review draft — **December 1, 2026** (`WP-5`)

---

## Section 4.1 — System & Harness Architecture Overview

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Ch. 4 opening):** Introduces the concrete engineering realization of the methodology presented in Chapter 3—showing how software modules, data pipelines, and execution backends are architected and wired together.
- **Committee Expectation:** Prove that the team has built a reproducible, instrumented, production-grade benchmarking and prototyping harness (not ad-hoc notebook scripts) capable of isolating KV-cache bottlenecks under strict consumer VRAM budgets.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Rather than forcing all experimental KV-cache algorithms into a single rigid runtime, we implemented a **Two-Tier Multi-Repo Architecture** (`kv_co-op_hub` coordination plane + `kv_bench` execution plane) backed by a **Dual-Backend Execution Harness** (Native **vLLM V1** for production serving & concurrency saturation + **Unified PyTorch Native SDPA** for white-box KV-cache prototyping and post-hoc diagnostic inspection), unified by an automated **SHA256 Provenance Manifest Pipeline** enforced by GitHub Actions CI.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Assumed all evaluations and prototypes would run directly inside vLLM with custom C++/CUDA kernels from Semester 1.
  - *Current State:* Split cleanly into **vLLM V1** (`vllm >= 0.20.0`, empirical cluster run on `vllm 0.30.0` in `WP-2`) for native baseline ceilings and multi-user concurrency (`1–16` users), and **PyTorch Native SDPA** (`WP1-4-1`, `WP3-1-3`) for rapid, white-box reproduction of all 11 literature candidates with per-layer/per-session retention telemetry (`Post-Hoc Diagnostic Inspector`). Custom CUDA kernel integration is deliberately scoped to Semester 2 (`WP-6`, `WP-8`) after evidence-driven policy selection.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Artifact / Component | Relative Path / Identifier | Status Tag | Usage in Section 4.1 |
| :--- | :--- | :--- | :--- |
| Two-tier workflow & CI manifest spec | `docs/WORKFLOW_SPEC.md` | `[READY]` | Architecture of `kv_co-op_hub` vs. `kv_bench` + GitHub Project #9 rollup |
| CI validation workflow | `.github/workflows/hub-ci.yml` | `[READY]` | Automated SHA256 & schema validation guardrail |
| Co-optimization axes & `ResultEntry` spec | `docs/METRICS_AND_AXES.md` | `[READY]` | Mapping of `STRUCTURAL`, `SPATIAL`, `TEMPORAL` to harness telemetry |
| Pinned models & engine parity spec | `docs/MODELS.md` | `[READY]` | `vllm >= 0.20.0`, `PyTorch >= 2.6.0`, CUDA toolchain, `--language-model-only` flag |
| vLLM baseline run manifest | `results/manifests/run_manifest_baseline_16gb.json` | `[READY]` | Cluster node (`gpu01.slurm.cpe.kmutt.ac.th`), Slurm Job `22868`, `kv_bench` SHA `3aee66fe` |
| PyTorch SDPA baseline run manifest | `results/manifests/run_manifest_baseline_pytorch_16gb.json` | `[READY]` | Slurm Job `24643`, `kv_bench` SHA `2a9338a2` |
| PyTorch SDPA hardware baseline JSON | `results/baselines/baseline_pytorch_16gb.json` | `[DRAFT/PR#21]` | Exists only in unmerged PR #21; do **not** cite as merged on `main` |
| `kv_bench` internal module layout (`src/core/`, `src/engines/`, `src/techniques/`) | `khemingkapat/kv_bench` @ commit `2a9338a2` | `[NEEDS_VERIFICATION]` | Exact class/method names and call graph must be verified directly from `kv_bench` before drawing final UML/component diagram |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  1. **Two-Tier Repository Separation:** Scientific authority & immutable manifests (`kv_co-op_hub`) vs. benchmark runner & technique adapters (`kv_bench`).
  2. **Dual-Backend Execution Engine:**
     - *Path A (vLLM V1 Production Backend):* PagedAttention KV block pool, continuous batching, two-tier VRAM capping on the 23.52 GiB physical RTX 4090 — Tier 1 CUDA process allocator cap (`cap_fraction = 0.680` → `16.00 GiB`) and Tier 2 vLLM KV-pool cap (`effective_gpu_utilization = 0.5783` → `13.60 GiB`).
     - *Path B (Unified PyTorch Native SDPA Backend):* Hookable `DynamicCache` / custom KV-cache wrappers enabling arbitrary structural compression, spatial CPU offloading, and temporal prefetching without modifying vLLM C++ internals.
  3. **Post-Hoc Diagnostic Inspector & Offline Semantic Grading Pipeline:** Captures per-sample `diagnostic` telemetry (`overall_retention_ratio`, `overall_effective_bitwidth`, `min_session_effective_bitwidth`, `most_degraded_session_idx`, `num_sessions`) paired with offline LLM semantic grading (`orchestrator-pro-semantic-v1` → `results/grades/*_grades.json` + `*_judged.json`).
  4. **Provenance & Reproducibility Pipeline:** Automated manifest generation (`results/manifests/run_manifest_*.json`) recording Slurm Job ID, GPU/CUDA/driver versions, `kv_bench` Git SHA, and SHA256 artifact checksums validated on every PR.
- **Required Figure Spec — Fig. 4.1 (`Dual-Backend kv_bench & Provenance Pipeline Architecture`):**
  - *Type:* System block diagram (Mermaid / vector draw.io).
  - *Panels:* (Left) Workload Staging (`PG-19`, `LongMemEval 150q`, `2D NIAH`) → (Center) Dual-Backend Runner (`vLLM V1` vs. `PyTorch SDPA + 11 Technique Wrappers`) → (Right) Telemetry & Quality Collectors (`ResultEntry`, `NVML/CUDA Allocator`, `Logit Ref Comparator`, `Post-Hoc Inspector`, `Offline Semantic Judge`) → (Bottom) `kv_co-op_hub` CI Gate (`run_manifest_*.json` SHA256 check).
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Should Section 4.1 include an explicit side-by-side environment callout contrasting the **two distinct cluster software stacks** — the **vLLM V1 env** (`gpu01`, `CUDA 13.0`, driver `580.178.04`, `vLLM 0.30.0`, `PyTorch 2.13.0+cu130`) vs. the **PyTorch SDPA env** (`gpu06` & `gpu01`, `CUDA 12.2`, `PyTorch 2.6.0`, matching `docs/MODELS.md` §4)? *(Recommended: Yes — makes the dual-backend parity boundary transparent.)*

---

## Section 4.2 — Mathematical Formulations of Core Metrics & Causal Attribution

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Ch. 4 mathematical subsections):** Provides formal, unambiguous mathematical definitions of all system resource models, latency/bandwidth metrics, and quality/error metrics used in experiments.
- **Committee Expectation:** Ensure every metric plotted in Chapter 5 has an exact equation, units, and clear operational definition (especially distinguishing micro-level logit divergence from macro-level semantic accuracy and causal attribution).

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Evaluating KV-cache co-optimization requires jointly quantifying **hardware capacity/bandwidth physics** (KV footprint, VRAM headroom, TTFT, TPOT, normalized PCIe bytes/token) and **Two Primary Quality Pillars**—(1) Micro-Level Logit Divergence and (2) Macro-Level LongMemEval with a **Causal Attribution Tree**—while relegating Perplexity ($\Delta\text{PPL}$) to a strictly optional diagnostic.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Treated Perplexity ($\Delta\text{PPL} \le 0.1$) as a co-equal primary quality gate alongside accuracy; did not yet formalize the Causal Attribution Tree or Post-Hoc Diagnostic Inspector.
  - *Current State:* Enforces the Two Primary Quality Pillars (`docs/METRICS_AND_AXES.md`) supported by literature (`illusion_of_equivalence.md`, `locomo.md`), adds **Effective Bitwidth / Retention Ratio** diagnostics, and formalizes the 4-way **Causal Verdict** (`PASS_FIDELITY_MAINTAINED`, `QUALITY_REGRESSION`, `BASELINE_LIMITATION`, `UNEXPECTED_GAIN`) to separate base-model errors from KV-cache compression regressions.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Metric / Formulation | Relative Path & Exact Keys | Status Tag | Notes |
| :--- | :--- | :--- | :--- |
| KV-cache memory formula | `docs/MODELS.md` (§3, lines 47–50) | `[READY]` | $4 \times L_{\text{full\_attn}} \times H_{KV} \times D_{\text{head}}$ bytes/token (BF16) |
| Hardware & latency metrics | `docs/METRICS_AND_AXES.md` (§2) + `results/baselines/baseline_hardware_16gb.json` (`peak_alloc_mb`, `peak_reserved_mb`, `host_rss_delta_mb`, `ttft_sec`, `tpot_mean_ms`, `tpot_p95_ms`, `throughput_tok_per_sec`, `memcpy_bytes_per_token`) | `[READY]` | Standardized `ResultEntry` schema |
| Pillar 1: Top-1 Agreement & KL Divergence | `docs/METRICS_AND_AXES.md` (§3.1) + `results/runs/wp3/verify_*_pytorch_16gb.json` & `*_longmem_pytorch_16gb.json` (`extra.top1_agreement`, `extra.kl_div_mean`, `extra.kl_div_p99`) | `[READY]` | Reference logits stored strictly in `results/baselines/references/` |
| Sparse Top-5 KL smoothing floor (`15.3301` / `15.4249`) | `khemingkapat/kv_bench` @ `2a9338a2` | `[NEEDS_VERIFICATION]` | Exact $\epsilon$ smoothing formula for truncated top-$k$ (`--logprobs 5`) KL divergence must be verified in `kv_bench` source |
| Pillar 2: LongMemEval Semantic Grading & Causal Attribution | `results/grades/*_judged.json` (`overall_accuracy`, `per_task_accuracy`, `causal_verdict_counts`, `samples[].causal_verdict`, `samples[].diagnostic`) | `[READY]` | 150 samples (30 per each of 5 capabilities); grader `orchestrator-pro-semantic-v1` |
| Supplementary Diagnostic: 2D NIAH | `results/runs/wp3/*_niah_pytorch_16gb.json` (`extra.niah_accuracy`, `extra.niah_depths`, `extra.niah_grid`) | `[READY]` | Evaluated at 2048 & 4096 ctx across 10 depth positions |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton (Equations to include):**
  1. **KV-Cache Memory Footprint (GQA in BF16):**
     - $\text{KV}_{\text{tok}} = 2 \times b_{\text{elem}} \times L_{\text{full\_attn}} \times H_{KV} \times D_{\text{head}}$ (for `Qwen3-4B-Instruct-2507`: $2 \times 2 \times 36 \times 8 \times 128 = 147{,}456\text{ B} = 144\text{ KiB/tok}$).
     - $\text{Total KV (GiB)} = (\text{KV}_{\text{tok}} \times S \times B) / 1024^3$.
  2. **System Latency, Throughput & PCIe Transfer Intensity:**
     - $\text{TTFT}$ (prefill wall time to first token), $\text{TPOT}_{\text{mean}} = T_{\text{decode}} / (N_{\text{gen}} - 1)$, $\text{Throughput} = N_{\text{gen}} / T_{\text{decode}}$ (or aggregate concurrent tok/s), and Normalized PCIe Volume $V_{\text{PCIe}} = \text{Bytes}_{\text{H2D+D2H}} / N_{\text{gen}}$ (`memcpy_bytes_per_token`).
     - Host RAM Footprint Delta: $\Delta\text{RSS} = \text{RSS}_{\text{post}} - \text{RSS}_{\text{pre-init}}$ (`host_rss_delta_mb`).
  3. **Primary Pillar 1 — Micro-Level Logit Divergence:**
     - $\text{Top-1 Agreement} = \frac{1}{T}\sum_{t=1}^T \mathbb{I}(\arg\max L_{\text{cand}}^{(t)} = \arg\max L_{\text{base}}^{(t)})$.
     - $\text{KL}(P_{\text{base}} \parallel P_{\text{cand}}) = \sum_{v \in \mathcal{V}_k} P_{\text{base}}(v) \log \frac{P_{\text{base}}(v)}{P_{\text{cand}}(v)}$, aggregated into `kl_div_mean` and tail `kl_div_p99`.
  4. **Primary Pillar 2 — Macro-Level LongMemEval & Causal Attribution Tree:**
     - Overall & per-task accuracy across 5 capabilities (`fact_recall`, `information_update`, `relative_timeframe`, `cross_session_reasoning`, `unanswerable_refusal`, $N=30$ each, total $150$).
     - **4-Way Causal Attribution Rule** comparing candidate verdict $c_i \in \{0,1\}$ against uncompressed BF16 baseline verdict $b_i \in \{0,1\}$:
       - $(b_i=1, c_i=1) \rightarrow \texttt{PASS\_FIDELITY\_MAINTAINED}$
       - $(b_i=1, c_i=0) \rightarrow \texttt{QUALITY\_REGRESSION}$ *(true KV-cache induced degradation)*
       - $(b_i=0, c_i=0) \rightarrow \texttt{BASELINE\_LIMITATION}$ *(base model failure, not KV-cache fault)*
       - $(b_i=0, c_i=1) \rightarrow \texttt{UNEXPECTED\_GAIN}$ *(noise / regularization artifact)*
     - **White-Box Effective Compression Diagnostics:** `overall_retention_ratio`, `overall_effective_bitwidth` ($b_{\text{eff}} = 16 \times \text{retention} \times (b_{\text{quant}}/16)$), and `compression_ratio` ($16 / b_{\text{eff}}$).
- **Required Table Spec — Table 4.1 (`Metric-to-Axis Correspondence & Mathematical Definitions`):**
  - *Columns:* `Metric Name | Symbol / Formula | Unit | ResultEntry / Judged Key | Co-Optimization Axis | Role (Primary / Diagnostic)`.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Verify exact top-$k$ logprob normalization and floor $\epsilon$ in `kv_bench` that produces `kl_div_p99 = 15.4249` on LongMemEval (`[NEEDS_VERIFICATION]`) so the equation in 4.2 matches the code exactly.

---

## Section 4.3 — Evaluated KV-Cache Candidate Implementations (WP-3 Taxonomy)

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Ch. 4 algorithm subsections):** Details the concrete algorithms implemented and evaluated during Semester 1, explaining how each mechanism operates on KV tensors and what configuration parameters govern its memory/quality trade-off.
- **Committee Expectation:** Show deep engineering command over the 11 implemented SOTA baselines—not just paper summaries, but exact reproduction configurations, axis classifications, and known PyTorch simulation boundaries.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** We implemented and empirically profiled **11 distinct KV-cache management algorithms** spanning all three co-optimization axes inside a single unified PyTorch SDPA testbed (`kv_bench` commit `2a9338a2`), enabling controlled apples-to-apples comparison on `Qwen3-4B-Instruct-2507` under the identical 16 GB VRAM cap.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Planned only 5 baseline comparisons (H2O, StreamingLLM, FlexGen, QServe, InfiniGen).
  - *Current State:* Expanded to **11 implemented & benchmarked mechanisms**, adding **ArkVale** (recallable paging), **CacheGen** (delta-encoded bitstream), **EpiCache** (episodic clustering), **ShadowKV** (low-rank Key + CPU Value offload), **HeadQ** (2-bit PCA residual key corrector), and **TurboQuant** (asymmetric K/V + QJL quantization).

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Candidate (`WP-3` ID) | Axes (`results/runs/wp3/*`) | Literature Spec (`[READY]`) | Run Manifest (`[READY]`) | Slurm Job / SHA | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StreamingLLM** (`WP3-1`, `WP3-1-3`) | `STRUCTURAL` | `docs/literature_survey/techniques/streamingllm.md` | `run_manifest_wp3_streamingllm_pytorch.json` *(+ legacy vLLM `run_manifest_wp3_streamingllm.json`)* | `23938` / `2a9338a2` *(vLLM: `23910` / `1241f558`)* | `[READY]` |
| **H2O** (`WP3-2`) | `STRUCTURAL` | `docs/literature_survey/techniques/h2o.md` | `run_manifest_wp3_h2o_pytorch.json` | `23958` / `2a9338a2` | `[READY]` |
| **TurboQuant** (`WP3-3`) | `STRUCTURAL` | `docs/literature_survey/techniques/turboquant.md` | `run_manifest_wp3_turboquant.json` | `25931` / `2a9338a2` | `[READY]` |
| **HeadQ** (`WP3-4`) | `STRUCTURAL` | `docs/literature_survey/techniques/headq.md` | `run_manifest_wp3_headq.json` | `25749` / `2a9338a2` | `[READY]` |
| **QServe KV4** (`WP3-5`) | `STRUCTURAL` | `docs/literature_survey/techniques/qserve.md` | `run_manifest_wp3_qserve_kv4.json` | `25748` / `2a9338a2` | `[READY]` |
| **FlexGen KV** (`WP3-6`) | `STRUCTURAL`, `SPATIAL` | `docs/literature_survey/techniques/flexgen.md` | `run_manifest_wp3_flexgen_kv.json` | `25110` / `2a9338a2` | `[READY]` |
| **ShadowKV** (`WP3-7`) | `STRUCTURAL`, `SPATIAL`, `TEMPORAL` | `docs/literature_survey/techniques/shadowkv.md` | `run_manifest_wp3_shadowkv.json` | `25947` / `2a9338a2` | `[READY]` |
| **CacheGen** (`WP3-8`) | `STRUCTURAL`, `SPATIAL` | `docs/literature_survey/techniques/cachegen.md` | `run_manifest_wp3_cachegen.json` | `25765` / `2a9338a2` | `[READY]` |
| **InfiniGen** (`WP3-9`) | `STRUCTURAL`, `SPATIAL`, `TEMPORAL` | `docs/literature_survey/techniques/infinigen.md` | `run_manifest_wp3_infinigen.json` | `25111` / `2a9338a2` | `[READY]` |
| **ArkVale** (`WP3-10`) | `STRUCTURAL`, `SPATIAL` | `docs/literature_survey/techniques/arkvale.md` | `run_manifest_wp3_arkvale.json` | `25751` / `2a9338a2` | `[READY]` |
| **EpiCache** (`WP3-11`) | `STRUCTURAL`, `SPATIAL` | `docs/literature_survey/techniques/epicache.md` | `run_manifest_wp3_epicache.json` | `25949` / `2a9338a2` | `[READY]` |
| *Pseudocode & fidelity notes* | *All 11* | `khemingkapat/kv_bench` @ `2a9338a2` | `analysis/wp3/*` (`[DRAFT/PR#21]`) | — | `[NEEDS_VERIFICATION]` |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton (Grouped by Architectural Paradigm):**
  1. **Group A — Eviction & Retention (`STRUCTURAL`):**
     - *StreamingLLM:* `sink_tokens=4`, `window_size=2048`, `block_size=16` (`effective_cache_tokens=2052`). Permanent middle-token eviction.
     - *H2O:* `heavy_hitter_tokens=1024`, `recent_window_tokens=1024`, `block_size=16` (`effective_cache_tokens=2048`). Cumulative attention-score eviction.
     - *EpiCache:* `num_episodes=4`, `episode_budget_tokens=1024`, `layer_sensitivity_alpha=2.0`. Block-wise prefill + episodic routing.
  2. **Group B — Quantization & Coding (`STRUCTURAL` / `SPATIAL`):**
     - *QServe KV4:* `kv_cache_bits=4`, `quant_group_size=32`, `smooth_attention_alpha=0.5` ($4.0\times$ theoretical compression).
     - *TurboQuant:* Asymmetric `k_bits=4`, `v_bits=3`, `qjl_v=True`, `qjl_k=False` ($3.66\times$ compression, pure-PyTorch QJL simulation).
     - *HeadQ:* `key_bits=2`, `val_bits=4`, `pca_rank=4`, `sidecode_bits=4` (`effective_key_bits=2.125`, $5.22\times$ theoretical compression).
     - *CacheGen:* Layer-tiered quantization (`shallow_bits=8`, `mid_bits=4`, `deep_bits=2`), `anchor_group_size=10`, delta-encoded bitstream.
  3. **Group C — Offloading, Recall & Speculative Prefetching (`SPATIAL` / `TEMPORAL`):**
     - *FlexGen KV:* `gpu_cache_tokens=1024`, `cpu_cache_tokens=32768`, `quantize_kv=True` (`group_size=64`), `swap_space_gb=4`, `cpu_delegation=False`.
     - *ArkVale:* `page_size=16`, `active_pages_budget=16`, `exempt_early_layers=2`, AABB bounding-box digests for recallable page eviction.
     - *InfiniGen:* `k_proxy_ratio=0.3`, `alpha_threshold=4.0`, `max_budget_ratio=0.35`, `swap_space_gb=4` — speculative next-layer prefetching from CPU.
     - *ShadowKV:* `svd_rank=160`, `chunk_size=8`, `topk_chunks=256`, `outlier_ratio=0.003` — low-rank compressed Keys on GPU + sparse Value offload on CPU.
- **Required Table Spec — Table 4.2 (`WP-3 Implemented Candidate Registry & Verified Hyperparameters`):**
  - *Columns:* `Technique | Axes | Core Mechanism | Key Hyperparameters (Verified from verify_*.json) | PyTorch Simulation Caveat`.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - How explicitly should Chapter 4 flag known PyTorch wrapper fidelity bugs (e.g., `headq` 0% LongMemEval, `epicache` 5.3% LongMemEval, `turboquant` pure-Python QJL latency) vs. deferring root-cause diagnosis to Section 5.2.3 after PR #21 is reviewed? *(Recommended: Mention simulation boundary briefly in 4.3, put quantitative root-cause attribution in 5.2.3 once PR #21 is merged.)*

---

## Section 4.4 — Proposed 3-Axis Co-Optimization Policy State Machine (`WP-4` — Working Direction)

> [!IMPORTANT]
> **Template / Working Direction Notice:** Per project governance, `WP-4` policy design is **not yet locked**. PR #21 (`[DRAFT/PR#21]`) has not been merged/reviewed, and `WP-4` micro-benchmarking has not yet validated tier-transition thresholds. Do **not** present an untested policy as finalized—frame Section 4.4 explicitly as an **evidence-driven architectural template with falsifiable design hypotheses and open decisions**.

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Ch. 4 proposed design):** Presents the project's novel algorithmic/system contribution—the unified 3-Axis Co-Optimization Policy Layer that bridges Semester 1 exploration (`WP-3`) and Semester 2 runtime integration (`WP-6`).
- **Committee Expectation:** Clearly articulate the proposed co-optimization architecture, state transitions, and scheduling logic, while transparently distinguishing validated design principles from open tuning parameters.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Guided by WP-3 empirical evidence (showing that pure eviction destroys multi-session recall, pure uniform quantization incurs high KL or compute overhead, and naive CPU offloading saturates PCIe), our proposed `WP-4` architecture coordinates a **3-Tier KV-Cache State Machine** (`Tier 0: GPU Active BF16` $\leftrightarrow$ `Tier 1: GPU Compressed / Low-Rank` $\leftrightarrow$ `Tier 2: Host DRAM Offload with Recallable Index`) driven by **asynchronous CUDA stream pipelining** and **layer-aware budgets**.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Sketched a fixed 3-layer heuristic (`L1 Priority Scoring → L2 Uniform INT4 Format → L3 Async CPU Transfer + CPU Dequantization`).
  - *Current State:* Evolved from a speculative heuristic into a **modular hypothesis-driven state machine** informed by WP-3's 11-method evidence:
    - *Insight 1 (Structural):* Preserve high-sensitivity early layers & sinks in full precision (ArkVale / StreamingLLM lesson) while applying low-distortion `KV4` (`QServe`-style) or low-rank Key compression (`ShadowKV`-style) rather than blind uniform eviction.
    - *Insight 2 (Spatial):* Offload cold Value blocks (not full K+V) to Host DRAM while retaining lightweight routing metadata (AABB digests or low-rank Key proxies) on GPU to bound PCIe traffic (`ShadowKV` transferred only `67.24 MB/tok` vs. `FlexGen`'s `1,646.96 MB/tok` at 32K).
    - *Insight 3 (Temporal):* Overlap sparse page recall with active layer compute via dual CUDA streams; hold **CPU dequantization delegation** as a falsifiable crossover hypothesis rather than an unconditional rule.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Component / Evidence | Relative Path / Identifier | Status Tag | Role in Section 4.4 |
| :--- | :--- | :--- | :--- |
| WP-4 roadmap deliverables | `WORK_PACKAGES.md` (`WP-4`, lines 55–60) | `[READY]` | State-machine tiers + async transfer pipeline + prototype smoke test |
| Proposal 3-layer heuristic baseline | `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` | `[READY]` | Explicit "before" reference for Proposal → S1 delta |
| WP-3 empirical evidence (32K VRAM, Memcpy, LongMemEval) | `results/runs/wp3/*_hardware_pytorch_16gb.json` & `results/grades/*_judged.json` | `[READY]` | Empirical motivation for hybrid tiering (`ShadowKV` 64.0% @ 67 MB/tok vs. `QServe KV4` 59.3% @ 0.01 MB/tok) |
| WP-3 component-level synthesis & WP-4 candidate hypotheses | `analysis/wp3/*` (PR #21) | `[DRAFT/PR#21]` | Pending manual review before locking exact WP-4 policy rules |
| WP-4 prototype implementation & micro-benchmarks | `khemingkapat/kv_bench` (`WP-4` issues) | `[OPEN_DECISION]` / `[PENDING_S2]` | Policy thresholds & CUDA stream overlap not yet benchmarked |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  1. **Design Synthesis from WP-3 Failures & Wins:** Why no single axis suffices (trilemma summary linking to Ch. 5).
  2. **3-Tier KV-Cache State Machine (Template):**
     - *State `T0` (GPU Active BF16):* Attention sinks, recent sliding window, and exempt early layers.
     - *State `T1` (GPU Compressed):* Medium-priority blocks quantized (e.g., group-wise `KV4` or low-rank Keys) to eliminate PCIe round-trips while cutting VRAM.
     - *State `T2` (Host DRAM Offloaded + GPU Digest):* Cold Value blocks/pages stored in pinned Host RAM with GPU-resident metadata (AABB / Key proxy) for sparse recall (`T2 → T0/T1`) without permanent amnesia.
  3. **Temporal Scheduling & Async Transfer Pipeline:** Layer-$l$ lookahead prefetching overlapped on a dedicated H2D CUDA stream while layer $l-1$ executes SDPA/FFN compute.
  4. **Falsifiable WP-4 Hypotheses (To Be Validated Before Sem 2 Lock-In):**
     - *H1:* Hybrid `ShadowKV`-style sparse V-offload + `QServe`-style KV4 compression achieves $\ge 60\%$ LongMemEval ($\ge 91\%$ of BF16 baseline) at $\le 10\text{ GB}$ peak VRAM on 32K context.
     - *H2:* Dedicated async CUDA H2D stream reduces effective decode stall overhead by $>50\%$ compared to synchronous PyTorch `.to()` copies.
- **Required Figure Spec — Fig. 4.2 (`Proposed 3-Axis KV-Cache State Machine & Async Stream Pipeline`):**
  - *Type:* State transition diagram (`T0 ↔ T1 ↔ T2`) + timeline swimlane (GPU Compute Stream vs. PCIe H2D Prefetch Stream).
  - *Visual Tag:* Mark transition thresholds ($\tau_{\text{hot}}, \tau_{\text{cold}}, B_{\text{VRAM}}$) explicitly as *tunable WP-4 parameters*.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  1. **Compression Codec Choice for Tier 1:** `QServe`-style symmetric/group-wise `KV4` vs. asymmetric `K4V3` (`TurboQuant`) vs. low-rank SVD Keys (`ShadowKV`)? *(Awaiting PR #21 review + WP-4 kernel feasibility check.)*
  2. **Offload Granularity for Tier 2:** Value-only chunk offload (`ShadowKV`) vs. page-level K+V offload with AABB recall (`ArkVale`)?
  3. **CPU Dequantization Delegation:** Worth implementing in Sem 2, or does GPU-side fused dequantization (as in `QServe`) strictly dominate on RTX 4090 / RTX 4060 PCIe 4.0?
  4. **Target Integration Runtime for Sem 2 (`WP-6`):** Deep vLLM V1 `LMCacheMPConnector` integration vs. custom vLLM KV-cache manager plugin?

---

## Section 4.5 — Experimental Setup & Hyperparameter Registry

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Ch. 4 setup):** Consolidates the exact hardware environment, software versions, model checkpoints, dataset splits, and sweep parameters so every result in Chapter 5 is fully reproducible.
- **Committee Expectation:** Zero ambiguity on hardware caps, random seeds, prompt lengths, and model quantization status (strictly BF16 base weights).

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** All Semester 1 empirical runs were executed under a strictly controlled, commit-pinned HPC testbed (`gpu01` / `gpu06.slurm.cpe.kmutt.ac.th`, RTX 4090 capped to 16.0 GiB process VRAM) using unmodified BF16 `Qwen/Qwen3-4B-Instruct-2507`, with exact dataset SHA256 verification.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Listed RTX 4060 8 GB / 16 GB generically and unpinned 7B–8B models (`Llama-3 8B`, `Qwen-2.5 7B`).
  - *Current State:* Primary 16 GB cluster experiments standardized on `Qwen/Qwen3-4B-Instruct-2507` (BF16, `8.06 GB` decimal / `7.506 GiB` binary weights, `144 KiB/tok` KV) to leave $\approx 7.94\text{ GB}$ decimal weights-only headroom (`docs/MODELS.md`) — and **`7.694 GiB` net usable Tier-1 KV headroom** after the `0.80 GiB` runtime overhead buffer (`16.0 - 7.506 - 0.80 GiB`) or **`5.294 GiB` Tier-2 vLLM pool headroom** (`13.6 GiB` pool) per `analysis/THEORETICAL_LIMITS.md`; local RTX 4060 8 GB adaptation (`Qwen3-1.7B`, `WP1-5`) is explicitly tracked as pending.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Configuration Dimension | Exact Provenance File & Keys | Status Tag | Verified Value |
| :--- | :--- | :--- | :--- |
| Hardware & allocator cap | `results/manifests/run_manifest_baseline_16gb.json`, `run_manifest_baseline_pytorch_16gb.json`, `run_manifest_wp3_*.json` (`execution_environment.*`) | `[READY]` | RTX 4090 (`23.52 GiB` physical): Tier 1 process cap `cap_fraction = 0.680` → `16.00 GiB`; Tier 2 vLLM pool cap `effective_gpu_utilization = 0.5783` → `13.60 GiB`. Nodes: `gpu01.slurm.cpe.kmutt.ac.th` (vLLM runs + `streamingllm_pytorch`, `h2o_pytorch`) and `gpu06.slurm.cpe.kmutt.ac.th` (PyTorch baseline + 9 WP-3 runs) |
| Software stack (Two Distinct Cluster Envs) | `run_manifest_baseline_16gb.json`, `run_manifest_wp3_streamingllm.json` (vLLM env); `run_manifest_baseline_pytorch_16gb.json` + 11 WP-3 PyTorch manifests (PyTorch env); `docs/MODELS.md` §4 | `[READY]` | **vLLM V1 env:** `CUDA 13.0`, Driver `580.178.04`, `vLLM 0.30.0`, `PyTorch 2.13.0+cu130` (`kv_bench` SHA `3aee66fe` / `1241f558`).<br>**PyTorch SDPA env:** `CUDA 12.2`, `PyTorch 2.6.0` (matches `docs/MODELS.md` §4), `kv_bench` SHA `2a9338a2` for **both** `WP1-4-1` and `WP-3`, with `longmem_job_id: "30137"` across all 12 PyTorch manifests |
| Primary model checkpoint | `docs/MODELS.md` (§2) | `[READY]` | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` (36L, 32Q/8KV, $D=128$, BF16, `144 KiB/tok`) |
| Datasets & SHA256 | `results/manifests/dataset_manifest.json` | `[READY]` | `LongMemEval` (`150` samples = `30 × 5` tasks, SHA256 `bc37f751...`); `PG-19` (`5` books: `10146`, `10321`, `10356`, `10762`, `12204`, up to `75,868` approx. tokens) |
| Sweep grids | `results/baselines/baseline_hardware_16gb.json`, `concurrent_baseline_16gb.json`, `results/runs/wp3/*` | `[READY]` | Single-stream vLLM: `512`–`65,536`; Concurrent vLLM: `{1,4,8,12,16}` users × `{1024,2048,4096,8192}` ctx; WP-3 PyTorch: `512`–`32,768`; NIAH: `{2048,4096}` × 10 depths; LongMemEval: `150q` @ `32,768` |
| Local 8 GB testbed (`WP1-5`) | `WORK_PACKAGES.md` (`WP-1`), `docs/MODELS.md` (`Qwen3-1.7B`) | `[NEEDS_VERIFICATION]` | Pending execution on local RTX 4060 8 GB |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  1. **Hardware & Allocator Capping Protocol** (how 16.0 GiB cap is enforced on 23.52 GiB RTX 4090 + planned 8 GB RTX 4060 local parity).
  2. **Controlled Model Suite** (primary `Qwen3-4B-Instruct-2507` + secondary dense/hybrid models pinned for Sem 2 generalization).
  3. **Dataset Staging & Integrity** (`LongMemEval` 150q balanced 5-task split + `PG-19` 5-book continuous stream).
  4. **Sweep Matrix & Hyperparameter Summary** (pointing to Appendix C for the full parameter table).
- **Required Table Spec — Table 4.3 (`Semester 1 Experimental Environment & Sweep Configuration`):**
  - *Columns:* `Dimension | Cluster 16 GB Testbed (Verified) | Local 8 GB Testbed (WP1-5 Target) | Provenance Source`.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Confirm whether `docs/METRICS_AND_AXES.md` (which still mentions "50 curated multi-turn conversational histories" from early WP-2) should be updated to `150` samples to match `dataset_manifest.json` and all WP-3 runs before final report drafting. *(Recommended: Yes — note the WP-2 50q → WP-3 150q expansion explicitly in Ch. 4 & Ch. 5.)*
