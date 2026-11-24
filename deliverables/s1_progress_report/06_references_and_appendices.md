# References & Appendices Blueprint

> **KMUTT Report Mapping:** References + Appendices A, B, C (`Progress_Sem1_ENG_Example2.pdf` back matter)  
> **Document Role:** Templating, Drafting & Agent Handoff Specification (**not** final polished thesis prose)  
> **Branch Baseline:** `docs/s1-progress-report-blueprint` (branched from `main`; PR #21 unmerged)  
> **Target Milestone:** First advisor review draft — **December 1, 2026** (`WP-5`)

---

## References — Complete Bibliography Inventory

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` References):** Provides a complete, consistently formatted academic bibliography (IEEE style standard for KMUTT CPE) covering every foundational systems paper, KV-cache technique, benchmark suite, and software runtime cited in Chapters 1–5.
- **Committee Expectation:** Every in-text citation in Chapters 1–5 must resolve to a verified entry with complete authors, title, venue/publisher, year, and arXiv ID / DOI.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Our Semester 1 bibliography is grounded in **20 completed, implementation-grade literature reviews** in `docs/literature_survey/` (3 Tier-1 Foundations, 11 Tier-2 Techniques, 6 Tier-3 Benchmarks), supplemented by core serving/proposal references (vLLM/PagedAttention, FlashAttention, KIVI, SnapKV, PyramidKV, PG-19, WikiText-2).
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Cited ~7 KV-cache papers and high-level benchmark references.
  - *Current State:* Expanded to **20 deep technical reviews** in `docs/literature_survey/`, adding 2024–2026 state-of-the-art techniques (**ArkVale**, **CacheGen**, **EpiCache**, **ShadowKV**, **HeadQ**, **TurboQuant**) and conversational/diagnostic benchmarks (**LongMemEval**, **LoCoMo**, **RULER**, **Memory NIAH**, **SEAL**, **RealTalk**) plus **The Illusion of Equivalence** (FP16 non-associativity).

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

#### A. Tier 1 — Systems & Architecture Foundations (3 Completed Reviews `[READY]`)
| Ref ID | Title / System | Authors / Venue / Year | Repo Review File (`[READY]`) | BibTeX / DOI Status |
| :--- | :--- | :--- | :--- | :--- |
| `[F1]` | *Mastering LLM Techniques: Inference Optimization* | NVIDIA Technical Blog | `docs/literature_survey/foundations/nvidia_inference_optimization.md` | `[NEEDS_VERIFICATION]` (exact publication date & URL) |
| `[F2]` | *The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference* | Chodavarapu (2026), `arXiv:2604.15409v1` | `docs/literature_survey/foundations/illusion_of_equivalence.md` | `[NEEDS_VERIFICATION]` (full author list & BibTeX) |
| `[F3]` | *How Long Can Open-Source LLMs Truly Promise on Context Length?* (LongEval / LongChat) | LMSYS Org Technical Report (June 29, 2023) | `docs/literature_survey/foundations/longeval.md` | `[NEEDS_VERIFICATION]` (exact author list & URL) |

#### B. Tier 2 — KV-Cache Optimization Techniques (11 Completed Reviews `[READY]`)
| Ref ID | Technique | Authors / Venue / Year / arXiv | Repo Review File (`[READY]`) | BibTeX / DOI Status |
| :--- | :--- | :--- | :--- | :--- |
| `[T1]` | **FlexGen** | Sheng et al., *ICML 2023*, `arXiv:2303.06865v2` | `docs/literature_survey/techniques/flexgen.md` | `[NEEDS_VERIFICATION]` (PMLR pages/DOI) |
| `[T2]` | **H2O** | Zhang et al., *NeurIPS 2023*, `arXiv:2306.14048v3` | `docs/literature_survey/techniques/h2o.md` | `[NEEDS_VERIFICATION]` (NeurIPS pages/DOI) |
| `[T3]` | **StreamingLLM** | Xiao et al., *ICLR 2024* (Oral), `arXiv:2309.17453v4` | `docs/literature_survey/techniques/streamingllm.md` | `[NEEDS_VERIFICATION]` (OpenReview/DOI) |
| `[T4]` | **CacheGen** | Liu et al., *ACM SIGCOMM 2024*, `arXiv:2310.07240v6` | `docs/literature_survey/techniques/cachegen.md` | `[NEEDS_VERIFICATION]` (ACM DOI, ISBN `979-8-4007-0614-1`) |
| `[T5]` | **QServe (QoQ)** | Lin, Tang, Yang et al., *MLSys 2025*, `arXiv:2405.04532v3` | `docs/literature_survey/techniques/qserve.md` | `[NEEDS_VERIFICATION]` (MLSys proceedings DOI) |
| `[T6]` | **InfiniGen** | Lee, Lee, Seo, Sim, *USENIX OSDI 2024*, `arXiv:2406.19707v1` | `docs/literature_survey/techniques/infinigen.md` | `[NEEDS_VERIFICATION]` (USENIX pages/ISBN) |
| `[T7]` | **ShadowKV** | Sun et al., *arXiv:2410.21465v3* (Oct 2024 / Apr 2025) | `docs/literature_survey/techniques/shadowkv.md` | `[NEEDS_VERIFICATION]` (confirm if published at ICML/ICLR 2025) |
| `[T8]` | **ArkVale** | Chen et al., *NeurIPS 2024* | `docs/literature_survey/techniques/arkvale.md` | `[NEEDS_VERIFICATION]` (arXiv ID & NeurIPS pages) |
| `[T9]` | **EpiCache** | Kim et al. (Apple), *ICML 2026*, `arXiv:2509.17396v4` | `docs/literature_survey/techniques/epicache.md` | `[NEEDS_VERIFICATION]` (PMLR volume/pages) |
| `[T10]` | **HeadQ** | Ruiz Williams, `arXiv:2605.03562v2` (May 2026) | `docs/literature_survey/techniques/headq.md` | `[NEEDS_VERIFICATION]` (BibTeX check) |
| `[T11]` | **TurboQuant / Statistical Quality Measures** | D'Alberto (AMD), `arXiv:2605.08114v1` (Apr 2026) | `docs/literature_survey/techniques/turboquant.md` | `[NEEDS_VERIFICATION]` (BibTeX check) |

#### C. Tier 3 — Benchmarks, Datasets & Diagnostics (6 Completed Reviews `[READY]`)
| Ref ID | Benchmark | Authors / Venue / Year / arXiv | Repo Review File (`[READY]`) | BibTeX / DOI Status |
| :--- | :--- | :--- | :--- | :--- |
| `[B1]` | **LoCoMo** | Maharana et al., *ACL 2024*, `arXiv:2402.17753v1` | `docs/literature_survey/benchmarks/locomo.md` | `[NEEDS_VERIFICATION]` (ACL Anthology DOI) |
| `[B2]` | **RULER** | Hsieh, Sun et al., *COLM 2024*, `arXiv:2404.06654v3` | `docs/literature_survey/benchmarks/ruler.md` | `[NEEDS_VERIFICATION]` (COLM proceedings URL) |
| `[B3]` | **Memory NIAH** | Nelson et al. (IBM Research), `arXiv:2407.01437v2` | `docs/literature_survey/benchmarks/memory_niah.md` | `[NEEDS_VERIFICATION]` (venue/DOI check) |
| `[B4]` | **LongMemEval** | Wu et al., *ICLR 2025*, `arXiv:2410.10813v2` | `docs/literature_survey/benchmarks/longmemeval.md` | `[NEEDS_VERIFICATION]` (OpenReview/DOI) |
| `[B5]` | **SEAL** | Lee, Jin, Cho, Park (POSTECH), `arXiv:2501.15225v1` | `docs/literature_survey/benchmarks/seal.md` | `[NEEDS_VERIFICATION]` (venue/DOI check) |
| `[B6]` | **RealTalk** | Lee, Maharana, Pujara et al., `arXiv:2502.13270v1` | `docs/literature_survey/benchmarks/realtalk.md` | `[NEEDS_VERIFICATION]` (venue/DOI check) |

#### D. Supplementary / Proposal Core References (No Local Review File Yet)
| Ref ID | Topic / Paper | Cited In | Status Tag |
| :--- | :--- | :--- | :--- |
| `[S1]` | **vLLM / PagedAttention** (Kwon et al., *SOSP 2023*) | Ch. 1–5, `docs/MODELS.md`, `docs/literature_survey/README.md` | `[NEEDS_VERIFICATION]` (BibTeX & DOI) |
| `[S2]` | **FlashAttention / FlashAttention-2** (Dao et al., *NeurIPS 2022 / ICLR 2024*) | Ch. 2, `docs/literature_survey/README.md` | `[NEEDS_VERIFICATION]` (BibTeX & DOI) |
| `[S3]` | **KIVI** (Liu et al., *ICML 2024*), **SnapKV** (Li et al., 2024), **PyramidKV** (Cai et al., 2024) | Ch. 2, `docs/METRICS_AND_AXES.md`, Proposal slides | `[NEEDS_VERIFICATION]` (BibTeX & DOI) |
| `[S4]` | **PG-19** (Rae et al., *ICLR 2020*) & **WikiText-2** (Merity et al., *ICLR 2017*) | Ch. 3–5, `results/manifests/dataset_manifest.json` | `[NEEDS_VERIFICATION]` (BibTeX & DOI) |
| `[S5]` | **KV-Cache Survey (Jiang et al., 2026)** & **LMCache** (`lmcache >= 0.5.0`) | `docs/literature_survey/README.md`, `docs/MODELS.md` | `[NEEDS_VERIFICATION]` (BibTeX & URL) |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:** Export all 25+ entries into a single `references.bib` file formatted in **IEEE style** (matching KMUTT CPE convention), ordered by first appearance in Chapters 1–5.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Confirm citation style with advisors: **IEEE numeric `[1]`** (standard in `Progress_Sem1_ENG_Example2.pdf`) vs. APA author-year. *(Default: IEEE numeric.)*

---

## Appendix A — Candidate Algorithm Pseudocode & Config Registry

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Appendix A):** Houses detailed algorithmic pseudocode and implementation configurations that would clutter the narrative flow of Chapter 4.
- **Committee Expectation:** Allow technical reviewers to inspect exact cache update, eviction, quantization, and offload rules for every reproduced baseline.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Documents the exact algorithmic pseudocode and runtime configuration dictionary for all 11 WP-3 techniques implemented in `kv_bench` (`commit 2a9338a2`), explicitly noting any PyTorch SDPA simulation approximations.
- **Delta from Proposal Defense:** New appendix; proposal contained no implementation pseudocode.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Artifact | Exact Source | Status Tag | Purpose |
| :--- | :--- | :--- | :--- |
| Verified runtime configs for all 11 methods | `results/runs/wp3/verify_<tech>_pytorch_16gb.json` (`results[0].extra`) | `[READY]` | Exact hyperparameters (`sink_tokens`, `window_size`, `heavy_hitter_tokens`, `kv_cache_bits`, `svd_rank`, `page_size`, etc.) |
| Exact Python wrapper implementations | `khemingkapat/kv_bench` @ `2a9338a2` (`src/techniques/*.py`) | `[NEEDS_VERIFICATION]` | Extract concise 10–20 line pseudocode blocks per technique directly from `kv_bench` source |
| Wrapper fidelity audit notes | `analysis/wp3/*` (PR #21) | `[DRAFT/PR#21]` | Annotate any known implementation divergences (e.g., `headq`, `epicache`, `turboquant`) |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  - `A.1` Group A Eviction Pseudocode & Configs (`StreamingLLM`, `H2O`, `EpiCache`)
  - `A.2` Group B Quantization/Codec Pseudocode & Configs (`QServe KV4`, `TurboQuant`, `HeadQ`, `CacheGen`)
  - `A.3` Group C Offload/Prefetch Pseudocode & Configs (`FlexGen KV`, `ArkVale`, `InfiniGen`, `ShadowKV`)
- **Required Listing Spec:** Uniform LaTeX `algorithm2e` / Markdown pseudocode template (`Input: K, V, Q, budget B`, `Prefill Update`, `Decode Step`, `Telemetry Hook`) for each of the 11 candidates.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Include all 11 pseudocode blocks in Appendix A, or only the top 5–6 representative candidates (`ShadowKV`, `QServe KV4`, `ArkVale`, `FlexGen KV`, `H2O`, `StreamingLLM`) plus a compact config table for all 11? *(Recommended: Compact config table for all 11 + pseudocode for the 4–5 core mechanisms informing WP-4.)*

---

## Appendix B — Sample Provenance Manifest & Output JSON Schemas

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Appendix B):** Provides concrete machine-readable schemas and sample records proving the reproducibility and engineering rigor of the data pipeline.
- **Committee Expectation:** Verifiable schema definitions linking raw benchmark outputs, semantic grading, diagnostic telemetry, and SHA256 manifests.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Every empirical number in Chapter 5 is traceable through four standardized, CI-validated JSON schemas: (1) `run_manifest_*.json`, (2) `ResultEntry` (`*_hardware_*.json` / `*_longmem_*.json`), (3) `*_grades.json`, and (4) `*_judged.json`.
- **Delta from Proposal Defense:** New appendix reflecting the production data-provenance infrastructure built in Semester 1.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec — Exact Verified Keys)
| Schema | Exact Verified File on `main` (`[READY]`) | Verified Top-Level & Nested Keys |
| :--- | :--- | :--- |
| **1a. Baseline Run Manifests** (2 files: vLLM V1 & PyTorch SDPA) | `results/manifests/run_manifest_baseline_16gb.json`, `run_manifest_baseline_pytorch_16gb.json` | `manifest_version`, `generated_at`, `provenance_type`, `milestone`, `work_package`, `target_workload`, `primary_model`, `execution_environment.{cluster_node, slurm_job_id, gpu_model, physical_vram_gib, process_allocator_cap_gib, cuda_version, pytorch_version}` *(vLLM baseline adds `nvidia_driver`, `vllm_version`; PyTorch baseline omits `nvidia_driver`/`vllm_version` and adds `longmem_job_id: "30137"`)*, `source_repository.{repo, commit_sha, clean_working_tree}`, `golden_artifacts.<id>.{path, sha256, description}` *(PyTorch baseline also records `size_bytes`)* |
| **1b. WP-3 PyTorch Run Manifests** (11 files) | `results/manifests/run_manifest_wp3_<tech>.json` (incl. `h2o_pytorch`, `streamingllm_pytorch`) | `manifest_version`, `generated_at`, `provenance_type`, `milestone`, `work_package`, `technique`, `engine`, `axes`, `technique_config`, `primary_model`, `execution_environment.{cluster_node, slurm_job_id, longmem_job_id ("30137"), gpu_model, physical_vram_gib, process_allocator_cap_gib}` *(all `11/11` omit `nvidia_driver`/`vllm_version`; `9/11` [`turboquant`–`epicache`] also omit `cuda_version`/`pytorch_version`/`cap_fraction`/`effective_gpu_utilization` inside `execution_environment`)*, `source_repository.{repo, commit_sha, clean_working_tree}`, `golden_artifacts.<id>.{path, sha256, size_bytes, description}` |
| **1c. Legacy WP-3 vLLM Run Manifest** (1 file) | `results/manifests/run_manifest_wp3_streamingllm.json` | `manifest_version`, `generated_at`, `provenance_type`, `milestone`, `work_package`, `technique`, `axes`, `technique_config`, `primary_model` *(no `engine`, `longmem_job_id`, or `golden_artifacts.<id>.size_bytes`)*, `execution_environment.{cluster_node, slurm_job_id, gpu_model, physical_vram_gib, process_allocator_cap_gib, cap_fraction, effective_gpu_utilization, cuda_version ("13.0"), nvidia_driver, vllm_version, pytorch_version}`, `source_repository`, `golden_artifacts.<id>.{path, sha256, description}` |
| **2. Dataset Manifest** (`dataset_manifest.json`) | `results/manifests/dataset_manifest.json` | `manifest_version`, `generated_at`, `is_dry_run`, `datasets.longmemeval.{name, role, upstream, file.{path, sha256, size_bytes, sample_count, task_distribution}}`, `datasets.pg19.{name, role, upstream, files[].{path, book_id, sha256, size_bytes, word_count, approx_tokens}}` *(distinct schema: no `milestone`/`work_package`/`slurm_job_id`/`commit_sha`)* |
| **3. Benchmark Output** (`ResultEntry`) | `results/baselines/baseline_hardware_16gb.json`, `results/runs/wp3/*_hardware_pytorch_16gb.json` | Top-level: `model`, `technique`, `axes`, `results[]`<br>Entry keys (20): `requested_ctx`, `status`, `axes`, `prompt_tokens`, `generated_tokens`, `ttft_sec`, `decode_time_sec`, `total_time_sec`, `throughput_tok_per_sec`, `tpot_mean_ms`, `tpot_p95_ms`, `cold_ttft_sec`, `cold_total_sec`, `ttft_p95_sec`, `peak_alloc_mb`, `peak_reserved_mb`, `host_rss_delta_mb`, `memcpy_bytes_per_token`, `error`, `extra` |
| **4. LongMemEval Diagnostic & Sample Record** | `results/runs/wp3/arkvale_longmem_pytorch_16gb.json` | `extra.sample_records[]`: `id`, `task_type`, `unanswerable`, `prompt_tokens`, `generated_tokens`, `finish_reason`, `question`, `expected_answer`, `prediction`, `diagnostic.{overall_retention_ratio, overall_effective_bitwidth, min_session_effective_bitwidth, most_degraded_session_idx, num_sessions}` |
| **5. Raw Grades** (`*_grades.json`) | `results/grades/baseline_grades.json` | Dict keyed by `sample_id` (150 keys) → `{correct: bool, reason: str, grader: "orchestrator-pro-semantic-v1"}` |
| **6. Judged Summary** (`*_judged.json`) | `results/grades/arkvale_judged.json`, `baseline_judged.json` | `technique`, `requested_ctx`, `source_result`, `source_grades`, `dataset_sha256`, `total_samples`, `correct_samples`, `overall_accuracy`, `per_task_accuracy`, `per_task_counts`, `causal_verdict_counts`, `samples[].{id, task_type, unanswerable, prompt_tokens, generated_tokens, finish_reason, correct, baseline_correct, causal_verdict, reason, grader, diagnostic}` *(note: internal `"source_result"` strings record pre-staging cluster paths such as `results/longmem_v2_baseline_pytorch_16gb.json` and `results/<tech>_longmem_pytorch_16gb.json`, which differ from the staged repo paths `results/baselines/...` and `results/runs/wp3/...`)* |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  - `B.1` Sample `run_manifest_baseline_16gb.json` + `run_manifest_wp3_shadowkv.json` (abridged side-by-side showing vLLM baseline vs. WP-3 PyTorch manifest keys).
  - `B.2` Standardized `ResultEntry` JSON Schema (hardware + `extra` quality/diagnostic keys).
  - `B.3` Sample `*_grades.json` & `*_judged.json` Record (1 passing + 1 `QUALITY_REGRESSION` sample with `diagnostic` block).
  - `B.4` Complete Semester 1 Manifest Registry — split cleanly into **(i) 14 `run_manifest_*.json` files** (`filename | milestone | wp | slurm_job_id | longmem_job_id | kv_bench_sha | artifact_count`) and **(ii) the 15th manifest (`dataset_manifest.json`)** summarized separately (`datasets | sample_count / file_count | sha256`).
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):** None — all schemas are `[READY]` on `main`.

---

## Appendix C — Master Hyperparameter & Semester 2 Ablation Study Configuration Tables

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` Appendix C):** Consolidates the full experimental hyperparameter registry (Semester 1) and the factorized ablation matrix (Semester 2 `WP-7` / `WP-8`) into clean reference tables.
- **Committee Expectation:** Complete parameter transparency for reproducibility and clear experimental control variables for Semester 2.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Provides a single-stop reference for (1) all 9 pinned BF16 model checkpoints (`docs/MODELS.md`), (2) all 11 WP-3 candidate hyperparameter configurations verified from `results/runs/wp3/verify_*.json`, and (3) the planned Semester 2 ablation & multi-model stress matrix (`WP-6`–`WP-8`).
- **Delta from Proposal Defense:** Replaces generic model names and unpinned sweep ranges with exact HF commit SHAs and verified WP-3 parameter dictionaries.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Table | Exact Provenance Source | Status Tag | Notes |
| :--- | :--- | :--- | :--- |
| Table C.1 — Pinned Model Registry (9 Checkpoints) | `docs/MODELS.md` (§2–§3) | `[READY]` / `[NEEDS_VERIFICATION]` | Exact HF repo IDs, 40-char Git commit SHAs, $L, H_Q/H_{KV}, D$, KV KiB/tok, `--language-model-only` requirement. *Note:* `Qwen 3.5 2B` and `Gemma 4 E4B Instruct` (`KV rate = Heterogeneous`) appear in `docs/MODELS.md` §2 but are omitted from §3's Empirical Allocation Matrix (`32K (1x)` / `10x 4K` cells → `N/A` / `[NEEDS_VERIFICATION]`) |
| Table C.2 — WP-3 Master Hyperparameter Registry (11 Methods) | `results/runs/wp3/verify_<tech>_pytorch_16gb.json` (`results[0].extra`) | `[READY]` | Verified exact runtime parameters used across all 11 runs |
| Table C.3 — Semester 2 Ablation & Cross-Model Stress Matrix | `WORK_PACKAGES.md` (`WP-6`, `WP-7`, `WP-8`) + `docs/MODELS.md` | `[PENDING_S2]` / `[OPEN_DECISION]` | Exact WP-4 tier thresholds (`tau_hot`, `tau_cold`, bitwidth) remain `[OPEN_DECISION]` until PR #21 & WP-4 complete |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton & Exact Table Column Specs:**
  - **Table C.1 (`Pinned BF16 Model Suite & Theoretical KV Footprint`):** `Role | Model Name | HF Commit SHA (short) | Arch Class | Layers / Q:KV Heads / D | KV KiB/tok | 32K (1x) | 10x 4K | Target Budget`.
  - **Table C.2 (`Verified WP-3 Candidate Hyperparameters`):** `Method | Axes | Block/Page Size | Budget / Compression Parameters | Offload / Prefetch Parameters | Theoretical Compression Ratio`.
  - **Table C.3 (`Semester 2 Factorized Ablation Matrix — WP-8`):** `Ablation ID | Structural Axis (Tier 1) | Spatial Axis (Tier 2) | Temporal Axis (Async Stream) | Target Models | Primary Evaluation Metrics`.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Finalize `WP-8` ablation parameter grid (e.g., `KV4` group sizes `{32, 64}`, offload trigger thresholds `{70%, 80%, 90%}` VRAM, and CPU-delegation toggle) once `WP-4` policy design is locked.
