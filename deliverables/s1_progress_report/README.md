# Semester 1 Progress Report — Blueprint & Agent Handoff Index

> **Project:** Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs  
> **Course / Institution:** CPE401 Senior Project I · Department of Computer Engineering, Faculty of Engineering, KMUTT  
> **Academic Year:** 2026–2027 (B.E. 2568–2569)  
> **Advisors:** Dr. Aye Hninn Khine (Advisor), Dr. Boonyarit Changaival (Co-Advisor)  
> **Target Milestone:** First advisor review draft due **December 1, 2026** (`WP-5`)  
> **Format Baseline:** KMUTT Computer Engineering Semester 1 Progress Report (`Progress_Sem1_ENG_Example2.pdf`, untracked local reference)  
> **Proposal Baseline:** [`deliverables/s1_proposal_defense/KV-Cache Co-optimization.pdf`](../s1_proposal_defense/KV-Cache%20Co-optimization.pdf) & [`Slide_Content_and_Executive_Summary.md`](../s1_proposal_defense/Slide_Content_and_Executive_Summary.md)

---

## 1. Purpose of This Directory

This directory contains **structured chapter blueprints and agent handoff specifications** — **not** final polished thesis prose or publication-ready plots.

Each chapter file serves two audiences simultaneously:
1. **Human Authors (Team & Advisor Review):** Clarifies what each KMUTT report section must accomplish, what core message to convey, what evolved since the Proposal Defense, and which open decisions still require advisor/team sign-off.
2. **Future Coding / Writing Agents:** Provides **strict data and source provenance maps** (exact repo file paths, JSON keys, manifest IDs, GitHub Project #9 issue numbers, and readiness tags) so future agents can draft prose, generate tables, and render charts without guessing or hallucinating.

---

## 2. Standard 4-Block Blueprint Pattern

Every chapter blueprint in this directory follows the same 4-block structure per major section:

1. **Section Purpose & KMUTT Baseline Alignment** — Role of the section in `Progress_Sem1_ENG_Example2.pdf` and committee expectations.
2. **Core Message & Delta from Proposal Defense** — Key narrative to convey + explicit diff from `KV-Cache Co-optimization.pdf`.
3. **Strict Data & Source Provenance Map (Agent Handoff Spec)** — Exact relative paths, JSON keys, manifest IDs, GitHub issues, and readiness tags:
   - `[READY]` — Verified present on `main` (and current branch `docs/s1-progress-report-blueprint`).
   - `[DRAFT/PR#21]` — Exists only in unmerged PR #21 (`docs/wp3-result-interpretation`); **do not cite as merged until PR #21 is reviewed and merged**.
   - `[NEEDS_VERIFICATION]` — Requires manual check, schema reconciliation, or quick re-run before final prose.
   - `[PENDING_S2]` — Planned for Semester 2 (`WP-6`–`WP-9`); describe as experimental plan / follow-up only.
   - `[OPEN_DECISION]` — Genuine unresolved design, scope, or administrative decision requiring team/advisor input (do **not** commit blindly).
4. **Draft Outline, Table/Figure Specs & Open Questions** — Bullet skeleton, exact column/axis/diagram specs, and unresolved questions.

---

## 3. Master Chapter Map & Verification Status

| File | Report Scope | KMUTT Baseline Mapping | Primary Repo Sources | Audit Status |
| :--- | :--- | :--- | :--- | :--- |
| [`00_front_matter.md`](00_front_matter.md) | Cover, Approval, Abstract, ToC, Lists of Tables/Figures/Symbols/Abbreviations | Front Matter (`i`–`ix`) | `deliverables/s1_proposal_defense/`, `docs/METRICS_AND_AXES.md`, `docs/MODELS.md` | ✅ Verified (`pro` Audit PASS) |
| [`01_introduction_and_timeline.md`](01_introduction_and_timeline.md) | Ch. 1: Problem, RQs/Objectives, Scope, WBS, Updated Gantt (Sem 1 & 2), Team Workload, Risks, Deliverables | Chapter 1 (`1.1`–`1.2`) | `WORK_PACKAGES.md`, GitHub Project #9 (35 items), `docs/proposal.md` | ✅ Verified (`pro` Audit PASS) |
| [`02_related_works.md`](02_related_works.md) | Ch. 2: Background, Core Theory, Tech Stack, Related Research (3 Axes + Benchmarks), Limitations, Design Rationale | Chapter 2 (`2.1`–`2.7`) | `docs/literature_survey/` (20 completed reviews across `foundations/`, `techniques/`, `benchmarks/`) | ✅ Verified (`pro` Audit PASS) |
| [`03_methodology_and_design.md`](03_methodology_and_design.md) | Ch. 3: Two-Tier Repo & Harness Design, 3-Axis Framework, Controlled Models & Hardware Caps, Evaluation Plan, Datasets | Chapter 3 (`3.1`–`3.4`) | `docs/MODELS.md`, `docs/METRICS_AND_AXES.md`, `docs/WORKFLOW_SPEC.md`, `workloads/download_datasets.py` | ✅ Verified (`pro` Audit PASS) |
| [`04_algorithms_and_implementation.md`](04_algorithms_and_implementation.md) | Ch. 4: Dual-Backend Harness Architecture, Metric Math, Post-Hoc Diagnostic Inspector, 11 Candidate Formulations, WP-4 Policy State-Machine Direction | Chapter 4 (`4.1`–`4.11`) | `docs/METRICS_AND_AXES.md`, `docs/WORKFLOW_SPEC.md`, `docs/literature_survey/techniques/`, `results/manifests/` | ✅ Verified (`pro` Audit PASS) |
| [`05_preliminary_results_and_followup.md`](05_preliminary_results_and_followup.md) | Ch. 5: Testbed & WP-3 Implementation Status, WP-1/WP-2 Baselines, WP-3 11-Method Comparative Snapshot, Sem 2 Full Eval & Ablation Plan, Next Steps | Chapter 5 (`5.1`–`5.4`) | `results/baselines/`, `results/runs/wp3/`, `results/grades/`, `results/manifests/`, `[DRAFT/PR#21]` | ✅ Verified (`pro` Audit PASS) |
| [`06_references_and_appendices.md`](06_references_and_appendices.md) | Complete Bibliography Inventory + Appendix A (Algorithms/Configs), B (Sample Provenance Manifests & Output Schemas), C (Hyperparameter & Ablation Matrix) | References & Appendices A–C | `docs/literature_survey/`, `results/manifests/`, `results/baselines/`, `results/runs/wp3/` | ✅ Verified (`pro` Audit PASS) |

---

## 4. High-Level Delta Summary: Proposal Defense → Current Semester 1 State

| Dimension | Proposal Defense (`KV-Cache Co-optimization.pdf`) | Current Verified State (`kv_co-op_hub` + GitHub Project #9) |
| :--- | :--- | :--- |
| **Model Suite** | Generic families (*Llama-3, Qwen-3, Gemma 3, Mistral*; summary also listed Llama-3 8B / Qwen-2.5 7B) | 9 exact HF commit-SHA pinned BF16 checkpoints in `docs/MODELS.md`, split into **Pure Dense GQA** (`Qwen3-4B-Instruct-2507` primary @ `144 KiB/tok`) vs. **Hybrid / SWA** (`Qwen3.5-4B`, `Gemma-4-E4B-it`) |
| **Execution Engine** | Assumed vLLM + custom C++/CUDA kernels for all experiments | **Dual-Backend Harness** (`WP2-5`, `WP3-1-3`): **vLLM** for production/concurrency baselines + **Unified PyTorch Native SDPA** for white-box KV-cache prototyping & post-hoc layer retention diagnostics (`WP1-4-1`) |
| **Quality Pillars** | Perplexity ($\Delta\text{PPL} \le 0.1$) listed alongside Logit Divergence & LongMemEval | **Two Primary Pillars** (`docs/METRICS_AND_AXES.md`): (1) Micro Logit Divergence (`top1_agreement`, `kl_div_mean`, `kl_div_p99` up to 32K PG-19) & (2) Macro LongMemEval (150q offline semantic judge + post-hoc layer survival), plus opt-in **2D NIAH**. Perplexity strictly supplementary. |
| **Literature & Candidates** | 7 surveyed techniques; 5 planned baselines | **20 completed literature reviews** (3 foundations, 11 techniques, 6 benchmarks); **all 11 techniques empirically reproduced & benchmarked** in WP-3 (adding CacheGen, EpiCache, HeadQ, TurboQuant) |
| **Project Tracking** | Static spreadsheet Gantt (4 ad-hoc sections, no live issue tracking) | **Two-Tier Multi-Repo Workflow** (`kv_co-op_hub` + `kv_bench`) unified on **GitHub Project #9** (`WP-1`–`WP-9`, 35 tracked items; 24 of 26 sub-issues `Done` as of 2026-11-24) |
| **WP-4 Policy Design** | Early fixed 3-layer heuristic (L1 Priority, L2 Format, L3 Transfer) | Held open for evidence-driven synthesis: WP-3 white-box analysis (`[DRAFT/PR#21]`) separates algorithmic merit from PyTorch wrapper bugs to define falsifiable WP-4 candidate hypotheses |

---

## 5. Repo-Wide Guardrails for Future Agents

- **Scope Compliance (`.agents/AGENTS.md`):** Never violate the 16 GB cluster / 8 GB local VRAM caps, never substitute pre-quantized GGUF/AWQ base weights for pinned BF16/FP16 checkpoints, and never promote Perplexity above Logit Divergence + LongMemEval.
- **PR #21 Boundary:** Files added in PR #21 (`analysis/wp3/*`, `analysis/figures/wp3/*`, `results/baselines/baseline_pytorch_16gb.json`) are **not yet merged into `main`** as of 2026-11-24. Always tag references to them as `[DRAFT/PR#21]`.
- **Untracked Reference PDF:** `Progress_Sem1_ENG_Example2.pdf` in the repo root is a local formatting reference only — **never commit it to git**.

---

## 6. Consolidated Open Decisions & Pre-Draft Checklist (For Team / Advisor Review)

These items are intentionally left open (`[OPEN_DECISION]` / `[NEEDS_VERIFICATION]`) across chapters so no prose or design choice is committed blindly before review:

1. **PR #21 Manual Review & Merge (`docs/wp3-result-interpretation`):** Merging PR #21 promotes `analysis/wp3/CONSOLIDATED_PATTERN_REPORT.md`, 11 technique analyses, 37 figures (`analysis/figures/wp3/*`), and `results/baselines/baseline_pytorch_16gb.json` from `[DRAFT/PR#21]` to `[READY]`.
2. **WP-4 Co-Optimization Policy Scope for S1 Defense (`01`, `04`, `05`):** Decide whether the S1 Progress Report frames WP-4 as an **evidence-driven architecture specification + falsifiable candidate hypotheses** or requires a **runnable prototype smoke test**; hold Tier-1 codec (`QServe KV4` vs. `TurboQuant` vs. `ShadowKV` SVD Keys), Tier-2 offload granularity (Value-only vs. page-level AABB), and CPU dequantization delegation vs. GPU fused kernels open until WP-4 micro-benchmarks run.
3. **WP1-5 Local 8 GB RTX 4060 Adaptation (`01`, `02`, `03`, `05`):** Decide whether `kv_bench#6` (`Qwen3-1.7B` on local RTX 4060 8 GB) executes before S1 defense or is formally deferred to Semester 2.
4. **Administrative & Formatting Details (`00`, `01`):** Confirm Examination Committee member names, exact CPE401 credits & program track, CE vs. B.E. Academic Year string (`2026–2027` vs. `B.E. 2568 / 2569–2570`), CITI Program Ethics Exam completion across all 4 members, LaTeX vs. Word/Docs toolchain, and final team role / WP ownership wording (`Table 1.3`).
5. **Repo Housekeeping (`01`, `03`, `04`, `06`):** Sync `WORK_PACKAGES.md` stale `🔴 Not started` badges and GitHub Project #9 `WP-3` epic status (`Todo` → `In Progress`, since `14/15` sub-issues are `Done`); reconcile `docs/METRICS_AND_AXES.md` minor schema/count drifts (`50` vs. `150` LongMemEval prompts, `longmem_macro_f1` vs. `longmem_f1_overall`, top-level vs. `extra` `memcpy_*` keys).

