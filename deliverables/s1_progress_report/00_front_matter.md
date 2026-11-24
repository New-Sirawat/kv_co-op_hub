# Front Matter Blueprint (`00_front_matter.md`)

> **Report Scope:** KMUTT Senior Project Semester 1 Progress Report — Front Matter (Cover, Approval, and Roman-numeral pages `ii`–`ix`)  
> **KMUTT Baseline:** `Progress_Sem1_ENG_Example2.pdf` (untracked local reference)  
> **Status:** Blueprint & Agent Handoff Spec (NOT final polished prose)  
> **Last Verified Against Workspace:** 2026-11-24

---

## Section FM-1: Cover Page, Committee Approval Page & Project Metadata Block

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Verified pagination of `Progress_Sem1_ENG_Example2.pdf`:
  1. **Cover Page (p. 1, unnumbered):** Formal KMUTT Faculty of Engineering / Department of Computer Engineering title page displaying the full English project title, course identifier (`CPE401 Senior Project I`), all four student names and 11-digit KMUTT student IDs, and Academic Year.
  2. **Project Committee Approval Page (p. 2, unnumbered / `i` implied):** Signature block for the Project Advisor, Co-Advisor, and Examination Committee members certifying review of the Semester 1 Progress Report.
  3. **Project Metadata Block (shares page `ii` / p. 3 with Abstract & Keywords in `Progress_Sem1_ENG_Example2.pdf`):** Standardized administrative summary table (Project Title, Degree Program, Credits, Student Roster, Advisors, Academic Year) used by department staff and examiners for quick verification.
- **Committee Expectations:** Exact spelling of student names, student IDs, advisor titles, and course metadata with zero discrepancies against university registration records.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Establish official academic authorship, supervision, and institutional framing for *Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs* under KMUTT CPE401 (Academic Year 2026–2027).
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf` & `docs/proposal.md`):**
  - `docs/proposal.md` header lists `Academic Year 2568`, whereas `Slide_Content_and_Executive_Summary.md` and `deliverables/s1_progress_report/README.md` write `Academic Year 2026–2027 (B.E. 2568–2569)` (note the CE/B.E. calendar offset: `2026–2027 CE` corresponds to `B.E. 2569–2570`, whereas `B.E. 2568` corresponds to `2025 CE`). The Progress Report must reconcile this against official KMUTT registrar records (`[NEEDS_VERIFICATION]`).
  - The Proposal Defense did not include a formal Examination Committee signature page; this page must now be populated once committee assignments are confirmed.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Data Element | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| Project Title, Advisors, 4 Members & IDs | `docs/proposal.md` (Lines 5–14) & `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` (Lines 3–13) | `[READY]` | Verified identical across both files: Kantawich Limwilai (`66070503406`), Khem Ingkapat (`66070503408`), Sirawat Sanboonsong (`66070503456`), Run Prajunban (`66070503480`); Advisor: Dr. Aye Hninn Khine; Co-Advisor: Dr. Boonyarit Changaival. |
| Course Code & Target Milestone | `deliverables/s1_progress_report/README.md` (Lines 4–7) | `[READY]` | `CPE401 Senior Project I`, Department of Computer Engineering, Faculty of Engineering, KMUTT; First advisor review draft due Dec 1, 2026 (`WP-5`). |
| Academic Year Format & CE/B.E. Offset | `docs/proposal.md` (Line 3: `2568`) vs. `README.md` (Line 5: `2026–2027 (B.E. 2568–2569)`) | `[NEEDS_VERIFICATION]` | Resolve CE/B.E. offset (`2026–2027 CE` = `B.E. 2569–2570`, while `docs/proposal.md` & `README.md` reference `2568`) and confirm exact registrar format. |
| Course Credits (`3` Credits?) | `Progress_Sem1_ENG_Example2.pdf` (local baseline) | `[OPEN_DECISION]` | Confirm exact credit count for CPE401 Senior Project I in the B.Eng. Computer Engineering curriculum before printing. |
| Examination Committee Members | *Not yet recorded in repository* | `[OPEN_DECISION]` | Obtain official committee member names/titles from department / advisor before finalizing Approval Page. |
| Metadata Block + Abstract Page Sharing | `Progress_Sem1_ENG_Example2.pdf` (p. 3 / `ii`) | `[OPEN_DECISION]` | Decide whether Project Metadata Block shares page `ii` with Abstract & Keywords (matching `Progress_Sem1_ENG_Example2.pdf`) or splits onto its own page. |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Draft Skeleton
- **Page 1 (unnumbered) — Cover Page**
  - KMUTT Emblem (per department LaTeX/Word template)
  - Report Type: `SENIOR PROJECT I PROGRESS REPORT`
  - Project Title: `Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs`
  - Author Block (4 students, ordered by Student ID; honorifics omitted until confirmed):
    1. Kantawich Limwilai — `66070503406`
    2. Khem Ingkapat — `66070503408`
    3. Sirawat Sanboonsong — `66070503456`
    4. Run Prajunban — `66070503480`
  - Degree & Affiliation: `A Senior Project Progress Report Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Engineering in Computer Engineering, Department of Computer Engineering, Faculty of Engineering, King Mongkut's University of Technology Thonburi`
  - Academic Year: `Academic Year 2026–2027` *(resolve CE/B.E. string — `[NEEDS_VERIFICATION]`)*
- **Page 2 (unnumbered / `i` implied) — Project Committee Approval Page**
  - Statement of submission and approval for `CPE401 Senior Project I`
  - Signature lines:
    - Advisor: `Dr. Aye Hninn Khine`
    - Co-Advisor: `Dr. Boonyarit Changaival`
    - Committee Chairperson / Members: `[TBD — OPEN_DECISION]`
- **Page `ii` (p. 3, shared with Abstract & Keywords in baseline — `[OPEN_DECISION]`) — Project Metadata Table**

#### Table Spec: Project Metadata Block
| Field | Value |
| :--- | :--- |
| **Project Title** | Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs |
| **Course** | CPE401 Senior Project I (`[OPEN_DECISION: confirm 3 credits]`) |
| **Program** | B.Eng. in Computer Engineering (International / Regular — `[OPEN_DECISION]`) |
| **Project Members** | Kantawich Limwilai (`66070503406`), Khem Ingkapat (`66070503408`), Sirawat Sanboonsong (`66070503456`), Run Prajunban (`66070503480`) |
| **Project Advisor** | Dr. Aye Hninn Khine |
| **Co-Advisor** | Dr. Boonyarit Changaival |
| **Academic Year** | `[NEEDS_VERIFICATION: reconcile 2026–2027 CE vs. B.E. 2568 / 2569–2570]` |

#### Open Questions / Decisions Needed
1. **`[OPEN_DECISION]` Committee Roster:** Who are the assigned Examination Committee members for the Semester 1 Progress Defense?
2. **`[OPEN_DECISION]` Credit Count & Program Track:** Confirm exact credits (e.g., 3 credits) and whether to state "Regular Program" or "International Program" on the metadata page.
3. **`[OPEN_DECISION]` Metadata Block + Abstract Pagination:** Should the Project Metadata Block share page `ii` with the Abstract & Keywords (exact match to `Progress_Sem1_ENG_Example2.pdf`) or be placed on separate pages?
4. **`[NEEDS_VERIFICATION]` Academic Year String & Student Honorifics:** Reconcile the CE/B.E. offset (`2026–2027 CE` = `B.E. 2569–2570` vs. `2568` in `docs/proposal.md` / `README.md`) and confirm whether student honorifics (e.g., Mr./Ms.) are required on the KMUTT cover/approval pages.

---

## Section FM-2: Abstract & Keywords

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Page `ii` (p. 3, shared with the Project Metadata Block) of `Progress_Sem1_ENG_Example2.pdf`. A concise, standalone English abstract (~250–350 words) followed by 5–7 comma-separated **Keywords**.
- **Committee Expectations:** Summarize the technical problem, why existing approaches fail, the project's 3-axis co-optimization framework, **verified Semester 1 progress** (testbed, baselines, 11-method WP-3 exploration), and the Semester 2 roadmap — without overclaiming unimplemented Semester 2 results.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Autoregressive LLM decoding on 8–16 GB consumer GPUs hits a severe KV-cache memory wall at long contexts (up to 32K tokens) and multi-user concurrency (10–15 SME sessions). Isolated fixes (eviction, quantization, offloading) trade one bottleneck for another. In Semester 1, we established a reproducible two-tier benchmark infrastructure (`kv_co-op_hub` + `kv_bench`), pinned unmodified BF16 base checkpoints (`Qwen3-4B-Instruct-2507` primary), profiled native vLLM OOM/latency ceilings under a 16 GB VRAM cap, and reproduced and micro-benchmarked 11 published KV-cache techniques across two primary quality pillars (Micro Logit Divergence and Macro LongMemEval) to ground our upcoming 3-axis co-optimization policy (`WP-4`) and Semester 2 runtime integration (`WP-6`–`WP-9`).
- **Delta from Proposal Defense:**
  - Moves from **prospective proposal claims** (*"we will achieve 30–40% higher throughput and 95–98% quality retention"*) to **verified Semester 1 empirical progress** plus clearly demarcated Semester 2 targets.
  - Reflects the shift to **Two Primary Quality Pillars** (Logit Divergence + LongMemEval; Perplexity strictly supplementary) and the **Dual-Backend Harness** (vLLM + PyTorch Native SDPA).

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Data Element | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| Project Goal & 3-Axis Framing | `WORK_PACKAGES.md` (Line 4), `docs/METRICS_AND_AXES.md` (Lines 7–21) | `[READY]` | Strict scope: `STRUCTURAL`, `SPATIAL`, `TEMPORAL` axes; 8–16 GB consumer GPU caps; up to 32K context. |
| Pinned Base Models & KV Footprint | `docs/MODELS.md` (Lines 24–63) | `[READY]` | Primary Dense GQA model: `Qwen/Qwen3-4B-Instruct-2507` (`144 KiB/tok`, BF16 weights ~8.06 GB, 1x 32K = 4.50 GB, 10x 32K = 45.0 GB OOM). |
| WP-2 Native vLLM Baseline Provenance | `results/manifests/run_manifest_baseline_16gb.json`, `results/baselines/baseline_hardware_16gb.json`, `results/baselines/concurrent_baseline_16gb.json` | `[READY]` | Slurm Job `22868`, RTX 4090 capped at `16.0` GiB, `vllm 0.30.0`, `kv_bench` SHA `3aee66fe...`. |
| WP-3 11-Method Benchmark Artifacts | `results/runs/wp3/` (44+ JSON files), `results/grades/` (24 JSON files), `results/manifests/run_manifest_wp3_*.json` (12 manifests) | `[READY]` | All 11 techniques empirically run and graded; exact headline numbers must be extracted only from verified JSONs. |
| WP-3 Comparative Interpretation & PyTorch Baseline | `analysis/wp3/*`, `results/baselines/baseline_pytorch_16gb.json` | `[DRAFT/PR#21]` | **Do NOT cite PR #21 headline numbers in the final Abstract until PR #21 is merged into `main`.** |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Draft Skeleton (4-Paragraph Structure, ~280–320 words)
1. **Problem & Motivation (2–3 sentences):** Linear KV-cache growth during autoregressive decode exhausts 8–16 GB consumer GPU VRAM under long contexts (up to 32K) and multi-user SME workloads (10–15 concurrent sessions); isolated eviction, quantization, or offloading shifts bottlenecks to permanent fact loss or PCIe stalls.
2. **Approach & Methodology (2–3 sentences):** Formalize KV-cache management as a 3-axis co-optimization problem (`STRUCTURAL`, `SPATIAL`, `TEMPORAL`) evaluated on unmodified, SHA-pinned BF16 models (`docs/MODELS.md`) using a dual-backend harness (vLLM + PyTorch Native SDPA) and two primary quality pillars (Micro Logit Divergence and Macro LongMemEval across 5 capabilities).
3. **Semester 1 Progress (3–4 sentences):** Completed hardware/software testbed setup on a 16 GB VRAM-capped RTX 4090 cluster (`WP-1`–`WP-2`), established golden uncompressed baselines (single-stream up to 32K and multi-user concurrency sweeps), completed 20 literature reviews, and reproduced/benchmarked 11 SOTA KV-cache techniques (`WP-3`) with full SHA256 provenance manifests.
4. **Remaining S1 & Semester 2 Outlook (2 sentences):** Finalizing 3-axis co-optimization policy synthesis (`WP-4`) and local 8 GB RTX 4060 adaptation (`kv_bench#6`), followed in Semester 2 by production vLLM integration (`WP-6`), 32K stress/LongMemEval benchmarking (`WP-7`), ablation studies (`WP-8`), and thesis/artifact release (`WP-9`).
- **Keywords (6–7 terms):** `KV-Cache Management`, `Large Language Model Inference`, `Consumer GPUs`, `Three-Axis Co-Optimization`, `Long-Context Serving`, `Logit Divergence`, `LongMemEval`.

#### Open Questions / Decisions Needed
1. **`[OPEN_DECISION]` Numerical Claims in Abstract:** Should the Semester 1 Abstract quote specific WP-2 OOM cliff numbers (e.g., from `results/baselines/baseline_hardware_16gb.json`) and/or WP-3 top-line numbers (contingent on PR #21 merge), or stay qualitative on WP-3 until PR #21 is merged?

---

## Section FM-3: Table of Contents, List of Tables & List of Figures Specifications

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Verified pagination of `Progress_Sem1_ENG_Example2.pdf`:
  - **Table of Contents (`iii`–`v`, pp. 4–6):** Hierarchical numbering matching Chapters 1–5, References, and Appendices A–C.
  - **List of Tables (`vi`):** Auto-generated index of all numbered tables (`Table 1.1`, `Table 2.1`, ..., `Table C.3`) with concise captions and page numbers.
  - **List of Figures (`vii`):** Auto-generated index of all numbered figures (`Figure 1.1`, ..., `Figure 5.3`) with concise captions and page numbers.
- **Committee Expectations:** Every table and figure listed must exist in the body or appendices, be referenced in prose, and carry unambiguous provenance.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Provide a complete, cross-chapter navigation map connecting the conceptual framework (Ch. 1–2), experimental methodology and algorithms (Ch. 3–4), verified Semester 1 empirical baselines/reproductions (Ch. 5), and References & Appendices (A–C).
- **Delta from Proposal Defense:** `docs/proposal.md` contained only two figures (Sem 1 & Sem 2 Gantt charts). The Semester 1 Progress Report expands to a full engineering inventory of architecture diagrams, WBS/Gantt charts, pinned model/metric tables, empirical WP-2/WP-3 plots, and reproducible appendices.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Cross-Chapter Inventory | Source Blueprint / Artifact | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| Ch. 1 Tables & Figures | `deliverables/s1_progress_report/01_introduction_and_timeline.md` | `[READY]` / `[OPEN_DECISION]` | WBS table (`[NEEDS_VERIFICATION]` board sync), Updated Gantt charts, Team Workload (`[OPEN_DECISION]`), Risk Matrix, Deliverables table. |
| Ch. 2 Tables & Figures | `deliverables/s1_progress_report/02_related_works.md`, `docs/literature_survey/` (20 reviews) | `[READY]` | 3-Axis taxonomy table, 11-technique comparative matrix, benchmark comparison table. |
| Ch. 3 Tables & Figures | `deliverables/s1_progress_report/03_methodology_and_design.md`, `docs/MODELS.md`, `docs/METRICS_AND_AXES.md`, `docs/WORKFLOW_SPEC.md` | `[READY]` | Two-Tier repo diagram, 9-model pinned registry & footprint table, 2-Pillar quality framework diagram, dataset manifest summary. |
| Ch. 4 Tables & Figures | `deliverables/s1_progress_report/04_algorithms_and_implementation.md` | `[READY]` / `[OPEN_DECISION]` | Dual-backend harness diagram, 11-method algorithmic summary (`[READY]` / `[DRAFT/PR#21]` audit notes), WP-4 policy state-machine direction (`[OPEN_DECISION]`). |
| Ch. 5 Empirical Tables & Charts | `deliverables/s1_progress_report/05_preliminary_results_and_followup.md`, `results/baselines/`, `results/runs/wp3/`, `results/grades/` | `[READY]` / `[DRAFT/PR#21]` | Raw JSONs & Fig. 5.1 / 5.3 inputs are `[READY]` on `main`; PR #21 overview charts & `analysis/figures/wp3/*` are `[DRAFT/PR#21]`. |
| References & Appendices (A–C) | `deliverables/s1_progress_report/06_references_and_appendices.md` | `[READY]` | Complete bibliography inventory (`[NEEDS_VERIFICATION]` BibTeX/DOIs), Appendix A (pseudocode/configs), Appendix B (manifest/JSON schemas), Appendix C (`Table C.1`–`Table C.3`). |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Master Planned List of Tables (Cross-Chapter Spec)
| Table ID | Proposed Caption | Chapter / Appendix | Readiness |
| :--- | :--- | :---: | :---: |
| `Table 1.1` | Proposal Defense vs. Semester 1 Scope & Methodology Evolution | Ch. 1 | `[READY]` |
| `Table 1.2` | Work Breakdown Structure (`WP-1`–`WP-9`) and Live GitHub Project #9 Status | Ch. 1 | `[NEEDS_VERIFICATION]` / `[OPEN_DECISION]` |
| `Table 1.3` | Team Workload Distribution & Ownership Matrix | Ch. 1 | `[OPEN_DECISION]` |
| `Table 1.4` | Semester 1 & Semester 2 Technical and Schedule Risk Matrix | Ch. 1 | `[READY]` / `[DRAFT/PR#21]` |
| `Table 1.5` | Project Deliverables Schedule (Semester 1 Actuals vs. Semester 2 Targets) | Ch. 1 | `[READY]` / `[NEEDS_VERIFICATION]` / `[OPEN_DECISION]` |
| `Table 2.x` | Literature Taxonomy (Foundations, 11 Techniques across 3 Axes, 6 Benchmarks) | Ch. 2 | `[READY]` |
| `Table 3.x` | Pinned Base Model Registry (`docs/MODELS.md`), VRAM Headroom Matrix, Metrics-to-Axis Mapping, Dataset Manifest | Ch. 3 | `[READY]` |
| `Table 4.1` | Metric-to-Axis Correspondence & Mathematical Definitions | Ch. 4 | `[READY]` |
| `Table 4.2` | 11 Candidate Technique Formulations & Parameterizations (`kv_bench`) | Ch. 4 | `[READY]` / `[NEEDS_VERIFICATION]` |
| `Table 4.3` | Semester 1 Experimental Environment & Sweep Configuration | Ch. 4 | `[READY]` |
| `Table 5.x` | Testbed Hardware/Software Specs, WP-2 Native vLLM Baseline Results (`[READY]`), WP-3 11-Method Comparative Matrix (`[READY]` raw / `[DRAFT/PR#21]` synthesis) | Ch. 5 | `[READY]` / `[DRAFT/PR#21]` |
| `Table C.1` | Pinned BF16 Model Suite & Theoretical KV Footprint (`docs/MODELS.md`) | Appendix C | `[READY]` / `[NEEDS_VERIFICATION]` |
| `Table C.2` | Verified WP-3 Candidate Hyperparameters (All 11 Techniques) | Appendix C | `[READY]` |
| `Table C.3` | Semester 2 Factorized Ablation Matrix (`WP-8`) | Appendix C | `[PENDING_S2]` / `[OPEN_DECISION]` |

#### Master Planned List of Figures (Cross-Chapter Spec)
| Figure ID | Proposed Caption | Chapter | Readiness |
| :--- | :--- | :---: | :---: |
| `Figure 1.1` | Work Package Dependency Graph (`WP-1`–`WP-9`) | Ch. 1 | `[READY]` |
| `Figure 1.2` | Updated Semester 1 Gantt Chart (Phase 1–3: Aug–Dec 2026, Plan vs. Actual) | Ch. 1 | `[READY]` / `[NEEDS_VERIFICATION]` |
| `Figure 1.3` | Semester 2 Gantt Chart (Phase 4–6: Jan–May 2027, Projected) | Ch. 1 | `[PENDING_S2]` |
| `Figure 2.x` | 3-Axis Co-Optimization Gap & Technique Positioning Map | Ch. 2 | `[READY]` |
| `Figure 3.x` | Two-Tier Multi-Repo Workflow (`kv_co-op_hub` + `kv_bench`) & Two-Pillar Quality Evaluation Pipeline | Ch. 3 | `[READY]` |
| `Figure 4.x` | Dual-Backend Harness Architecture (`[READY]`) & Candidate WP-4 Tier Transition State Machine (`[OPEN_DECISION]`) | Ch. 4 | `[READY]` / `[OPEN_DECISION]` |
| `Figure 5.1` | Native vLLM Context Scaling & Multi-User Concurrency Bottleneck Curves (`results/baselines/` on `main`) | Ch. 5 | `[READY]` |
| `Figure 5.2` | 32K Pareto Frontier Scatter — Peak VRAM vs. LongMemEval Semantic Accuracy (`analysis/figures/wp3/overview/pareto_vram_vs_quality.png`) | Ch. 5 | `[DRAFT/PR#21]` |
| `Figure 5.3` | Stacked Bar Chart of LongMemEval Causal Attribution Verdicts across 11 WP-3 Techniques (`results/grades/*_judged.json` on `main`) | Ch. 5 | `[READY]` |

#### Open Questions / Decisions Needed
1. **`[OPEN_DECISION]` Typesetting Toolchain:** Will the final Semester 1 Progress Report be compiled in **LaTeX** (Overleaf / KMUTT thesis class) or **Microsoft Word / Google Docs**? (Determines whether ToC/LoT/LoF are auto-generated and whether Mermaid diagrams are exported to SVG/PDF or redrawn in TikZ/draw.io.)

---

## Section FM-4: List of Symbols & List of Technical Vocabulary / Abbreviations

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Pages `viii`–`ix` of `Progress_Sem1_ENG_Example2.pdf`:
  - **List of Symbols (`viii`):** Alphabetical/logical glossary of mathematical variables, operators, and units used in memory footprint equations, quality divergence formulas, and latency models.
  - **List of Technical Vocabulary and Abbreviations (`ix`):** Alphabetical table expanding domain acronyms (hardware, systems, ML, and evaluation terminology).
- **Committee Expectations:** Every symbol appearing in equations in Ch. 1–5 must be defined here with consistent, collision-free notation (e.g., reserving $L$ strictly for layer count and $\mathbf{z}^{(t)}$ for logit vectors).

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Standardize mathematical notation and systems terminology across all chapters so examiners can verify KV-cache memory math (`docs/MODELS.md`), logit divergence equations (`docs/METRICS_AND_AXES.md`), and hardware profiling metrics without ambiguity.
- **Delta from Proposal Defense:**
  - `docs/proposal.md` (§2.1) used informal prose notation (`num_layers`, `num_heads`, `dim_head`, `precision_in_bytes`) and omitted GQA full-attention layer distinctions.
  - Current verified notation in `docs/MODELS.md` introduces $L_{\text{full\_attn}}$, $H_{KV}$, and $D_{\text{head}}$ (paired with $H_Q$ from `docs/literature_survey/foundations/illusion_of_equivalence.md`) to accurately model Pure Dense GQA vs. Architectural Hybrid / Sliding-Window models (`Qwen3.5-4B` with GDN layers, `Gemma-4-E4B-it`).
  - Added formal Pillar 1 quality symbols ($\mathbf{z}_{\text{baseline}}^{(t)}$, $\mathbf{z}_{\text{structural}}^{(t)}$, $\text{KL}(P_{\text{baseline}} \parallel P_{\text{structural}})$, $\mathcal{V}$, $\mathbb{I}(\cdot)$, $Q_{\text{err}}$) from `docs/METRICS_AND_AXES.md` (resolving the raw $L^{(t)}$ vs. layer-count $L$ symbol collision).

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Symbol / Term Group | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| KV-Cache Memory Symbols ($L, L_{\text{full\_attn}}, H_{KV}, D_{\text{head}}, \text{Memory}_{\text{KV}}$) | `docs/MODELS.md` (Lines 28–50) | `[READY]` | Exact GQA BF16 formula: $4 \times L_{\text{full\_attn}} \times H_{KV} \times D_{\text{head}}$ bytes/token. |
| Query Head Count Symbol ($H_Q$) | `docs/literature_survey/foundations/illusion_of_equivalence.md` | `[READY]` | Matches `Attn Heads (Q / KV)` column in `docs/MODELS.md`. |
| Sequence & Batch Symbols ($S_{\text{context}}, B_{\text{batch}}, N$) | `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` (Slide 02, Line 57), `results/manifests/run_manifest_baseline_16gb.json` (Line 57) | `[READY]` | $N \in \{1, 4, 8, 12, 16\}$ concurrent users in `concurrent_baseline_16gb.json`. |
| Quality & Divergence Symbols ($\mathbf{z}_{\text{baseline}}^{(t)}, \mathbf{z}_{\text{structural}}^{(t)}, \text{KL}, P_{\text{baseline}}, P_{\text{structural}}, \mathcal{V}, T, Q_{\text{err}}, \Delta\text{PPL}$) | `docs/METRICS_AND_AXES.md` (Lines 45–97) | `[READY]` | Pillar 1 equations on Lines 76–77 (renamed from $L^{(t)}$ to $\mathbf{z}^{(t)}$ to avoid collision with layer count $L$); lossless property $Q_{\text{err}} = 0$ on Lines 45–46. |
| Latency Overlap & Priority Formula Symbols ($T_{\text{Compute}}, T_{\text{PCIe Transfer}}, \alpha, \beta, W$) | `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` (Lines 178–201) | `[NEEDS_VERIFICATION]` | Proposal-stage priority/overlap notation; retain in List of Symbols only if used in Ch. 4 WP-4 design. |
| Abbreviations & System Terms | `docs/MODELS.md`, `docs/METRICS_AND_AXES.md`, `docs/WORKFLOW_SPEC.md`, `WORK_PACKAGES.md` | `[READY]` | Distinguishes consumer GDDR6/GDDR6X VRAM (RTX 4060/4090) from datacenter HBM. |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Table Spec: List of Symbols
| Symbol | Meaning / Definition | Unit / Domain | Primary Source |
| :--- | :--- | :--- | :--- |
| $L$, $L_{\text{full\_attn}}$ | Total Transformer layers / number of full-attention layers holding KV-cache | Count ($\mathbb{Z}^+$) | `docs/MODELS.md` |
| $H_Q$, $H_{KV}$ | Number of Query attention heads / Key-Value attention heads (GQA) | Count ($\mathbb{Z}^+$) | `illusion_of_equivalence.md`, `docs/MODELS.md` |
| $D_{\text{head}}$ | Dimensionality per attention head | Elements ($\mathbb{Z}^+$) | `docs/MODELS.md` |
| $S_{\text{context}}$ | Input + generated sequence (context) length | Tokens | `Slide_Content_and_Executive_Summary.md` (Slide 02) |
| $B_{\text{batch}}$, $N$ | Batch size / number of concurrent user sessions | Sessions | `Slide_Content_and_Executive_Summary.md` (Slide 02), `run_manifest_baseline_16gb.json` |
| $\text{Memory}_{\text{KV}}$ | Total KV-cache VRAM footprint | Bytes / KiB / GiB | `Slide_Content_and_Executive_Summary.md` (Slide 02), `docs/MODELS.md` |
| $T$ | Total number of generated decode steps evaluated for logit fidelity | Steps ($\mathbb{Z}^+$) | `docs/METRICS_AND_AXES.md` |
| $\mathcal{V}$ | Model vocabulary set | Token IDs | `docs/METRICS_AND_AXES.md` |
| $\mathbf{z}_{\text{baseline}}^{(t)}, \mathbf{z}_{\text{structural}}^{(t)}$ | Output logit vectors at decode step $t$ (uncompressed BF16 vs. structural method; avoids collision with layer count $L$) | $\mathbb{R}^{\|\mathcal{V}\|}$ | `docs/METRICS_AND_AXES.md` |
| $P_{\text{baseline}}, P_{\text{structural}}$ | Softmax next-token probability distributions (baseline vs. structural method) | $[0, 1]^{\|\mathcal{V}\|}$ | `docs/METRICS_AND_AXES.md` |
| $\text{KL}(P_{\text{baseline}} \parallel P_{\text{structural}})$ | Forward Kullback–Leibler divergence between baseline and candidate distributions | Nats | `docs/METRICS_AND_AXES.md` |
| $\mathbb{I}(\cdot)$ | Indicator function ($1$ if condition holds, $0$ otherwise) | $\{0, 1\}$ | `docs/METRICS_AND_AXES.md` |
| $Q_{\text{err}}$ | Numerical quality distortion error ($Q_{\text{err}} = 0$ for lossless spatial/temporal ops) | Dimensionless | `docs/METRICS_AND_AXES.md` |
| $\Delta\text{PPL}$ | Change in token-level Perplexity vs. uncompressed baseline *(supplementary only)* | Perplexity delta | `docs/METRICS_AND_AXES.md` |
| $\alpha, \beta$ | Proposal-stage weighting coefficients for instantaneous attention vs. persistence in Priority Engine *(prune if unused in Ch. 4)* | $\mathbb{R}^+$ | `Slide_Content_and_Executive_Summary.md` (Slide 09) |
| $W$ | Recent sliding-window token count preserved in full precision *(prune if unused in Ch. 4)* | Tokens ($\mathbb{Z}^+$) | `Slide_Content_and_Executive_Summary.md` (Slide 09) |
| $T_{\text{Compute}}, T_{\text{PCIe Transfer}}$ | GPU decode compute duration vs. host-to-device PCIe transfer duration | Milliseconds (ms) | `Slide_Content_and_Executive_Summary.md` (Slide 10) |

#### Table Spec: List of Technical Vocabulary & Abbreviations
| Abbreviation / Term | Full Expansion & Project Context |
| :--- | :--- |
| **BF16 / FP16** | Brain Floating Point 16-bit / IEEE Half-Precision 16-bit (pinned uncompressed base model formats in `docs/MODELS.md`) |
| **CPU** | Central Processing Unit (host processor handling DRAM offloading, scheduling, and optional CPU dequantization) |
| **CUDA** | Compute Unified Device Architecture (NVIDIA GPU parallel computing platform and dual-stream execution API) |
| **DMA** | Direct Memory Access (asynchronous host-to-device transfers over PCIe) |
| **DRAM** | Dynamic Random-Access Memory (host CPU system memory tier) |
| **EM / F1** | Exact Match / Token-Level F1 Score (lexical QA scoring diagnostics contrasted with semantic grading) |
| **FP8 / INT4** | 8-bit Floating Point / 4-bit Integer (KV-cache quantization formats under the `STRUCTURAL` axis) |
| **GDN** | Gated DeltaNet (linear-attention recurrent layers in hybrid `Qwen3.5` models) |
| **GPU** | Graphics Processing Unit (target consumer accelerators: RTX 4090 capped at 16 GB; local RTX 4060 8 GB) |
| **GQA** | Grouped-Query Attention (sharing KV heads across multiple Query heads) |
| **HBM** | High-Bandwidth Memory (stacked datacenter GPU memory, e.g., A100/H100; contrasted with consumer GDDR6/GDDR6X VRAM) |
| **HPC / Slurm** | High-Performance Computing / Simple Linux Utility for Resource Management (KMUTT CPE GPU cluster) |
| **KL-Divergence** | Kullback–Leibler Divergence (Primary Pillar 1 micro-level numerical fidelity metric) |
| **KV-Cache** | Key-Value Cache (stored past attention projections avoiding $O(S^2)$ decode recomputation) |
| **LLM** | Large Language Model |
| **LongMemEval** | Long-Term Memory Evaluation benchmark (Primary Pillar 2 macro-level adherence across 5 capabilities) |
| **MHA** | Multi-Head Attention (standard Transformer attention with $H_{KV} = H_Q$, contrasted with GQA) |
| **NIAH** | Needle-In-A-Haystack (diagnostic positional retrieval test; opt-in 2D NIAH in WP-3) |
| **OOM** | Out-Of-Memory (CUDA allocation failure when VRAM exceeds physical/capped budget) |
| **P95 / P99** | 95th / 99th Percentile tail metric (used for TTFT, TPOT, and KL-Divergence tail monitoring) |
| **PCIe** | Peripheral Component Interconnect Express (CPU–GPU interconnect bus) |
| **PPL** | Perplexity (supplementary/optional language-modeling diagnostic; deprioritized as primary gate) |
| **RSS** | Resident Set Size (process host DRAM footprint, tracked via `host_rss_delta_mb`) |
| **RULER** | Comprehensive synthetic long-context diagnostic benchmark suite (referenced in `WORK_PACKAGES.md` WP-7 & `docs/literature_survey/benchmarks/ruler.md`) |
| **SDPA** | Scaled Dot-Product Attention (PyTorch native white-box backend used in WP-3) |
| **SHA256** | Secure Hash Algorithm 256-bit (checksum enforced by Hub CI for dataset and result provenance) |
| **SME** | Small and Medium Enterprise (target multi-user deployment persona: 10–15 concurrent users) |
| **SOTA** | State-of-the-Art (published KV-cache management baselines surveyed and reproduced in WP-3) |
| **SWA** | Sliding-Window Attention (architectural hybrid pattern, e.g., `Gemma-4-E4B-it`) |
| **TPOT** | Time Per Output Token (decode inter-token latency, in ms) |
| **TTFT** | Time To First Token (prefill + queueing latency, in seconds) |
| **vLLM** | High-throughput LLM serving engine with PagedAttention (production baseline backend in `kv_bench`) |
| **VRAM (GDDR6 / GDDR6X)** | Video Random-Access Memory on consumer GPUs (e.g., GDDR6 on RTX 4060 8 GB, GDDR6X on RTX 4090 capped at 16 GB) |
| **WBS / WP** | Work Breakdown Structure / Work Package (`WP-1` through `WP-9`) |

#### Open Questions / Decisions Needed
1. **`[NEEDS_VERIFICATION]` Notation Pruning:** After `04_algorithms_and_implementation.md` and `05_preliminary_results_and_followup.md` are finalized, prune any unused symbols (e.g., $\alpha, \beta, W$ if WP-4 adopts a different policy formulation) and add any method-specific symbols actually appearing in Ch. 4 equations.
