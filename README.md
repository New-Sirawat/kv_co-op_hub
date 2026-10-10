# Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs

> **Central Hub Repository** for Senior Engineering Project (Academic Year 2568, B.Eng. Computer Engineering, KMUTT).

* **Project Board**: [GitHub Project #9: KV-Cache Co-Optimization](https://github.com/users/khemingkapat/projects/9)
* **Master Proposal**: [docs/proposal.md](docs/proposal.md)
* **Roadmap & Work Packages**: [WORK_PACKAGES.md](WORK_PACKAGES.md)

---

## 🎯 Project Overview

Large Language Models (LLMs) suffer from linearly growing Key-Value (KV) cache memory footprints during autoregressive decoding. On consumer-grade GPUs (8–16 GB VRAM), long-context inference (up to 32K context) quickly hits the "memory wall" or suffers severe latency penalties over standard PCIe connections.

This project addresses the **co-optimization gap** by jointly coordinating three dimensions of KV-cache optimization under strict hardware budgets:
1. **Structural Axis**: Selective compression and quantization (FP16 high-priority vs. 4-bit low-priority).
2. **Spatial Axis**: Tiered CPU host memory offloading and selective recall.
3. **Temporal Axis**: Asynchronous transfer scheduling and compute-copy overlapping to prevent GPU decode stalls.

---

## 🗂️ Repository Ecosystem

```mermaid
flowchart TD
    Hub["<b>kv_co-op_hub (This Repo)</b><br/>• Central Proposal & Specs<br/>• Work Package Tracking<br/>• Golden Results & Manifests<br/>• Datasets & Seeds<br/>• Academic Deliverables"]

    GHProj["<b>GitHub Project #9</b><br/>• Unified Project Board<br/>• Work Packages WP 1–9"]
    GHProj --- Hub

    Bench["<b>kv_bench</b><br/>• Benchmark Runner & Harness<br/>• Slurm HPC & Local Scripts<br/>• Profiling & Baselines"]
    Engine["<b>kv-engine (Upcoming)</b><br/>• Tiered Cache Manager<br/>• vLLM Policy Engine Integration<br/>• Custom CUDA / CPU Kernels"]

    Bench -->|Publishes results & manifests| Hub
    Engine -->|Referenced by commit SHA| Hub
    Hub -->|Standardized workloads & configs| Bench

    GHProj -.->|Sub-tasks & PRs| Bench
    GHProj -.->|Sub-tasks & PRs| Engine
```

### Connected Repositories
* **[kv_bench](https://github.com/khemingkapat/llm_bench)**: Unified benchmarking harness for profiling baselines and evaluating co-optimization techniques on consumer GPUs.
* **kv-engine** *(Planned)*: The modular policy layer, tiered KV-cache manager, and runtime integration (e.g. vLLM).

---

## 📁 Repository Structure

```
.
├── WORK_PACKAGES.md           # Master roadmap and deliverable definitions
├── docs/                      # Core documentation & background
│   ├── proposal.md            # Approved senior project proposal document
│   ├── architecture/          # 3-Axis state machine & pipeline specifications
│   ├── literature_survey/     # Paper summaries and comparative matrices
│   └── meetings/              # Advisor meeting logs & action items
├── workloads/                 # Unified evaluation data & trace definitions
│   ├── download_datasets.sh   # Deterministic downloader for LongMemEval, PG-19
│   ├── synthetic_traces/      # Simulated multi-turn SME user chat profiles (10–15 users)
│   └── seeds.json             # Pinned random seeds & evaluation splits
├── results/                   # Golden benchmark results & provenance manifests
│   ├── baselines/             # Verified reference data (vLLM, FlexGen, StreamingLLM)
│   ├── runs/                  # Structured evaluation outputs by milestone
│   └── manifests/             # Run manifests linking commit SHAs, GPU specs, and results
├── analysis/                  # Data analysis & chart generation
│   ├── notebooks/             # Jupyter notebooks for data aggregation & Pareto curves
│   └── figures/               # Publication-grade SVG/PDF figures for reports
└── deliverables/              # Academic deliverables & presentations
    ├── s1_proposal_defense/   # Proposal document, slide deck, CITI certification
    ├── s1_progress_report/    # Semester 1 progress report & presentation
    └── s2_thesis/             # Senior project thesis LaTeX source & final defense deck
```

---

## 🗓️ Milestones & Work Packages

| WP | Title | Semester / Phase | Status |
|---|---|---|---|
| **WP 1** | Testbed Setup & Baseline Reproduction | Sem 1 / Phase 1: Environment & Baselines 🛠️ | 🔴 Backlog |
| **WP 2** | Hardware & System Bottleneck Profiling | Sem 1 / Phase 1: Environment & Baselines 🛠️ | 🔴 Backlog |
| **WP 3** | Multi-Axis Strategy Exploration & Candidate Benchmarking | Sem 1 / Phase 2: Exploration & Co-Design 🔬 | 🔴 Backlog |
| **WP 4** | 3-Axis Co-Optimization Policy Design & Prototype Architecture | Sem 1 / Phase 2: Exploration & Co-Design 🔬 | 🔴 Backlog |
| **WP 5** | Semester 1 Milestones, Sanity Checks & Progress Defense | Sem 1 / Phase 3: S1 Milestones & Defense 📋 | 🔴 Backlog |
| **WP 6** | Runtime Engine Integration & Output Correctness Verification | Sem 2 / Phase 4: Runtime Integration ⚙️ | 🔴 Backlog |
| **WP 7** | End-to-End Stress & Long-Context Quality Benchmarking | Sem 2 / Phase 5: Comprehensive Evaluation 📊 | 🔴 Backlog |
| **WP 8** | Ablation Studies & Performance Bottleneck Tuning | Sem 2 / Phase 5: Comprehensive Evaluation 📊 | 🔴 Backlog |
| **WP 9** | Reproducibility Artifacts, Thesis Documentation & Final Defense | Sem 2 / Phase 6: Synthesis & Thesis 🎓 | 🔴 Backlog |

For detailed breakdown and deliverables, see [WORK_PACKAGES.md](WORK_PACKAGES.md).
