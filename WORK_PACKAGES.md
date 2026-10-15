# WP-KV-BENCH: Co-optimization of KV-Cache Management for LLM Inference on Consumer GPUs

## Project Goal
> Jointly co-optimize KV-cache management across structural (compression/quantization), spatial (CPU offloading), and temporal (transfer scheduling/pipelining) dimensions to enable long-context (up to 32K) LLM inference on 8–16 GB consumer GPUs during the decode phase without offsetting transfer or compute overhead.

---

## Work Package Overview

| WP | Name | Semester / Phase | Status |
|---|---|---|---|
| 1 | Testbed Setup & Native Reference Baseline | Sem 1 / Phase 1: Environment & Baselines 🛠️ | 🔴 Not started |
| 2 | Hardware & System Bottleneck Profiling | Sem 1 / Phase 1: Environment & Baselines 🛠️ | 🔴 Not started |
| 3 | Literature Candidate Exploration & Micro-Benchmarking | Sem 1 / Phase 2: Exploration & Co-Design 🔬 | 🔴 Not started |
| 4 | 3-Axis Co-Optimization Policy Design & Prototype Architecture | Sem 1 / Phase 2: Exploration & Co-Design 🔬 | 🔴 Not started |
| 5 | Semester 1 Milestones, Sanity Checks & Progress Defense | Sem 1 / Phase 3: S1 Milestones & Defense 📋 | 🔴 Not started |
| 6 | Runtime Engine Integration & Output Correctness Verification | Sem 2 / Phase 4: Runtime Integration ⚙️ | 🔴 Not started |
| 7 | End-to-End Stress & Long-Context Quality Benchmarking | Sem 2 / Phase 5: Comprehensive Evaluation 📊 | 🔴 Not started |
| 8 | Ablation Studies & Performance Bottleneck Tuning | Sem 2 / Phase 5: Comprehensive Evaluation 📊 | 🔴 Not started |
| 9 | Reproducibility Artifacts, Thesis Documentation & Final Defense | Sem 2 / Phase 6: Synthesis & Thesis 🎓 | 🔴 Not started |

---

## Semester 1: Research, Profiling & Prototype Co-Design

### Phase 1: Environment & Baselines 🛠️

#### WP-1: Testbed Setup & Native Reference Baseline
Establish stable consumer hardware environments, dataset ingestion, and execution infrastructure to reproduce initial native vLLM uncompressed inference behavior.
- Cluster testbed configuration & verification (Slurm HPC with 16GB VRAM allocation caps on RTX 4090 node).
- Model selection & version pinning for 16GB consumer memory budget (unmodified FP16/BF16 weights).
- Evaluation dataset ingestion and staging (LongMemEval multi-turn chat, PG-19 long context, WikiText-2 reference).
- Unified metrics harness instrumentation (Hardware VRAM/TTFT/TPOT/PCIe + Quality logit reference and LongMemEval adherence).
- Empirical uncompressed reference baseline run (standard native vLLM FP16/BF16).
- Local testbed adaptation (RTX 4060 8GB, 16GB RAM, standard PCIe with downscaled 3B model profile).

#### WP-2: Hardware & System Bottleneck Profiling
Push native vLLM to the limit across scaling context lengths and concurrent workloads on consumer hardware to map exact saturation points.
- VRAM Out-Of-Memory (OOM) ceiling mapping across context lengths (512 to 32K).
- Time-to-First-Token (TTFT) and Time-Per-Output-Token (TPOT) tail latency profiling (P95/P99 under 10–15 concurrent users).
- Host-to-device PCIe bandwidth and saturation measurements during decode phase.
- Baseline empirical profiling report and target benchmark curves.

---

### Phase 2: Exploration & Co-Design 🔬

#### WP-3: Literature Candidate Exploration & Micro-Benchmarking
Survey and micro-benchmark candidate algorithms from published literature organized by research topic against consumer-grade hardware constraints.
- Eviction / Retention: Evaluate priority scoring and retention mechanisms (e.g., StreamingLLM attention sinks, H2O heavy-hitters, sliding window).
- Compression & Quantization: Evaluate candidate compression codecs (e.g., FP8, INT4, QServe, HeadQ-style residuals) and profile CPU dequantization latency.
- Offloading & Tiering: Evaluate data movement models and offload granularities (e.g., FlexGen uniform swapping, InfiniGen prefetching, ShadowKV value-only offloading).
- Comparative trade-off matrix synthesizing candidates against PCIe and compute constraints.

#### WP-4: 3-Axis Co-Optimization Policy Design & Prototype Architecture
Design and implement the unified policy coordination layer that manages cache entries across multiple memory and representation tiers.
- State-machine design for KV-cache tier transitions (GPU Active $\rightarrow$ GPU Compressed $\rightarrow$ Host RAM Offload).
- Asynchronous transfer scheduling pipeline overlapping host-to-device PCIe copy with GPU decode execution.
- Prototype KV-cache manager integration and initial functional smoke tests.

---

### Phase 3: S1 Milestones & Defense 📋

#### WP-5: Semester 1 Milestones, Sanity Checks & Progress Defense
Consolidate Semester 1 experimental findings, complete institutional milestones, and present progress.
- Completion of CITI Program Ethics Exam and formal project proposal documentation.
- Onsite Senior Project Proposal Defense.
- Prototype sanity validation on the hardware testbed.
- Semester 1 Progress Report and oral presentation.

---

## Semester 2: System Integration, Evaluation & Final Synthesis

### Phase 4: Runtime Integration ⚙️

#### WP-6: Runtime Engine Integration & Output Correctness Verification
Integrate the co-optimized policy layer into a production-grade inference engine and verify output fidelity.
- Deep integration with open-source runtime engine (e.g., vLLM PagedAttention architecture).
- Numerical output correctness verification: Mean and P99 KL-divergence against uncompressed FP16.
- Top-1 token agreement and perplexity delta verification under varying context lengths.
- Memory defragmentation and multi-session allocation checks.

---

### Phase 5: Comprehensive Evaluation 📊

#### WP-7: End-to-End Stress & Long-Context Quality Benchmarking
Perform full-spectrum system and model-quality evaluation under scaled and realistic operational workloads.
- Long-context scaling benchmarks up to 32K context window.
- Multi-user concurrency simulation: 10–15 concurrent SME users with bursty conversation histories.
- Tail latency evaluation (P95/P99 TTFT and TPOT).
- Task quality evaluation using LongMemEval across 5 core conversational capabilities.
- Diagnostic positional retrieval evaluation using RULER suite for anomaly analysis.

#### WP-8: Ablation Studies & Performance Bottleneck Tuning
Isolate contributions of individual axes and optimize critical path kernels.
- Isolation of performance gains: priority scoring vs. offloading vs. compression.
- Trade-off analysis: CPU delegation overhead vs. PCIe transfer savings.
- Kernel-level optimizations and stream pipelining refinements based on profiling data.

---

### Phase 6: Synthesis & Thesis 🎓

#### WP-9: Reproducibility Artifacts, Thesis Documentation & Final Defense
Produce final scientific deliverables, open-source reproducibility artifacts, and academic documentation.
- Publication-quality visualization charts and Pareto frontier comparison curves.
- Consolidated reproducibility repository: automated benchmarking scripts, configuration manifests, and datasets.
- Final Senior Project Thesis document.
- Final slide deck, rehearsal, and Senior Project Defense presentation.

---

## Dependency Graph

```mermaid
flowchart TD
    subgraph S1 ["Semester 1: Research, Profiling & Architecture Co-Design"]
        WP1["WP 1: Testbed Setup & Native Reference Baseline"] --> WP2["WP 2: Hardware & Bottleneck Profiling"]
        WP1 --> WP3["WP 3: Literature Candidate Exploration"]
        WP2 --> WP3
        WP3 --> WP4["WP 4: 3-Axis Co-Optimization Policy Design"]
        WP2 --> WP5["WP 5: S1 Milestones & Progress Defense"]
        WP4 --> WP5
    end

    subgraph S2 ["Semester 2: System Integration, Evaluation & Final Synthesis"]
        WP5 --> WP6["WP 6: Runtime Engine Integration & Correctness"]
        WP6 --> WP7["WP 7: End-to-End Stress & Quality Benchmarking"]
        WP6 --> WP8["WP 8: Ablation Studies & Performance Tuning"]
        WP7 --> WP9["WP 9: Artifacts, Thesis & Final Defense"]
        WP8 --> WP9
    end
```
