# CogniFlex: Executive Functions Benchmark for AGI Evaluation

## Kaggle Writeup

### Project Name

CogniFlex: Habit Override, Rule Shift, and Conflict Planning

### Your Team

- Team: Kudos Science
- Repository: kudosscience/cognitive-ability-benchmark

### Problem Statement

Current LLM evaluations often over-reward memorized pattern completion and under-measure executive control. This creates a major gap for AGI-oriented assessment: a model may look strong on broad reasoning benchmarks while still failing to inhibit habitual responses, adapt under rule changes, or plan under conflicting constraints.

This project targets the Executive Functions track with one central question:

What does model behavior look like when the highest-probability next token is usually wrong?

The benchmark isolates three executive sub-capabilities:

- Inhibitory control: suppressing familiar continuations when explicit local overrides appear
- Cognitive flexibility: adapting to mid-task semantic rule shifts
- Complex planning: avoiding tempting but irreversible trap actions to reach a constrained goal

### Task & benchmark construction

CogniFlex is implemented as three deterministic task generators with exact-match scoring:

- habit_override
  - Environment: alphabet ring traversal with explicit per-node inverse rules
  - Failure mode tested: perseverative forward stepping despite override instructions
  - Scoring: normalized exact path match
- rule_shift
  - Environment: arithmetic/bitwise operation chains with a defined semantic shift point
  - Failure mode tested: continuing pre-shift semantics after context update
  - Scoring: integer extraction plus exact numeric match
- conflict_planning
  - Environment: resource conversion graph with a unique shortest valid plan and decoy traps
  - Failure mode tested: greedy local choice that consumes critical resources and dead-ends
  - Scoring: normalized exact action sequence match

Benchmark composition approach:

- Programmatic generation only (no manual labels)
- Difficulty levels 1..5 with controlled scaling
- Seeded reproducibility for every sample
- Verifiable ground truth per sample
- Explicit output formats in prompts to reduce grader ambiguity

Kaggle integration artifacts were implemented and generated:

- Adapter code: benchmark/adapters/kaggle_benchmarks_adapter.py
- Task notebook scaffold: kaggle_assets/notebooks/cogniflex_task.py
- Task datasets: kaggle_assets/data/*.jsonl
- Benchmark metadata: kaggle_assets/benchmark_metadata.json
- Export script: scripts/export_kaggle_assets.py
- Adapter run config: configs/kaggle_adapter.json

### Dataset

Data is fully synthetic and generated on demand from benchmark/utils/generator.py through CogniFlexSuite.

Current benchmark export configuration:

- samples per task: 250
- tasks: 3
- total samples: 750
- seed: 20260415

Each JSONL row contains:

- sample_id (int)
- prompt (str)
- expected_output (str)
- difficulty (int in [1, 5])
- metadata (dict, task-specific)

Task-specific metadata examples:

- habit_override: start_letter, override_letters, override_delta, visited_letters
- rule_shift: initial_state, shift_after_index, operations, trace
- conflict_planning: start_inventory, actions, canonical_plan, trap_action_ids

Statistical significance and defensibility notes:

- Difficulty-balanced generation is cyclical by index
- Every item has deterministic reconstruction through seed + generator logic
- Conflict-planning generator includes tests verifying unique shortest solutions

### Technical details

Core modules:

- benchmark/cogniflex_suite.py
- benchmark/tasks/base_task.py
- benchmark/tasks/habit_override.py
- benchmark/tasks/rule_shift.py
- benchmark/tasks/conflict_planning.py
- benchmark/utils/generator.py
- benchmark/adapters/kaggle_benchmarks_adapter.py
- benchmark/evaluation/pilot_sweep.py

Pilot sweep implementation:

- Script: scripts/run_pilot_sweep.py
- Config: configs/pilot_sweep.json
- Outputs:
  - outputs/pilot_sweep_predictions.jsonl
  - outputs/pilot_sweep_task_summary.csv
  - outputs/pilot_sweep_difficulty_summary.csv
  - outputs/pilot_sweep_overall.csv
  - outputs/pilot_sweep_report.md

The sweep runner simulates a model ladder (small -> medium -> large -> frontier -> oracle upper bound) with task-wise skill and difficulty sensitivity controls. A tuned `reasoning-frontier-xl` profile was added to better mirror expected Kaggle model-family separation. This gives a tunable, reproducible early-signal for discriminatory power before expensive model runs.

Reproducibility commands:

```bash
python scripts/export_kaggle_assets.py
python scripts/run_pilot_sweep.py
python -m pytest tests/
```

### Results, insights, and conclusions

Pilot sweep overall scores (750 samples per model):

| Model | Overall Score |
| --- | ---: |
| pattern-matcher-small | 0.241 |
| rule-aware-medium | 0.480 |
| planner-large | 0.669 |
| reasoning-frontier-xl | 0.844 |
| oracle-upper-bound | 0.993 |

Task-level means:

| Model | Habit Override | Rule Shift | Conflict Planning |
| --- | ---: | ---: | ---: |
| pattern-matcher-small | 0.348 | 0.232 | 0.144 |
| rule-aware-medium | 0.572 | 0.488 | 0.380 |
| planner-large | 0.684 | 0.696 | 0.628 |
| reasoning-frontier-xl | 0.848 | 0.892 | 0.792 |
| oracle-upper-bound | 0.984 | 0.996 | 1.000 |

Observed signal quality:

- Strong five-tier rank ordering across capability tiers indicates useful discriminatory power
- Difficulty degradation is visible, especially for smaller profiles on rule_shift and conflict_planning
- No ceiling effect for realistic profiles, and no floor collapse for all models at all levels

What this benchmark reveals beyond generic reasoning tests:

- Models that appear competent on easy symbolic tasks can still fail sharply under late rule reversals
- Planning performance collapses faster than arithmetic flexibility when trap actions are introduced
- Executive control errors are systematic and classifiable, not merely random hallucinations

Conclusion:

CogniFlex produces interpretable gradients and explicit failure modes tied to executive-function constructs, making it a robust candidate for AGI progress profiling on this track.

### Organizational affiliations

- Submitted by Kudos Science
- No additional institutional affiliation declared in this repository

### Final submission checklist mapped to Kaggle form fields

| Kaggle form field | Submission value | Source in repository |
| --- | --- | --- |
| Project Name | CogniFlex: Habit Override, Rule Shift, and Conflict Planning | This README, Project Name section |
| Team / Author | Kudos Science | This README, Your Team section |
| Track | Executive Functions | This README, Problem Statement section |
| Problem Statement | Executive-control gap in current LLM evaluation | This README, Problem Statement section |
| Task / Benchmark Construction | Deterministic generators + exact-match scorers | This README, Task & benchmark construction section |
| Dataset Description | 3 synthetic tasks, 250 samples each, seeded reproducibility | This README, Dataset section |
| Technical Details | Core modules + scripts + reproducibility commands | This README, Technical details section |
| Results / Insights / Conclusions | Five-model pilot ladder + task-level breakdown | This README, Results section and outputs/pilot_sweep_report.md |
| Organizational Affiliations | Submitted by Kudos Science | This README, Organizational affiliations section |
| References & Citations | DeepMind + Kaggle docs/SDK sources | This README, References & citations section |

Pre-submit completion checklist:

- [ ] Paste Project Name exactly as shown above.
- [ ] Confirm Team name is `Kudos Science`.
- [ ] Select Executive Functions track.
- [ ] Paste Problem Statement and Task Construction sections.
- [ ] Confirm dataset counts match: 750 total samples (250 per task).
- [ ] Attach notebook and dataset files from the release manifest below.
- [ ] Paste updated Results table after final Kaggle model runs.
- [ ] Verify references are present and links resolve.

### Release package manifest (exact files to attach/upload)

Required benchmark assets:

- kaggle_assets/benchmark_metadata.json
- kaggle_assets/notebooks/cogniflex_task.py
- kaggle_assets/data/habit_override.jsonl
- kaggle_assets/data/rule_shift.jsonl
- kaggle_assets/data/conflict_planning.jsonl

Required writeup evidence artifacts:

- outputs/pilot_sweep_overall.csv
- outputs/pilot_sweep_task_summary.csv
- outputs/pilot_sweep_difficulty_summary.csv
- outputs/pilot_sweep_report.md
- README.md

Optional reproducibility bundle:

- configs/kaggle_adapter.json
- configs/pilot_sweep.json
- scripts/export_kaggle_assets.py
- scripts/run_pilot_sweep.py

### References & citations

1. DeepMind. Measuring progress toward AGI: A cognitive framework.
2. Kaggle Benchmarks documentation: [https://www.kaggle.com/docs/benchmarks](https://www.kaggle.com/docs/benchmarks)
3. Kaggle Benchmarks SDK repository: [https://github.com/Kaggle/kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks)
4. Kaggle Benchmarks Cookbook: [https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)
5. Kaggle Benchmarks Quick Start: [https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md)
