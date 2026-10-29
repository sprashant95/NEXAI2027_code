# Code for "When to Verify: A Checkpoint-Interval Law for Agentic Workflows with Imperfect Verifiers"

Everything in the paper's tables and figures is produced by the scripts here. Seeds are fixed in the scripts.

Requirements: Python 3.11+, numpy, pandas, scipy, matplotlib, pillow.

Order of execution (run from this folder):

1. `python abstract_sim.py` : validates Eqs. (3)-(4) against event-level simulation (24 settings) and runs the 756-setting grid for Eq. (7). Writes `results/abstract_A1A2.json`.
2. `python abstract_sim2.py` : unequal error rates (dynamic program), bursty failures, and the online controller. Writes `results/abstract_A3A4A5.json`.
3. `python calibrate.py` : calibrates per-kind failure rates, verifier recall, false-alarm rate and cost on 60 calibration pipelines. Writes `results/calibration.json`.
4. `python run_testbed.py` : end-to-end runs (79,200 workflow executions, roughly 15 minutes on two cores). Resumable: finished configurations are cached in `results/parts/`. Writes `results/testbed_runs.csv`.
5. `python analysis2.py` : bootstrap intervals, paired ratios, model predictions. Writes `results/summary.json`.
6. `python figs1.py` and `python figs2.py` : figures (600 dpi PNG) into `figs/`.

File guide:
- `model.py` : the analytic model (Eqs. 1-10): acceptance probability, expected cost, success probability, closed-form interval, overhead, dynamic program.
- `testbed.py` : executable pipelines, six kinds of silent failure, five verifiers (T1-T5), and the run harness.
- `results/` : raw outputs used in the paper.

Notes for readers: the executor in the testbed is a fault injector, not a language model. Verifier T5 is an emulated judge (label noise on top of T3, assumed cost of one step unit). Costs for T1-T4 are measured wall-clock ratios (see `results/calibration.json`).
