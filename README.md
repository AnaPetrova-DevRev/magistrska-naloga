# Magistrska naloga — thesis code

Code for cleaning sync-unit records, forecasting event rates and time to the next
event, training reinforcement-learning sync policies, and evaluating their cost
and data freshness. The repository includes implementation, configurations,
tests, and invented fixtures. Real datasets, trained models, and historical
experiment results are not included.

## Install

Use Python 3.14. The clean installation was verified with Python 3.14.2 on macOS
ARM64; other platforms and GPU execution have not been verified. The following
commands use a bash/zsh shell. Run them from the repository root unless stated
otherwise.

```sh
git clone https://github.com/AnaPetrova-DevRev/magistrska-naloga.git
cd magistrska-naloga
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The base requirements cover statistical and recurrent forecasting, RL, and the
dashboards. For NHiTS, TFT, and Chronos adapters, also install:

```sh
python -m pip install -r requirements-forecasting.txt
python -m pip check
```

The forecasting file includes the base requirements. Dependency versions are
ranges rather than a lockfile. Pretrained Chronos weights require a separate
download when first used; pretrained inference was not part of the clean
verification. If pip reports no matching distribution, inspect earlier messages
for DNS, network, or package-index errors before changing Python versions.

## Run the synthetic pipeline

With the virtual environment active, create a separate temporary workspace:

```sh
repo_root="$PWD"
demo_root="$(mktemp -d)"
python scripts/create_synthetic_fixture.py --root "$demo_root" --run-pipeline
```

Expect `Completed 6 cleanup and preprocessing actions`. The command creates two
invented input chunks, a sync-unit registry, cleaned records, time deltas, and
minute event counts under `$demo_root/data`. It targets only
`test_org/test_system/test_project`, refuses to overwrite an existing `data/`
directory, and does not train models or policies.

The six recipes are `configs/synthetic_*.json`. To inspect or rerun one action
against this workspace, keep the same shell and use absolute script/config paths:

```sh
(
  cd "$demo_root"
  python "$repo_root/config_processor.py" "$repo_root/configs/synthetic_event_counts.json" --dry-run
  python "$repo_root/config_processor.py" "$repo_root/configs/synthetic_event_counts.json"
)
```

For your own workflows, `config_processor.py CONFIG.json` dispatches the action,
parameters, and sync units defined in a JSON config; `--dry-run` previews the
selection. It reads `data/sync_units_register.json` relative to the working
directory. Model training also expects `model_configs/` relative to that
directory, so run those workflows from the repository root after preparing your
own inputs. Historical experiment configs require the omitted datasets and, in
some cases, saved model IDs; they are not ready-to-run synthetic examples.

## Open the dashboards

To explore the synthetic data in the main GUI, use the variables above:

```sh
(
  cd "$demo_root"
  python -m streamlit run "$repo_root/gui_app.py"
)
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
Select the synthetic sync unit to inspect its data. Config selection and model
training need the repository's `configs/` and `model_configs/` in the working
directory; the temporary workspace above is intended for data exploration.

From the repository root, start the policy evaluation dashboard separately:

```sh
python -m streamlit run sync_policy_dashboard.py --server.port 8502
```

This dashboard searches the repository's `data/` directory, regardless of the
working directory. It needs artifacts produced by `evaluate_sync_policy` under
`data/<organization>/<external_system>/<project>/rl_policies/`. A fresh clone
shows `No evaluation artifacts found`; the cleanup quickstart does not populate
policy charts. Stop either server with Ctrl+C.

## Run tests and small training checks

Run tests from a disposable checkout without your own data: some tests create
and remove shared synthetic paths under `data/test_org/test_system/test_project`.

```sh
CUDA_VISIBLE_DEVICES="" python -m unittest tests.test_synthetic_pipeline tests.test_rl_integration -v
python -m unittest discover -s tests -p "test_*.py" -v
```

The focused command exercises synthetic preprocessing, a small ARIMA
fit/predict/save/load cycle, tiny PPO/DQN/A2C training, and policy evaluation.
The default suite leaves heavier neural tests gated. With forecasting extras
installed, run this bounded CPU subset explicitly:

```sh
RUN_HEAVY_TESTS=1 FORCE_CPU_TRAINING=1 python -m unittest \
  tests.test_rnn_model \
  tests.test_foundation_models.TestNHiTSModel \
  tests.test_foundation_models.TestTFTModel \
  tests.test_foundation_models.TestGroundTruthContextPrediction -v
```

Other heavy tests can download pretrained weights or train larger ensembles.
Running these checks validates the synthetic workflows. Reproducing the thesis
measurements requires the omitted source data and a complete experiment setup;
inspecting historical results requires the saved experiment artifacts.

## Code layout

- `data_cleanup/`, `data_preprocessing/`: parsing, cleaning, and derived datasets.
- `modeling/`, `model_training/`, `model_serving/`, `model_lifecycle/`: forecasting,
  persistence, inference, and model management.
- `rl_training/`, `sync_policy_evaluation/`: sync environments, policies, and evaluation.
- `configs/`, `model_configs/`: action recipes and model templates.
- `analysis/`, `dataset_comparison/`: scientific analysis and dataset comparisons.
- `tests/`: unittest coverage and synthetic fixtures.

See [LICENSE](LICENSE) for the repository license.
