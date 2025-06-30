# Evaluation

The baseline system can be run using

```bash
python run.py
```

This is equivalent to running the following steps

```bash
python -m scripts.setup
python -m scripts.enhance
python -m scripts.validate
python -m scripts.prepare
python -m scripts.evaluate
python -m scripts.report
```

Results will appear in the reports directory defined in `config/paths.yaml`. Results
are reported at three levels:

- The device level, `report.dev.<device>._._.json` - i.e. accumulated over all
 sessions.
- The session level, `report.dev.<device>.<session>._.json` - i.e. for a specific
 session and given device.
- The participant level, `report.dev.<device>.<session>.<PID>.json` - i.e. for a
 specific participant within a session for a given device.

For the `dev` set there will be 2, 24 (2 devices x 12 session) and 72 (2 devices
 x 12 session x 3 participants) of these files respectively.

The reports are stored as a dictionary with an entry for each metric. Each metric,
in turn, is presented as a dictionary storing the `mean`, `standard deviation`,
`standard error`, `min value`, `max value`, and the `number of segments`.

For each `json` file there will also be a similarly named `csv` file containing
the metric data on which the statistics were computed.

## <a id="configuration">4. Configuring the baseline</a>

The system uses [Hydra](https://hydra.cc/) for configuration management.
 This allows for a flexible and hierarchical way to manage settings.

The main configuration files are located in the `config` directory:

- `main.yaml`: Main configuration, imports other specific configurations.
- `shared.yaml`: Shared parameters used across different scripts (e.g., dataset paths,
general settings).
- `setup.yaml`: Configuration for the data setup stage (`scripts/setup.py`).
- `enhance.yaml`: Configuration for the enhancement stage (`scripts/enhance.py`).
- `validate.yaml`: Configuration for the validate stage (`scripts/validate.py`).
- `prepare.yaml`: Configuration for the preparationstage (`scripts/prepare.py`).
- `evaluate.yaml`: Configuration for the evaluation stage (`scripts/evaluate.py`).
- `report.yaml`: Configuration for the reporting stage (`scripts/report.py`).
- `metrics.yaml`: Configuration for the metrics used in evaluation.
- `paths.yaml`: Defines paths for data, models, and outputs.

You can override any configuration parameter from the command line.

For `run.py`, which executes the entire pipeline:

```bash
# Example: Run with a specific dataset configuration and disable GPU usage
# for enhancement
python run.py shared.dataset=my_custom_dataset enhance.use_gpu=false
```

For individual scripts like `scripts/evaluate.py`:

```bash
# Example: Evaluate a specific submission directory
python scripts/evaluate.py evaluate.submission=<submission_dir>

# Example: Evaluate with specific test data
python scripts/evaluate.py evaluate.submission=data/submission
```

Key configurable parameters include:

- **Dataset:** `shared.dataset` allows you to specify different dataset configurations.
- **Device Settings:** Parameters like `enhance.use_gpu` (true/false) and
 `enhance.device` (e.g., 'cuda:0', 'cpu') control hardware usage.
- **Evaluation:**
  - `evaluate.submission`: Path to the enhanced audio or transcriptions to be evaluated.
  - `evaluate.n_batches`, `evaluate.batch`: Control parallel processing during
 evaluation by splitting the data into batches.
