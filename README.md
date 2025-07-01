# CHiME-9 Task 2 - ECHI Baseline

### Enhancing Conversation to address Hearing Impairment

This repository contains the baseline system for the CHiME-9 challenge, Task 2 ECHI. For detailed information on how to participate in the challenge and for obtaining the datasets, please refer to the [challenge website](https://www.chimechallenge.org/current/task2/index)

## About the Challenge

**CHiME-9 Task 2: Enhancing Conversations for Hearing Impairment (ECHI)** addresses the task of trying to separate conversations between known participants from noisy cafeteria-like backgrounds with low latency processing. Solutions to this problem would benefit huge numbers of people with mild hearing loss.

![concept](images/echi_concept.png)

### Key Challenge Features

-- **Natural conversation** each of over 30 minutes with up to four persons; 49 sessions and 196 accent-diverse speakers.
-- **High degree of overlap** both within the conversation and with distractor talkers.
-- **Noisy background** simulating cafeteria-like environments with varying loudness.
-- **Hearing aid audio** with 4-channel input worn by a participant; **Aria glasses audio** with 7-channel input.
-- **Headtracking** for all conversation participants.

### Evaluation Metrics

Systems are evaluated on a suite of metrics and listening tests,

1. **Reference based** - For estimating both quality and intelligibility
2. **Reference free** - DNN-based intelligibility and quality metric predictors
3. **Listening tests** - Submitted systems will be subject to speech quality and intelligibility listing tests.

### Challenge documentation

For further information about the challenge, please refer to:

- **[Challenge Overview](https://www.chimechallenge.org/current/task2/index)** - Complete challenge description and scenario details
- **[Data Description](<https://www.chimechallenge.org/current/task2/data>** - Dataset structure, formats, and download instructions  
- **[Baseline System](https://www.chimechallenge.org/current/task2/baseline)** - Architecture and components of the baseline system
- **[Challenge Rules](dhttps://www.chimechallenge.org/current/task2/rules)** - Participation rules, allowed datasets, and evaluation details

## Getting Started

### Sections

1. <a href="#install">Installing the software</a>
2. <a href="#data">Installing the dataset</a>
3. <a href="#stages">Stages</a>
4. <a href="#troubleshooting">Troubleshooting</a>

## <a id="#install">1. Installing the software</a>

Clone this repository from GitHub

```bash
git clone git@github.com:CHiME9-ECHI/CHiME9-ECHI.git
cd CHiME9-ECHI
```

The installation of the necessary tools is detailed in `install.sh`.
We recommend following it step-by-step and adjusting for your system if needed.
The script will build a conda environment called `echi_recipe` and install the
dependencies listed in `environment.yaml`.

When running the system, remember to activate the conda environment and set the
necessary environment variables,

```bash
conda activate echi_recipe
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

To make the `PYTHONPATH` setting persistent across terminal sessions, you can add
 the `export` command to your shell's configuration file (e.g., `~/.bashrc` for
 bash or `~/.zshrc` for zsh).

## <a id="data"> 2. Installing the data </a>

Our dataset is hosted on HuggingFace. To download the dataset, you must first
request access at the
[CHiME9-ECHI Dataset page](https://huggingface.co/datasets/CHiME9-ECHI/CHiME9-ECHI).
Full details of the dataset can be found at that link. Once you have been
granted access, you can download the dataset using the following commands:

```bash
# Navigate to your baseline project directory if you aren't already there
# cd CHiME9-ECHI

# Create a data directory (if it doesn't exist)
mkdir -p data
cd data

# Export your Hugging Face token (replace 'your_actual_token_here'
# with your actual token)
export HF_TOKEN=your_actual_token_here

# Download the development set
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/CHiME9-ECHI/CHiME9-ECHI/resolve/main/data/chime9_echi.dev.v1_0.tar.gz

# Untar the downloaded files
tar -zxovf chime9_echi.dev.v1_0.tar.gz
```

The baseline system expects the dataset to be placed in the `data/chime9_echi`
 directory by default. The paths to various subsets of the data (e.g., training,
 development, evaluation) are defined in the `config/paths.yaml` file. If you
 choose to place the data in a different location, you will need to update
 `config/paths.yaml` accordingly. However, adhering to the default directory
 structure (`data/chime9_echi`) is recommended for ease of use.

Note that `data/chime9_echi` can also be a symbolic link to another location,
allowing flexibility in where the dataset is physically stored.

## <a id="stages">3. Stages</a>

This repository is set up to handle all phases of training, enhancement and evaluation.
 Each of these has its own pipeline, which will prepare the data and perform the
 intended task. All participants are free to modify the **train** and
 **enhancement** code to obtain the best results possible.

- **Train:** Prepares speech segments of the dataset and then trains using them.
 Details can be found on the [training page](docs/training.md).
- **Enhancement:** Given a system, the enhancement pipeline produces
 audio for each full session and saves it with the correct formatting. Default
 options for passthrough and the baseline are provided, but custom options
 can be added. Details can be found on the
 [enhancement page](docs/enhancement.md).
- **Evaluation:** Given a directory containing all the enhanced files, this
 script computes all the specified metrics over all sessions. Details can be
 found on the [evaluation page](docs/evaluation.md).

> **⚠️ WARNING:**
> **Evaluation code should be considered read-only.**
> Any modifications to the evaluation scripts could lead to invalid results.
> If there are any problems which cannot be resolved without editing the code,
> please raise an issue and we will respond accordingly.

## <a id="troubleshooting">4. Troubleshooting</a>

If you encounter issues, here are some common troubleshooting steps:

- **Activate Conda Environment:** Ensure your Conda environment (`echi_recipe`) is
 activated:

  ```bash
  conda activate echi_recipe
  ```

- **Check PYTHONPATH:** Verify that your `PYTHONPATH` environment variable is correctly
 set to include the `src` directory of this project:

  ```bash
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  # (or ensure this is in your .bashrc or equivalent shell startup script)
  echo $PYTHONPATH
  ```

- **Verify Data Paths:** Double-check that the dataset paths in `config/paths.yaml`
 match the actual location of your CHiME-9 ECHi data. The default expected location
 is `data/chime9_echi`.
- **Hydra Log Files:** For detailed error messages and execution logs, inspect the
 Hydra log files. These are typically found in the `exp/<experiment_name>/hydra/`
 directory (e.g., `exp/main/hydra/`). The exact path will be printed at the start
 of a run.
- **Common Python Issues:** Check for common Python package installation problems
 or version conflicts within the Conda environment. Sometimes, reinstalling a
 problematic package can help.
- **Other Issues:** If you have any other problems with this repository that
cannot be fixed, please raise a GitHub Issue and we will do our best to resolve
it for you.
