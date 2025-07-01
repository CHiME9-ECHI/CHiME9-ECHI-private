# Training a Model

The script to train a model from scratch can be run using

```bash
python run_train.py
```

This is equivalent to running the following steps

```bash
python -m scripts.train.unpack
python -m scripts.train.train
```

This will unpack the data downloaded and then use them to trian a model.

## Unpack

Given your downloaded dataset, stored in `paths.echi_dir`, the unpacking script
 will prepare the audio for training. Given a `model_sample_rate` and `device`,
 this script will:

- Resample the `device` audio and the corresponding refrences to
`model_sample_rate` and then segment it all in to just the speech segments,
- Resample the rainbow passages to `model_sample_rate`.

The outputs of this stage will be saved into `paths.working dir` under
`train_segments` for the device and reference audio, and `participant` for the
rainbow passages by default. Parameters for this stage are found in
`config.train.unpack.yaml`.

If you want to use the same data segmentation for each training run, this
stage will only need to be run once. If you want to modify the unpacking script,
you should specify a new `paths.working_dir` to avoid overwriting any data
that's already being used.

This stage can be run on CPU or GPU.

## Training

The training loop will create a dataloader which loads the data above, and
then defines a model to train with the given training parameters. There are
multiple configs associated with this stage, stored in `configs.train`:

- `dataloading`: Stores the paths of the audio files to load, and dataloading
parameters.
- `train`: The parameters/details for the training cycle, including epochs,
learning rate, loss functions, etc.
- `model`: Stores parameters for how to prepare the audio for input to the
model and the parameters for the model architecture.
- `wandb`: Information for logging runs to
[Weights and Biases](https://wandb.ai/site/models/). Setting values to `null`
will stop any logging here.

This stage should only be run on GPU.
