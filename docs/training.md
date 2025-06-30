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

When training using this style of segmentation, you
