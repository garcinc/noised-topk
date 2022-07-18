## Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification

This is the code of the paper [Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification](https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf) published at ICML2022.

The code of our newly introduced losses can be found in losses/noised_losses.py.

For running the experiments, pytorch, torchvision and tqdm are needed.

You can create a conda environment containing these modules by running `conda env create -f environment.yml`.

For running the experiment on Pl@ntNet-300K, you can download the dataset [here](https://zenodo.org/record/5645731#.YfMvN-rMJPY)

Command lines that were run for obtaining the results can be found in the *cmd_lines* file.
