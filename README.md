# Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification

This is the code of the paper [Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification](https://proceedings.mlr.press/v162/garcin22a/garcin22a.pdf) published at ICML2022.
If you use this code for your work, please cite the paper:
```
@inproceedings{pmlr-v162-garcin22a,
  title = 	 {Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification},
  author =       {Garcin, Camille and Servajean, Maximilien and Joly, Alexis and Salmon, Joseph},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {7208--7222},
  year = 	 {2022},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
}
```

## The pytopk package

The pytopk package contains the code for the balanced and imbalanced top-k losses as well as a differentiable top-k function.

### Installation
It can be installed as follows:

```console
pip install pytopk
```

### Top-k losses

Our losses can be used as standard pytorch loss functions:

```python
import torch
from pytopk import BalNoisedTopK, ImbalNoisedTopK

scores = torch.tensor([[2.0, 1.5, -3.0],
                       [7.5, 4.0, -1.5]])
labels = torch.tensor([0, 2])

k = 2

criteria_bal = BalNoisedTopK(k=k, epsilon=1.0)
criteria_imbal = ImbalNoisedTopK(k=k, epsilon=0.01, max_m=0.3, cls_num_list=[17, 23, 55])

loss_batch_bal = criteria_bal(scores, labels)
loss_batch_imbal = criteria_imbal(scores, labels)
```

### Smooth top-k function

We also provide a differentiable top-k function for tensors of any size that can be plugged into any neural network architecture:

```python
import torch
from pytopk import NoisedTopK

smooth_topk = NoisedTopK(k=3, dim=-1)
x = torch.tensor([[-1.5, 2.0, 0.7, 3.8],
                  [-1.1, -5.4, 0.1, 2.3]], requires_grad=True)
out = smooth_topk(x)
print(out)

>> tensor([ 0.4823, -1.4710], grad_fn=<_NoisedTopKBackward>)

out.sum().backward()
print(x.grad)

>> tensor([[0.0000, 0.4000, 0.6000, 0.0000],
          [0.8000, 0.0000, 0.2000, 0.0000]])
```

## Experiments


For running the experiments, you can download the required packages running:

```console
pip install -r requirements.txt
``` 


For running the experiments on Pl@ntNet-300K, you first need to download the dataset [here](https://zenodo.org/record/5645731#.YfMvN-rMJPY)
and then place it as *plantnet* inside a folder named *data*.

Command lines that were run for obtaining the results can be found in the *cmd_lines.txt* file.
