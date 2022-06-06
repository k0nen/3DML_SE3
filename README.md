# 3DML project: SE(3)-Transformers

## Use of existing code

We partially modified the [NVIDIA's implementation](https://github.com/NVIDIA/DeepLearningExamples/blob/master/DGLPyTorch/DrugDiscovery/SE3Transformer/se3_transformer/model/basis.py) of basis pre-computation into `dataloader/basis.py` of our code, as it supports JIT compile support for faster computation.

## Requirements

```
torch
tensorboard
dgl-cu110  # or whatever CUDA version of your choice
e3nn
tqdm
easydict
```

### Preparation

- N-body: Make sure you have two files `data/nbody_train.pkl` and `data/nbody_test.pkl`, which can be downloaded from [here](https://drive.google.com/drive/folders/1OrVxlTjdkjMpduZ3XnjdsbNWkp5DFpZ7?usp=sharing)
- QM9: The dataset is automatically downloaded upon running `preprocess_qm9.py`.

## List of improvements

### Fused Key/Value computation

- In the attention module, Q is graph node feature; K and V are graph edge feature.
    - Fuse K and V projection operations, then slice them to obtain K and V.
    - This implementation halves the number of small CUDA kernels launched for K, V computation in total.
    - Radial functions are also fused, so the number of total parameters slightly decrease.

### Basis pre-computation

Since the input graphs do not change (no augmentation, no iterative position refinement),
we can precompute the basis of all graph before training. Those basis files can be re-used across multiple experiment runs on the same dataset.

- The precomputed basis for N-body training dataset is about 1.3GB.
- The precomputed basis for the whole QM9 dataset is around 60GB.

## How to train your own model

You must pre-compute the basis of dataset, then train the model. 
The best model is saved in `data/nbody_best.pt`. 
The arguments for training are listed in `train_nbody.py`, and the default values are provided in the comment of each argument.

```
python preprocess_nbody.py
python train_nbody.py
```

For QM9 experiments, substitute `nbody` to `qm9` in the paragraph above.

### N-body experiments

The train/test splits were generated using code from the author's repository [here](https://github.com/FabianFuchsML/se3-transformer-public/tree/master/experiments/nbody/data_generation). We report the results averaged over five runs.

|       | MSE x                | MSE v             |
| ----- | -------------------- | ----------------- |
| Paper | 0.0076 $\pm$ 0.0002  | 0.075 $\pm$ 0.001 |
| Ours  | 0.0042 $\pm$ 0.00008 | 0.071 $\pm$ 0.001 |

### QM9 experiments

The dataset is divided into train/val/test sets with size 100k, 18k, and 12,831 samples respectively.

|       | $\alpha$ (bohr^3) | $\Delta \epsilon$ (meV) | $\epsilon_{HOMO}$ (meV) | $\epsilon_{LUMO}$ (meV) | $\mu$ (D) | $C_\nu$ (cal/mol K) |
| ----- | ----------------- | ----------------------- | ----------------------- | ----------------------- | --------- | ------------------- |
| Paper | 0.142             | 53.0                    | 35.0                    | 33.0                    | 0.051     | 0.054               |
| Ours  | 0.3478            | 98.23                   | 60.97                   | 57.7                    | 0.1161    | 0.147               |


### Pretrained models
Pretrained models are available via Google Drive.
- [N-body](https://drive.google.com/drive/folders/1OrVxlTjdkjMpduZ3XnjdsbNWkp5DFpZ7?usp=sharing)
- [QM9](https://drive.google.com/drive/folders/1jHuqq64NPesx69v7a39-O2EE6Ev3mT5m?usp=sharing)

## References

- [Authors' official code](https://github.com/FabianFuchsML/se3-transformer-public)
- [NVIDIA's implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer)
