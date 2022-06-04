# 3DML project: SE(3)-Transformers

### TODO

- Add more content to README
- Add reference links to README
- Add QM9 dataset results

## Requirements

### Libraries

This list is manually written; please update this section if you need more libraries!
- torch
- tensorboard
- dgl-cu110 (Or whatever CUDA version you use)
- e3nn
- tqdm
- easydict

### Preparation

- N-body: Make sure you have two files `data/nbody_train.pkl` and `data/nbody_test.pkl`.
- QM9: The dataset is automatically downloaded upon running `preprocess_qm9.py`.

## List of improvements

### Fused Key/Value computation

- Q is graph node feature; K and V are graph edge feature.
    - Fuse K and V projection operations, then slice them to obtain K and V.
    - This implementation halves the number of small CUDA kernels launched for K, V computation in total.
    - Radial functions are also fused, so the number of total parameters slightly decrease.

### Basis pre-computation

Since the input graphs do not change (no augmentation, no iterative position refinement),
we can precompute the basis of all graph before training.

- The precomputed basis for N-body training dataset is about 1.3GB.
- The precomputed basis for the whole QM9 dataset is around 60GB.

## How to train your own model

You must pre-compute the basis of dataset, then train the model.  
The best model is saved in `data/nbody_best.pt`. 

```
python preprocess_nbody.py
python train_nbody.py
```

The arguments for training are provided in `train_nbody.py`, and the default values are provided in the comment of each argument.

### N-body experiments

The train/test splits were generated using code from the author's repository.

Best test loss: 0.03814424
- MSE_x:  0.00410201
- MSE_v:  0.07218647
