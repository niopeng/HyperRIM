# Dataloader

- use opencv (`cv2`) to read and process images.

- read from **image** files OR from **.lmdb** for fast IO speed.
    - How to create .lmdb file? Please see [`create_lmdb.py`](https://github.com/niopeng/HyperRIM/blob/main/code/data/create_lmdb.py).

## Contents

- `LR_dataset`: only reads LR images in test phase where there is no target images.
- `LRHR_dataset`: reads LR and HR pairs from image folder or lmdb files. If only HR images are provided, downsample the images on-the-fly.
- `LRHR_four_levels_dataset`: similar to LRHR_dataset, include intermediate targets

## Data Augmentation

We use random crop, random flip/rotation, (random scale) for data augmentation. 
