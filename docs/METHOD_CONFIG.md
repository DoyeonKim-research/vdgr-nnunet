# Method configuration

| Setting | Reported value |
|---|---:|
| Backbone | nnU-Net v2 3D full resolution |
| Epochs | 100 |
| Patch size | 128 x 128 x 128 |
| Batch size | 2 |
| Distance-stratum percentiles | 33.3333, 66.6667 |
| Distance classes | background, deep interior, intermediate, boundary proximal |
| Auxiliary class weights | 0.1, 1.0, 1.5, 2.0 |
| Background in auxiliary CE | ignored |
| `lambda_dist` | 0.10 |
| Gate | `clip(P_B + alpha P_I, 0, 1)` |
| `alpha` | 0.25 |
| Context kernels | 5 x 5 x 5 and 9 x 9 x 9 |
| Checkpoint retention | every 5 epochs |

The binary segmentation objective is the standard nnU-Net deep-supervision
loss. The auxiliary cross-entropy is applied to the highest-resolution distance
head output, matching the experiment code. The leading `0.1` entry in the
four-class weight vector is retained for class-index compatibility but has no
effect because background voxels are assigned the loss ignore index.
