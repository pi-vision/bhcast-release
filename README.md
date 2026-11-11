# BHCast v3 audit code

This branch contains the audit scripts with configurable input paths. The frozen
release is the inference implementation under audit.

```bash
git clone https://github.com/pi-vision/bhcast-release.git source/release
git -C source/release checkout --detach 7cb2f93423f49fc4edd05cad97f9e643c7202e53
curl -fL https://github.com/pi-vision/bhcast-release/releases/download/v1.0.1/model.pt \
    -o source/release/checkpoints/model.pt
./run.sh released.py
```

The eight public inputs are frame 0 of each `source/release/images/sgra_spin*.npy`.
The replay writes the summary to `results/released.json` and the 256-step
`step_change` and `flux` arrays to eight `results/released_*.npz` files.
Compare both arrays per case against the recorded outputs before rescoring.

Set `BHCAST_AUDIT_PYTHON` to select the Python environment and
`BHCAST_CHECKPOINT` to use an existing checkpoint file. The original replay
used Torch 2.7.0+cu126, GPU FP32, and disabled cuDNN TF32. Run the paper replay
on the data host with `BHCAST_MOVIES` and `BHCAST_CLASSIFIER` pointing to the
local inputs. The released-frame script can also use `BHCAST_MOVIES` for source
matching and `BHCAST_TRAINING_MANIFEST` for checkpoint provenance.

- `./run.sh paper.py 20 --historical` runs the historical split with the frozen release.
- `./run.sh paper.py 20 --historical --batch-stats` switches just BatchNorm for the diagnostic.
- `./run.sh ablations.py --tag gpu baseline clamp roundtrip batch tf32 fp64` runs the public-frame ablations.
- `./run.sh compare.py REFERENCE_JSON PAPER_CACHE` compares local outputs with supplied reference inputs. The JSON maps each sample name to three motion ratios followed by three flux ratios. Use a trusted historical cache.

The saved audit output fixture and Illinois v3 movies are stored separately.
