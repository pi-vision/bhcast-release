# Trace the release artifacts and replay the eight public inputs.
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "source/release"
CHECKPOINT = Path(os.environ.get("BHCAST_CHECKPOINT", RELEASE / "checkpoints/model.pt"))
MOVIES = Path(os.environ.get("BHCAST_MOVIES", ROOT / "movies"))
sys.path.insert(0, str(RELEASE))
from bhcast_pipeline import GRMHDInference  # noqa: E402

BINS = [(1, 4), (5, 16), (17, 64), (65, 256)]
FLUX_STEPS = [16, 60, 256]


def load_model(device, checkpoint=CHECKPOINT):
    model = GRMHDInference(str(checkpoint), str(RELEASE / "checkpoints/data_processor_sgra.pth"),
                           device=device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    print("Strict load:", model.model.load_state_dict(state, strict=True))
    return model


def change(frames):
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2), dtype=np.float64)


def replay(device):
    print("Torch", torch.__version__, "device", device)
    model = load_model(device)
    if manifest_path := os.environ.get("BHCAST_TRAINING_MANIFEST"):
        manifest = torch.load(manifest_path, weights_only=True, map_location="cpu")
        print("Training manifest", manifest)
    print("Normalizers", vars(model.data_processor.in_normalizer),
          vars(model.data_processor.out_normalizer))
    rows = {}
    for path in tqdm(sorted((RELEASE / "images").glob("sgra_spin*.npy")), desc="Released samples"):
        frames = np.load(path)
        assert frames.shape == (3, 100, 100) and frames.dtype == np.float32
        name = path.stem.removeprefix("sgra_spin")
        spin, inc = name.split("_inc")
        sources = {}
        for suffix in ("", "_blurred"):
            movie_path = MOVIES / f"Ma{spin}_Rh10_i{inc}_w4{suffix}.pt"
            if not movie_path.is_file():
                continue
            movie = torch.load(movie_path, map_location="cpu", weights_only=True).numpy().squeeze()
            sources[movie_path.name] = [
                np.flatnonzero(np.all(movie == frame, axis=(1, 2))).tolist() for frame in frames
            ]
        has_anchor = any(any(i + 1 in b and i + 2 in c for i in a)
                         for a, b, c in sources.values())
        pred = model.predict(frames[0], horizon=256)
        assert pred.shape == (256, 100, 100) and np.isfinite(pred).all()
        assert np.array_equal(pred[:60], model.predict(frames[0], horizon=60))
        step_change = change(np.concatenate((frames[:1], pred)))
        flux = pred.sum(axis=(1, 2), dtype=np.float64) / frames[0].sum(dtype=np.float64)
        rows[name] = {
            "sources": sources,
            "change_relative": [step_change[a - 1:b].mean() / step_change[:4].mean()
                                for a, b in BINS],
            "flux_relative": flux[np.asarray(FLUX_STEPS) - 1].tolist(),
            "forecast_change_1_2": step_change[:2].tolist(),
            "truth_change_1_2": change(frames).tolist() if has_anchor else None,
        }
        np.savez(ROOT / "results" / f"released_{name}.npz", step_change=step_change, flux=flux)
    (ROOT / "results/released.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    replay(args.device)
