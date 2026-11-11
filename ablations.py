# Change one inference factor per rollout on the eight public inputs.
import argparse
import json

import h5py
import numpy as np
import torch
from tqdm import tqdm

from released import BINS, FLUX_STEPS, MOVIES, RELEASE, ROOT, change, load_model
from paper import INCS, SPINS


def fit_extrema(normalizer):
    # Historical training uses 799 pair starts, from 0 through 798 inclusive.
    extrema = [[], []]
    for spin in SPINS:
        for inc in INCS:
            path = MOVIES / f"Ma{spin:+g}_Rh10_i{inc}_w4.pt"
            movie = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
            for offset in (0, 1):
                x = movie[offset:offset + 799]
                extrema[offset].append((x.min(), x.max()))
    values = torch.tensor([(min(lo for lo, _ in group), max(hi for _, hi in group))
                           for group in extrema])
    return (values + normalizer.log_offset).log().tolist()


@torch.inference_mode()
def rollout(model, frames, variant, batch):
    inputs = model.data_processor.in_normalizer
    outputs = model.data_processor.out_normalizer
    dtype = next(model.model.parameters()).dtype
    predictions = []
    for group in np.array_split(frames, 1 if batch else len(frames)):
        x = inputs.transform(torch.as_tensor(group[:, None], dtype=dtype, device=model.device))
        steps = []
        for _ in range(256):
            y = model.model(x)
            if variant == "clamp":
                y = y.clamp(0, 1)
            physical = outputs.inverse_transform(y)
            x = inputs.transform(physical) if variant == "roundtrip" else y
            steps.append(physical[:, 0].cpu().numpy())
        predictions.append(np.stack(steps, axis=1))
    return np.concatenate(predictions)


def measure(start, forecast):
    delta = change(np.concatenate((start[None], forecast)))
    flux = forecast.sum(axis=(1, 2), dtype=np.float64) / start.sum(dtype=np.float64)
    return [delta[a - 1:b].mean() / delta[:4].mean() for a, b in BINS] + [
        flux[s - 1] for s in FLUX_STEPS]


def raw_metadata():
    output = {}
    for window in ("w4", "w5"):
        root = MOVIES / "Ma+0.5/Rh10/i10" / window
        paths = sorted(root.glob("img*.h5"))
        with h5py.File(paths[0]) as a:
            movie = torch.load(MOVIES / f"Ma+0.5_Rh10_i10_{window}.pt", weights_only=True)
            raw = a["unpol"][:].astype(np.float32)
            output[window] = {"first": paths[0].name, "last": paths[-1].name, "n": len(paths),
                              "shape": raw.shape, "keys": list(a),
                              "tensor_equals_unpol": bool(np.array_equal(movie[0].numpy().squeeze(), raw))}
    full = MOVIES.parent / "full_grmhd/Ma+0.94/Rh10/i10/w4"
    with h5py.File(full / "img04000.h5") as a, h5py.File(full / "img04001.h5") as b:
        output["full_w4"] = {
            "time": float(a["header/t"][()]),
            "cadence": float(b["header/t"][()] - a["header/t"][()]),
            "fov_uas": float(a["header/camera/fovx_dsource"][()]),
            "scale": float(a["header/scale"][()]),
        }
    return output


def run(args):
    model = load_model(args.device)
    inputs = model.data_processor.in_normalizer
    outputs = model.data_processor.out_normalizer
    paths = sorted((RELEASE / "images").glob("sgra_spin*.npy"))
    frames = np.stack([np.load(path)[0] for path in paths])
    names = [p.stem.removeprefix("sgra_spin") for p in paths]
    result = {"torch": torch.__version__, "device": args.device, "variants": {}}
    if "training" in args.variants:
        extrema = fit_extrema(inputs)
        result["provenance"] = {
            "refitted": extrema,
            "release": [[float(n.min.item()), float(n.max.item())] for n in (inputs, outputs)],
            "raw_metadata": raw_metadata(),
        }
    for variant in tqdm(args.variants, desc="Ablations"):
        torch.backends.cudnn.allow_tf32 = variant == "tf32"
        if variant == "training":
            for n, (lo, hi) in zip((inputs, outputs), extrema):
                n.min.fill_(lo)
                n.max.fill_(hi)
        if variant == "fp64":
            model.model.double()
            inputs.double()
            outputs.double()
        forecast = rollout(model, frames, variant, variant == "batch")
        result["variants"][variant] = {name: measure(x, y)
                                        for name, x, y in zip(names, frames, forecast)}
        if variant == "training":
            for n, (lo, hi) in zip((inputs, outputs), result["provenance"]["release"]):
                n.min.fill_(lo)
                n.max.fill_(hi)
        if variant == "fp64":
            model.model.float()
            inputs.float()
            outputs.float()
        (ROOT / "results" / f"ablations_{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", required=True)
    parser.add_argument("variants", nargs="+",
                        choices=("baseline", "training", "clamp", "roundtrip", "batch", "tf32", "fp64"))
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    run(args)
