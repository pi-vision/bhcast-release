# Compare the fresh measurements with a supplied reference and paper cache.
import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np

from paper import INCS, SPINS, radial_psd
from released import ROOT

RESULTS = ROOT / "results"


def compare_ablations(reference):
    released = json.loads((RESULTS / "released.json").read_text())
    keys = list(reference)
    expected = np.asarray([reference[k] for k in keys])
    baseline = np.array([released[k]["change_relative"][1:] + released[k]["flux_relative"]
                         for k in keys])
    for path in sorted(RESULTS.glob("ablations_*.json")):
        for variant, rows in json.loads(path.read_text())["variants"].items():
            values = np.asarray([rows[k][1:] for k in keys])
            delta = np.abs(values - baseline)
            err = np.abs(values - expected)
            label = path.stem.removeprefix("ablations_") + "/" + variant
            idx = np.unravel_index(delta.argmax(), delta.shape)
            print(label, "median/max Δ", np.median(delta), delta.max(),
                  "largest move", keys[idx[0]], idx[1], values[idx] - baseline[idx],
                  "max ref error", err.max(), "outside", (err > .02).sum())


def compare_quality(cache):
    lpips, psd = {}, {}
    for path in sorted(cache.glob("result_*.pkl")):
        with path.open("rb") as f:
            row = pickle.load(f)
        key = (row["spin"], row["inclination"])
        if key[0] not in SPINS or key[1] not in INCS or row["filename"] != f"Ma{key[0]:+g}_Rh10_i{key[1]}_w4.pt":
            continue
        lpips[key] = row["our_forecast_gt_lpips_errors"]
        psd[key] = radial_psd(row["our_forecast_frames"])
    assert len(lpips) == 32
    for p in sorted(RESULTS.glob("paper_20_historical*_quality.npz")):
        with np.load(p) as result:
            keys = list(zip(result["spins"], result["inclinations"]))
            lpips_ref = np.stack([lpips[k] for k in keys])
            psd_ref = np.stack([psd[k] for k in keys])
            pred = result["lpips_aligned"]
            errors = pred - lpips_ref
            print(p.stem, "LPIPS mean", pred.mean(), "cache mean", lpips_ref.mean(),
                  "mean absolute curve difference", np.abs(errors.mean(axis=0)).mean())
            tag = p.stem.removesuffix("_quality")
            with (RESULTS / f"{tag}_curves.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "group", "step", "radial_bin", "paper_cache",
                                 "paper_cache_se", "fresh", "delta"])
                for label, incs in (("all", INCS), ("face-on", (10, 30, 150, 170)),
                                    ("edge-on", (50, 70, 110, 130))):
                    mask = np.isin(result["inclinations"], incs)
                    for metric, ref, observed in (
                        ("LPIPS", lpips_ref, pred),
                        ("PSD", psd_ref[:, :60], result["psd"][:, 0, 1:]),
                    ):
                        ref, observed = ref[mask], observed[mask]
                        n = ref.shape[0]
                        mean, se = ref.mean(axis=0), ref.std(axis=0, ddof=1) / np.sqrt(n)
                        fresh = observed.mean(axis=0)
                        for index in np.ndindex(mean.shape):
                            step = index[0] + 1
                            bin_idx = index[1] if len(index) > 1 else ""
                            writer.writerow([metric, label, step, bin_idx, mean[index],
                                             se[index], fresh[index], fresh[index] - mean[index]])
                        if metric == "LPIPS":
                            print(label, "steps outside cache 2 SE", (abs(fresh - mean) > 2 * se).sum())
                            for step in (1, 6, 16, 60, 100):
                                j = step - 1
                                print(step, "cache/SE/fresh", mean[j], se[j], fresh[j])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    compare_ablations(json.loads(args.reference.read_text()))
    compare_quality(args.cache)
