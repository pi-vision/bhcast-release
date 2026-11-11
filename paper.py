# Recompute Sgr A* metrics using frozen release prediction and feature extraction.
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import json
import multiprocessing as mp
import os
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm

from released import MOVIES, RELEASE, ROOT, load_model
from bhcast_pipeline import (DEFAULT_FEATURE_ORDER, decode_prediction, extract_features,
                             feature_vector_from_metrics, load_label_metadata,
                             write_forecast_to_hdf5)

SPINS = (-0.94, -0.5, 0.5, 0.94)
INCS = (10, 30, 50, 70, 110, 130, 150, 170)
STARTS = range(0, 40, 2)
INC_GROUP = {inc: i // 2 for i, inc in enumerate(INCS)}
RADIUS_FACTORS = list(np.linspace(0.75, 1.5, 5))


def load_test(spin, inc, suffix="", input_offset=900):
    movie = torch.load(MOVIES / f"Ma{spin:+g}_Rh10_i{inc}_w4{suffix}.pt", mmap=True,
                       map_location="cpu", weights_only=True)
    assert len(movie) == 1000
    return movie[input_offset:].numpy().reshape(1000 - input_offset, 100, 100)


def features(movie, spin, inc):
    with TemporaryDirectory(dir=ROOT / "results") as temp, open(os.devnull, "w") as quiet:
        with redirect_stdout(quiet):
            write_forecast_to_hdf5(movie, temp, spin, inc, 200, 39.764, 5)
            result = extract_features(temp, movie, spin, inc, 39.764, None, 5.2, 360,
                                      RADIUS_FACTORS.copy(), False, 0.0, 1.0)
    return {key: (None if result[key] is None or not np.isfinite(result[key])
                  else float(result[key])) for key in DEFAULT_FEATURE_ORDER}


def measure(spin, inc, start, forecast, sharp):
    result = {"spin": spin, "inc": inc, "start": start,
              "forecast": features(forecast, spin, inc)}
    if sharp is not None:
        result["aligned"] = features(sharp[-60:], spin, inc)
        if len(sharp) == 61:
            result["same_index"] = features(sharp[:-1], spin, inc)
    return result


@torch.inference_mode()
def lpips_steps(metric, frames, truth):
    def prepare(images):
        x = torch.as_tensor(images, device="cuda").unsqueeze(1)
        low = x.amin(dim=(-1, -2), keepdim=True)
        span = x.amax(dim=(-1, -2), keepdim=True) - low
        return ((x - low) / span.clamp_min(torch.finfo(x.dtype).tiny) * 2 - 1).repeat(1, 3, 1, 1)

    return torch.cat([metric.net(a, b).reshape(-1)
                      for a, b in zip(prepare(frames).split(16), prepare(truth).split(16))]
                     ).cpu().numpy()


def radial_psd(frames):
    freq = np.fft.fftfreq(100)
    radius = np.hypot(freq[:, None], freq[None, :])
    edges = np.linspace(0, np.sqrt(0.5), 51)
    power = np.abs(np.fft.fft2(frames)) ** 2
    masks = [(radius >= lo) & (radius < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return np.stack([power[..., mask].mean(axis=-1) for mask in masks], axis=-1)


def summary(rows, xgb, labelmap, historical):
    pairs, order, spins = labelmap
    classified = [row for row in rows if row["start"] < (939 if historical else 940)]
    vectors = [feature_vector_from_metrics(row["forecast"], order) for row in classified]
    pred = np.argmax(xgb.predict_proba(np.stack(vectors)), axis=1)
    decoded = [decode_prediction(int(p), pairs, spins, "sgra") for p in pred]
    inc = np.array([INC_GROUP[i] == INC_GROUP[row["inc"]] for (_, i), row in zip(decoded, classified)])
    spin = np.array([(s > 0) == (row["spin"] > 0) for (s, _), row in zip(decoded, classified)])
    out = {"classification": {"n": len(classified), "inclination_correct": int(inc.sum()),
                              "spin_correct": int(spin.sum())}}
    for truth in ("aligned", "same_index"):
        if truth not in rows[0]:
            continue
        out[truth] = []
        for key in DEFAULT_FEATURE_ORDER:
            stats = []
            for s in SPINS:
                for i in INCS:
                    subset = [row for row in rows if row["spin"] == s and row["inc"] == i]
                    values = np.array([[row[truth][key], row["forecast"][key]] for row in subset],
                                      dtype=np.float64)
                    if key == "pitch_angle_deg":
                        values = (values + 90) / 180
                    if key == "asymmetry_A_over_C":
                        values = abs(values)
                    stats.append(np.nanmean(values, axis=0))
            data = np.asarray(stats)
            for s in (*SPINS, "all"):
                group = data if s == "all" else data[np.repeat(np.asarray(SPINS) == s, len(INCS))]
                group = group[np.isfinite(group).all(axis=1)]
                errors = abs(group[:, 0] - group[:, 1])
                out[truth].append(dict(feature=key, spin=s, n=len(errors), mae=errors.mean(),
                                       se=errors.std(ddof=1) / np.sqrt(len(errors)),
                                       r=np.corrcoef(group.T)[0, 1]))
    return out


def run(scenario, workers, historical, batch_stats):
    model = load_model("cuda")
    if batch_stats:
        for layer in model.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.train()
    lab = load_label_metadata(str(RELEASE / "checkpoints/sgra_Ma_forecast.pkl"))
    xgb = joblib.load(os.environ.get("BHCAST_CLASSIFIER", RELEASE / "checkpoints/sgra_Ma_forecast.joblib"))
    if scenario == "20":
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").cuda().eval()
    movies = [(spin, inc) for spin in SPINS for inc in INCS]
    rows, quality = [], []
    normalizer = model.data_processor.in_normalizer
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for spin, inc in tqdm(movies, desc=f"Paper {scenario}"):
            sharp = load_test(spin, inc)
            offset = 899 if historical else 900
            blurred = load_test(spin, inc, "_blurred", offset)
            sharp_input = load_test(spin, inc, input_offset=offset) if scenario in ("25", "30") else None
            jobs = []
            starts = range(0, 41, 2) if historical and scenario == "20" else STARTS
            for start in starts:
                frame = blurred[start]
                if scenario in ("25", "30"):
                    x = normalizer.transform(torch.from_numpy(sharp_input[start][None, None].copy()).cuda())
                    sigma = int(scenario) / (2 * np.sqrt(8 * np.log(2)))
                    x = torch.from_numpy(gaussian_filter(x.cpu().numpy(), (0, 0, sigma, sigma))).cuda()
                    frame = normalizer.inverse_transform(x)[0, 0].cpu().numpy()
                elif scenario == "noise":
                    frame = frame.copy()
                    indices = torch.randperm(frame.size, device="cuda")[:100].cpu().numpy()
                    salt = torch.rand(100, device="cuda").cpu().numpy() > 0.5
                    frame.flat[indices] = np.where(salt, frame.max(), frame.min())
                elif scenario == "shift":
                    frame = np.pad(frame[:, 5:], ((0, 0), (0, 5)))
                forecast = model.predict(frame, horizon=60)
                if scenario == "20" and start == 0:
                    quality_forecast = model.predict(frame, horizon=100) if historical else forecast
                    quality.append((spin, inc, lpips_steps(metric, quality_forecast,
                                                           sharp[:100] if historical else sharp[1:61]),
                                    lpips_steps(metric, quality_forecast, sharp[:len(quality_forecast)]),
                                    radial_psd(np.stack((np.concatenate((frame[None], forecast)),
                                                         sharp[:61])))))
                truth = sharp[start:start + (60 if historical else 61)].copy() if scenario == "20" else None
                jobs.append(pool.submit(measure, spin, inc, offset + start, forecast, truth))
            rows.extend(job.result() for job in jobs)
    tag = f"paper_{scenario}{'_historical' if historical else ''}{'_batch_stats' if batch_stats else ''}"
    (ROOT / "results" / f"{tag}_rows.json").write_text(json.dumps(rows) + "\n")
    (ROOT / "results" / f"{tag}.json").write_text(
        json.dumps(summary(rows, xgb, lab, historical), indent=2) + "\n")
    if quality:
        spins, incs, aligned, same_index, psd = zip(*quality)
        np.savez(ROOT / "results" / f"{tag}_quality.npz", spins=spins, inclinations=incs,
                 lpips_aligned=aligned, lpips_same_index=same_index, psd=psd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("20", "25", "30", "noise", "shift"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--batch-stats", action="store_true")
    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = args.historical
    run(args.scenario, args.workers, args.historical, args.batch_stats)
