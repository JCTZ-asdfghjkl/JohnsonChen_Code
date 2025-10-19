#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIDI → MFCC features (no FluidSynth / no soundfont) → Nested CV (10-fold outer, Bayesian inner)
Multi-output RandomForestRegressor; metric = RMSE

Usage:
  python midi_mfcc_nestedcv_rf_nosf2.py \
      --midi_dir /path/to/midis \
      --targets_csv /path/to/targets.csv \
      --out_csv results_nestedcv.csv

Notes:
  - Synthesis: simple additive synth (fundamental + 2 harmonics) with ADSR envelope.
  - MFCC features: MFCC and Δ-MFCC, aggregated by mean & std → n_features = n_mfcc * 4 (default 80).
  - Targets CSV must have 6 columns (optionally a 'filename' column to align).
"""

import os
import glob
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import pretty_midi
import librosa

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.utils.validation import check_random_state




# ----------------------------- Lightweight Synth ----------------------------- #
def hz_from_midi(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

def adsr_envelope(n_samples: int, sr: int,
                  attack=0.01, decay=0.05, sustain_level=0.8, release=0.05):
    a = int(sr * attack)
    d = int(sr * decay)
    r = int(sr * release)
    s = max(0, n_samples - (a + d + r))
    env = np.zeros(n_samples, dtype=np.float32)
    # Attack
    if a > 0:
        env[:a] = np.linspace(0, 1, a, endpoint=False)
    # Decay
    if d > 0:
        env[a:a+d] = np.linspace(1, sustain_level, d, endpoint=False)
    # Sustain
    if s > 0:
        env[a+d:a+d+s] = sustain_level
    # Release
    if r > 0:
        start = a + d + s
        env[start:start+r] = np.linspace(sustain_level, 0, r, endpoint=False)
    # Tail pad if needed
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)))
    return env

def synthesize_track(pm, sr: int = 22050) -> np.ndarray:
    """
    Simple additive synthesis:
      - For each note: sum sinusoids at f, 2f, 3f with per-harmonic weights.
      - Velocity scales amplitude. Overlapping notes are mixed.
    Returns mono float32 waveform.
    """
    if not pm.instruments:
        return np.zeros(int(sr), dtype=np.float32)

    end_time = max([note.end for inst in pm.instruments for note in inst.notes], default=0.0)
    n_samples = max(int(np.ceil(end_time * sr)) + sr // 10, sr // 10)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Simple timbre map by instrument: (w1, w2, w3)
    # Defaults give a mild brightness; percussive channel ignored for pitched notes.
    def timbre_for(program: int, is_drum: bool):
        if is_drum:
            return (1.0, 0.0, 0.0)  # ignore drums (or you could add noise bursts)
        if program in range(24, 32):      # Guitars
            return (1.0, 0.4, 0.2)
        if program in range(40, 48):      # Strings
            return (1.0, 0.6, 0.3)
        if program in range(0, 8):        # Pianos
            return (1.0, 0.5, 0.25)
        if program in range(56, 64):      # Brass
            return (1.0, 0.7, 0.4)
        if program in range(64, 72):      # Reeds
            return (1.0, 0.6, 0.35)
        if program in range(72, 80):      # Pipes
            return (1.0, 0.5, 0.3)
        return (1.0, 0.5, 0.25)

    for inst in pm.instruments:
        w1, w2, w3 = timbre_for(inst.program, inst.is_drum)
        for note in inst.notes:
            f0 = hz_from_midi(note.pitch)
            start = int(note.start * sr)
            stop = max(start + 1, int(note.end * sr))
            if stop > n_samples:
                pad = stop - n_samples
                audio = np.pad(audio, (0, pad))
                n_samples = audio.shape[0]

            t = np.arange(stop - start) / sr
            #print(len(t))
            if len(t)<0.1*sr: #TODO: test short notes length.
                continue
            # fundamental + 2 harmonics
            sig = (w1 * np.sin(2 * np.pi * f0 * t) +
                   w2 * np.sin(2 * np.pi * 2 * f0 * t) * 0.5 +
                   w3 * np.sin(2 * np.pi * 3 * f0 * t) * 0.33)

            # ADSR; shorten release for staccato notes
            dur = (stop - start) / sr
            rel = min(0.05, max(0.01, 0.2 * dur))
            env = adsr_envelope(len(sig), sr, attack=0.005, decay=0.02, sustain_level=0.85, release=rel)

            # velocity scaling (0..127)
            amp = np.clip(note.velocity / 127.0, 0.05, 1.0)
            audio[start:stop] += (sig * env * amp).astype(np.float32)

    # Prevent clipping
    m = np.max(np.abs(audio)) or 1.0
    if m > 1.0:
        audio = audio / m
    return audio.astype(np.float32)


# ------------------------- Feature Extraction (MFCC) ------------------------- #
def extract_feature_vector(y: np.ndarray, sr: int, n_mfcc: int = 20, hop_length: int = 512) -> np.ndarray:
    if y.size == 0:
        # fixed length for all features
        return np.zeros(n_mfcc*4 + 24 + 14 + 2, dtype=np.float32)

    feats = []

    # -------------------- MFCC + Δ -------------------- #
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    if mfcc.shape[1] == 0:
        mfcc = np.zeros((n_mfcc,1))
    dmfcc = librosa.feature.delta(mfcc)
    feats.extend(mfcc.mean(axis=1))
    feats.extend(mfcc.std(axis=1))
    feats.extend(dmfcc.mean(axis=1))
    feats.extend(dmfcc.std(axis=1))

    # -------------------- Chroma -------------------- #
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12)
    if chroma.shape[1] == 0:
        chroma = np.zeros((12,1))
    feats.extend(chroma.mean(axis=1))
    feats.extend(chroma.std(axis=1))  # 12+12 = 24

    # -------------------- Spectral Contrast -------------------- #
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    if contrast.shape[1] == 0:
        contrast = np.zeros((7,1))
    feats.extend(contrast.mean(axis=1))
    feats.extend(contrast.std(axis=1))  # 7+7 = 14

    # -------------------- Tempo / Rhythm -------------------- #
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    feats.append(float(tempo))
    feats.append(len(beats) / (len(y)/sr))
    return np.array(feats).astype(float)#, dtype=np.float32)



def build_feature_matrix(midi_files: List[str], sr: int = 22050, n_mfcc: int = 20, hop_length: int = 512, quiet: bool = False) -> np.ndarray:
    feats = []
    iterator = midi_files if quiet else tqdm(midi_files, desc="Synth → Features (MFCC+Chroma+Contrast+Tonnetz+Rhythm)")
    for path in iterator:
        y = synthesize_track(pretty_midi.PrettyMIDI(path), sr=sr)
        x_i = extract_feature_vector(y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        feats.append(x_i)
    return np.vstack(feats)



# ------------------------------- Targets Loading ------------------------------ #
def load_targets(targets_csv: str, midi_files: List[str]) -> np.ndarray:
    df = pd.read_csv(targets_csv)
    if 'filename' in df.columns:
        df = df.set_index('filename').reindex([os.path.basename(p) for p in midi_files])
        if df.isnull().any().any():
            missing = df.index[df.isnull().any(axis=1)].tolist()
            raise ValueError(f"Targets CSV missing rows for files: {missing}")
    target_cols = [c for c in df.columns if c.lower().startswith('y')] or df.columns[:6].tolist()
    if len(target_cols) < 6:
        raise ValueError("Targets CSV must have 6 target columns (e.g., y1..y6).")
    Y = df[target_cols[:6]].to_numpy(dtype=float)
    return Y


# ------------------------------ Nested CV Training ---------------------------- #
def nested_cv_random_forest(X: np.ndarray, Y: np.ndarray,
                            #outer_splits: int = 10, inner_splits: int = 3,
                            n_iter: int = 32, random_state: int = 42) -> Tuple[pd.DataFrame, dict, np.ndarray]:
    rng = check_random_state(random_state)

    rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Define pipeline
    pipe = Pipeline([
        ("select", SelectKBest(score_func=f_regression)),   # feature selection
        #("rf", RandomForestRegressor(random_state=rng.randint(0, 2**31 - 1), n_jobs=-1))
        ("rf", Ridge(random_state=2025)),
    ])

    # Search space includes both feature selector and RF hyperparams
    search_space = {
        "select__k": Integer(3, int(np.sqrt(X.shape[1]))),   # number of features to select
        #"rf__n_estimators": Integer(2,5),#100, 900),
        #"rf__max_depth": Integer(2, 3),#32),
        #"rf__min_samples_split": Integer(2, 10),#20),
        #"rf__min_samples_leaf": Integer(1, 5),#10),
        # "rf__max_features": Categorical(["auto", "sqrt", "log2", None]),
        # "rf__bootstrap": Categorical([True, False]),
    }

    outer_cv = LeaveOneOut()#KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    fold_rows, best_params = [], {}

    ypte = np.zeros((Y.shape[0], Y.shape[1]))
    for fold_idx, (tr, te) in enumerate(outer_cv.split(X, Y), start=1):
        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y[tr], Y[te]

        inner_cv = LeaveOneOut()#KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

        opt = BayesSearchCV(
            estimator=pipe,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=rmse_scorer,   # minimize RMSE
            n_jobs=4,
            refit=True,
            random_state=random_state,
            verbose=0,
        )

        opt.fit(X_tr, Y_tr[:,0])
        best_params[fold_idx] = opt.best_params_

        y_pred = opt.predict(X_te).reshape(-1,1)
        ypte[te] = y_pred
        rmse_overall = mean_squared_error(Y_te, y_pred)
        per_target = [mean_squared_error(Y_te[:, j], y_pred[:, j]) for j in range(Y.shape[1])]

        row = {"fold": fold_idx, "rmse_overall": rmse_overall}
        row.update({f"rmse_y{j+1}": v for j, v in enumerate(per_target)})
        fold_rows.append(row)

        print(f"[Outer Fold {fold_idx:02d}] RMSE={rmse_overall:.4f} | best={opt.best_params_}")

    results_df = pd.DataFrame(fold_rows)
    summary = results_df.mean(numeric_only=True).to_dict()
    print("\n=== Nested CV Summary (mean over outer folds) ===")
    print("RMSE overall: {:.4f} | per-target: {}".format(
        summary["rmse_overall"],
        ", ".join([f"y{j+1}:{summary[f'rmse_y{j+1}']:.4f}" for j in range(Y.shape[1])])
    ))
    return results_df, best_params, ypte


# ------------------------------------ Main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi_dir', type=str, required=True, help="Directory with .mid files")
    parser.add_argument('--targets_csv', type=str, required=True, help="CSV with 6 targets; optional 'filename' column")
    parser.add_argument('--out_csv', type=str, default='results_nestedcv.csv', help="Per-fold metrics output CSV")
    parser.add_argument('--sr', type=int, default=22050, help="Synthesis & analysis sample rate")
    parser.add_argument('--n_mfcc', type=int, default=20, help="Number of MFCCs")
    parser.add_argument('--hop_length', type=int, default=512, help="Hop length for MFCC")
    parser.add_argument('--quiet', action='store_true', help="Less verbose progress")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    #midi_files = sorted(sum([glob.glob(os.path.join(args.midi_dir, ext))
    #                         for ext in ("*.mid", "*.MID", "*.midi", "*.MIDI")], []))
    midi_files = sorted(glob.glob(os.path.join(args.midi_dir, "*.mid")))
    if len(midi_files) == 0:
        raise FileNotFoundError(f"No MIDI files found in {args.midi_dir}")
    if len(midi_files) != 40:
        print(f"[WARN] Found {len(midi_files)} MIDI files; proceeding anyway.")

    print(f"Found {len(midi_files)} MIDI files. Synthesizing with built-in additive synth (no SF2).")

    X = build_feature_matrix(midi_files, sr=args.sr, n_mfcc=args.n_mfcc,
                             hop_length=args.hop_length, quiet=args.quiet)
    X = X.astype(float)
    print(f"Feature matrix shape: {X.shape}")

    Y = load_targets(args.targets_csv, midi_files)
    Y = Y.astype(float)
    Y = Y[:,1:]  # remove delta power because it is very low

    Y = Y[:,[0]]

    #if Y.shape[0] != X.shape[0] or Y.shape[1] != 5:
    #    raise ValueError(f"Targets shape must be (N_files, 6); got {Y.shape}, N_files={X.shape[0]}")
    print(f"Targets shape: {Y.shape}")

    # remove X rows with all zeros
    non_zero_rows = np.any(X != 0, axis=1)
    if not np.all(non_zero_rows):
        print(f"[WARN] Removing {np.sum(~non_zero_rows)} rows with all-zero features.")
        X = X[non_zero_rows]
        Y = Y[non_zero_rows]
    #print(X)
    #print(Y)

    # compuete univariate correlation between each feature of X and each target in Y
    #corrs = np.corrcoef(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:]
    #print("Feature-target correlations (first 5 features):")
    #TODO X=MFCC features: what is MFCC, which feature is most correlated with each target?

    results_df, best_params, ypred = nested_cv_random_forest(X, Y, n_iter=32, random_state=42)
    breakpoint()
    results_df.to_csv(args.out_csv, index=False)
    print(f"Saved per-fold metrics to {args.out_csv}")

    # Save best params per fold
    import json
    with open(os.path.splitext(args.out_csv)[0] + "_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("Saved best hyperparameters per outer fold.")


if __name__ == "__main__":
    main()
