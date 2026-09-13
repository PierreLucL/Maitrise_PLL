"""Compare seed replicates of the SAME session, retaining unit-level diagnostics.

Uses saved float32 J/RNN. Unit × time maps are computed one anatomical pair
at a time; original pickles retain all inputs needed to regenerate them.
No alignment across mice, hypothesis test, or causal interpretation.
"""
import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
from compute_curbd_currents_mouse410 import load_pickle_compatible


def metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ac, bc = a-a.mean(axis=-1, keepdims=True), b-b.mean(axis=-1, keepdims=True)
    va, vb = np.sum(ac*ac, axis=-1), np.sum(bc*bc, axis=-1)
    denom = np.sqrt(va*vb)
    r = np.divide(np.sum(ac*bc, axis=-1), denom, out=np.full_like(denom, np.nan), where=denom > 0)
    ratio = np.sqrt(np.divide(vb, va, out=np.full_like(va, np.nan), where=va > 0))
    return r, ratio


def finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in np.atleast_1d(values)]


def audit(paths, output):
    data = []
    for path in paths:
        d = load_pickle_compatible(path)
        # Drop bulky replay inputs after reading; originals remain untouched.
        data.append({k: d[k] for k in ['parameters', 'row', 'reproducibility', 'masque_sub', 'regions', 'tRNN', 'J_final', 'RNN_final']})
        del d
    ref = data[0]
    params = {k: v for k, v in ref['parameters'].items() if k != 'seed'}
    seeds = [d['parameters']['seed'] for d in data]
    if len(data) != 3 or len(set(seeds)) != 3:
        raise ValueError('Expected three distinct seeds per session')
    for d in data:
        if {k: v for k, v in d['parameters'].items() if k != 'seed'} != params:
            raise ValueError('Parameters differ beyond seed')
        for key in ['timeseries_sha256', 'source_sha256', 'packages', 'threads']:
            if d['reproducibility'][key] != ref['reproducibility'][key]:
                raise ValueError(f'Provenance mismatch: {key}')
        for key in ['masque_sub', 'tRNN']:
            if not np.array_equal(d[key], ref[key], equal_nan=(key == 'masque_sub')):
                raise ValueError(f'Mismatch: {key}')
        if list(d['regions'][:, 0]) != list(ref['regions'][:, 0]):
            raise ValueError('Anatomical labels differ')
        for i in range(len(ref['regions'])):
            if not np.array_equal(d['regions'][i, 1], ref['regions'][i, 1]):
                raise ValueError('Unit identities differ')
        if not all(np.isfinite(d[k]).all() for k in ['J_final', 'RNN_final']):
            raise ValueError('Nonfinite model')
    names = list(ref['regions'][:, 0])
    rows = []
    recurrent_rows = []
    for ti, target in enumerate(names):
        target_idx = np.asarray(ref['regions'][ti, 1], dtype=int)
        recurrent = [np.zeros((len(target_idx), len(ref['tRNN'])), dtype=np.float64) for _ in data]
        for si, source in enumerate(names):
            source_idx = np.asarray(ref['regions'][si, 1], dtype=int)
            maps = [d['J_final'][np.ix_(target_idx, source_idx)] @ d['RNN_final'][source_idx] for d in data]
            for accumulator, current in zip(recurrent, maps):
                accumulator += current
            totals = [m.sum(axis=0, dtype=np.float64) for m in maps]
            for ai, bi in itertools.combinations(range(3), 2):
                unit_r, unit_ratio = metrics(maps[ai], maps[bi])
                total_r, total_ratio = metrics(totals[ai], totals[bi])
                rows.append(dict(target=target, source=source, seed_a=seeds[ai], seed_b=seeds[bi],
                                 total_pearson=finite_list(total_r)[0], total_std_ratio_b_over_a=finite_list(total_ratio)[0],
                                 total_mean_a=float(totals[ai].mean()), total_mean_b=float(totals[bi].mean()),
                                 total_rmse=float(np.sqrt(np.mean((totals[ai]-totals[bi])**2))),
                                 unit_ids=target_idx.tolist(), unit_pearson=finite_list(unit_r),
                                 unit_std_ratio_b_over_a=finite_list(unit_ratio),
                                 unit_pearson_median=float(np.nanmedian(unit_r))))
            del maps
        for ai, bi in itertools.combinations(range(3), 2):
            unit_r, _ = metrics(recurrent[ai], recurrent[bi])
            total_r, total_ratio = metrics(recurrent[ai].sum(axis=0), recurrent[bi].sum(axis=0))
            rnn_r, _ = metrics(data[ai]['RNN_final'][target_idx], data[bi]['RNN_final'][target_idx])
            recurrent_rows.append(dict(target=target, seed_a=seeds[ai], seed_b=seeds[bi],
                                       total_pearson=finite_list(total_r)[0],
                                       total_std_ratio_b_over_a=finite_list(total_ratio)[0],
                                       unit_pearson_median=float(np.nanmedian(unit_r)),
                                       rnn_unit_pearson_median=float(np.nanmedian(rnn_r))))
        del recurrent
        print(f"Mouse {params['mouse']}: target {ti+1}/{len(names)}", flush=True)
    result = dict(parameters=params, seeds=seeds, files=[str(p.resolve()) for p in paths],
                  provenance_checks='passed: identical parameters except seed, data hash, source hashes, packages, threads, mask, unit mapping, time grid',
                  runs=[d['row'] for d in data], region_names=names, comparisons=rows,
                  recurrent_all_sources=recurrent_rows,
                  limitations=['Same training session only; no held-out generalization test.',
                               'Fixed segmentation seed; segmentation uncertainty is not measured.',
                               'Totals can hide cancellation across target units; inspect unit metrics.',
                               'Current maps use saved float32 J/RNN; originals preserved for reconstruction.',
                               'No age effect can be inferred from these two M6 sessions.'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    del data
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for mouse in [410, 415]:
        paths = sorted(args.input.glob(f'*mouse{mouse}_*.pkl'))
        audit(paths, args.output / f'mouse{mouse}.json')
