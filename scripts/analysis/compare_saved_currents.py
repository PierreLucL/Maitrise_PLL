"""Compare deux NPZ de courants sur une grille temporelle strictement commune.

Les NPZ attendus sont ceux de compute_curbd_currents_mouse410.py.
La correlation est calculee par paire source/cible, jamais sur un vecteur
tronque. Les amplitudes restent celles des courants totaux sauvegardes.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_currents(path):
    """Charge les fichiers locaux de confiance; region_names utilise dtype=object."""
    with np.load(path, allow_pickle=True) as data:
        return validate_currents(
            data['currents'], data['labels'], data['tRNN'], data['region_names']
        )


def validate_currents(currents, labels, times, region_names):
    currents = np.asarray(currents, dtype=float)
    labels = np.asarray(labels)
    times = np.asarray(times, dtype=float)
    region_names = np.asarray(region_names)
    if region_names.ndim != 1 or not all(isinstance(x, str) and x for x in region_names):
        raise ValueError('Les noms de regions doivent etre des chaines non vides.')
    names = list(region_names)
    if not names or len(names) != len(set(names)):
        raise ValueError('Les noms de regions doivent etre uniques.')
    n = len(names)
    if times.ndim != 1 or len(times) < 3 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError('Les temps doivent etre finis et strictement croissants (au moins 3).')
    if currents.shape != (n*n, len(times)) or not np.all(np.isfinite(currents)):
        raise ValueError('Courants non finis ou dimensions incompatibles avec regions/temps.')
    if labels.shape != (n*n, 2) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Les labels doivent etre des couples entiers (cible, source).')
    keys = [tuple(map(int, p)) for p in labels]
    expected = {(target, source) for target in range(n) for source in range(n)}
    if len(set(keys)) != n*n or set(keys) != expected:
        raise ValueError('Les labels doivent couvrir chaque paire cible/source exactement une fois.')
    # La cle anatomique permet de reordonner sans supposer le meme ordre numerique.
    by_pair = {(names[target], names[source]): currents[k]
               for k, (target, source) in enumerate(keys)}
    return {'names': names, 'times': times, 'by_pair': by_pair}


def trace_metrics(a, b):
    """Forme, moyennes et amplitudes separees; une trace constante donne r=None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 3:
        raise ValueError('Les traces doivent avoir exactement les memes dimensions.')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Les traces doivent etre finies.')
    ac, bc = a-a.mean(), b-b.mean()
    na, nb = np.linalg.norm(ac), np.linalg.norm(bc)
    constant = bool(np.all(a == a[0]) or np.all(b == b[0]))
    r = None if constant or na == 0 or nb == 0 else float(np.clip(np.dot(ac,bc)/(na*nb), -1, 1))
    return {'pearson': r, 'pearson_status': 'constant' if r is None else 'ok',
            'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'std_a': float(a.std()), 'std_b': float(b.std()),
            'std_ratio_b_over_a': None if na == 0 else float(nb/na),
            'rmse_raw': float(np.sqrt(np.mean((a-b)**2)))}


def compare_currents(a, b):
    if set(a['names']) != set(b['names']):
        raise ValueError('Les noms anatomiques des regions ne correspondent pas.')
    if not np.array_equal(a['times'], b['times']):
        raise ValueError('Grilles temporelles differentes: aucun recadrage ou resampling implicite.')
    names = a['names']
    pairs, totals = [], []
    for target in names:
        total_a = np.zeros_like(a['times'])
        total_b = np.zeros_like(b['times'])
        for source in names:
            ca, cb = a['by_pair'][(target,source)], b['by_pair'][(target,source)]
            pairs.append({'target':target, 'source':source, **trace_metrics(ca,cb)})
            total_a += ca
            total_b += cb
        totals.append({'target':target, **trace_metrics(total_a,total_b)})
    valid_r = [p['pearson'] for p in pairs if p['pearson'] is not None]
    return {
        'method': 'Pearson par paire sur toute la session; moyennes et amplitudes brutes separees',
        'limitations': 'Les NPZ seuls ne prouvent ni la meme session, ni le meme masque, ni des graines independantes. Les noms servent de correspondance anatomique declaree.',
        'region_names': names,
        'n_time': len(a['times']), 'time_start':float(a['times'][0]), 'time_end':float(a['times'][-1]),
        'n_pairs':len(pairs), 'n_valid_pearson':len(valid_r),
        'median_pearson':float(np.median(valid_r)) if valid_r else None,
        'n_negative_pearson':sum(r < 0 for r in valid_r),
        'pairs':pairs, 'total_recurrent_by_target':totals,
    }


def plot_comparison(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = result['region_names']
    n = len(names)
    matrix = np.array([np.nan if p['pearson'] is None else p['pearson'] for p in result['pairs']]).reshape(n,n)
    fig, ax = plt.subplots(figsize=(max(6,n), max(5,n-1)), layout='constrained')
    im = ax.imshow(matrix, cmap='bwr', vmin=-1, vmax=1)
    ax.set_xticks(range(n),names,rotation=45,ha='right')
    ax.set_yticks(range(n),names)
    ax.set_xlabel('Source'); ax.set_ylabel('Cible')
    ax.set_title('Similarité des courants par paire\nSession complète, grille temporelle identique')
    for i in range(n):
        for j in range(n):
            value = matrix[i,j]
            ax.text(j,i,'NA' if not np.isfinite(value) else f'{value:.2f}',ha='center',va='center',
                    color='white' if abs(value) > .6 else 'black')
    fig.colorbar(im,ax=ax,label='Pearson r')
    fig.savefig(output,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a',type=Path)
    parser.add_argument('b',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    result=compare_currents(load_currents(args.a),load_currents(args.b))
    result['file_a']=str(args.a.resolve())
    result['file_b']=str(args.b.resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if args.plot:
        plot_comparison(result,args.output_dir/'pearson_by_pair.png')
    print(json.dumps({k:result[k] for k in ['n_pairs','n_valid_pearson','median_pearson','n_negative_pearson']}))


if __name__=='__main__':
    main()
