"""Render the descriptive seed audit; no biological inference or cutoff selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path('results/reproducibility_2614685')
    datasets = [json.loads((root / f'mouse{m}.json').read_text()) for m in [410, 415]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    lines = ['# Reproductibilité des courants — job 2614685', '',
             '## Conclusion', '',
             'Cet audit complémentaire porte sur les sommes signées et les corrélations par unité sans alignement. Certaines traces sont anticorrélées entre graines, mais cela ne contredit pas la reproductibilité des dynamiques dominantes déjà observée en PCA–CCA. Il faut distinguer dynamique, signe, amplitude et spécificité régionale avant de choisir une signature du vieillissement.', '',
             '## Mise en contexte historique — ajout du 11 septembre', '',
             'Les six modèles avaient déjà été rapatriés et analysés le 10 septembre dans « Analyser les discussions de maîtrise ». Les corrélations médianes des normes PCA10, selon les trois paires de graines, vont de 0,848 à 0,899 pour 410 et de 0,903 à 0,917 pour 415 ; les premières CCA médianes sont respectivement 0,986–0,987 et 0,988–0,989. Le calcul suit une description méthodologique, sans identité vérifiée ligne par ligne avec les scripts des auteurs.', '',
             'Les contrôles temporels et de source existent également : CCA test médiane 0,982 / 0,935 ; bonne source première en CCA 89/108 / 29/108, pour 410 / 415. Ces comparaisons sont dépendantes et le réseau a appris la session entière. Le point ouvert est donc notamment la spécificité régionale, surtout chez 415, et non une absence générale de reproductibilité. Voir ../../docs/contexte_discussions_maitrise.md et les bilans sources qui y sont référencés.', '',
             '## Données et méthode', '',
             'Deux dossiers C9/M6 (410 et 415), trois graines (2026, 2027, 2028), 100 entraînements, segmentation fixée à 0. Les six sorties sont locales. Vérifications réussies entre graines de chaque souris : paramètres hors graine, empreinte des données, code, versions, threads, masque (NaN hors cortex), identités des unités et grille temporelle.', '',
             'Courant calculé comme J[cible, source] × RNN[source, temps], avec les tableaux float32 sauvegardés. Comparaison temporelle de toute la session entre graines de la même session. Analyse des sommes régionales et des traces de chaque unité cible, sans aligner les souris entre elles. Les cartes unités × temps sont calculées par blocs ; les PKL originaux conservent leurs facteurs pour les régénérer sans perte supplémentaire.', '',
             '## Résultats descriptifs', '',
             '| Souris | pVar finale (min–max) | Médiane r des sommes source→cible | Comparaisons r < 0 | Médiane r du courant toutes sources |',
             '|---|---|---|---|---|']
    for row, d in enumerate(datasets):
        entries = d['comparisons']
        rs = np.array([x['total_pearson'] for x in entries])
        rec = np.array([x['total_pearson'] for x in d['recurrent_all_sources']])
        pv = [x['pVar_finale'] for x in d['runs']]
        lines.append(f"| {d['parameters']['mouse']} | {min(pv):.4f}–{max(pv):.4f} | {np.median(rs):.3f} | {sum(rs<0)}/{len(rs)} | {np.median(rec):.3f} |")
        names = d['region_names']
        labels = [n.replace('Rég. ', '') for n in names]
        for col, field in enumerate(['total_pearson', 'unit_pearson_median']):
            # Worst of three seed pairs for each anatomical source/target.
            matrix = np.array([[min(x[field] for x in entries if x['target']==t and x['source']==s) for s in names] for t in names])
            ax = axes[row, col]
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(names)), labels, rotation=30, ha='right')
            ax.set_yticks(range(len(names)), labels)
            ax.set_xlabel('Région source'); ax.set_ylabel('Région cible')
            title = 'Somme des unités cibles' if col==0 else 'Médiane des corrélations par unité'
            ax.set_title(f"Souris {d['parameters']['mouse']} — {title}")
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color='white' if abs(matrix[i,j])>.6 else 'black')
        fig.colorbar(im, ax=axes[row,:], label='Plus faible r parmi les 3 paires de graines')
    fig.suptitle('Courants CURBD : stabilité entre graines sur la même session\n100 entraînements — valeurs descriptives, sans seuil de validation', fontsize=14)
    fig.savefig(root/'stabilite_graines.png', dpi=160)
    plt.close(fig)
    lines += ['', 'Chaque souris fournit 36 paires anatomiques × 3 comparaisons de graines = 108 valeurs ; ces valeurs sont dépendantes et ne sont pas 108 répétitions biologiques. Le courant toutes sources donne 6 cibles × 3 comparaisons = 18 valeurs. Les médianes ne constituent pas un test statistique.', '']
    for d in datasets:
        entries=d['comparisons']; recurrent=d['recurrent_all_sources']
        ratio=np.array([x['total_std_ratio_b_over_a'] for x in entries])
        lines += [f"### Souris {d['parameters']['mouse']}", '',
                  f"- Corrélation des sommes source→cible : minimum {min(x['total_pearson'] for x in entries):.3f}, maximum {max(x['total_pearson'] for x in entries):.3f}.",
                  f"- Rapport d’écart-type b/a : {ratio.min():.2f} à {ratio.max():.2f}. Une corrélation élevée ne garantit donc pas une amplitude comparable.",
                  f"- Médiane des médianes par unité, par paire anatomique et paire de graines : {np.median([x['unit_pearson_median'] for x in entries]):.3f}.",
                  f"- Courant récurrent toutes sources : r régional entre {min(x['total_pearson'] for x in recurrent):.3f} et {max(x['total_pearson'] for x in recurrent):.3f} ; médiane des médianes par unité {np.median([x['unit_pearson_median'] for x in recurrent]):.3f}.",
                  f"- Activité RNN : médiane des corrélations médianes par unité et cible {np.median([x['rnn_unit_pearson_median'] for x in recurrent]):.3f}.", '']
    lines += ['## Conséquence pour le programme longitudinal', '',
              'La stabilité du courant total malgré des contributions variables serait compatible avec des compensations entre sources ; elle ne démontrerait pas une décomposition régionale unique. La figure montre également les corrélations par unité pour éviter de conclure uniquement à partir de sommes régionales.', '',
              '1. À la fin du run 300 de 410/graine2026, vérifier à données et segmentation identiques si les contributions et leurs amplitudes changent par rapport au run 100. Un seul run 300 ne teste pas la reproductibilité à 300.',
              '2. Si cette prolongation est utile, compléter de façon ciblée les autres graines à 300 pour 410 avant de généraliser. Choisir selon la stabilité des caractéristiques et le coût, pas seulement pVar.',
              '3. Pendant ce temps, préparer le panel longitudinal décrit dans ../longitudinal_inventory/rapport.md : trois souris avec dossiers M6 et M18, plus âges intermédiaires. Confirmer les métadonnées et faire le contrôle qualité anatomique.',
              '4. Exploiter les contrôles temporels et régionaux déjà réalisés : examiner les dimensions suivantes, le signal partagé, les amplitudes et les cartes par unité, particulièrement chez 415. Étudier la redondance de l’activité et la sensibilité à la normalisation avant de choisir les caractéristiques biologiques.', '',
              '## Limites', '',
              '- Deux souris à M6 seulement : aucune conclusion sur le vieillissement.',
              '- Reconstruction sur la session entraînée ; aucune validation hors échantillon ici.',
              '- Segmentation fixée : sa variabilité reste à étudier.',
              '- Les courants sont des interactions fonctionnelles inférées ; ni connexions synaptiques ni preuve causale.',
              '- Métadonnées de session issues des sorties et dossiers ; les empreintes établissent la concordance entre graines, pas l’exactitude biologique des étiquettes.', '',
              '## Reproduction', '',
              '`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MPLCONFIGDIR=/tmp/curbd-mpl python3 scripts/analysis/audit_seed_currents.py results/narval_reproducibility_pix15_410_415/2614685 results/reproducibility_2614685`', '',
              '`MPLCONFIGDIR=/tmp/curbd-mpl python3 scripts/analysis/report_seed_audit.py`', '',
              'Les détails numériques, métriques par unité, sources et vérifications sont dans mouse410.json et mouse415.json.', '']
    (root/'rapport.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
