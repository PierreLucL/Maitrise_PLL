"""Rapport descriptif des pilotes tt/tt-1 ; mêmes échelles pour les deux souris."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/comparison_indices_20260915')
def main():
    ds=[json.loads((ROOT/f'mouse{m}/comparison.json').read_text()) for m in [410,415]]
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    summary=[]
    for i,(mouse,d) in enumerate(zip([410,415],ds)):
        rows=d['comparisons'];controls=d['controls']
        matrices=[np.array([r['pca_norm_r'] for r in rows]).reshape(6,6),np.array([np.mean(r['cca']) for r in rows]).reshape(6,6),np.array([r['centered_rms_ratio'] for r in rows]).reshape(6,6)]
        for j,(mat,title) in enumerate(zip(matrices,['Corrélation des normes PCA10','Moyenne des 10 CCA · session','Amplitude RMS centrée · tt-1 / tt'])):
            ax=axes[i,j]
            if j<2: im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis')
            else:
                from matplotlib.colors import TwoSlopeNorm
                im=ax.imshow(mat,cmap='coolwarm',norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2))
            for y in range(6):
                for x in range(6): ax.text(x,y,f'{mat[y,x]:.2f}',ha='center',va='center',fontsize=9,color='white' if j<2 and mat[y,x]<.65 else 'black')
            names=[x.replace('Rég. ','') for x in d['region_names']]
            ax.set_xticks(range(6),names,rotation=45,ha='right',fontsize=8);ax.set_yticks(range(6),names,fontsize=8)
            ax.set_title(f'Souris {mouse} · {title}',fontsize=11);ax.set_xlabel('Région source');ax.set_ylabel('Région cible');fig.colorbar(im,ax=ax,shrink=.7)
        p=[r['pvar'] for r in d['reconstruction']]
        s=dict(mouse=mouse,pvar=p,gain_percentage_points=100*(p[1]-p[0]),residual_mse_reduction=1-(1-p[1])/(1-p[0]),norm_correlation_median=float(np.median(matrices[0])),cca10_median=float(np.median(matrices[1])),centered_rms_ratio_median=float(np.median(matrices[2])),centered_rms_ratio_range=[float(matrices[2].min()),float(matrices[2].max())],unit_r_median=float(np.median([r['unit_r_median'] for r in rows])),held_cca2_10_median=float(np.median([c['own']['test_cca_2_10'] for c in controls])),source_wins_cca2_10=sum(c['cca_2_10_margin']>0 for c in controls),source_margin_median=float(np.median([c['cca_2_10_margin'] for c in controls])))
        summary.append(s)
    fig.suptitle('Effet du changement d’indice tt → tt-1 · 100 passages · graine 2026\n36 courants régionaux par souris ; valeurs descriptives, une graine par version',fontsize=15)
    fig.savefig(ROOT/'comparaison_courants.png',dpi=160);fig.savefig(ROOT/'comparaison_courants.pdf');plt.close(fig)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Comparaison appariée tt / tt-1 — 15 septembre 2026','', 'Deux souris, même graine2026 et100 passages. Données, unités, initialisations, bruit, paramètres, versions et threads vérifiés. Seule modification du moteur : r_slice utilise tt-1. Cibles normalisées sauvegardées, sans nouveau lissage.','', '| Souris | pVar tt | pVar tt-1 | Réduction erreur quadratique | r norme PCA10 médian | CCA10 moyenne médiane | Ratio RMS centrée médian (étendue) |', '|---|---:|---:|---:|---:|---:|---|']
    for s in summary:
        lines.append(f"| {s['mouse']} | {s['pvar'][0]:.6f} | {s['pvar'][1]:.6f} | {s['residual_mse_reduction']:.1%} | {s['norm_correlation_median']:.3f} | {s['cca10_median']:.3f} | {s['centered_rms_ratio_median']:.3f} ({s['centered_rms_ratio_range'][0]:.3f}–{s['centered_rms_ratio_range'][1]:.3f}) |")
    lines += ['', '## Contrôle temporel et spécificité', 'PCA et CCA apprises avant230s, évaluées après250s, signes et axes figés. Le RNN a été entraîné sur toute la session : ce découpage ne constitue pas une validation du RNN sur de nouvelles données.']
    for s in summary: lines.append(f"- {s['mouse']} : CCA2–10 test moyenne médiane {s['held_cca2_10_median']:.3f} ; même source classée première {s['source_wins_cca2_10']}/36 ; marge médiane {s['source_margin_median']:+.3f}.")
    lines += ['', '## Portée', 'Comparaison entre versions à une graine par souris, pas une mesure de reproductibilité inter-graines de tt-1. CCA autorise un changement de base et ne garantit ni amplitude ni signe des courants par unité. Les rapports RMS centrés mesurent séparément leur amplitude. Ne pas mélanger des versions dans une comparaison d’âge ; une différence de méthode pourrait devenir une différence biologique apparente.', '', 'Traces : six unités par souris, chacune proche de la pVar régionale médiane du modèle tt, sans sélection sur le gain. Même échelle par ligne. Fenêtres60–120s et zoom75–90s. Ce sont des exemples, pas toutes les unités. Cartes des courants par unité conservées dans maps_and_learning.npz ; aucune somme signée utilisée comme métrique principale.']
    (ROOT/'bilan.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
