"""Rapport descriptif du contrôle partagé, avec comparaison appariée des modes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_training_durations import sha256


def report(out):
    ds=[json.loads((out/f'mouse{m}.json').read_text()) for m in [410,415]]
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    lines=['# Spécificité après projection du signal partagé — 14 septembre 2026','','## Méthode et portée','','Six modèles historiques à 100 passages (410/415, graines 2026–2028). Le prédicteur commun est la moyenne à poids égaux des six moyennes régionales de Adata, interpolée linéairement sur tRNN. Il inclut les régions source et cible. Pour chaque unité cible de chaque courant J[cible,source] @ RNN[source], l’ordonnée et la pente de la projection sont ajustées uniquement pour tRNN < 230 s. Les coefficients restent figés ensuite. Aucun modèle n’est réentraîné et aucune GSR n’est ajoutée au prétraitement.','','PCA exacte à dix composantes non blanchies, centrage et axes ajustés sur le même premier bloc. CCA ajustée sur ce bloc puis corrélations des dix paires canoniques évaluées après 250 s, sans réordonner les dimensions ni réorienter leurs signes au test. La moyenne des dimensions 2–10 est complémentaire de CCA1. Pour une cible et une source dans la graine A, la même source dans B doit dépasser chacune des cinq autres sources pour gagner. Les trois paires de graines sont 2026/2027, 2026/2028 et 2027/2028, dans cet ordre.','','108 comparaisons dépendantes par souris et par mode. Ce sont des descriptions, pas des répétitions biologiques indépendantes ni des p-values. Le RNN a appris toute la session : seul le transfert temporel des transformations est évalué. La composante projetée peut contenir un signal biologique et le résidu ne constitue pas une vérité terrain régionale. Le contrôle original est recalculé dans la même implémentation pour isoler l’effet de projection ; les décalages temporels historiques ne sont pas recalculés.','','## Résultats','','| Souris | Mode | CCA1 médiane test | Bonne source CCA1 | Marge CCA1 médiane | CCA2–10 médiane | Bonne source CCA2–10 | Bonne source norme |','|---|---|---:|---:|---:|---:|---:|---:|']
    summary=[]
    for col,d in enumerate(ds):
        stats={}
        for mode,color,label in [('original','#1765ad','Courants originaux'),('residual','#d77817','Après projection')]:
            rows=[r for r in d['comparisons'] if r['mode']==mode];assert len(rows)==108
            cc=np.array([r['matches'][r['source']]['cca_test'] for r in rows]);profile=np.median(cc,axis=0)
            axes[0,col].plot(range(1,11),profile,'o-',color=color,label=label)
            axes[0,col].fill_between(range(1,11),np.percentile(cc,25,axis=0),np.percentile(cc,75,axis=0),color=color,alpha=.10)
            stats[mode]={metric:float(np.median([r[metric] for r in rows])) for metric in ['cca1','cca2_10','cca_mean10','norm','cca1_margin','cca2_10_margin']}
            for metric in ['cca1','cca2_10','cca_mean10','norm']:stats[mode][metric+'_wins']=sum(r[metric+'_wins'] for r in rows)
            s=stats[mode];lines.append(f'| {d["mouse"]} | {label} | {s["cca1"]:.3f} | {s["cca1_wins"]}/108 | {s["cca1_margin"]:+.4f} | {s["cca2_10"]:.3f} | {s["cca2_10_wins"]}/108 | {s["norm_wins"]}/108 |')
            x=np.arange(3)+(-.19 if mode=='original' else .19);v=[s[m+'_wins'] for m in ['cca1','cca2_10','norm']]
            bars=axes[1,col].bar(x,v,width=.36,color=color,label=label)
            axes[1,col].bar_label(bars,fontsize=10,padding=3)
        axes[0,col].set(title=f'Souris {d["mouse"]} — même source entre graines',xlabel='Dimension canonique (ordre fixé au premier bloc)',ylabel='Corrélation sur le second bloc',xticks=range(1,11),ylim=(-.05,1.05));axes[0,col].grid(alpha=.2)
        axes[1,col].set(xticks=range(3),xticklabels=['CCA1','Moyenne CCA2–10','Norme PCA10'],ylim=(0,120),ylabel='Bonne source strictement première / 108',title='Reconnaissance de la région source')
        axes[0,col].legend(fontsize=9)
        e=np.array([r['test_remaining_variance'] for r in d['energy']]);stats['remaining_test_variance_median']=float(np.median(e));stats['mouse']=d['mouse'];summary.append(stats)
    fig.suptitle('Signal partagé : reproductibilité et spécificité des courants\nAjustement avant 230 s · évaluation après 250 s · médiane et intervalle interquartile',fontsize=13)
    fig.savefig(out/'specificite_signal_partage.png',dpi=180);fig.savefig(out/'specificite_signal_partage.pdf');plt.close(fig)
    lines+=['','## Variabilité résiduelle','','Le rapport ci-dessous compare les variances temporelles résiduelles/originales dans le bloc test, centrées dans ce bloc uniquement pour calculer la variance. La pente de projection reste apprise dans le premier bloc. Un rapport supérieur à 1 serait possible hors bloc d’ajustement.']
    for s in summary:lines.append(f'- Souris {s["mouse"]} : variance résiduelle médiane = {s["remaining_test_variance_median"]:.1%} de l’originale (36 contributions × 3 graines).')
    lines+=['','## Validation et provenance','','Tests : retrait exact d’un signal linéaire partagé ; changement massif du bloc test sans modification des coefficients appris ; transfert CCA sous mélange inversible ; conservation des corrélations négatives au test. Identité des cibles, temps, segmentations, régions et provenance entre graines vérifiée. Les empreintes des six PKL sont dans les JSON de résultats. Les scripts et fonctions PCA/corrélation réutilisées sont identifiés dans provenance.json.','','Les JSON contiennent chaque comparaison aux six sources, les dix corrélations canoniques, les marges et les fractions de variance conservées par PCA10 dans le premier bloc. Aucun résultat biologique sur l’âge ne découle directement de ce contrôle.']
    lines+=['','## Interprétation actualisée','','Le contrôle original retrouve les nombres historiques : CCA1 89/108 chez 410 et 29/108 chez 415 ; norme 38/108 et 67/108. La nouvelle analyse des dimensions suivantes précise la conclusion : avec la moyenne CCA2–10, la bonne source gagne 108/108 chez 410 et 107/108 chez 415 avant projection. Les marges médianes sont respectivement +0,132 et +0,144. La moyenne des dix dimensions donne les mêmes nombres gagnants. La spécificité ambiguë de CCA1 ne doit donc plus être généralisée à toute la représentation multidimensionnelle.','','La projection porte les classements CCA1 à 100/108 et 65/108. Elle ne procure pas un gain général : CCA2–10 reste à 108/108 et passe à 106/108, et la corrélation des normes chez 410 baisse de 0,826 à 0,394. Les sous-espaces demeurent alignables après projection, mais leurs normes ne deviennent pas uniformément plus reproductibles. Aucun de ces résultats ne justifie automatiquement de retirer le signal commun du pipeline.','','Pour la maîtrise, conserver la représentation originale et rapporter plusieurs dimensions canoniques, leurs marges de spécificité et les amplitudes séparément. La moyenne CCA2–10 constitue ici un diagnostic exploratoire, pas une nouvelle mesure biologique validée ni un seuil à optimiser. Les axes appris pour chaque paire ne définissent pas encore une caractéristique directement comparable entre sessions spontanées non alignées. La comparaison longitudinale doit encore tenir compte des acquisitions, couvertures et états comportementaux. Les contrôles Narval à 300 passages et tt-1 restent complémentaires.']
    (out/'bilan.md').write_text('\n'.join(lines)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2))
    paths=[Path('scripts/analysis')/n for n in ['shared_signal_specificity.py','report_shared_signal_specificity.py','compare_training_durations.py','compute_curbd_currents_mouse410.py']]
    (out/'provenance.json').write_text(json.dumps({str(p):sha256(p) for p in paths},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();report(a.output)
