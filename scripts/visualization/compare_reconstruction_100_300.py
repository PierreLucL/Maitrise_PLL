"""Traces appariées : cible sauvegardée, RNN 100 et RNN 300 passages.

Une unité par région, choisie près de la pVar médiane à 100 sur toute la
session, avant examen du gain à 300. Aucun lissage ou z-score ajouté.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
from compute_curbd_currents_mouse410 import load_pickle_compatible


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir',type=Path,default=Path('results/comparison_train100_300_410_2026'))
    args=parser.parse_args();out=args.comparison_dir
    comparison=json.loads((out/'comparison.json').read_text())
    a,b=[load_pickle_compatible(Path(p)) for p in comparison['files']]
    for key in ['Adata','tData','tRNN','masque_sub']:
        np.testing.assert_array_equal(a[key],b[key])
    for ra,rb in zip(a['regions'],b['regions']):
        assert ra[0]==rb[0]
        np.testing.assert_array_equal(ra[1],rb[1])
    t=np.asarray(a['tData'],float);tr=np.asarray(a['tRNN'],float)
    idx=np.array([np.abs(tr-v).argmin() for v in t])
    mismatch=float(np.max(np.abs(tr[idx]-t)))
    assert mismatch <= np.median(np.diff(tr))/2+1e-5
    target=np.asarray(a['Adata'],float)
    preds=[np.asarray(d['RNN_final'][:,idx],float) for d in [a,b]]
    assert all(p.shape==target.shape and np.isfinite(p).all() for p in preds)
    variance=np.var(target,axis=1)
    scores=[np.divide(np.mean((p-target)**2,axis=1),variance,out=np.full(len(target),np.nan),where=variance>0) for p in preds]
    scores=[1-s for s in scores]
    selected=[]
    for name,ids in a['regions']:
        ids=np.asarray(ids,int);valid=ids[np.isfinite(scores[0][ids])]
        if not len(valid):raise ValueError(f'Aucune unité variable dans {name}')
        med=float(np.median(scores[0][valid]));unit=int(valid[np.argmin(np.abs(scores[0][valid]-med))])
        selected.append(dict(region=str(name),unit=unit,median_unit_pvar100=med,pvar100=float(scores[0][unit]),pvar300=float(scores[1][unit])))
    full=(t>=60)&(t<120);zoom=(t>=75)&(t<90)
    if not full.any() or not zoom.any():raise ValueError('Fenêtres hors session')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    colors=['#2376b7','#d65f19'];target_color='#333333'
    fig,axes=plt.subplots(len(selected),3,figsize=(17,13),layout='constrained',gridspec_kw={'width_ratios':[1.25,1.25,1]})
    for row,item in enumerate(selected):
        unit=item['unit'];y=target[unit]
        low=min(y[full].min(),*(p[unit,full].min() for p in preds));high=max(y[full].max(),*(p[unit,full].max() for p in preds));pad=max((high-low)*.08,1e-4)
        for col in range(3):
            ax=axes[row,col];keep=zoom if col==2 else full
            ax.plot(t[keep],y[keep],color=target_color,lw=1.8,alpha=.65,zorder=1)
            for di in ([0,1] if col==2 else [col]):
                ax.plot(t[keep],preds[di][unit,keep],color=colors[di],lw=1.2,ls='--' if di==0 else '-',zorder=2+di)
            ax.set_ylim(low-pad,high+pad)
            ax.set_xlim((75,90) if col==2 else (60,120))
            ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
            ax.set_xticks([75,80,85,90] if col==2 else [60,75,90,105,120])
            if col==0:ax.set_ylabel(f"{item['region'].replace('Rég. ','')} · unité {unit}\nActivité normalisée",fontsize=10)
            if col in [0,1]:
                ax.text(.015,.95,f"pVar unité, session : {item['pvar100' if col==0 else 'pvar300']:.3f}",transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=2))
            if row==len(selected)-1:ax.set_xlabel('Temps (s)')
            else:ax.tick_params(labelbottom=False)
    for ax,title in zip(axes[0],['100 passages + cible','300 passages + cible','Superposition · zoom 75–90 s']):ax.set_title(title,pad=28,fontsize=12)
    fig.suptitle('Reconstruction des mêmes unités · souris 410 · graine 2026\nUne unité proche de la pVar médiane à 100 par région · mêmes échelles sur chaque ligne',fontsize=15)
    handles=[Line2D([],[],color=target_color,lw=2,label='Cible traitée sauvegardée (Adata)'),Line2D([],[],color=colors[0],ls='--',lw=1.5,label='RNN · 100 passages'),Line2D([],[],color=colors[1],lw=1.5,label='RNN · 300 passages')]
    # Légende dans la marge au-dessus de la première ligne, sans masquer les traces.
    axes[0,1].legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,fontsize=8,frameon=False)
    fig.savefig(out/'reconstruction_traces_100_300.png',dpi=180)
    fig.savefig(out/'reconstruction_traces_100_300.pdf')
    plt.close(fig)
    metadata=dict(files=comparison['files'],sha256=comparison['sha256'],selection='Unité la plus proche de la pVar médiane de sa région à 100, calculée sur toute la session ; égalités départagées par ordre des unités. Aucun critère basé sur le gain à 300.',
                  units=selected,window=[60,120],zoom=[75,90],target='Adata : cible traitée et normalisée sauvegardée ; ne représente pas le signal avant lissage.',
                  grids='RNN échantillonné au temps le plus proche sur tData, sans troncature ni lissage supplémentaire.',max_time_mismatch=mismatch,
                  scaling='Valeurs internes du modèle ; échelle commune aux trois panneaux de chaque ligne, échelles distinctes entre régions.',
                  caveat='Exemples de six unités, pas une preuve de performance pour toutes les unités ni une validation hors entraînement.')
    (out/'reconstruction_traces_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(selected,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
