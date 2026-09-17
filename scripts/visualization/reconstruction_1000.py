"""Reconstruction à 1000 passages : une unité médiane par région et souris."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from compute_curbd_currents_mouse410 import load_pickle_compatible
from compare_training_durations import sha256

root=Path('results/narval_long_npixels25_g08_sigma4_410_415_ntrain1000/1901587')
out=Path('results/reconstruction_1000_1901587');out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(6,4,figsize=(18,12),layout='constrained',gridspec_kw={'width_ratios':[1.5,1,1.5,1]})
reports=[]
for mi,mouse in enumerate([410,415]):
    paths=list(root.glob(f'*mouse{mouse}_*.pkl'));assert len(paths)==1
    p=paths[0];d=load_pickle_compatible(p)
    t=np.asarray(d['tData'],float);tr=np.asarray(d['tRNN'],float)
    idx=np.array([np.abs(tr-v).argmin() for v in t]);delta=np.max(np.abs(tr[idx]-t))
    assert delta<=np.median(np.diff(tr))/2+1e-5
    a=np.asarray(d['Adata'],float);r=np.asarray(d['RNN_final'][:,idx],float)
    assert a.shape==r.shape and np.isfinite(a).all() and np.isfinite(r).all()
    score=1-np.mean((a-r)**2)/np.var(a);assert abs(score-float(d['row']['pVar_finale']))<1e-5
    var=np.var(a,axis=1);unit=1-np.divide(np.mean((a-r)**2,axis=1),var,out=np.full(len(a),np.nan),where=var>0)
    finite=unit[np.isfinite(unit)]
    report=dict(mouse=mouse,file=str(p.resolve()),sha256=sha256(p),pvar=score,units=len(a),unit_pvar_percentiles=np.percentile(finite,[5,25,50,75,95]).tolist(),fraction_units_below_05=float(np.mean(finite<.5)),max_time_mismatch=float(delta),selected=[])
    color=['#1765ad','#d77817'][mi];full=(t>=60)&(t<120);zoom=(t>=75)&(t<90)
    for row,(name,ids) in enumerate(d['regions']):
        ids=np.asarray(ids,int);valid=ids[np.isfinite(unit[ids])];median=np.median(unit[valid]);i=int(valid[np.argmin(np.abs(unit[valid]-median))])
        report['selected'].append(dict(region=str(name),unit=i,pvar=float(unit[i]),region_median=float(median)))
        low=min(a[i,full].min(),r[i,full].min());high=max(a[i,full].max(),r[i,full].max());pad=max((high-low)*.12,1e-4)
        for sub,keep in enumerate([full,zoom]):
            ax=axes[row,mi*2+sub];ax.plot(t[keep],a[i,keep],color='#333333',alpha=.7,lw=1.8);ax.plot(t[keep],r[i,keep],color=color,lw=1.1,ls='--')
            ax.set_xlim((60,120) if sub==0 else (75,90));ax.set_ylim(low-pad,high+pad);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
            if sub==0:ax.set_ylabel(str(name).replace('Rég. ','')+'\nActivité normalisée',fontsize=9);ax.text(.02,.96,f'Unité {i} · pVar session {unit[i]:.3f}',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
            if row==5:ax.set_xlabel('Temps (s)')
            else:ax.tick_params(labelbottom=False)
            if row==0:ax.set_title(f'Souris {mouse} · '+('60–120 s' if sub==0 else 'zoom 75–90 s'),fontsize=12,pad=30)
    reports.append(report)
fig.suptitle('Reconstruction à 1 000 passages · résolution cible 25 pixels · σ = 4\nUne unité proche de la pVar médiane par région · échelle commune extrait/zoom pour chaque unité',fontsize=14)
handles=[Line2D([],[],color='#333333',lw=2,label='Cible traitée sauvegardée'),Line2D([],[],color='#1765ad',ls='--',label='RNN 410'),Line2D([],[],color='#d77817',ls='--',label='RNN 415')]
axes[0,1].legend(handles=handles,loc='lower center',bbox_to_anchor=(1,1.02),ncol=3,fontsize=9,frameon=False)
fig.savefig(out/'traces_1000.png',dpi=180);fig.savefig(out/'traces_1000.pdf');plt.close(fig)
metadata=dict(models=reports,selection='Unité la plus proche de la pVar médiane par région sur la session entière ; aucune sélection du meilleur exemple',target='Adata prétraitée et normalisée, sigma4 ; aucun lissage ajouté',window=[60,120],zoom=[75,90],caveat='Reconstruction sur données apprises. Unités et échelles distinctes entre souris. Anciens runs sans provenance complète des graines ; comparaison non appariée avec les runs récents à pixels15.')
(out/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False))
print(json.dumps(reports,indent=2,ensure_ascii=False))
