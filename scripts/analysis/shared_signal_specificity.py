"""Sensibilité de la spécificité inter-graines à une projection du signal commun.

Régression, PCA et CCA ajustées avant 230 s ; évaluation après 250 s.
Diagnostic sur modèles déjà entraînés sur la session entière, sans nouvel entraînement.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from compare_training_durations import pca_scores, correlation, sha256
from compute_curbd_currents_mouse410 import load_pickle_compatible


def residualize(current,g,fit):
    # Chaque unité garde son propre coefficient. Le bloc test ne choisit rien.
    mean=current[:,fit].mean(1,keepdims=True)
    gm=g[fit].mean(); gc=g-gm
    beta=((current[:,fit]-mean)@gc[fit])/(gc[fit]@gc[fit])
    return current-mean-beta[:,None]*gc[None,:],beta,mean,gm


def cca_test(x,y,fit,test):
    mx=x[fit].mean(0);my=y[fit].mean(0)
    qx,rx=np.linalg.qr(x[fit]-mx,mode='reduced');qy,ry=np.linalg.qr(y[fit]-my,mode='reduced')
    if np.linalg.matrix_rank(rx)<10 or np.linalg.matrix_rank(ry)<10:raise ValueError('CCA rang insuffisant')
    u,s,vh=np.linalg.svd(qx.T@qy,full_matrices=False)
    a=(x[test]-mx)@np.linalg.solve(rx,u);b=(y[test]-my)@np.linalg.solve(ry,vh.T)
    return [correlation(a[:,i],b[:,i]) for i in range(10)]


def analyze(root,out,mouse):
    features={};energy=[];provenance=[];reference=None
    for seed in [2026,2027,2028]:
        paths=list(root.glob(f'*mouse{mouse}_*seed{seed}_*.pkl'));assert len(paths)==1
        path=paths[0];d=load_pickle_compatible(path)
        if reference is not None:
            for key in ['Adata','tData','tRNN','masque_sub']:np.testing.assert_array_equal(d[key],reference[key])
            for a,b in zip(d['regions'],reference['regions']):
                assert a[0]==b[0];np.testing.assert_array_equal(a[1],b[1])
            for key in ['timeseries_sha256','source_sha256','packages','threads']:assert d['reproducibility'][key]==reference['reproducibility'][key]
        else:reference={k:d[k] for k in ['Adata','tData','tRNN','masque_sub','regions','reproducibility']}
        t=np.asarray(d['tRNN'],float);fit=t<230;test=t>=250
        means=np.stack([np.asarray(d['Adata'][np.asarray(ids,int)],float).mean(0) for _,ids in d['regions']])
        g=np.interp(t,np.asarray(d['tData'],float),means.mean(0))
        j=np.asarray(d['J_final'],float);r=np.asarray(d['RNN_final'],float)
        assert np.isfinite(j).all() and np.isfinite(r).all()
        for ti,(_,ix) in enumerate(d['regions']):
            ix=np.asarray(ix,int)
            for si,(_,iy) in enumerate(d['regions']):
                iy=np.asarray(iy,int);raw=j[np.ix_(ix,iy)]@r[iy]
                residual,beta,mu,gm=residualize(raw,g,fit)
                record=dict(seed=seed,target=ti,source=si)
                for mode,c in [('original',raw),('residual',residual)]:
                    scores,variance=pca_scores(c,fit)
                    features[seed,ti,si,mode]=scores
                    record[mode+'_pca10_fit_variance']=variance
                for label,select in [('fit',fit),('test',test)]:
                    a=raw[:,select]-raw[:,select].mean(1,keepdims=True)
                    b=residual[:,select]-residual[:,select].mean(1,keepdims=True)
                    record[label+'_remaining_variance']=float(np.sum(b*b)/np.sum(a*a))
                energy.append(record)
            print(f'{mouse} graine {seed} cible {ti+1}/6 : PCA originale et résiduelle',flush=True)
        provenance.append(dict(path=str(path.resolve()),sha256=sha256(path)))
        del d,j,r
    rows=[]
    for seed_a,seed_b in itertools.combinations([2026,2027,2028],2):
        for mode in ['original','residual']:
            for ti in range(6):
                for si in range(6):
                    x=features[seed_a,ti,si,mode];matches=[]
                    for sj in range(6):
                        y=features[seed_b,ti,sj,mode];cc=cca_test(x,y,fit,test)
                        matches.append(dict(source=sj,cca_test=cc,cca1=cc[0],cca2_10=float(np.mean(cc[1:])),cca_mean10=float(np.mean(cc)),norm=correlation(np.linalg.norm(x[test],axis=1),np.linalg.norm(y[test],axis=1))))
                    same=matches[si];record=dict(seeds=[seed_a,seed_b],mode=mode,target=ti,source=si,matches=matches)
                    for metric in ['cca1','cca2_10','cca_mean10','norm']:
                        other=max(m[metric] for m in matches if m['source']!=si)
                        record[metric]=same[metric];record[metric+'_margin']=same[metric]-other;record[metric+'_wins']=same[metric]>other
                    rows.append(record)
            print(f'{mouse} {seed_a}/{seed_b} {mode} comparé',flush=True)
    result=dict(mouse=mouse,names=[str(v[0]) for v in reference['regions']],provenance=provenance,
                fit='tRNN < 230 s',test='tRNN >= 250 s',global_signal='Moyenne à poids égaux des six moyennes régionales Adata ; interpolation linéaire sur tRNN',energy=energy,comparisons=rows)
    (out/f'mouse{mouse}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for mouse in [410,415]:analyze(a.root,a.output,mouse)
