"""Comparaison appariée de deux durées, avec cartes par unité et PCA–CCA.

Les PKL doivent être de confiance. Aucun entraînement ni changement du moteur.
PCA exacte centrée par unité, dix composantes non blanchies, CCA descriptive.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compute_curbd_currents_mouse410 import load_pickle_compatible


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x-x.mean(), y-y.mean()
    den = np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.clip(np.sum(x*y)/den, -1, 1)) if den else None


def pca_scores(current, fit, k=10):
    ### On apprend le centrage et les axes uniquement sur les temps demandés.
    mean = current[:, fit].mean(axis=1, keepdims=True)
    z = current[:, fit]-mean
    values, vectors = np.linalg.eigh(z@z.T)
    order = np.argsort(values)[::-1][:k]
    if len(order) != k or values[order[-1]] <= values[order[0]]*1e-12:
        raise ValueError('PCA de rang insuffisant pour dix composantes')
    scores = (current-mean).T@vectors[:, order]
    return scores, float(values[order].sum()/values.clip(0).sum())


def cca(x, y, xv=None, yv=None):
    mx, my = x.mean(0), y.mean(0)
    qx, rx = np.linalg.qr(x-mx, mode='reduced')
    qy, ry = np.linalg.qr(y-my, mode='reduced')
    if np.linalg.matrix_rank(rx) < rx.shape[0] or np.linalg.matrix_rank(ry) < ry.shape[0]:
        raise ValueError('CCA de rang insuffisant')
    u, s, vh = np.linalg.svd(qx.T@qy, full_matrices=False)
    test = None
    if xv is not None:
        ax = (xv-mx)@np.linalg.solve(rx, u[:, 0])
        by = (yv-my)@np.linalg.solve(ry, vh.T[:, 0])
        test = correlation(ax, by)
    return s.clip(0, 1).tolist(), test


def validate_pair(a, b):
    checked = []
    for key in ['Adata', 'tData', 'tRNN', 'masque_sub', 'J0', 'inputWN', 'initial_state', 'iTarget', 'activity_scale']:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        checked.append(key)
    for key in ['parameters', 'model_parameters']:
        aa = {k:v for k,v in a[key].items() if k not in ['nRunTrain', 'nRunTot']}
        bb = {k:v for k,v in b[key].items() if k not in ['nRunTrain', 'nRunTot']}
        if aa != bb:
            raise ValueError(f'{key} diffèrent au-delà de la durée')
        checked.append(key+' hors durée')
    for key in ['timeseries_sha256', 'source_sha256', 'packages', 'threads']:
        if a['reproducibility'][key] != b['reproducibility'][key]:
            raise ValueError(f'Provenance différente : {key}')
        checked.append(key)
    if list(a['regions'][:,0]) != list(b['regions'][:,0]):
        raise ValueError('Noms anatomiques différents')
    for x,y in zip(a['regions'], b['regions']):
        np.testing.assert_array_equal(x[1], y[1])
    n = len(a['RNN_final'])
    ids = np.concatenate([np.asarray(x[1],int) for x in a['regions']])
    np.testing.assert_array_equal(np.sort(ids), np.arange(n))
    for d in [a,b]:
        for key in ['J_final','RNN_final','Adata','tData','tRNN','pVar','chi2']:
            if not np.isfinite(d[key]).all():
                raise ValueError(f'Valeurs non finies : {key}')
        if d['J_final'].shape != (n,n) or d['RNN_final'].shape != (n,len(d['tRNN'])):
            raise ValueError('Dimensions du modèle incompatibles')
        if not np.all(np.diff(d['tRNN'])>0):
            raise ValueError('Temps non croissants')
    checked.append('partition et identité des unités ; matrices et temps finis')
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a',type=Path)
    parser.add_argument('b',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-b-sha256')
    args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    hashes=[sha256(p) for p in [args.a,args.b]]
    if args.expected_b_sha256 and hashes[1]!=args.expected_b_sha256:
        raise ValueError('Empreinte locale différente de Narval')
    ds=[load_pickle_compatible(p) for p in [args.a,args.b]]
    a,b=ds
    checks=validate_pair(a,b)
    print('Appariement vérifié',flush=True)
    t=np.asarray(a['tRNN']);all_times=np.ones(len(t),bool)
    fit=t<230;test=t>=250;window=(t>=60)&(t<120)
    if not all(x.any() for x in [fit,test,window]):
        raise ValueError('Session trop courte pour les fenêtres prédéfinies')
    names=[str(r[0]) for r in a['regions']]
    recon=[]
    for d in ds:
        idx=np.array([np.abs(t-time).argmin() for time in d['tData']])
        r=np.asarray(d['RNN_final'][:,idx],float);target=np.asarray(d['Adata'],float)
        pvar=1-np.mean((r-target)**2)/np.var(target)
        if abs(pvar-d['row']['pVar_finale'])>1e-5:
            raise ValueError('pVar recalculée différente')
        regions=[]
        for name,ids in d['regions']:
            ids=np.asarray(ids,int);x=target[ids];y=r[ids]
            regions.append(dict(region=str(name),pvar=float(1-np.mean((y-x)**2)/np.var(x)),
                                derivative_r=correlation(np.diff(x,axis=1),np.diff(y,axis=1))))
        recon.append(dict(nRunTrain=d['parameters']['nRunTrain'],pvar=float(pvar),regions=regions,
                          runtime_sec=d['row']['runtime_sec']))
    nfirst=a['parameters']['nRunTrain']
    prefix_delta=float(np.max(np.abs(a['pVar'][:nfirst]-b['pVar'][:nfirst])))
    rows=[];features=[{},{}];maps={'time':t[window]};orders=[]
    for ti,(target,ix) in enumerate(a['regions']):
        ix=np.asarray(ix,int)
        order=np.argsort(np.argmax(a['RNN_final'][ix][:,window],axis=1),kind='stable')
        orders.append(order.tolist())
        for di,d in enumerate(ds):maps[f'activity_{di}_{ti}']=d['RNN_final'][ix][:,window]
        for si,(source,iy) in enumerate(a['regions']):
            iy=np.asarray(iy,int);cs=[];scores=[];variances=[];norms=[]
            for di,d in enumerate(ds):
                c=np.asarray(d['J_final'][np.ix_(ix,iy)],float)@np.asarray(d['RNN_final'][iy],float)
                cs.append(c)
                score,variance=pca_scores(c,all_times)
                scores.append(score);variances.append(variance);norms.append(np.linalg.norm(score,axis=1))
                split,_=pca_scores(c,fit)
                features[di][ti,si]=split
                maps[f'current_{di}_{ti}_{si}']=c[:,window].astype(np.float32)
            cc,_=cca(*scores)
            uc=[c-c.mean(axis=1,keepdims=True) for c in cs]
            denom=np.linalg.norm(uc[0],axis=1)*np.linalg.norm(uc[1],axis=1)
            unit_r=np.divide(np.sum(uc[0]*uc[1],axis=1),denom,out=np.full(len(ix),np.nan),where=denom>0)
            rows.append(dict(target=str(target),source=str(source),target_index=ti,source_index=si,
                             pca_norm_r=correlation(*norms),pca_norm_ratio=float(norms[1].mean()/norms[0].mean()),
                             cca=cc,pca10_variance=variances,signed_r=correlation(*[c.sum(0) for c in cs]),
                             unit_r_median=float(np.nanmedian(unit_r)),
                             unit_r=[float(v) if np.isfinite(v) else None for v in unit_r],
                             centered_rms_ratio=float(np.linalg.norm(uc[1])/np.linalg.norm(uc[0])),
                             raw_rms_ratio=float(np.linalg.norm(cs[1])/np.linalg.norm(cs[0]))))
            print(f'Courants {ti+1}/6, source {si+1}/6',flush=True)
    controls=[]
    for ti in range(len(names)):
        for si in range(len(names)):
            x=features[0][ti,si];matches=[]
            for sj in range(len(names)):
                y=features[1][ti,sj];_,held=cca(x[fit],y[fit],x[test],y[test])
                matches.append(dict(source=sj,test_cca=held,norm_r=correlation(np.linalg.norm(x[test],axis=1),np.linalg.norm(y[test],axis=1))))
            own=matches[si];wrong=[m for m in matches if m['source']!=si]
            controls.append(dict(target=ti,source=si,matches=matches,own=own,
                                 cca_margin=own['test_cca']-max(m['test_cca'] for m in wrong),
                                 norm_margin=own['norm_r']-max(m['norm_r'] for m in wrong)))
    meta=dict(files=[str(args.a.resolve()),str(args.b.resolve())],sha256=hashes,
              analysis_sha256=sha256(Path(__file__)),checks=checks,reconstruction=recon,
              training_prefix_pvar_max_abs_difference=prefix_delta,region_names=names,
              sizes=[len(x[1]) for x in a['regions']],orders=orders,comparisons=rows,controls=controls,
              method='PCA exacte par unité, centrage temporel, 10 scores non blanchis ; CCA QR/SVD. Contrôle PCA/CCA appris avant 230 s et évalué après 250 s. Même convention que les analyses historiques.',
              limitations='Une souris, une graine ; pas de test de stabilité entre graines à 300. Réseau entraîné sur toute la session. CCA1 ne garantit ni signe ni amplitude. Pas de preuve biologique ou causale ; pas de reproduction vérifiée ligne par ligne des scripts des auteurs.')
    (out/'comparison.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    maps['pvar_a']=a['pVar'];maps['pvar_b']=b['pVar']
    np.savez_compressed(out/'maps_and_learning.npz',**maps)
    print('Analyse sauvegardée',flush=True)


if __name__=='__main__':
    main()
