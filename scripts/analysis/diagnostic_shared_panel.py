"""Diagnostic descriptif du signal partagé et contrôle visuel du panel, sans entraînement."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from compute_curbd_currents_mouse410 import load_pickle_compatible
from maitrise_curbd.masks import reduce_atlas_to_6_regions, remove_thin_label_artifacts


def shared_fraction(x, g):
    """Projection temporelle par unité, puis fraction de variance agrégée (pas somme signée)."""
    z = np.asarray(x, float)
    z = z-z.mean(axis=1, keepdims=True)
    g = g-g.mean()
    total = np.sum(z*z)
    if total == 0 or g@g == 0:
        raise ValueError('Signal constant')
    per_unit = (z@g)**2/(g@g)
    return float(per_unit.sum()/total)


def currents(root, out):
    rows=[]
    for p in sorted(root.glob('*.pkl')):
        d=load_pickle_compatible(p)
        names=[str(r[0]) for r in d['regions']]
        data=np.asarray(d['Adata'],float)
        means=np.stack([data[np.asarray(ids,int)].mean(0) for _,ids in d['regions']])
        # Chaque région contribue également au prédicteur, sans normaliser sa variance.
        g=means.mean(0)
        times=np.asarray(d['tRNN']); td=np.asarray(d['tData'])
        idx=np.array([np.abs(times-v).argmin() for v in td])
        r=np.asarray(d['RNN_final'][:,idx],float)
        j=np.asarray(d['J_final'],float)
        fractions=np.zeros((6,6))
        for ti,(_,ix) in enumerate(d['regions']):
            ix=np.asarray(ix,int)
            for si,(_,iy) in enumerate(d['regions']):
                iy=np.asarray(iy,int)
                c=j[np.ix_(ix,iy)]@r[iy]
                fractions[ti,si]=shared_fraction(c,g)
        row=dict(path=str(p.resolve()),mouse=int(d['parameters']['mouse']),seed=int(d['parameters']['seed']),
                 names=names,current_shared_fraction=fractions.tolist(),
                 activity_shared_fraction=shared_fraction(data,g),regional_activity_correlation=np.corrcoef(means).tolist(),
                 frames=len(td),dt=float(np.median(np.diff(td))))
        rows.append(row)
        (out/'shared_signal.json').write_text(json.dumps(rows,indent=2))
        print('Courants',row['mouse'],row['seed'],np.median(fractions),flush=True)
    plot_shared(rows,out)
    return rows


def plot_shared(rows,out):
    names=['MII','MI','Som','Ass','Vis','Ret']
    fig,axes=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    for col,mouse in enumerate([410,415]):
        subset=[r for r in rows if r['mouse']==mouse]
        assert len(subset)==3
        for rr,key,title in [(0,'regional_activity_correlation','Corrélation des activités régionales'),(1,'current_shared_fraction','Fraction de variance des courants liée au signal commun')]:
            a=np.mean([r[key] for r in subset],axis=0)
            ax=axes[rr,col]; im=ax.imshow(a,vmin=0 if rr else -1,vmax=1,cmap='viridis' if rr else 'coolwarm')
            ax.set(xticks=range(6),yticks=range(6),xticklabels=names,yticklabels=names,title=f'{mouse} — '+('Activités régionales' if rr==0 else 'Courants : fraction liée au signal commun'))
            ax.set_xlabel('Source' if rr else 'Région');ax.set_ylabel('Cible' if rr else 'Région')
            for i in range(6):
                for k in range(6):ax.text(k,i,f'{a[i,k]:.2f}',ha='center',va='center',fontsize=8,color='black' if a[i,k]>.65 else 'white')
            fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('M6 • 100 passages • σ = 4 • moyenne descriptive des trois graines\nSignal commun = moyenne des six activités régionales ; projection sur toute la session',fontsize=12)
    fig.savefig(out/'signal_partage.png',dpi=180);plt.close(fig)


def panel(root,out):
    specs=[(3,316,[6,12,18]),(6,374,[6,10,18]),(9,415,[6,8,18])]
    fig,axes=plt.subplots(3,3,figsize=(12,11),layout='constrained');rows=[]
    for rr,(cohort,mouse,months) in enumerate(specs):
        for cc,month in enumerate(months):
            folder=root/f'C{cohort}_M{month}'/'Data'/f'RS_M{mouse}'
            atlas=np.load(folder/'atlas.npy');mask=tifffile.imread(folder/'roi_mask.tif')>0
            cleaned=remove_thin_label_artifacts(atlas,size=5,min_fraction=.25)
            reduced=reduce_atlas_to_6_regions(cleaned,mask)
            with tifffile.TiffFile(folder/'GCaMP.tif') as tif:
                shape=tif.series[0].shape
                indices=np.linspace(0,shape[0]-1,32,dtype=int)
                sample=np.stack([tif.pages[int(i)].asarray() for i in indices])
                description=tif.pages[0].description
            assert sample.shape[1:]==atlas.shape==mask.shape
            std=np.std(sample.astype(float),axis=0)
            counts=[int(np.sum(reduced==i)) for i in range(6)]
            row=dict(cohort=cohort,mouse=mouse,month_label=month,path=str(folder),shape=list(shape),
                     mask_pixels=int(mask.sum()),assigned_pixels=sum(counts),parent_region_pixels=counts,
                     atlas_labels_inside_mask=np.unique(atlas[mask]).tolist(),
                     cleaned_fraction_inside_mask=float(np.mean(atlas[mask]!=cleaned[mask])),
                     finite_sample_fraction=float(np.isfinite(sample[:,mask]).mean()),
                     sampled_frames=indices.tolist(),tiff_description=description,
                     status='visual_screening_not_anatomical_registration_validation')
            rows.append(row)
            ax=axes[rr,cc]; valid=std[mask & np.isfinite(std)];lo,hi=np.percentile(valid,[2,98])
            ax.imshow(std,cmap='gray',vmin=lo,vmax=hi)
            ax.imshow(np.ma.masked_invalid(reduced),cmap=matplotlib.colors.ListedColormap(['#0047AB','#FF7F00','#00A550','#A020F0','#E60026','#00B7EB']),vmin=0,vmax=5,alpha=.20)
            ax.contour(mask,levels=[.5],colors='white',linewidths=.6)
            for k,color in enumerate(['#0047AB','#FF7F00','#00A550','#A020F0','#E60026','#00B7EB']):
                if counts[k]:ax.contour(reduced==k,levels=[.5],colors=[color],linewidths=.55)
            ax.set_title(f'C{cohort} · {mouse} · M{month}\n{sum(counts):,} pixels attribués / {mask.sum():,} masqués',fontsize=10);ax.axis('off')
            print('Panel',mouse,month,counts,flush=True)
    fig.suptitle('Panel longitudinal : atlas nettoyé et masque sur variabilité GCaMP\n32 frames réparties dans la session • contraste propre à chaque image • aucune registration entre sessions',fontsize=12)
    fig.legend(handles=[matplotlib.lines.Line2D([0],[0],color=c,label=n) for c,n in zip(['#0047AB','#FF7F00','#00A550','#A020F0','#E60026','#00B7EB'],['MII','MI','Som','Ass','Vis','Ret'])],loc='lower center',ncol=6,bbox_to_anchor=(.5,-.015))
    fig.savefig(out/'panel_atlas.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (out/'panel_qc.json').write_text(json.dumps(rows,indent=2))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['currents','panel'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    (currents if a.mode=='currents' else panel)(a.root,a.output)
