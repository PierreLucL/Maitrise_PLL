"""Cartes unités × temps : activité du modèle et contributions régionales.

Exemple : python scripts/analysis/plot_curbd_maps.py model.pkl --start 60 --end 120
Les PKL doivent provenir d’une source de confiance.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use('default')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'#333333','text.color':'#333333','xtick.color':'#555555','axes.titlepad':10,'figure.facecolor':'white','savefig.facecolor':'white'})
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('model',type=Path)
parser.add_argument('--output-dir',type=Path,default=Path('results/curbd_maps'))
parser.add_argument('--start',type=float,default=60)
parser.add_argument('--end',type=float,default=120)
parser.add_argument('--panel-aspect',type=float,default=1,help='Hauteur/largeur : 1 carré, 1.5 rectangle étroit')
parser.add_argument('--row-heights',choices=['units','equal'],default='units',help='Hauteurs proportionnelles aux unités (défaut) ou égales')
parser.add_argument('--current-percentile',type=float,default=99,help='Percentile de saturation commun aux courants')
args=parser.parse_args()
if not (args.start<args.end and args.panel_aspect>0 and 0<args.current_percentile<=100):parser.error('Fenêtre, aspect ou percentile invalide')
with args.model.open('rb') as f:d=pickle.load(f)
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
t=np.asarray(d['tRNN']);keep=(t>=args.start)&(t<args.end)
if keep.sum()<2 or args.start<t[0] or args.end>t[-1]+np.median(np.diff(t)):parser.error('Fenêtre hors données ou trop courte')
regions=d['regions'];n=len(regions);names=[str(r[0]).replace('Rég. ','') for r in regions]
colors=[plt.get_cmap('tab10')(i%10) for i in range(n)]
r=np.asarray(d['RNN_final']);j=np.asarray(d['J_final'])
assert r.shape[1]==len(t) and j.shape==(len(r),len(r))
assert np.all(np.diff(t)>0) and np.isfinite(r).all() and np.isfinite(j).all()
assert np.array_equal(np.sort(np.concatenate([np.asarray(x[1],int) for x in regions])),np.arange(len(r)))
blocks=[];orders=[]
for ti in range(n):
 ix=np.asarray(regions[ti,1],int);a=d['RNN_final'][ix][:,keep];order=np.argsort(np.argmax(a,axis=1),kind='stable');orders.append(order)
 blocks.append([np.asarray(d['J_final'][np.ix_(ix,np.asarray(regions[si,1],int))],float)@np.asarray(d['RNN_final'][np.asarray(regions[si,1],int)][:,keep],float) for si in range(n)])
# One symmetric scale for all contributions, without normalizing each panel.
limit=float(np.quantile(np.abs(np.concatenate([c.ravel() for row in blocks for c in row])),args.current_percentile/100))
if limit==0:limit=1.0
sizes=np.array([len(x[1]) for x in regions])
ratios=sizes/sizes.mean() if args.row_heights=='units' else np.ones(n)
fig,axes=plt.subplots(n,n+1,gridspec_kw={'height_ratios':ratios},figsize=(2.15*(n+1),2.15*n*args.panel_aspect+1),layout='constrained',squeeze=False)
cmap=LinearSegmentedColormap.from_list('activity',['#101010','#ffffff','#b21818'])
for ti in range(n):
 ix=np.asarray(regions[ti,1],int);order=orders[ti]
 im_a=axes[ti,0].imshow(d['RNN_final'][ix][:,keep][order],aspect='auto',cmap=cmap,vmin=0,vmax=1,extent=[args.start,args.end,len(ix),0],interpolation='nearest')
 axes[ti,0].set_ylabel(f'{names[ti]}\n{len(ix)} unités',color=colors[ti],fontsize=11)
 for si in range(n):
  im_c=axes[ti,si+1].imshow(blocks[ti][si][order],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[args.start,args.end,len(ix),0],interpolation='nearest')
 for j,ax in enumerate(axes[ti]):
  ax.set_box_aspect(args.panel_aspect*ratios[ti]);ax.set_yticks([]);ax.set_xticks([args.start,(args.start+args.end)/2,args.end] if ti==n-1 else []);ax.tick_params(labelsize=8,length=3,width=.6,pad=3)
  for sp in ax.spines.values():sp.set_color(colors[ti] if j==0 else colors[j-1]);sp.set_linewidth(1.1)
  if ti==0:ax.set_title('Activité\ndu modèle' if j==0 else 'Depuis\n'+names[j-1],color='black' if j==0 else colors[j-1],fontsize=12)
  if ti==n-1:ax.set_xlabel('Temps (s)',fontsize=9)
bar_a=fig.colorbar(im_a,ax=axes[:,0],location='bottom',shrink=.8,pad=.035,aspect=24,label='Activité du modèle')
bar_c=fig.colorbar(im_c,ax=axes[:,1:],location='bottom',shrink=.45,pad=.035,aspect=45,label='Courant signé · échelle commune',extend='neither')
for bar in (bar_a,bar_c):
 bar.outline.set_linewidth(.6)
 bar.outline.set_edgecolor('#777777')
 bar.ax.tick_params(labelsize=9,length=3,width=.6)
params=d.get('parameters',{})
fig.suptitle(f"Souris {params.get('mouse','?')} · graine {params.get('seed','inconnue')} · {params.get('nRunTrain','?')} passages · {args.start:g}–{args.end:g} s",fontsize=13)
fig.savefig(out/'maps_modele.png',dpi=220)
plt.close(fig)
metadata=dict(model=str(args.model.resolve()),start=args.start,end=args.end,panel_aspect=args.panel_aspect,row_heights=args.row_heights,units_per_region=sizes.tolist(),current_limit=limit,current_percentile=args.current_percentile,sort='pic de RNN dans la fenêtre, stable',orders=[o.tolist() for o in orders],activity_limits=[0,1],activity_clipped_fraction=float(np.mean((r[:,keep]<0)|(r[:,keep]>1))),note='Somme des contributions = J@R, pas directement RNN. Échelle commune aux courants ; pas de normalisation par panneau.')
(out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
print(out.resolve())
