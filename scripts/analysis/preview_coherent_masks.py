"""Aperçu local de la segmentation candidate, sans entraînement."""
import sys,json
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from maitrise_curbd.masks import (reduce_atlas_to_6_regions,remove_thin_label_artifacts,repair_anatomical_outliers,subdivide_mask_by_spatial_clustering,make_connected_subgroups,build_parent_regions_dict,validate_region_assignment)
mouse=int(sys.argv[1]) if len(sys.argv)>1 else 410
root=Path('results/audit_masks_20260916');folder=root/'inputs'/str(mouse)
atlas=np.load(folder/'atlas.npy');roi=tifffile.imread(folder/'roi_mask.tif')>0
raw=reduce_atlas_to_6_regions(atlas,roi)
legacy=reduce_atlas_to_6_regions(remove_thin_label_artifacts(atlas,size=5,min_fraction=.25),roi)
new,report=repair_anatomical_outliers(legacy,roi)
mask,info=subdivide_mask_by_spatial_clustering(new,target_size=15,random_state=0)
mask,info=make_connected_subgroups(new,mask)
qc=validate_region_assignment(new,mask,info,build_parent_regions_dict(info),len(info))
changed=np.isfinite(legacy)&np.isfinite(new)&(legacy!=new)
report.update(qc=qc,changed_vs_legacy=int(np.sum(~((new==legacy)|(np.isnan(new)&np.isnan(legacy))))),roi_pixels=int(roi.sum()))
(root/f'coherent_{mouse}.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(root/f'coherent_{mouse}.npz',raw=raw,legacy=legacy,parent=new,parcels=mask)
fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
for ax,a,title in zip(axes,[raw,legacy,new,np.where(changed,1,np.nan)],['Parents bruts','Nettoyage historique','Proposition coherent_v1','Pixels réattribués (historique → proposition)']):
 ax.imshow(a,cmap='tab10' if title!= 'Pixels réattribués (historique → proposition)' else 'autumn',vmin=0,vmax=5 if title!='Pixels réattribués (historique → proposition)' else 1,interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
fig.suptitle(f'Souris {mouse} · six groupes bilatéraux · proposition à valider anatomiquement')
fig.savefig(root/f'coherent_{mouse}.png',dpi=160);plt.close(fig)
print(json.dumps(report,indent=2))
