"""Contrôles ciblés des conventions de masque, sans modifier le pipeline."""
import sys
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maitrise_curbd.masks import remove_thin_label_artifacts,reduce_atlas_to_6_regions,clean_reduced_atlas,build_parent_regions_dict

def main():
    findings={}
    a=np.ones((7,7));a[3,3]=np.nan
    b=remove_thin_label_artifacts(a,size=5,min_fraction=.25)
    assert np.isnan(b[3,3]);findings['NaN_hole_preserved']=True
    a[3,3]=0
    b=remove_thin_label_artifacts(a,size=5,min_fraction=.25)
    assert b[3,3]==1;findings['zero_hole_can_be_relabelled']=True
    b=reduce_atlas_to_6_regions(np.array([[0,1,69,np.nan]]))
    assert np.isnan(b[0,[0,2,3]]).all() and b[0,1]==0
    findings['unmapped_labels_excluded']=True
    b=clean_reduced_atlas(np.full((3,3),np.nan),brain_mask=np.ones((3,3),bool))
    assert np.all(b==0);findings['unused_cleaner_invents_region_zero_when_no_seed']=True
    try:remove_thin_label_artifacts(np.full((3,3),np.nan))
    except ValueError as e:findings['empty_atlas_error']=str(e)
    d={2:{'parent_region':0},0:{'parent_region':0},1:{'parent_region':1}}
    r=build_parent_regions_dict(d)
    assert list(r[0,1])==[2,0];findings['dict_preserves_insertion_not_sorted_ids']=True
    out=ROOT/'results/audit_masks_20260916';out.mkdir(exist_ok=True)
    (out/'synthetic_checks.json').write_text(json.dumps(findings,indent=2)+'\n')
    print(json.dumps(findings,indent=2))
if __name__=='__main__':main()
