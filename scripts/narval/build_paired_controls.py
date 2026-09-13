"""Construit deux copies figées à partir du snapshot historique vérifié."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from compute_curbd_currents_mouse410 import load_pickle_compatible


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.parent/'snapshot_manifest.json').read_text())['files']
    keep=[n for n in manifest if n.startswith('src/') or n in ['scripts/curbd/loop.py','tests/test_reproducibility.py']]
    for variant in ['legacy','corrected']:
        for n in keep:
            src=a.parent/n;assert digest(src)==manifest[n],n
            dest=a.output/variant/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    changed=a.output/'corrected/src/maitrise_curbd/curbd.py'
    old='r_slice = RNN[iTarget, tt].reshape(number_learn, 1)';new='r_slice = RNN[iTarget, tt-1].reshape(number_learn, 1)'
    text=changed.read_text();assert text.count(old)==1;changed.write_text(text.replace(old,new))
    tasks=[];base=json.loads((a.parent/'scripts/narval/configs/reproducibility_pix15_410_415.json').read_text())
    repo=Path(__file__).resolve().parents[2]
    specs=[('corrected',410,2026,100),('corrected',415,2026,100),('legacy',410,2027,300),('legacy',410,2028,300),('legacy',415,2026,300),('legacy',415,2027,300),('legacy',415,2028,300)]
    (a.output/'configs').mkdir()
    for i,(variant,mouse,seed,duration) in enumerate(specs):
        refs=list((repo/'results/narval_reproducibility_pix15_410_415/2614685').glob(f'*mouse{mouse}_*seed{seed}_*.pkl'));assert len(refs)==1
        ref=load_pickle_compatible(refs[0]);expected=ref['reproducibility']['source_sha256']
        for n,h in expected.items():assert digest(a.output/'legacy'/n)==h,n
        task=dict(index=i,variant=variant,mouse=mouse,seed=seed,nRunTrain=duration,
                  reference='/scratch/pllar11/Maitrise_PLL/results/narval_reproducibility_pix15_410_415/2614685/'+refs[0].name,reference_sha256=digest(refs[0]))
        tasks.append(task)
        config=json.loads(json.dumps(base));config['experiment_name']=f'paired_{variant}_{mouse}_{seed}_{duration}';config['datasets']=[[9,6,mouse]];config['sweep']={'seed':[seed]};config['base_params']['nRunTrain']=duration
        (a.output/'configs'/f'{i}.json').write_text(json.dumps(config,indent=2))
    (a.output/'plan.json').write_text(json.dumps({'parent_snapshot':'repro15_b3abdc905a6b5302','tasks':tasks,'purpose':'Durée inter-graines et pilote indice temporel ; sigma4 inchangé'},indent=2))
    for n in ['run_paired_controls.py','run_paired_controls.sbatch','submit_paired_controls.sh']:shutil.copyfile(repo/'scripts/narval'/n,a.output/n)
    files={str(f.relative_to(a.output)):digest(f) for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'manifest.json').write_text(json.dumps(files,indent=2))
    print('Snapshot apparié vérifié et construit :',a.output)

if __name__=='__main__':main()
