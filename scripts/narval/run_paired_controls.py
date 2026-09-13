"""Exécute un contrôle figé et vérifie son appariement à la référence 100."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
from pathlib import Path
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    root=a.root.resolve(); plan=json.loads((root/'plan.json').read_text()); task=plan['tasks'][a.task]
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
    variant=root/task['variant'];os.chdir(variant);sys.path.insert(0,str(variant/'src'))
    reference=Path(task['reference'])
    # On vérifie aussi le résultat de référence, pas seulement son nom de fichier.
    h=hashlib.sha256()
    with reference.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==task['reference_sha256'], 'Référence modifiée'
    with reference.open('rb') as f:ref=pickle.load(f)
    for package,version in ref['reproducibility']['packages'].items():
        assert importlib.metadata.version(package)==version,(package,version)
    for name,value in ref['reproducibility']['threads'].items():
        assert os.environ.get(name)==value,(name,value)
    for name,digest in ref['reproducibility']['source_sha256'].items():
        if task['variant']=='corrected' and name=='src/maitrise_curbd/curbd.py':continue
        assert hashlib.sha256((variant/name).read_bytes()).hexdigest()==digest,name
    config=root/'configs'/f'{a.task}.json'
    for name in ['GCaMP.tif','atlas.npy','roi_mask.tif']:
        path=Path(os.environ['MAITRISE_DATA_DIR'])/'C9_M6'/'Data'/f'RS_M{task["mouse"]}'/name
        assert path.is_file() and path.stat().st_size>0,path
    spec=importlib.util.spec_from_file_location('paired_loop',variant/'scripts/curbd/loop.py');loop=importlib.util.module_from_spec(spec);spec.loader.exec_module(loop)
    original_prepare=loop.prepare_timeseries
    def checked_prepare(*args,**kwargs):
        prepared=original_prepare(*args,**kwargs)
        assert loop.reproducibility_metadata(prepared['ts'])['timeseries_sha256']==ref['reproducibility']['timeseries_sha256'],'Cible prétraitée différente : arrêt avant entraînement'
        np.testing.assert_array_equal(prepared['masque_sub'],ref['masque_sub'])
        for new,old in zip(prepared['regions'],ref['regions']):
            assert new[0]==old[0];np.testing.assert_array_equal(new[1],old[1])
        print('Cible, segmentation et régions appariées avant entraînement.',flush=True)
        return prepared
    loop.prepare_timeseries=checked_prepare
    output=f'narval_paired_controls_20260913/{os.environ.get("SLURM_ARRAY_JOB_ID","preflight")}/task{a.task}'
    sys.argv=['loop.py','--config',str(config),'--job-index','0','--data-dir',os.environ['MAITRISE_DATA_DIR'],'--output-dir',output]+(['--dry-run'] if a.dry_run else [])
    loop.main()
    if a.dry_run:return
    folder=variant/'results'/output
    with (folder/'loop_summary_job_0000.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['status']=='done',rows
    with Path(rows[0]['save_path']).open('rb') as f:result=pickle.load(f)
    for name in ['Adata','tData','tRNN','J0','inputWN','initial_state','iTarget','activity_scale','masque_sub']:
        np.testing.assert_array_equal(result[name],ref[name],err_msg=name)
    for name in ['parameters','model_parameters']:
        aa={k:v for k,v in result[name].items() if k not in ['nRunTrain','nRunTot']}
        bb={k:v for k,v in ref[name].items() if k not in ['nRunTrain','nRunTot']}
        assert aa==bb,name
    if task['variant']=='legacy':np.testing.assert_array_equal(result['pVar'][:100],ref['pVar'][:100])
    (folder/'pairing_verified.json').write_text(json.dumps({'task':task,'checks':'Identité des entrées, initialisation, paramètres ; préfixe pVar100 exact pour durée legacy','pvar_final':result['row']['pVar_finale']},indent=2))
    print('Résultat et appariement confirmés.',flush=True)

if __name__=='__main__':main()
