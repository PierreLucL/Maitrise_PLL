"""Fige les quatre réplications de l'indice corrigé depuis les pilotes vérifiés."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
def main():
    parent=ROOT/'results/narval_paired_controls_20260913/snapshot'
    out=ROOT/'results/narval_ttminus1_replicates_20260916/snapshot'
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((parent/name).read_bytes()).hexdigest()==digest,name
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent/'corrected',out/'corrected',ignore=shutil.ignore_patterns('__pycache__','results'))
    (out/'configs').mkdir()
    oldplan=json.loads((parent/'plan.json').read_text());tasks=[]
    for i,oldindex in enumerate([2,3,5,6]):
        task=dict(oldplan['tasks'][oldindex]);task.update(index=i,variant='corrected',nRunTrain=100);tasks.append(task)
        config=json.loads((parent/'configs'/f'{oldindex}.json').read_text())
        config['experiment_name']=f'ttminus1_{task["mouse"]}_{task["seed"]}_100'
        config['base_params']['nRunTrain']=100
        (out/'configs'/f'{i}.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'plan.json').write_text(json.dumps(dict(parent_snapshot='paired_controls_20260913_v1',purpose='Reproductibilité inter-graines tt-1 à 100 passages, souris410/415',tasks=tasks),indent=2)+'\n')
    runner=(parent/'run_paired_controls.py').read_text().replace('narval_paired_controls_20260913/','narval_ttminus1_replicates_20260916/')
    (out/'run_paired_controls.py').write_text(runner)
    sbatch=(parent/'run_paired_controls.sbatch').read_text().replace('curbd_pairs','curbd_ttminus1').replace('96:00:00','36:00:00').replace('0-6%2','0-3%2').replace('paired_controls_%A','ttminus1_replicates_%A')
    (out/'run_paired_controls.sbatch').write_text(sbatch)
    submit=(parent/'submit_paired_controls.sh').read_text().replace('legacy corrected','corrected').replace('0 1 2 3 4 5 6','0 1 2 3')
    (out/'submit_paired_controls.sh').write_text(submit)
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
