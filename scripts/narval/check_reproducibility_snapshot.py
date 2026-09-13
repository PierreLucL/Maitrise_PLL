"""Controle les empreintes du snapshot et la presence des fichiers sources."""
import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / 'snapshot_manifest.json').read_text())
for name, expected in manifest['files'].items():
    path = root / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Empreinte de code differente: {name}')
config = json.loads((root / 'scripts/narval/configs/reproducibility_pix15_410_415.json').read_text())
assert config['sweep'] == {'seed': [2026, 2027, 2028]}
assert config['datasets'] == [[9, 6, 410], [9, 6, 415]]
data_root = Path(os.environ.get('MAITRISE_DATA_DIR', '/scratch/pllar11/Datasets'))
for cohort, month, mouse in config['datasets']:
    for name in ('GCaMP.tif', 'atlas.npy', 'roi_mask.tif'):
        path = data_root / f'C{cohort}_M{month}' / 'Data' / f'RS_M{mouse}' / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f'Fichier source absent ou vide: {path}')
print('Snapshot et six fichiers sources verifies.')
