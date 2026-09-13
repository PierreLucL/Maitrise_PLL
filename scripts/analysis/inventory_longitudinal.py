"""Inventory actual dataset folders without interpreting directory labels as metadata."""
import argparse
import datetime
import json
import re
from pathlib import Path


def inventory(root):
    rows = []
    for folder in sorted(root.glob('C*_M*/Data/RS_M*')):
        match = re.fullmatch(r'C(\d+)_M(\d+)/Data/RS_M(\d+)', folder.relative_to(root).as_posix())
        if not match or not folder.is_dir():
            continue
        cohort, month, mouse = map(int, match.groups())
        files = {p.name: p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file() and not p.name.startswith('.')}
        required = ['GCaMP.tif', 'atlas.npy', 'roi_mask.tif']
        rows.append(dict(cohort=cohort, month_label=month, mouse=mouse,
                         relative_path=folder.relative_to(root).as_posix(), files=files,
                         pipeline_files_present=all(files.get(k, 0) > 0 for k in required),
                         missing_files=[k for k in required if not files.get(k, 0)],
                         acquisition_date=None, exact_age_days=None, sex=None,
                         behavioral_state=None, session_id=None, quality_status='not_reviewed'))
    return dict(root=str(root), observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                caveat='M is a directory age label; identity continuity, actual age and session multiplicity need metadata confirmation. Presence is not quality validation.',
                sessions=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2))
