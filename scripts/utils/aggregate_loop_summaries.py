#!/usr/bin/env python3
### Agrege les petits CSV de jobs array en un seul loop_summary.csv.
### Chaque task SLURM ecrit son CSV pour eviter que 200 jobs se pilent sur les pieds.

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV final. Defaut: run_dir/loop_summary.csv.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    output = args.output or run_dir / "loop_summary.csv"
    files = sorted(run_dir.glob("loop_summary_job_*.csv"))

    if not files:
        raise SystemExit(f"Aucun loop_summary_job_*.csv trouve dans {run_dir}")

    rows = []
    fieldnames = []

    for path in files:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue

            for fieldname in reader.fieldnames:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)

            rows.extend(reader)

    def sort_key(row):
        ### Tri stable par config/dataset quand les colonnes existent.
        key = []
        for column in ("i_config", "cohort", "month", "mouse"):
            value = row.get(column, "")
            try:
                value = int(float(value))
            except (TypeError, ValueError):
                pass
            key.append(value)
        return tuple(key)

    rows = sorted(rows, key=sort_key)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(files)} fichiers agreges -> {output}")
    print(f"{len(rows)} lignes")


if __name__ == "__main__":
    main()
