#!/usr/bin/env python3
"""Synchronise le code et juste les fichiers de donnees lus par le loader."""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


### On ajoute src au path pour reutiliser la liste officielle des datasets.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from maitrise_curbd.io import DATASETS  # noqa: E402


LOADER_FILES = ("GCaMP.tif", "atlas.npy", "roi_mask.tif")
DEFAULT_REMOTE = "Narval"
DEFAULT_REMOTE_PROJECT = "/scratch/pllar11/Maitrise_PLL"
DEFAULT_REMOTE_DATA = "/scratch/pllar11/Datasets"


def parse_dataset(value: str) -> tuple[int, int, int]:
    ### Format compact C,M,souris, par exemple 3,6,316.
    try:
        cohort, month, mouse = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Un dataset doit avoir le format cohorte,mois,souris (ex: 3,6,316)."
        ) from exc
    return cohort, month, mouse


def all_datasets() -> list[tuple[int, int, int]]:
    ### Aplatit la petite map de io.py pour ne pas maintenir deux listes differentes.
    return [
        (cohort, month, mouse)
        for cohort, months in DATASETS.items()
        for month, mice in months.items()
        for mouse in mice
    ]


def dataset_files(datasets: list[tuple[int, int, int]]) -> list[str]:
    ### rsync recoit des chemins relatifs et recree automatiquement l'arborescence.
    return [
        f"C{cohort}_M{month}/Data/RS_M{mouse}/{filename}"
        for cohort, month, mouse in datasets
        for filename in LOADER_FILES
    ]


def run(command: list[str], apply: bool, retries: int = 1) -> None:
    print("$", shlex.join(command))
    if not apply:
        return

    for attempt in range(1, retries + 1):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            wait_seconds = min(30 * attempt, 120)
            print(
                f"Connexion coupee (tentative {attempt}/{retries}). "
                f"Reprise dans {wait_seconds} secondes..."
            )
            time.sleep(wait_seconds)


def split_complete_datasets(
    data_root: Path,
    datasets: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], list[str]]]:
    ### Un GCaMP sans atlas/masque ne sert pas au loader, alors on skip le trio au complet.
    complete = []
    incomplete = {}
    for dataset in datasets:
        files = dataset_files([dataset])
        missing = [relative for relative in files if not (data_root / relative).is_file()]
        if missing:
            incomplete[dataset] = missing
        else:
            complete.append(dataset)
    return complete, incomplete


def sync_code(args: argparse.Namespace) -> None:
    ### --delete nettoie les vieux scripts dans le projet distant seulement.
    command = [
        "rsync",
        "-avz",
        "--delete",
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.egg-info/",
        "--exclude=data/",
        "--exclude=results/",
        f"{REPO_ROOT}/",
        f"{args.remote}:{args.remote_project}/",
    ]
    run(command, args.apply)


def sync_data(args: argparse.Namespace) -> None:
    data_root = args.local_data_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Dossier de donnees introuvable: {data_root}")

    requested_datasets = args.dataset or all_datasets()
    datasets, incomplete = split_complete_datasets(data_root, requested_datasets)
    available = dataset_files(datasets)

    if incomplete:
        print(
            f"Attention: {len(incomplete)} dataset(s) incomplet(s), "
            "le trio au complet sera skip."
        )
        for cohort, month, mouse in incomplete:
            missing_names = ", ".join(
                Path(relative).name for relative in incomplete[(cohort, month, mouse)]
            )
            print(f"  - C{cohort}_M{month}/RS_M{mouse}: {missing_names}")

    if not datasets:
        raise SystemExit("Aucun dataset complet attendu par le loader n'a ete trouve.")

    ### Le manifest garantit qu'aucun fichier bonus des dossiers de souris ne voyage.
    with tempfile.NamedTemporaryFile("w", prefix="curbd_data_", delete=False) as manifest:
        manifest.write("\n".join(available) + "\n")
        manifest_path = Path(manifest.name)

    try:
        run(
            [
                "rsync",
                "-av",
                "--partial",
                "--progress",
                "--prune-empty-dirs",
                "-e",
                "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10",
                f"--files-from={manifest_path}",
                f"{data_root}/",
                f"{args.remote}:{args.remote_data}/",
            ],
            args.apply,
            retries=5,
        )
    finally:
        manifest_path.unlink(missing_ok=True)

    print(
        f"{len(available)} fichier(s) selectionne(s) pour "
        f"{len(datasets)} dataset(s) complet(s)."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronise le projet CURBD et les donnees minimales vers Narval."
    )
    parser.add_argument(
        "--local-data-dir",
        type=Path,
        default=os.environ.get("MAITRISE_DATA_DIR"),
        help="Racine locale des datasets; sinon utilise MAITRISE_DATA_DIR.",
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="Alias SSH.")
    parser.add_argument("--remote-project", default=DEFAULT_REMOTE_PROJECT)
    parser.add_argument("--remote-data", default=DEFAULT_REMOTE_DATA)
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        help="Dataset C,M,souris. Repetable; sans option, prend toute la map de io.py.",
    )
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute les commandes. Sans ceci, montre seulement le plan.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.code_only and args.data_only:
        raise SystemExit("Choisis --code-only ou --data-only, pas les deux.")
    if not args.code_only and args.local_data_dir is None:
        raise SystemExit(
            "Donne --local-data-dir ou configure MAITRISE_DATA_DIR."
        )

    ### On cree les destinations avant rsync; ca evite les surprises au premier upload.
    destinations = [args.remote_project]
    if not args.code_only:
        destinations.append(args.remote_data)
    run(["ssh", args.remote, "mkdir", "-p", *destinations], args.apply)

    if not args.data_only:
        sync_code(args)
    if not args.code_only:
        sync_data(args)

    if not args.apply:
        print("\nDry-run seulement. Relance avec --apply quand le plan est beau.")


if __name__ == "__main__":
    main()
