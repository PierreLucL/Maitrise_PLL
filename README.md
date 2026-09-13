# Maitrise_PLL

Pipeline d'analyse de donnees GCaMP pour extraire des series temporelles
regionales, entrainer un reseau recurrent contraint par les donnees, puis
calculer une decomposition des courants avec CURBD.

## Objectif

Ce projet accompagne mon travail de maitrise sur l'analyse des interactions
entre regions cerebrales a partir de donnees d'imagerie calcium GCaMP.

Le pipeline principal effectue les etapes suivantes:

1. Chargement des fichiers H5 GCaMP
2. Nettoyage du masque de regions
3. Subdivision spatiale des regions
4. Extraction des series temporelles par sous-region
5. Pretraitement des signaux
6. Entrainement d'un RNN multi-region
7. Calcul des courants CURBD
8. Sauvegarde et visualisation des resultats

## Structure du projet

```text
Maitrise_PLL/
  src/maitrise_curbd/       Package Python principal
  scripts/curbd/            Scripts d'entrainement et de screening CURBD
  scripts/analysis/         Diagnostics, benchmarks et analyses post-run
  scripts/visualization/    Scripts de visualisation et animations
  notebooks/examples/       Exemples conserves
  notebooks/archive/        Anciennes explorations
  results/                  Sorties de runs locales ou archivees
```

## Donnees

Les donnees lourdes ne sont pas versionnees dans Git. Le package cherche les
datasets dans cet ordre:

1. l'argument `data_root` passe a `load_dataset(...)`
2. la variable d'environnement `MAITRISE_DATA_DIR`
3. le dossier local `data/`

La structure attendue est:

```text
$MAITRISE_DATA_DIR/
  C8_M6/
    Data/
      RS_M409/
        GCaMP.tif
        atlas.npy
        roi_mask.tif
```

Sur Narval, une configuration typique ressemble a:

```bash
export MAITRISE_DATA_DIR=/scratch/pllar11/Datasets
PYTHONPATH=src python scripts/curbd/loop.py \
  --data-dir "$MAITRISE_DATA_DIR" \
  --output-dir narval_train_m6
```

### Synchronisation selective vers Narval

Le script `scripts/narval/sync_to_narval.py` envoie le projet propre et,
pour chaque souris choisie, uniquement les trois fichiers lus par
`load_dataset`: `GCaMP.tif`, `atlas.npy` et `roi_mask.tif`.

Commence par un apercu sans transfert:

```bash
python scripts/narval/sync_to_narval.py \
  --local-data-dir "/Volumes/Toute ma vie/Datasets"
```

Quand la selection est correcte, lance le vrai transfert:

```bash
python scripts/narval/sync_to_narval.py \
  --local-data-dir "/Volumes/Toute ma vie/Datasets" \
  --apply
```

Pour n'envoyer que quelques souris, repete `--dataset`:

```bash
python scripts/narval/sync_to_narval.py \
  --local-data-dir "/Volumes/Toute ma vie/Datasets" \
  --dataset 3,6,316 \
  --dataset 3,6,322 \
  --apply
```

`--code-only` et `--data-only` permettent de separer les deux transferts.
La synchronisation du code utilise `--delete` seulement dans le dossier distant
`/scratch/pllar11/Maitrise_PLL`; elle retire donc les anciens scripts qui ne
font plus partie du projet local, sans toucher aux datasets ni aux resultats.

En local, tu peux soit creer un dossier `data/` a la racine du repo, soit
pointer vers un autre emplacement:

```bash
export MAITRISE_DATA_DIR="/Volumes/Toute ma vie/Datasets"
```

## Loop CURBD generale

Le fichier `scripts/curbd/loop.py` remplace les petites loops tres
similaires. La configuration se fait en haut du fichier:

- `DATASETS`: les souris a analyser
- `BASE_PARAMS`: les parametres fixes
- `SWEEP`: les parametres a faire varier

Exemples:

```python
SWEEP = {"P0": [0.1, 0.3, 1.0, 3.0]}
SWEEP = {"n_pixels": [50, 80, 100, 150]}
SWEEP = {"tauRNN": [0.083, 0.167, 0.33], "dtFactor": [2, 4]}
```

Toutes les sorties sont forcees dans `results/`:

```bash
PYTHONPATH=src python scripts/curbd/loop.py --data-dir "$MAITRISE_DATA_DIR"
PYTHONPATH=src python scripts/curbd/loop.py --dry-run
```

## Entrainements reproductibles

`trainMultiRegionRNN(..., seed=2026)` utilise un generateur local MT19937.
La graine controle la permutation des unites, le bruit et les poids initiaux.
`seed=None` conserve le comportement historique du generateur NumPy global.
La graine de segmentation est distincte (`segmentation_seed=0` par defaut).

Pour preparer trois repetitions a configuration fixe sur 410 et 415:

```bash
python scripts/curbd/loop.py \
  --config scripts/narval/configs/reproducibility_pix15_410_415.json \
  --dry-run
```

La grille contient six entrainements (2 souris x 3 graines), avec pix15,
100 passages d'apprentissage et les memes pretraitements. Les 20 passages
libres reutilisent le meme bruit et ne constituent pas 20 repetitions.
Les graines apparaissent dans les noms des PKL, les parametres et les CSV.

Les nouveaux PKL conservent aussi `J0`, `inputWN`, `initial_state`,
`J_final_full_precision`, `model_parameters`, `iTarget`, `activity_scale`,
`masque_sub`, `info_masque_sub` et `reproducibility` (versions, empreintes
du code et des traces, configuration des threads). Les anciens champs restent
disponibles au meme format. Le bruit et les matrices de replay sont conserves
en pleine precision : les fichiers seront plus volumineux.

Ces informations permettent de rejouer la trajectoire a poids fixes apres
apprentissage lorsqu'au moins un passage libre a ete execute. Elles ne sont
pas un checkpoint de reprise de l'apprentissage (la matrice RLS PJ n'est pas
sauvegardee). L'identite numerique est testee dans le meme environnement;
elle n'est pas garantie entre machines ou versions de bibliotheques.

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

## Signal GCaMP

Les fichiers `GCaMP.tif` recus sont deja en DeltaF/F. Le pipeline ne recalcule
donc pas de DeltaF/F avec une baseline glissante.

Regle de travail:

- ne pas appliquer de nouveau DeltaF/F;
- extraire les series temporelles directement depuis `GCaMP.tif`;
- appliquer ensuite seulement les etapes explicites choisies, par exemple GSR
  et lissage temporel.

## Diagnostic des sources raw

Le script `scripts/analysis/diagnostic_raw_sources.py` regarde les fichiers
GCaMP tels qu'ils sont recus:

- distribution raw par souris
- offset global moyen/median au fil du temps
- fraction de valeurs negatives
- fraction de valeurs proches de zero
- zoom raw sur les regions ciblees si un CSV de regions problematiques est fourni

```bash
PYTHONPATH=src python scripts/analysis/diagnostic_raw_sources.py \
  --data-dir "$MAITRISE_DATA_DIR" \
  --resume
```

Sorties:

- `results/diagnostics/raw_sources/.../raw_source_summary.csv`
- `results/diagnostics/raw_sources/.../raw_breaking_regions.csv`
- figures PNG de distribution, offset global et regions raw problematiques
