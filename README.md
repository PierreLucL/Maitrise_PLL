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
  Coding CURBD 2026/        Code principal actuel
  Coding CURBD 2024/        Explorations et anciens notebooks
  Premiere nightrun/        Premiers essais de batch runs
  night_run_J_CURBD_36courbes/ Resultats de runs