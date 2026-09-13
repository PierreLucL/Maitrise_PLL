# Audit des courants CURBD — première séance

Analyse du 7 septembre 2026, effectuée sur les fichiers locaux. Aucune nouvelle consigne de direction signalée par Pierre-Luc pendant la séance.

**Résultat principal : la décomposition source→cible varie fortement entre deux bons modèles pix25, tandis que le courant total reçu par chaque région reste très similaire. Les calculs vérifiés ne présentent pas d’inversion source/cible ni d’erreur d’agrégation.**

## Ce qui a été examiné

L’inventaire des CSV, dédoublonné par fichier PKL résolu, retrouve 304 fichiers PKL présents, correspondant à 13 souris à M6. Le comptage indépendant sur disque retrouve également 304 PKL. Les CSV d’archives mentionnent sept résultats supplémentaires marqués terminés dont les PKL ne sont pas retrouvés localement, et deux échecs. Un CSV agrégé et ses CSV individuels ne comptent pas comme des répétitions distinctes.

Les vérifications numériques détaillées concernent **12 PKL de la souris 410 et leurs 12 NPZ de courants**, ainsi que les 66 comparaisons possibles entre eux. Elles ne constituent pas un audit numérique des 304 modèles. Les données du disque externe n’ont pas été retraitées, et aucun calcul Narval n’a été lancé.

## Calcul et sauvegardes

Le modèle utilise J[cible, source]. Pour une source S et une cible T, le script calcule :

`I(T ← S, t) = somme sur i dans T et j dans S de J[i,j] × R[j,t]`.

Cette quantité est la somme, sur les unités cibles, de la décomposition complète produite par `computeCURBD`. Elle est exprimée dans les unités internes du modèle, sans calibration en courant électrique physiologique.

- Réseau synthétique asymétrique : accord entre le calcul optimisé et la décomposition complète, erreur relative 3.21e-08.
- Pour les 12 fichiers : matrices et temps finis, dimensions compatibles, régions couvrant toutes les unités exactement une fois, mêmes noms régionaux entre modèles.
- Les 12 grilles temporelles sont identiques : 11 519 échantillons RNN, de 0 à environ 479,917 s. La troncature silencieuse du comparateur historique n’a donc pas faussé ces 12 comparaisons.
- Vérification du cache sur **257 instants répartis sur chaque session** : erreur relative maximale entre courants sauvegardés et recalcul direct en float64 = 8.06e-07. Les petits écarts sont compatibles avec les calculs et sauvegardes en float32.
- Conservation sur ces mêmes instants : la somme des six contributions sources redonne le courant récurrent total agrégé `J @ R`, à une erreur relative maximale de 3.98e-15.
- Le pVar a été recalculé sur **toute la session** avec les grilles sauvegardées. Écart absolu maximal : 3.37e-08 pour les 11 modèles à pVar positif ; 5.53e-06 pour le modèle divergent à pVar ≈ −110.

La conservation vérifiée porte sur le courant récurrent. Elle n’est pas une vérification complète de l’équation d’évolution incluant le bruit et le terme de fuite.

## Résultats de similarité

Toutes les corrélations ci-dessous utilisent la session complète. « Médiane par paire » donne le même poids à chacune des 36 paires. « Ancien score » est la corrélation des 36 traces concaténées après centrage et division par √N cible.

| Comparaison | Ancien score | Médiane par paire | Corrélations négatives | Corrélation du courant total, selon la cible |
|---|---:|---:|---:|---:|
| pix25 : 100 / 300 | 0.514 | 0.626 | 5/36 | 0.858–0.926 |
| pix25 : 300 / 1 000 | 0.097 | -0.008 | 19/36 | 0.866–0.932 |
| 100 entraînements : pix20 / pix15 | 0.679 | 0.772 | 0/36 | 0.894–0.942 |

![Corrélations par courant](/Users/pierre-luclarouche/Documents/Codex/2026-09-07/sal/outputs/audit_courants/correlations_par_courant.png)

Pour pix25/100, pix25/300 et pix25/1000, la cible `Adata` est identique octet pour octet. Les effectifs des six régions sont également identiques : 341, 245, 315, 94, 172 et 357 unités. Les différences de courants ne s’expliquent donc pas ici par un changement de cible ni d’effectif régional. La correspondance spatiale exacte n’a pas été reconstruite à partir des fichiers sources pendant cet audit.

Entre pix25/300 et pix25/1000, la contribution associative vers visuelle a une corrélation d’environ −0,818. Pourtant, le courant récurrent total vers la région visuelle a une corrélation de 0,866. Des contributions différentes se compensent et conduisent à des sommes semblables.

![Somme et contributions](/Users/pierre-luclarouche/Documents/Codex/2026-09-07/sal/outputs/audit_courants/somme_et_contributions.png)

La figure utilise la fenêtre fixe 60–120 s, sans lissage supplémentaire. La moyenne de chaque trace est retirée sur toute la session. Les axes verticaux ont des échelles distinctes, indiquées par les graduations. La cible visuelle a été choisie parce qu’elle présente la plus faible médiane de corrélation de ses six entrées entre ces deux modèles ; cet exemple illustre le cas le plus discordant, il n’est pas présenté comme une région typique. Les corrélations affichées sont calculées sur la session complète.

Une corrélation négative indique ici des fluctuations opposées **entre modèles**. Elle ne démontre pas une interaction inhibitrice entre régions.

## Ce qu’on peut conclure

Les résultats montrent une sensibilité de la décomposition malgré des reconstructions de bonne qualité. Ils sont compatibles avec plusieurs répartitions des contributions capables de produire une dynamique totale semblable. Ils ne suffisent pas à identifier la cause de cette sensibilité ni à invalider CURBD.

Les différences entre 300 et 1 000 entraînements ne sont pas un effet isolé de la durée : les graines, `J0` et `inputWN` ne sont pas sauvegardés dans les 12 PKL. On ne peut pas établir qu’il s’agit d’un même apprentissage poursuivi plus longtemps. Les poids initiaux et le bruit sont des facteurs non contrôlés dans cette comparaison historique.

La ressemblance pix20/pix15 est encourageante : 36 corrélations positives, médiane 0,772. Mais il s’agit d’une seule comparaison sur une seule souris, avec des cibles et des tailles de réseau différentes. Ce n’est pas encore une démonstration de robustesse.

## Limites techniques repérées

1. Le comparateur historique tronque au minimum des longueurs sans vérifier les temps. Aucun effet constaté sur les 12 fichiers, mais comportement à éviter pour les prochains jeux de données.
2. Son score global dépend du mode d’affichage et pondère implicitement les grandes fluctuations. Il ne doit pas remplacer les corrélations par paire et l’examen des amplitudes.
3. Les noms de dossiers des caches omettent plusieurs paramètres et la graine, et arrondissent le pVar à trois décimales. Le cache n’est pas lié à une empreinte du PKL. Les caches audités concordent sur les instants vérifiés, mais ce nommage n’est pas adapté à de nombreuses répétitions futures.
4. Les anciennes figures limitent l’axe vertical au 99e percentile absolu : les extrêmes peuvent sortir du cadre. Les nouvelles figures de traces utilisent l’étendue visible complète.
5. La boucle calcule `J0`, `inputWN` et le facteur de normalisation, mais ne les conserve pas dans les PKL finaux. Elle ne conserve pas non plus le masque spatial et les métadonnées de subdivision.
6. Les passages `nRunFree` réinitialisent le même état avec le même bruit et la même matrice. Un petit test synthétique de cinq unités a reproduit trois pVar libres exactement identiques. Ces passages ne sont pas des répétitions indépendantes ni une validation sur une nouvelle session.

## Code ajouté et vérification

Un nouveau comparateur est disponible dans le dépôt :

`/Users/pierre-luclarouche/Documents/Codex/2026-08-19/yo/Maitrise_PLL_shallow/scripts/analysis/compare_saved_currents.py`

Il vérifie les dimensions, les noms régionaux uniques, la couverture des paires, les valeurs finies et l’identité exacte des grilles temporelles. Il réordonne les courants selon les noms anatomiques déclarés, refuse toute troncature implicite, et sépare corrélation, moyenne, écart-type, ratio d’écarts-types et erreur quadratique. Les corrélations de traces constantes sont indiquées comme indéfinies.

Huit tests passent, notamment pour les temps décalés, les longueurs différentes, les régions réordonnées, les traces constantes et les courants qui se compensent. Le résultat du comparateur sur pix25/300 et pix25/1000 concorde avec le script d’audit indépendant pour les 36 corrélations à moins de 10⁻¹². La figure exportée a été inspectée visuellement.

Le cœur d’apprentissage, les anciennes analyses, les résultats et les modifications Git préexistantes n’ont pas été modifiés. Les deux ajouts sont le comparateur et son fichier de tests.

Exemple depuis le dépôt :

```bash
python3 scripts/analysis/compare_saved_currents.py \
  results/curbd_currents_mouse410/mouse410_pix25_N1524_train300_pVar0.925/curbd_currents_total.npz \
  results/curbd_currents_mouse410/mouse410_pix25_N1524_train1000_pVar0.932/curbd_currents_total.npz \
  --output-dir results/comparison_pix25_300_1000 --plot
```

Les NPZ seuls ne contiennent pas assez de métadonnées pour prouver la même session ou le même traitement : vérifier aussi les PKL avant d’interpréter une comparaison. Les amplitudes de courants totaux entre résolutions différentes demandent une convention d’agrégation justifiée ; le ratio brut n’est pas une différence physiologique.

## Suite proposée

Avant de soumettre de nouveaux calculs, ajouter à la boucle une graine explicite effectivement utilisée et sauvegarder les informations nécessaires à la reproduction : `J0`, `inputWN`, facteur de normalisation, masque et correspondance des sous-régions, version du code et paramètres. Vérifier avec un petit cas que deux exécutions à graine identique redonnent les mêmes sorties et que des graines différentes produisent des initialisations différentes.

Puis tester quelques répétitions **à mêmes données, segmentation, durée et paramètres**, en variant seulement la graine. Pix15/100 reste un candidat ; le choix avec pix20/100 doit tenir compte du budget de calcul. Trois répétitions par souris donneraient une première estimation descriptive, sans suffire à une validation définitive. Pour isoler ensuite l’effet 100/300/1000, apparier les initialisations et le bruit, ou utiliser des points de sauvegarde d’un même entraînement.

La question prioritaire devient : **quelle part de la décomposition source→cible est reproductible à configuration fixe ?** La comparaison longitudinale pourra ensuite être évaluée relativement à cette incertitude.

Les chemins sources, paramètres, empreintes de code, contrôles numériques et les 66 comparaisons sont conservés dans `donnees_audit.json`. Les valeurs de la comparaison principale sont aussi dans `comparaison_pix25/comparison.json`.
