# Souris 410, graine 2026 — comparaison appariée 100 / 300

Analyse du 13 septembre 2026. Le modèle à 300 passages a terminé en 56 h 06 min 51 s sur Narval (2796272_0).

## Intégrité et appariement

Empreinte SHA-256 du PKL 300 vérifiée contre Narval : `1903d7389d1f48c6262428347f8e68d291b80846ab9772f3871864afb149402e`.

Contrôles réussis : Adata, tData, tRNN, masque_sub, J0, inputWN, initial_state, iTarget, activity_scale, parameters hors durée, model_parameters hors durée, timeseries_sha256, source_sha256, packages, threads, partition et identité des unités ; matrices et temps finis.

Écart maximal de pVar sur les 100 premiers passages communs : 0. Les deux entraînements repartent des mêmes conditions ; il ne s’agit pas d’une reprise depuis checkpoint. La version historique tt est conservée.

## Reconstruction

| Mesure | 100 passages | 300 passages |
|---|---:|---:|
| pVar recalculée sur la session | 0.945365 | 0.958582 |
| Temps total enregistré dans le PKL | 22.89 h | 56.11 h |

Gain de pVar : 1.32 points ; baisse de l’erreur quadratique de 24.2 % sur cette cible commune. Ce gain ne constitue pas une validation sur de nouvelles données.

| Région | pVar 100 | pVar 300 | Corrélation dérivées 100 | Corrélation dérivées 300 |
|---|---:|---:|---:|---:|
| Rég. M.II | 0.9582 | 0.9684 | 0.8211 | 0.8438 |
| Rég. M.I | 0.9548 | 0.9654 | 0.8277 | 0.8497 |
| Rég. Som. | 0.9435 | 0.9569 | 0.8380 | 0.8583 |
| Rég. Ass. | 0.9212 | 0.9401 | 0.8368 | 0.8553 |
| Rég. Vis. | 0.9143 | 0.9355 | 0.8349 | 0.8540 |
| Rég. Rét. | 0.9306 | 0.9479 | 0.8340 | 0.8542 |

La corrélation des dérivées utilise les différences temporelles de chaque unité, puis une corrélation sur les valeurs regroupées de la région ; ce n’est pas la médiane des corrélations par unité.

## Courants par unité et dynamiques dominantes

- Corrélation des normes PCA10 : médiane **0.942**, étendue 0.841–0.977.
- Première CCA : médiane **0.992** ; moyenne des dix CCA : médiane **0.936**.
- Ratio moyen des normes PCA10 (300/100) : médiane **1.248**, étendue **1.029–1.862**.
- Corrélation des sommes signées, diagnostic complémentaire : médiane 0.941 ; 0/36 négatives.
- Médiane des 36 corrélations médianes par unité sans alignement : 0.938.

Chaque contribution est calculée par J[cible,source] @ RNN[source], sans somme préalable des unités cibles. La PCA est centrée temporellement par unité et conserve dix composantes non blanchies. Les CCA décrivent des sous-espaces et ne garantissent pas signe ou amplitude identiques. Ces conventions reprennent les analyses historiques ; leur identité exacte avec les scripts des auteurs reste à vérifier.

## Transfert temporel et correspondance des sources

PCA et CCA apprises avant 230 s, puis évaluées à partir de 250 s. CCA test médiane : **0.990**. La bonne source dépasse les cinq autres en CCA dans **36/36** cas ; marge médiane +0.0048. Selon la norme : 32/36 cas.

Ces contrôles comparent 100 à 300 pour une seule graine. Ils ne mesurent pas une amélioration de la spécificité entre graines à 300 et ne sont pas directement équivalents aux 108 comparaisons historiques. Les contrôles par décalage n’ont pas été répétés ici. Le réseau a appris toute la session ; seul l’alignement est évalué sur un autre bloc.

## Figures

- comparaison_pca_cca.png : les 36 contributions, sans sélection de région.
- maps_100.png et maps_300.png : fenêtre fixée à 60–120 s, activité du modèle à gauche, sources en colonnes, mêmes unités et ordre fixés depuis le modèle 100, hauteurs proportionnelles aux effectifs.
- Échelle commune des courants : ±0.58555, P99 absolu regroupé sur les deux modèles. Les extrêmes sont saturés à l’affichage ; aucune normalisation par panneau. Fractions de saturation dans plot_metadata.json.
- apprentissage.png : les passages libres sont séparés des passages entraînés.

## Portée pour la maîtrise

Ce test isole la durée à données, graine et version communes, sous les contrôles numériques rapportés. Il permet de décrire ce qui change entre 100 et 300, pas de conclure que 300 rend les courants plus reproductibles entre graines ou entre sessions. Il faudra retenir des caractéristiques dont la variabilité technique est suffisamment faible pour les comparaisons longitudinales ; aucune différence d’âge n’est testée ici.

Les 36 contributions sont dépendantes et ne sont pas 36 souris. Une CCA1 élevée, une somme stable ou une pVar améliorée ne constitue pas une preuve causale ou anatomique.

Détails et provenance : comparison.json. Cartes de la fenêtre et courbes : maps_and_learning.npz. Les PKL complets permettent de régénérer les cartes sur les autres temps. Scripts : scripts/analysis/compare_training_durations.py et report_training_durations.py.
