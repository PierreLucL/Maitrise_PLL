# Comparaison appariée tt / tt-1 — 15 septembre 2026

Deux souris, même graine2026 et100 passages. Données, unités, initialisations, bruit, paramètres, versions et threads vérifiés. Seule modification du moteur : r_slice utilise tt-1. Cibles normalisées sauvegardées, sans nouveau lissage.

| Souris | pVar tt | pVar tt-1 | Réduction erreur quadratique | r norme PCA10 médian | CCA10 moyenne médiane | Ratio RMS centrée médian (étendue) |
|---|---:|---:|---:|---:|---:|---|
| 410 | 0.945365 | 0.962845 | 32.0% | 0.989 | 0.985 | 0.849 (0.703–1.251) |
| 415 | 0.914296 | 0.927272 | 15.1% | 0.994 | 0.992 | 0.912 (0.818–1.029) |

## Contrôle temporel et spécificité
PCA et CCA apprises avant230s, évaluées après250s, signes et axes figés. Le RNN a été entraîné sur toute la session : ce découpage ne constitue pas une validation du RNN sur de nouvelles données.
- 410 : CCA2–10 test moyenne médiane 0.982 ; même source classée première 36/36 ; marge médiane +0.431.
- 415 : CCA2–10 test moyenne médiane 0.989 ; même source classée première 36/36 ; marge médiane +0.468.

## Portée
Comparaison entre versions à une graine par souris, pas une mesure de reproductibilité inter-graines de tt-1. CCA autorise un changement de base et ne garantit ni amplitude ni signe des courants par unité. Les rapports RMS centrés mesurent séparément leur amplitude. Ne pas mélanger des versions dans une comparaison d’âge ; une différence de méthode pourrait devenir une différence biologique apparente.

Traces : six unités par souris, chacune proche de la pVar régionale médiane du modèle tt, sans sélection sur le gain. Même échelle par ligne. Fenêtres60–120s et zoom75–90s. Ce sont des exemples, pas toutes les unités. Cartes des courants par unité conservées dans maps_and_learning.npz ; aucune somme signée utilisée comme métrique principale.
