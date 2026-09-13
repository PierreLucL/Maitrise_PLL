# Test contrôlé de la correction temporelle CURBD

Deux copies isolées du même code local ; seule la ligne de r_slice change de tt à tt-1. Le code de production et le snapshot Narval ne sont pas modifiés.

Données synthétiques : 18 unités, 240 échantillons à 12 Hz (20 s), combinaisons de sinusoïdes, 100 passages et 2 passages libres ; tauRNN=0,33, g=0,8, ampInWN=0,01. Trois graines et trois dtFactor. Dans chaque paire, J0, inputWN, Adata et iTarget sont vérifiés strictement identiques. Ces trajectoires synthétiques ne constituent pas une vérité terrain de connectivité.

| dtFactor | Graine | pVar ancienne | pVar corrigée | Écart |
|---|---|---|---|---|
| 1 | 2026 | 0.104 | 0.560 | +0.456 |
| 1 | 2027 | 0.694 | 0.578 | -0.116 |
| 1 | 2028 | 0.407 | 0.677 | +0.270 |
| 2 | 2026 | 0.573 | 0.497 | -0.075 |
| 2 | 2027 | 0.474 | 0.532 | +0.058 |
| 2 | 2028 | 0.768 | 0.683 | -0.085 |
| 4 | 2026 | 0.568 | 0.481 | -0.087 |
| 4 | 2027 | 0.690 | 0.739 | +0.049 |
| 4 | 2028 | 0.674 | 0.534 | -0.140 |

La correction modifie les résultats et ne donne pas une amélioration uniforme sur ce jeu. À dtFactor=2, utilisé dans nos expériences, les écarts de pVar vont de -0,085 à +0,058. Ce petit réseau n’est pas représentatif des 2 538 unités réelles et ne permet ni d’invalider la correction, ni de prédire son impact sur les modèles Narval. Il justifie de distinguer les versions avant de comparer les expériences.

Source du changement : https://github.com/rajanlab/CURBD/commit/08bb0192d919a9eb5e6ded42d0a0f15c128ed65e
