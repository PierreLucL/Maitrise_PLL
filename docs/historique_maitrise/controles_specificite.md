# Contrôles locaux : spécificité et alignement temporel

Six modèles : souris 410 et 415, graines 2026–2028, 100 passages. Pour chaque souris, trois paires de graines et 36 contributions, soit 108 comparaisons dépendantes.

## Protocole

PCA à dix dimensions ajustée séparément à chaque courant sur 0–230 s uniquement. Centrage appris sur ce bloc, scores projetés ensuite sur 250 s–fin : intervalle de garde de 20 s. CCA ajustée sur le premier bloc, axes et orientation du signe figés avant évaluation sur le second. La norme est celle des scores non blanchis. Les matrices RNN ont été entraînées sur toute la session : ceci valide le transfert temporel de l’alignement, pas la généralisation du réseau à des données inédites.

Pour chaque cible, la contribution d’une source dans la graine A est comparée à la même source dans B et aux cinq autres sources vers cette même cible. Le résultat « meilleure » impose de dépasser toutes les cinq autres correspondances.

Contrôles temporels : 40 décalages circulaires déterministes dans le bloc d’évaluation, séparations d’au moins 30 s dans les deux directions. Aucune nouvelle optimisation CCA après décalage. Comparaison au percentile 95 de ces contrôles ; ce percentile n’est pas une p-value. Les décalages introduisent une jointure circulaire et n’épuisent pas tous les contrôles possibles.

| Souris | Bonne source meilleure, norme | Bonne source meilleure, CCA | CCA test médiane | Marge CCA médiane contre meilleure autre source | Au-dessus du P95 décalé, norme / CCA |
|---|---|---|---|---|---|
| 410 | 38/108 | 89/108 | 0.982 | +0.0022 | 108/108 / 108/108 |
| 415 | 67/108 | 29/108 | 0.935 | -0.0096 | 108/108 / 108/108 |

## Interprétation

Les correspondances à temps aligné dépassent les contrôles décalés, et l’alignement CCA se transfère au second bloc. La similarité n’est donc pas seulement obtenue en ajustant les axes sur les instants où elle est évaluée.

La spécificité de région source est plus ambiguë. Chez 410, les premières CCA de la bonne source et des autres sources sont souvent très proches malgré une majorité de bonnes correspondances gagnantes. Chez 415, une autre source gagne plus souvent que la bonne en CCA. Ces résultats sont compatibles avec une composante dynamique largement partagée, sans démontrer son origine biologique ou technique. La force du premier alignement ne suffit pas à identifier la contribution régionale.

Ces nombres sont descriptifs : les blocs, graines et comparaisons se recouvrent, et aucune indépendance ou significativité statistique n’est revendiquée. Le rang parmi six sources ne fournit pas directement un test au hasard. Il faudra examiner les dimensions suivantes, les amplitudes et les cartes par unité, et éventuellement des contrôles spatiaux ou un signal commun.
