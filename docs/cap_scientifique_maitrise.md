# Cap scientifique de la maîtrise

Objectif explicitement fixé par Pierre-Luc le 10 septembre 2026, à conserver comme fil directeur des prochains mois.

## Objectif final

Caractériser l’évolution des interactions fonctionnelles entre régions cérébrales au cours du vieillissement à partir des courants inférés par CURBD. Exploiter les jeux de données longitudinaux pour étudier les similarités et différences entre jeunes et vieilles souris, la variabilité entre souris du même âge, et les changements au sein d’une même souris suivie dans le temps.

## Rôle de la validation

La validation du pipeline est le socle méthodologique de l’étude biologique. Elle doit établir quelles caractéristiques des courants sont reproductibles et comparables entre sessions, souris et âges. Ne pas faire de la maximisation de pVar ou de l’unicité des poids J une fin en soi.

Chaque décision méthodologique doit servir la comparaison biologique : séparer variabilité due aux graines, versions et paramètres de celle entre sessions, souris et âges ; contrôler la spécificité régionale et la comparabilité des amplitudes, normalisations, atlas et acquisitions.

## Principes d’analyse

- Distinguer longitudinal (même souris), transversal (souris différentes) et effet de cohorte. Ne pas confondre âge et lot expérimental.
- Les souris sont les unités biologiques ; les graines ne sont pas des répétitions biologiques indépendantes.
- En activité spontanée, comparer des caractéristiques temporelles, dynamiques et liées aux états comportementaux plutôt que supposer des événements alignés entre souris.
- Conserver les cartes unités × temps pour explorer les contributions régionales sans sommer prématurément les unités cibles.
- Interpréter les courants comme interactions fonctionnelles inférées, sans présumer des connexions synaptiques ou causales.
- Accepter aussi des résultats de stabilité avec l’âge : ne pas chercher uniquement des différences.

## Prochaine étape structurante

Inventorier souris × âge × session × cohorte et métadonnées pertinentes ; sélectionner un panel pilote jeune/vieux avec suivis répétés lorsque disponibles, avant de généraliser le pipeline à toute la collection.

## Continuité des décisions et résultats

Consulter [le contexte consolidé des trois discussions](contexte_discussions_maitrise.md). Les analyses PCA–CCA et les contrôles temporels/de spécificité des six graines étaient déjà réalisés le 10 septembre. Les sommes signées sont un diagnostic complémentaire ; elles ne remplacent pas l’analyse des cartes unités × temps et des dynamiques dominantes. L’inventaire initial a été effectué le 11 septembre : le prochain travail porte sur sa validation et celle du panel, avec les limites de reproductibilité et de spécificité déjà documentées.
