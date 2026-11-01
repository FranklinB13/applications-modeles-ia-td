# Projet Usine 4.0

Applications et Modèles en IA, ING3 option IA, CY Tech

## Le besoin

Une aciérie contrôle aujourd'hui ses tôles à l'œil : des opérateurs regardent défiler les tôles et signalent
les défauts de surface. Le contrôle est fatigant, irrégulier d'une équipe à l'autre, et certains défauts
passent. La direction se demande si une caméra et un modèle de vision pourraient faire ce travail, ou au moins
aider les opérateurs.

Elle vous confie l'étude. Elle dispose de photos de défauts déjà annotées (le jeu NEU-DET : six types de
défauts, avec leur position sur l'image), mais certains défauts sont rares sur la ligne : pour ceux-là, on
n'aura que quelques images. Le système devra tourner
sur le PC de la ligne, sans carte graphique. Elle veut savoir, à la fin de l'étude, s'il faut lancer le projet,
et à quelles conditions.

## Votre mission

Concevoir, évaluer et défendre un système d'inspection des tôles, puis faire une recommandation à l'usine.

Le système doit au minimum :

- reconnaître le type d'un défaut et le localiser sur la tôle ;
- fonctionner dans les contraintes de la ligne (CPU, temps disponible par tôle) ;
- être accompagné d'un plan de suivi une fois en service.

Vous ajoutez un volet au choix :

- les défauts rares : comment traiter un type de défaut dont on n'a qu'une quinzaine d'images ? Vous
  choisissez le type de défaut et le réduisez vous-mêmes à 15 images.
- la maintenance : l'usine voudrait aussi anticiper les pannes de ses pompes à partir de leurs capteurs (jeu
  SKAB, anomalies réelles étiquetées). Ce volet ne concerne pas les tôles : il est traité comme un second
  besoin de la même usine.

Le reste est à vous : les modèles, la façon de découper le problème (un classifieur puis un détecteur, un
détecteur seul, autre chose), les métriques, les seuils, les données supplémentaires si vous en trouvez de
publiques. Les dernières étapes des TD 1 à 4 et le TD 6 donnent des points de départ ; vous pouvez vous en
écarter, à condition de le justifier.

## Ce que vous devez montrer

Quelle que soit votre approche, votre étude doit permettre à l'usine de vous faire confiance. Pour cela :

1. Une solution simple sert de référence : une méthode sans deep learning, réglée sérieusement. Le deep
   learning doit montrer ce qu'il apporte par rapport à elle.
2. Les mesures sont honnêtes : tous les choix se font sur une validation, et le test ne sert qu'une fois, à la
   fin. Les méthodes comparées le sont sur les mêmes données, avec la même métrique.
3. Les choix sont chiffrés. Vous fixez vous-mêmes des hypothèses sur ce que coûte un défaut non vu et une
   fausse alerte, vous les écrivez, et vos seuils en découlent.
4. Les limites sont dites : sur quels défauts, dans quelles conditions le système se trompe, et ce que cela
   coûterait.
5. Le travail se relance : un autre ingénieur doit pouvoir reproduire vos résultats à partir de votre dépôt.

## Organisation

Groupes de 3 ou 4. Chaque groupe envoie sa composition à l'enseignant.

Avant le Jour 4, chaque groupe envoie une note de cadrage d'une page : ce que fera votre système, le volet
choisi, les métriques envisagées, la répartition du travail. Elle n'est pas notée ; elle sert à valider votre
périmètre et à vous faire un retour tôt.

## Rendu

À rendre deux semaines après le Jour 6, à 23 h 59, sur le dépôt privé du groupe (l'enseignant, FranklinB13,
en collaborateur). C'est le dernier commit poussé avant cette heure qui est corrigé.

- Le code : les notebooks marimo du projet dans `projet/`, organisés comme vous le souhaitez (le gabarit
  `projet/brique_modele.py` peut servir de point de départ), avec un `projet/README.md` qui dit dans quel ordre
  les lancer, et les copies avec sorties enregistrées par marimo dans `projet/__marimo__/`.
- Un rapport de 10 pages au maximum, `projet/rapport.pdf`, écrit pour la direction de l'usine : le problème,
  votre solution, les résultats sur le test, les limites, et votre recommandation (déployer, déployer sous
  conditions, ou ne pas déployer), argumentée.
- Un historique Git régulier pendant tout le module.

## Usage des outils d'IA

Utilisez le moins possible les outils d'IA générative (assistants de code, générateurs de texte), aussi bien
pour le code que pour le rapport. Le projet sert à vérifier que vous savez mener cette étude vous-mêmes.

Le code doit être commenté : un lecteur doit comprendre ce que fait chaque partie et pourquoi. Le rapport doit
être bien rédigé, avec vos propres mots. Une mise en page simple écrite par vous est préférable à un document
soigné produit par une IA.

## Notation

Une seule note pour le groupe, sur 20.

| Critère | Points |
|---|---|
| Rigueur de la démarche : référence sans deep learning, validation et test, comparaisons équitables | 6 |
| Pertinence des choix : découpage du problème, modèles, métriques, hypothèses de coût et seuils | 5 |
| Mise en production : contraintes de la ligne, optimisation, robustesse, plan de suivi | 3 |
| Analyse critique et recommandation : limites, cas d'échec, conclusion argumentée, rapport bien rédigé | 4 |
| Reproductibilité et qualité du dépôt, code commenté | 2 |

L'ambition compte, mais un résultat modeste bien mesuré vaut mieux qu'un bon score mal mesuré. Sont pénalisés :
un réglage fait sur le test, des chiffres qui ne viennent d'aucune exécution, des méthodes comparées sur des
données différentes, un dépôt qui ne se relance pas.

## Données et outils

NEU-DET est à télécharger sur Kaggle et à placer dans `donnees/NEU/` (README du dépôt, section Données). SKAB se
télécharge avec `tp.charger_skab()`. Les fonctions de `outils/tp_utils.py` restent utilisables dans `projet/`
(`tp.charger_neu`, `tp.voc_vers_yolo`, etc.) ; les fonctions écrites pendant les TD ne s'importent pas, il faut
les recopier.
