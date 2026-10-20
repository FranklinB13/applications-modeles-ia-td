# Applications et Modèles en IA — TD

Module ING3 option IA, CY Tech. Six TD, un projet fil rouge (Usine 4.0).

Chaque TD est un notebook **marimo** : un simple fichier Python que l'on ouvre dans le navigateur, avec des
cellules à exécuter, du code à compléter et des tests automatiques. Les fichiers se versionnent proprement
avec Git : c'est votre copie de ce dépôt, complétée, que vous rendez à la fin du module.

Les TD sont mis en ligne ici au fur et à mesure, un par séance. **Vous clonez ce dépôt une seule fois**, au
TD 1. Ensuite, au début de chaque séance, une seule commande récupère le nouveau TD dans votre dossier, à côté
de votre travail, sans rien écraser :

```bash
uv run python recuperer_td.py
```

Le détail est dans « Récupérer le TD suivant ».

## Installation (une seule fois, environ 15 minutes)

L'environnement est géré par **uv**. uv installe aussi la bonne version de Python : inutile d'installer
Python vous-même.

1. Installez uv, une fois par ordinateur (aucun droit d'administrateur n'est nécessaire), puis fermez et
   rouvrez le terminal :

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   # macOS et Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Récupérez le dépôt, créez l'environnement et vérifiez l'installation :

   ```bash
   git clone https://github.com/FranklinB13/applications-modeles-ia-td
   cd applications-modeles-ia-td
   uv sync
   uv run python verifier_installation.py --donnees
   ```

3. Créez votre dépôt de rendu, celui que vous rendrez à la fin du module. C'est à faire une seule fois, au
   TD 1, et c'est l'étape la plus délicate : suivez-la dans l'ordre.

   a. Dites à Git qui vous êtes (une fois par ordinateur) :

      ```bash
      git config --global user.name "Prénom Nom"
      git config --global user.email "votre.adresse@cyu.fr"
      ```

   b. Sur GitHub, créez un dépôt **privé** et **vide** : ne cochez ni README, ni `.gitignore`, ni licence.
      Dans ce dépôt, ajoutez l'enseignant comme collaborateur (Settings, Collaborators : `FranklinB13`).

   c. Dans le dossier du dépôt cloné, reliez les deux dépôts (remplacez `<votre-compte>/<votre-depot>`) :

      ```bash
      git remote rename origin prof
      git remote add origin https://github.com/<votre-compte>/<votre-depot>
      git push -u origin main
      ```

   Ne faites pas de *fork* : le fork d'un dépôt public est public, et tout le monde verrait vos réponses.
   Après cette étape, votre dossier est relié à deux dépôts : `origin`, le vôtre, où part votre travail, et
   `prof`, celui-ci, d'où arrivent les nouveaux TD. Vérifiez-le avec `git remote -v`.

`uv sync` crée l'environnement dans `.venv/`, avec exactement les versions inscrites dans `uv.lock` : les mêmes
pour tout le monde, celles avec lesquelles les TD ont été vérifiés. PyTorch y est installé en version CPU, plus
légère, qui suffit pour tous les TD. Ne modifiez pas `uv.lock` et n'ajoutez pas de bibliothèque sans en parler
à l'enseignant.

## Lancer un TD

```bash
uv run marimo edit tp1_vision.py
```

`uv run` lance la commande dans l'environnement du dépôt : il n'y a rien à activer.

Le notebook s'ouvre dans le navigateur. Vos modifications sont enregistrées dans le fichier `.py`.

Trois choses à savoir sur marimo :

- quand vous modifiez une cellule, toutes celles qui utilisent ses résultats se relancent automatiquement ;
- une variable ne peut être définie que dans **une seule cellule**. Dans vos cellules de test, préfixez vos
  variables d'un tiret bas (`_x`, `_modele`) pour qu'elles restent locales à la cellule. Sinon, marimo affiche
  une erreur du type *« This cell redefines variables from other cells »* ;
- pour arrêter marimo, fermez l'onglet puis faites Ctrl + C dans le terminal.

## Récupérer le TD suivant

Au début de chaque séance, dans le dossier du dépôt :

```bash
uv run python recuperer_td.py
```

Le script fait trois choses, et affiche ce qu'il a fait :

1. il enregistre votre travail en cours (un commit), pour que rien ne se perde ;
2. il récupère les nouveaux fichiers de l'enseignant : le notebook du TD, ses tests dans `outils/`, et pour
   certains TD un modèle fourni dans `modeles/`. Il vous donne la commande pour ouvrir le nouveau TD ;
3. il envoie le tout sur votre dépôt de rendu (`git push`).

Votre travail n'est jamais écrasé. Deux cas particuliers, que le script gère et explique :

- si vous avez modifié un fichier fourni (dans `outils/` par exemple) et que l'enseignant l'a mis à jour, le
  script reprend la version de l'enseignant ;
- si l'enseignant a corrigé un notebook que vous avez déjà commencé, le script annule la récupération et ne
  touche à rien : prévenez l'enseignant.

Si une bibliothèque a changé de version, `uv run` met l'environnement à jour tout seul au lancement suivant :
il n'y a rien d'autre à faire.

Sans le script, les mêmes étapes à la main :

```bash
git add -A
git commit -m "TD 1 : fin de séance"
git pull --no-rebase --no-edit prof main
git push
```

À la main, un conflit sur un fichier de `outils/` se règle en reprenant la version de l'enseignant :
`git checkout prof/main -- outils/`, puis `git commit --no-edit`.

## Données

| Données | Utilisées dans | Comment les obtenir |
|---|---|---|
| Oxford-IIIT Pet | TD 1 | Téléchargé automatiquement (`verifier_installation.py --donnees`) |
| Construction-PPE (EPI sur chantier) | TD 2 | Téléchargé automatiquement (`verifier_installation.py --donnees`) |
| ETTh1 (transformateur électrique) | TD 3 | Téléchargé automatiquement |
| SKAB (capteurs d'une pompe) | Projet, brique 3 | Téléchargé automatiquement |
| Fashion-MNIST (vêtements) | TD 4, pour aller plus loin | Téléchargé automatiquement |
| Cora (articles scientifiques et citations) | TD 5a | Téléchargé automatiquement |
| CartPole, MountainCar (simulateurs) | TD 5b | Aucune donnée : fournis par la bibliothèque Gymnasium |
| NEU-DET, défauts de surface d'acier | Projet, briques 1, 2 et 4 | Rechercher « NEU-DET » sur Kaggle (compte gratuit) : choisir une version qui contient les images **et** les annotations `.xml`, et décompresser le tout dans `donnees/NEU/` |

Les dossiers `donnees/`, `cache/`, `sorties/` et les poids téléchargés dans `modeles/` ne sont pas versionnés
(voir `.gitignore`). Seuls les modèles fournis pour les TD (`modeles/tp2_*.pt`, `modeles/tp4_*.pt`) le sont.

## Organisation du dépôt

```
tp1_vision.py ... tp6_industrialisation.py   les TD, à compléter, mis en ligne un par un (le Jour 5 compte deux fichiers :
                                            tp5a_graphes.py le matin, tp5b_renforcement.py l'après-midi)
outils/                                     code fourni (chargement, affichage, tests) : ne pas modifier
recuperer_td.py                             récupère le TD suivant (uv run python recuperer_td.py)
modeles/                                    modèles fournis et poids pré-entraînés téléchargés
projet/                                     vos briques du projet Usine 4.0 (partez de brique_modele.py)
sorties/                                    résultats d'entraînement (non versionnés)
__marimo__/                                 copies des notebooks avec leurs sorties, créées par marimo (à versionner)
```

## Rendu

Faites des commits réguliers (au moins un par séance) et poussez-les sur votre dépôt privé (`git push`) :
l'historique Git fait partie de l'évaluation.

Un notebook marimo est un fichier Python : il contient votre code et vos réponses, mais pas vos résultats
(figures, tableaux, scores). Pendant que vous travaillez, marimo enregistre donc à côté de chaque notebook une
copie avec ses sorties, dans un dossier `__marimo__/` (par exemple `__marimo__/tp1_vision.html`, ou
`projet/__marimo__/brique1_classification.html`). Ajoutez ces fichiers `.html` à vos commits : ce sont eux
qui montrent ce que vous avez obtenu.
