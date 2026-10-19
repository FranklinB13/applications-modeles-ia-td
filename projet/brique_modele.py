import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="Usine 4.0 - Brique à nommer",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Le dossier outils/ est à la racine du dépôt, un niveau au-dessus de projet/.
    sys.path.insert(0, str(Path(mo.notebook_dir()).parent / "outils"))
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return mo, np, plt, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Projet Usine 4.0 — Brique ... : titre de la brique

    **Groupe :** ... · **Membres :** ...

    **Mode d'emploi de ce modèle**

    - Copiez ce fichier dans `projet/` sous le nom de la brique indiqué à la fin du TD
      (`brique1_classification.py`, `brique2_localisation.py`, `brique3_anomalies.py`, `brique4_defauts_rares.py`),
      puis ouvrez la copie : `marimo edit projet/brique1_classification.py`.
    - Les fonctions fournies restent accessibles par `tp.` (chargement des données, affichage, cache).
      Les fonctions que vous avez écrites dans un TD ne s'importent pas d'un notebook à l'autre : copiez dans
      une cellule de cette brique celles dont vous avez besoin.
    - Dans une cellule, une variable dont le nom commence par un tiret bas reste locale à la cellule.
    - Fixez les graines (`tp.fixer_graine(0)`) et gardez les résultats des calculs longs
      (`tp.sauver_cache`, `tp.charger_cache`) pour ne pas les relancer à chaque ouverture.
    - Ne réglez rien sur le jeu de test : gardez une partie des données pour la validation, et ne mesurez sur
      le test qu'une fois vos choix faits.
    - Remplacez les `...` par votre travail et supprimez ce mode d'emploi avant le rendu.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. La question et les données

    Quelle question métier cette brique traite-t-elle ? Quelles données, en quelle quantité, comment
    sont-elles découpées (entraînement, validation, test) ?

    ...
    """)
    return


@app.cell
def _(tp):
    # Chargement et découpage des données. Exemple pour les images NEU (voir outils/tp_utils.py) :
    # (images_train, y_train, images_val, y_val, images_test, y_test,
    #  noms) = tp.charger_neu(n_train_par_classe=30, n_val_par_classe=20, n_test_par_classe=100)
    dossier_donnees = tp.DONNEES
    return (dossier_donnees,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. La baseline classique

    Une méthode sans réseau de neurones, raisonnable pour ce problème : laquelle, pourquoi, quel score ?

    ...
    """)
    return


@app.cell
def _():
    # Votre baseline : remplacez cette ligne par votre code ; réglez-la sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    score_baseline = None
    return (score_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. L'approche par deep learning

    Quel modèle, quelle stratégie d'adaptation, quels réglages essayés (comparés sur la validation) ?

    ...
    """)
    return


@app.cell
def _():
    # Votre modèle : remplacez cette ligne par votre code ; réglez-le sur la validation,
    # puis gardez son score sur le test, mesuré une seule fois.
    score_modele = None
    return (score_modele,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Résultats et analyse

    Tableau comparatif (mêmes données, même métrique pour toutes les méthodes), cas d'échec, seuil ou réglage
    retenu et sa justification en termes de coût métier.

    ...
    """)
    return


@app.cell
def _(mo, score_baseline, score_modele, tp):
    # Une ligne par méthode, la même métrique pour toutes ; ajoutez ensuite vos figures et vos cas d'échec.
    tableau_final = [
        {"méthode": "baseline classique", "score sur le test": score_baseline},
        {"méthode": "deep learning", "score sur le test": score_modele},
    ]
    tp.tableau(tableau_final, mo)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Fiche de décision

    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | ... |
    | Baseline classique et son score | ... |
    | Modèle retenu et pourquoi | ... |
    | Métrique principale et score obtenu | ... |
    | Seuil ou réglage retenu, justification métier | ... |
    | Coût (annotation, entraînement, inférence, matériel) | ... |
    | Principaux cas d'échec | ... |
    | Le deep learning est-il justifié ici ? | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... |
    """)
    return


if __name__ == "__main__":
    app.run()
