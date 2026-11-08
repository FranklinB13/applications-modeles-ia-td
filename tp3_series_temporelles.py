import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="TD 3 - Séries temporelles",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
    import tests_tp3 as tests
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return mo, nn, np, plt, tests, time, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TD 3 — Séries temporelles

    **Module Application au deep learning · ING3 IA · durée : 3h30 (l'étape 9 peut se terminer en autonomie)**

    Capteurs d'usine, consommation électrique, trafic réseau : une grande partie des données industrielles
    sont des séries temporelles. Deux questions reviennent toujours : **que va-t-il se passer ?** (prévision)
    et **se passe-t-il quelque chose d'anormal ?** (détection d'anomalies). Vous allez traiter les deux sur
    les relevés d'un transformateur électrique, en comparant à chaque fois des méthodes classiques et des
    réseaux de neurones. Attendez-vous à des surprises : ici, le deep learning ne gagne pas toujours.

    **Rappel du fonctionnement du notebook**

    - Exécuter une cellule : bouton de lecture à sa droite, ou Ctrl + Entrée (Cmd + Entrée sur Mac).
    - Cellules **À compléter** : remplacez le `raise NotImplementedError` par votre code, puis lisez la cellule
      de tests juste en dessous. Une étape se débloque quand les tests de l'étape précédente sont réussis.
    - Les calculs longs ne partent qu'au clic sur leur bouton, et leurs résultats sont gardés en cache.
    - Le code de certaines cellules fournies est replié : vous pouvez l'afficher depuis le menu de la cellule.
    - Cellules **Votre réponse** : double-cliquez sur le texte pour l'éditer.
    - Pour tester du code librement, ajoutez une cellule avec le bouton + qui apparaît entre deux cellules.
    - **Particularité de marimo** : une variable ne peut être définie que dans **une seule cellule**. Dans vos
      cellules de test, donnez à vos variables un nom qui commence par un tiret bas (`_x`, `_modele`) : elles
      restent alors locales à la cellule. Sinon, marimo affiche une erreur du type
      *« This cell redefines variables from other cells »* : il suffit de renommer la variable.

    | Étape | Contenu | Durée indicative |
    |---|---|---|
    | 0 | Vérification de l'installation | 5 min |
    | 1 | Le cours en bref, avec démonstrations | 20 min |
    | 2 | Explorer et préparer les données | 30 min |
    | 3 | Baselines classiques de prévision | 20 min |
    | 4 | Trois réseaux pour la prévision : LSTM, TCN, Transformer | 50 min |
    | 5 | Détection d'anomalies par autoencodeur | 40 min |
    | 6 | Choisir un seuil d'alerte : le coût métier | 15 min |
    | 7 | Fiche de décision ingénieur | 10 min |
    | 8 | Bilan : deep learning ou classique ? | 10 min |
    | 9 | Brique projet Usine 4.0 (volet au choix) : anticiper les pannes d'une pompe | 30 min |
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch, tp):
    mo.vstack([mo.md("## Étape 0 — Vérification de l'installation"),
               mo.callout(mo.md(f"- Matériel utilisé : **{tp.appareil()}**\n- Threads CPU : {torch.get_num_threads()}\n"
                                "- Tout ce TD tourne confortablement sur CPU : les modèles sont petits."), kind="info")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 1 — Le cours en bref

    ### 1.1 Rappel : deux briques que vous connaissez déjà

    - **La convolution** (vue avec les CNN) : un filtre glisse sur l'image. Sur une série, il glisse sur le
      temps : c'est une convolution 1D. Elle détecte des motifs locaux (un pic, une montée).
    - **L'attention** (vue en NLP, rappelée au TD 1) : chaque élément d'une séquence regarde tous les autres.
      Ici, les éléments seront des morceaux de série.

    ### 1.2 Trois règles pratiques, plus importantes que le choix du modèle

    1. **On ne mélange jamais le temps.** Le test doit être *postérieur* à l'entraînement. Mélanger les points
       au hasard revient à prédire le passé avec le futur.
    2. **On normalise avec les statistiques de l'entraînement uniquement.** Utiliser la moyenne du test est une
       fuite d'information.
    3. **On commence par une baseline.** « Demain ressemblera à aujourd'hui » est souvent difficile à battre.

    ### 1.3 Trois familles de réseaux pour les séquences

    **Réseaux récurrents : LSTM et GRU.** Le réseau lit la série pas à pas en entretenant un **état caché**,
    sa mémoire. À chaque pas, des *portes* décident ce qu'il oublie, ce qu'il retient et ce qu'il transmet.
    Deux limites : l'information ancienne se dilue sur les longues séquences, et le calcul est séquentiel
    (impossible de traiter tous les pas en parallèle).

    **TCN, Temporal Convolutional Network.** Un empilement de convolutions 1D **causales** (la sortie à
    l'instant $t$ ne dépend que des instants $\le t$) et **dilatées** (le filtre saute des points : écart de 1,
    puis 2, 4, 8...). La dilatation fait croître très vite le **champ réceptif**, c'est-à-dire la longueur de
    passé que voit le réseau. Tout se calcule en parallèle.
    """)
    return


@app.cell
def _(mo):
    demo_blocs = mo.ui.slider(1, 7, value=4, label="Nombre de couches", show_value=True)
    demo_dilate = mo.ui.checkbox(value=True, label="Dilatation doublée à chaque couche")
    mo.hstack([demo_blocs, demo_dilate], justify="start", gap=2)
    return demo_blocs, demo_dilate


@app.cell(hide_code=True)
def _(demo_blocs, demo_dilate, mo, plt):
    _n, _k = demo_blocs.value, 3
    _dil = [2**i if demo_dilate.value else 1 for i in range(_n)]
    _champ = 1 + (_k - 1) * sum(_dil)
    _fig, _ax = plt.subplots(figsize=(7.5, 0.5 + 0.45 * (_n + 1)))
    _largeur = max(_champ + 2, 20)
    _actifs = {_largeur - 1}
    for _c in range(_n, -1, -1):
        _ax.scatter(range(_largeur), [_c] * _largeur, s=14, color="#cfd6dd")
        _ax.scatter(sorted(_actifs), [_c] * len(_actifs), s=22, color="tab:blue")
        if _c > 0:
            _d = _dil[_c - 1]
            _suivants = set()
            for _t in _actifs:
                for _j in range(_k):
                    _src = _t - _j * _d
                    if _src >= 0:
                        _ax.plot([_src, _t], [_c - 1, _c], color="tab:blue", lw=0.5, alpha=0.5)
                        _suivants.add(_src)
            _actifs = _suivants
    _ax.set_yticks(range(_n + 1), ["entrée"] + [f"couche {i + 1} (dilatation {_dil[i]})" for i in range(_n)], fontsize=7)
    _ax.set_xticks([])
    _ax.set_xlabel("temps (le point le plus à droite est l'instant présent)", fontsize=8)
    for _s in ("top", "right", "bottom"):
        _ax.spines[_s].set_visible(False)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Champ réceptif : la dernière sortie voit **{_champ} pas de temps** dans le passé "
                           f"(noyau de taille 3, {_n} couches).")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Transformers temporels : PatchTST.** Appliquer l'attention point par point coûte cher (le coût croît
    comme le carré de la longueur) et un point isolé porte peu de sens. PatchTST découpe la série en **patches**
    de quelques dizaines de points, exactement comme ViT découpe une image, et applique l'attention entre patches.

    ### 1.4 Détecter des anomalies sans exemples d'anomalies

    Les pannes sont rares : on n'a presque jamais assez d'exemples pour entraîner un classifieur. On apprend
    donc ce qui est **normal**. Un **autoencodeur** compresse une fenêtre de signal dans un petit vecteur puis
    la reconstruit. Entraîné sur des données normales, il reconstruit mal ce qu'il n'a jamais vu : l'**erreur
    de reconstruction** sert de score d'anomalie. Il reste à choisir un seuil, et c'est une décision métier.

    Les équivalents classiques existent : l'**ACP** (PCA) fait la même chose avec une compression linéaire,
    et l'**Isolation Forest** isole les points atypiques avec des arbres de décision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 2 — Explorer et préparer les données

    **ETTh1** : deux ans de relevés horaires d'un transformateur électrique. Six mesures de charge
    (HUFL, HULL, MUFL, MULL, LUFL, LULL) et la **température d'huile OT**, la variable à prévoir :
    une huile trop chaude signale un transformateur en surcharge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_ett = mo.ui.run_button(label="Charger ETTh1")
    bouton_ett
    return (bouton_ett,)


@app.cell(hide_code=True)
def _(bouton_ett, mo, tp):
    _deja_la = (tp.DONNEES / "ett" / "ETTh1.csv").exists()
    mo.stop(not (tp.lance(bouton_ett) or _deja_la), mo.md("Cliquez sur le bouton pour charger les données (2,5 Mo)."))
    ett = tp.charger_ett()
    return (ett,)


@app.cell(hide_code=True)
def _(ett, mo, plt):
    _fig, _ax = plt.subplots(3, 1, figsize=(9, 6.2))
    _ax[0].plot(ett["date"], ett["OT"], lw=0.5)
    _ax[0].set_title("Température d'huile (OT) sur deux ans", fontsize=9)
    _deux_sem = ett.iloc[24 * 200:24 * 214]
    _ax[1].plot(_deux_sem["date"], _deux_sem["OT"], label="OT")
    _ax[1].plot(_deux_sem["date"], _deux_sem["HUFL"], label="HUFL (charge)", alpha=0.7)
    _ax[1].set_title("Zoom sur deux semaines", fontsize=9)
    _ax[1].legend(fontsize=7)
    _par_heure = ett.groupby(ett["date"].dt.hour)["OT"].mean()
    _ax[2].bar(_par_heure.index, _par_heure.values - _par_heure.mean(), color="tab:blue")
    _ax[2].set_title("Écart moyen à la moyenne selon l'heure de la journée (°C)", fontsize=9)
    _ax[2].set_xlabel("heure")
    _fig.tight_layout()
    mo.vstack([mo.md(f"**{len(ett)}** relevés horaires, du {ett['date'].min():%d/%m/%Y} au {ett['date'].max():%d/%m/%Y}."), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Quelles régularités voyez-vous (journée, saison) ? Y a-t-il une tendance de long terme ?
    2. Le niveau de la température en fin de période est-il le même qu'au début ? Quelle difficulté
       cela pose-t-il pour un modèle entraîné sur le début et testé sur la fin ?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. ...
    2. ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 1 : découper dans le temps et normaliser

    Entrée : `valeurs`, un tableau `(T, C)` (T instants, C canaux), et trois tailles.

    1. Découpez **dans l'ordre chronologique** : les `n_train` premiers points, puis `n_val`, puis `n_test`.
    2. Calculez la moyenne et l'écart-type de **chaque canal sur l'entraînement uniquement** (`axis=0`).
    3. Normalisez les trois parties avec ces mêmes statistiques : $(x - \text{moyenne}) / \text{écart-type}$.

    Renvoyez `(train, val, test, moyenne, ecart_type)`.
    """)
    return


@app.function
def decouper_et_normaliser(valeurs, n_train, n_val, n_test):
    """Découpage chronologique puis normalisation avec les statistiques de l'entraînement."""
    # TODO 1
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo1, _affichage = tp.verifier(tests.tests_decoupage(decouper_et_normaliser), mo)
    _affichage
    return (ok_todo1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 2 : fenêtres glissantes

    Un modèle de prévision reçoit une fenêtre de passé et prédit la suite. À partir d'une série 1D de longueur
    $T$, construisez toutes les paires (entrée, cible) :

    - `X[i]` = les `taille_entree` points à partir de l'instant `i` ;
    - `Y[i]` = les `horizon` points **qui suivent immédiatement**.

    Il y a $T - \text{taille\_entree} - \text{horizon} + 1$ fenêtres. Renvoyez `X` de forme `(N, taille_entree)`
    et `Y` de forme `(N, horizon)`.
    """)
    return


@app.cell
def _(np):
    def creer_fenetres(serie, taille_entree, horizon):
        """Série 1D -> X (N, taille_entree), Y (N, horizon)."""
        # TODO 2
        raise NotImplementedError("À compléter")
    return (creer_fenetres,)


@app.cell(hide_code=True)
def _(creer_fenetres, mo, tests, tp):
    ok_todo2, _affichage = tp.verifier(tests.tests_fenetres(creer_fenetres), mo)
    _affichage
    return (ok_todo2,)


@app.cell
def _(creer_fenetres, ett, mo, np, ok_todo1, ok_todo2, tp):
    tp.bloquer_si(not (ok_todo1 and ok_todo2), mo, "Terminez les TODO 1 et 2 pour préparer les données.")
    ENTREE, HORIZON = 96, 24  # 4 jours de passé pour prévoir les 24 prochaines heures
    _valeurs = ett.drop(columns="date").to_numpy(dtype=np.float32)
    # Découpage standard de ce jeu : 12 mois d'entraînement, 4 de validation, 4 de test
    serie_tr, serie_va, serie_te, moyennes, ecarts = decouper_et_normaliser(_valeurs, 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24)
    X_tr, Y_tr = creer_fenetres(serie_tr[:, -1], ENTREE, HORIZON)  # la dernière colonne est OT
    X_va, Y_va = creer_fenetres(serie_va[:, -1], ENTREE, HORIZON)
    X_te, Y_te = creer_fenetres(serie_te[:, -1], ENTREE, HORIZON)
    ecart_ot = float(ecarts[-1])
    mo.md(f"Fenêtres : **{len(X_tr)}** d'entraînement, **{len(X_va)}** de validation, **{len(X_te)}** de test. "
          f"Entrée : {ENTREE} heures, horizon : {HORIZON} heures. Un écart de 1 en unité normalisée vaut {ecart_ot:.1f} °C.")
    return (
        ENTREE,
        HORIZON,
        X_te,
        X_tr,
        X_va,
        Y_te,
        Y_tr,
        Y_va,
        ecart_ot,
        serie_te,
        serie_tr,
        serie_va,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 3 — Baselines classiques de prévision

    ### À compléter — TODO 3 : deux prévisions sans apprentissage

    - **Naïve** : la prévision est la dernière valeur observée, répétée sur tout l'horizon.
    - **Saisonnière** : « il fera demain à 15 h la même température qu'aujourd'hui à 15 h ». La prévision à
      l'instant $t + h$ est la valeur observée à $t + h - \text{periode}$. Avec `horizon` $\le$ `periode`,
      ce sont des valeurs prises parmi les `periode` derniers points de la fenêtre.

    `X` a la forme `(N, taille_entree)`. Les deux fonctions renvoient un tableau `(N, horizon)`.
    """)
    return


@app.cell
def _(np):
    def prevision_naive(X, horizon):
        """Répète la dernière valeur observée."""
        # TODO 3a
        raise NotImplementedError("À compléter")

    def prevision_saisonniere(X, horizon, periode=24):
        """Recopie ce qui s'est passé une période plus tôt (horizon <= periode)."""
        # TODO 3b
        raise NotImplementedError("À compléter")
    return prevision_naive, prevision_saisonniere


@app.cell(hide_code=True)
def _(mo, prevision_naive, prevision_saisonniere, tests, tp):
    ok_todo3, _affichage = tp.verifier(tests.tests_baselines(prevision_naive, prevision_saisonniere), mo)
    _affichage
    return (ok_todo3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On ajoute une troisième baseline, **avec** apprentissage mais sans réseau de neurones : une **régression
    linéaire régularisée** (Ridge). Chaque heure future est une somme pondérée des 96 heures passées.
    La métrique est l'erreur absolue moyenne (MAE), convertie en degrés.
    """)
    return


@app.cell
def _(
    HORIZON,
    X_te,
    X_tr,
    Y_te,
    Y_tr,
    ecart_ot,
    mo,
    np,
    ok_todo3,
    prevision_naive,
    prevision_saisonniere,
    time,
    tp,
):
    tp.bloquer_si(not ok_todo3, mo, "Terminez le TODO 3 pour évaluer les baselines.")
    from sklearn.linear_model import Ridge

    def mae_degres(prediction, cible):
        return float(np.abs(prediction - cible).mean() * ecart_ot)

    _t = time.perf_counter()
    _ridge = Ridge(alpha=10).fit(X_tr, Y_tr)
    _duree = time.perf_counter() - _t
    predictions_classiques = {
        "naïve (dernière valeur)": prevision_naive(X_te, HORIZON),
        "saisonnière (24 h plus tôt)": prevision_saisonniere(X_te, HORIZON),
        "régression linéaire (Ridge)": _ridge.predict(X_te),
    }
    resultats_classiques = [
        {"méthode": nom, "famille": "classique", "MAE test (°C)": round(mae_degres(p, Y_te), 3),
         "entraînement (s)": round(_duree, 2) if "Ridge" in nom else 0.0,
         "paramètres": X_tr.shape[1] * HORIZON + HORIZON if "Ridge" in nom else 0}
        for nom, p in predictions_classiques.items()
    ]
    tp.tableau(resultats_classiques, mo)
    return mae_degres, predictions_classiques, resultats_classiques


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question** : la prévision saisonnière fait-elle mieux que la naïve ? Est-ce ce que vous attendiez après
    l'exploration de l'étape 2 ? Proposez une explication.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 4 — Trois réseaux pour la prévision

    Les trois modèles reçoivent une fenêtre `(lot, 96)` et renvoient une prévision `(lot, 24)`.

    ### 4.1 LSTM (fourni)

    Le LSTM lit les 96 valeurs une à une. On garde son état caché final, qui résume la fenêtre, et une couche
    linéaire le transforme en 24 valeurs prévues.
    """)
    return


@app.cell
def _(HORIZON, nn):
    class PrevisionLSTM(nn.Module):
        def __init__(self, cache=64):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=cache, batch_first=True)
            self.tete = nn.Linear(cache, HORIZON)

        def forward(self, x):                      # x : (lot, 96)
            sorties, _ = self.lstm(x.unsqueeze(-1))  # (lot, 96, cache) : l'état caché à chaque pas
            return self.tete(sorties[:, -1])       # on ne garde que le dernier

    return (PrevisionLSTM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 TCN

    ### À compléter — TODO 4 : la convolution causale

    Une `nn.Conv1d` ordinaire de noyau $k$ et de dilatation $d$ raccourcit la séquence de $(k - 1) \times d$
    points. Pour garder la longueur **et** interdire de regarder le futur, on ajoute ce nombre de zéros
    **à gauche uniquement** (dans le passé) avant la convolution.

    Complétez `forward` : `torch.nn.functional.pad(x, (gauche, droite))` ajoute des zéros sur la dernière dimension.
    Le test vérifie la causalité : modifier le futur ne doit rien changer aux sorties passées.
    """)
    return


@app.cell
def _(nn, torch):
    class ConvCausale(nn.Module):
        def __init__(self, canaux_entree, canaux_sortie, noyau, dilatation):
            super().__init__()
            self.marge = (noyau - 1) * dilatation
            self.conv = nn.Conv1d(canaux_entree, canaux_sortie, noyau, dilation=dilatation)

        def forward(self, x):  # x : (lot, canaux, temps)
            """Convolution qui conserve la longueur et ne regarde jamais le futur."""
            # TODO 4
            raise NotImplementedError("À compléter")
    return (ConvCausale,)


@app.cell(hide_code=True)
def _(ConvCausale, mo, tests, tp):
    ok_todo4, _affichage = tp.verifier(tests.tests_conv_causale(ConvCausale), mo)
    _affichage
    return (ok_todo4,)


@app.cell
def _(ConvCausale, HORIZON, nn, torch):
    class BlocTCN(nn.Module):
        """Deux convolutions causales dilatées + une connexion résiduelle (comme dans ResNet)."""

        def __init__(self, canaux_entree, canaux, dilatation):
            super().__init__()
            self.conv1 = ConvCausale(canaux_entree, canaux, 3, dilatation)
            self.conv2 = ConvCausale(canaux, canaux, 3, dilatation)
            self.raccourci = nn.Conv1d(canaux_entree, canaux, 1) if canaux_entree != canaux else nn.Identity()
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            y = self.dropout(torch.relu(self.conv1(x)))
            y = self.dropout(torch.relu(self.conv2(y)))
            return torch.relu(y + self.raccourci(x))

    class PrevisionTCN(nn.Module):
        def __init__(self, canaux=32, n_blocs=5):
            super().__init__()
            # dilatations 1, 2, 4, 8, 16 : champ réceptif de 125 pas, de quoi couvrir les 96 heures d'entrée
            self.blocs = nn.Sequential(*[BlocTCN(1 if i == 0 else canaux, canaux, 2**i) for i in range(n_blocs)])
            self.tete = nn.Linear(canaux, HORIZON)

        def forward(self, x):                           # x : (lot, 96)
            y = self.blocs(x.unsqueeze(1))              # (lot, canaux, 96)
            return self.tete(y[:, :, -1])               # représentation du dernier instant

    return (PrevisionTCN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Transformer à patches (PatchTST simplifié)

    ### À compléter — TODO 5 : découper la série en patches

    Entrée : `x` de forme `(lot, longueur)`. Sortie : `(lot, n_patches, taille)`, où le patch $j$ contient les
    points `j * pas` à `j * pas + taille - 1`. Il y a $(\text{longueur} - \text{taille}) / \text{pas} + 1$ patches.

    Une ligne suffit avec `x.unfold(dimension, taille, pas)`.
    """)
    return


@app.function
def decouper_en_patches(x, taille, pas):
    """(lot, longueur) -> (lot, n_patches, taille)."""
    # TODO 5
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo5, _affichage = tp.verifier(tests.tests_patches(decouper_en_patches), mo)
    _affichage
    return (ok_todo5,)


@app.cell
def _(ENTREE, HORIZON, nn, torch):
    class PrevisionPatchTST(nn.Module):
        def __init__(self, taille_patch=16, pas=8, dim=64):
            super().__init__()
            self.taille_patch, self.pas = taille_patch, pas
            n_patches = (ENTREE - taille_patch) // pas + 1          # 11 patches de 16 heures, décalés de 8
            self.projection = nn.Linear(taille_patch, dim)           # un patch devient un vecteur (un « token »)
            self.position = nn.Parameter(torch.zeros(1, n_patches, dim))  # encodage de position appris
            couche = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(couche, num_layers=2)
            self.tete = nn.Linear(n_patches * dim, HORIZON)

        def forward(self, x):                                        # x : (lot, 96)
            patches = decouper_en_patches(x, self.taille_patch, self.pas)   # (lot, 11, 16)
            tokens = self.projection(patches) + self.position        # (lot, 11, dim)
            return self.tete(self.transformer(tokens).flatten(1))    # attention entre patches, puis prévision

    return (PrevisionPatchTST,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Une astuce qui compte plus que l'architecture

    Vous l'avez vu à l'étape 2 : le niveau de la température dérive au fil des mois. Un réseau entraîné sur
    une période « chaude » voit arriver en test des valeurs qu'il n'a jamais rencontrées.

    La parade est simple : **normaliser chaque fenêtre par rapport à elle-même**. On retire à la fenêtre sa
    dernière valeur, le réseau prédit donc des *écarts* par rapport au présent, puis on rajoute cette valeur à
    la prévision. Le réseau ne voit plus le niveau absolu, seulement la forme.

    ### À compléter — TODO 6 : normalisation par fenêtre

    `PrevisionNormalisee` enveloppe n'importe quel modèle. Dans `forward` : calculez `dernier = x[:, -1:]`
    (forme `(lot, 1)`), appliquez le modèle à `x - dernier`, et ajoutez `dernier` au résultat.
    """)
    return


@app.cell
def _(nn):
    class PrevisionNormalisee(nn.Module):
        def __init__(self, modele):
            super().__init__()
            self.modele = modele

        def forward(self, x):  # x : (lot, longueur)
            """Le modèle interne prédit des écarts par rapport à la dernière valeur observée."""
            # TODO 6
            raise NotImplementedError("À compléter")
    return (PrevisionNormalisee,)


@app.cell(hide_code=True)
def _(PrevisionNormalisee, mo, tests, tp):
    ok_todo6, _affichage = tp.verifier(tests.tests_norm_fenetre(PrevisionNormalisee), mo)
    _affichage
    return (ok_todo6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.5 Entraîner et comparer

    Lancez une première fois **sans** la normalisation par fenêtre, puis une seconde fois **avec**.
    Chaque lancement entraîne les trois réseaux (2 à 3 minutes sur CPU avec 6 époques). Les résultats
    s'ajoutent au tableau comparatif.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cfg_norm = mo.ui.checkbox(value=False, label="Normalisation par fenêtre (TODO 6)")
    cfg_epoques = mo.ui.slider(2, 12, value=6, label="Époques", show_value=True)
    bouton_prevision = mo.ui.run_button(label="Entraîner LSTM, TCN et PatchTST")
    mo.vstack([mo.hstack([cfg_norm, cfg_epoques], justify="start", gap=2), bouton_prevision])
    return bouton_prevision, cfg_epoques, cfg_norm


@app.cell
def _(
    PrevisionLSTM,
    PrevisionNormalisee,
    PrevisionPatchTST,
    PrevisionTCN,
    X_te,
    X_tr,
    X_va,
    Y_tr,
    Y_va,
    bouton_prevision,
    cfg_epoques,
    cfg_norm,
    mo,
    ok_todo4,
    ok_todo5,
    ok_todo6,
    torch,
    tp,
):
    compteur_entrainements = 0
    if not (ok_todo4 and ok_todo5 and ok_todo6):
        _sortie = mo.callout(mo.md("Terminez les TODO 4, 5 et 6 pour débloquer l'entraînement."), kind="warn")
    elif not tp.lance(bouton_prevision):
        _sortie = mo.md("Cliquez sur le bouton pour lancer l'entraînement.")
    else:
        _suffixe = "norm" if cfg_norm.value else "brut"
        for _nom, _classe in [("LSTM", PrevisionLSTM), ("TCN", PrevisionTCN), ("PatchTST", PrevisionPatchTST)]:
            tp.fixer_graine(0)
            _modele = PrevisionNormalisee(_classe()) if cfg_norm.value else _classe()
            _modele, _hist = tp.entrainer_prevision(
                _modele, X_tr, Y_tr, X_va, Y_va, epoques=cfg_epoques.value,
                progression=lambda it, n=_nom: mo.status.progress_bar(it, title=f"{n} ({_suffixe})"))
            with torch.no_grad():
                _pred = _modele(torch.as_tensor(X_te, dtype=torch.float32).to(tp.appareil())).cpu().numpy()
            tp.sauver_cache(f"tp3_prev_{_nom}_{_suffixe}", pred=_pred, duree=_hist["duree"],
                            params=tp.nombre_parametres(_modele), epoques=cfg_epoques.value)
            compteur_entrainements += 1
        _sortie = mo.md(f"Entraînement terminé ({_suffixe}). Le tableau ci-dessous est à jour.")
    _sortie
    return (compteur_entrainements,)


@app.cell
def _(Y_te, compteur_entrainements, mae_degres, mo, resultats_classiques, tp):
    _ = compteur_entrainements  # relance cette cellule après chaque entraînement
    predictions_deep, _lignes = {}, []
    for _nom in ["LSTM", "TCN", "PatchTST"]:
        for _suffixe, _libelle in [("brut", "sans normalisation par fenêtre"), ("norm", "avec normalisation par fenêtre")]:
            _c = tp.charger_cache(f"tp3_prev_{_nom}_{_suffixe}")
            if _c is not None:
                predictions_deep[f"{_nom}, {_libelle}"] = _c["pred"]
                _lignes.append({"méthode": f"{_nom}, {_libelle}", "famille": "deep learning",
                                "MAE test (°C)": round(mae_degres(_c["pred"], Y_te), 3),
                                "entraînement (s)": round(float(_c["duree"]), 1), "paramètres": int(_c["params"])})
    tableau_prevision = sorted(resultats_classiques + _lignes, key=lambda r: r["MAE test (°C)"])
    tp.tableau(tableau_prevision, mo)
    return (predictions_deep,)


@app.cell(hide_code=True)
def _(X_te, mo):
    exemple = mo.ui.slider(0, len(X_te) - 1, step=24, value=24 * 40, label="Fenêtre de test affichée", show_value=True)
    exemple
    return (exemple,)


@app.cell(hide_code=True)
def _(
    ENTREE,
    HORIZON,
    X_te,
    Y_te,
    ecart_ot,
    exemple,
    plt,
    predictions_classiques,
    predictions_deep,
):
    _i = exemple.value
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.plot(range(-48, 0), X_te[_i, -48:] * ecart_ot, color="black", label="passé (48 dernières heures)")
    _ax.plot(range(HORIZON), Y_te[_i] * ecart_ot, color="black", ls="--", label="réalité")
    for _nom, _p in {**predictions_classiques, **predictions_deep}.items():
        if "sans normalisation" in _nom:
            continue
        _ax.plot(range(HORIZON), _p[_i] * ecart_ot, lw=1, label=_nom)
    _ax.axvline(0, color="gray", lw=0.6)
    _ax.set_xlabel(f"heures (0 = instant présent ; la fenêtre d'entrée complète fait {ENTREE} heures)")
    _ax.set_ylabel("OT, écart à la moyenne (°C)")
    _ax.legend(fontsize=7, ncol=2)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Sans normalisation par fenêtre, les réseaux battent-ils la prévision naïve ? Et avec ?
       Qu'en concluez-vous sur l'importance relative du prétraitement et de l'architecture ?
    2. Quel est le meilleur modèle du tableau ? Combien de paramètres et de temps d'entraînement lui faut-il,
       comparé aux réseaux ?
    3. Le Transformer est-il meilleur que le LSTM et le TCN ici ? Dans quelles conditions (volume de données,
       longueur d'horizon, nombre de séries) pourrait-il prendre l'avantage ?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. ...
    2. ...
    3. ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 5 — Détection d'anomalies par autoencodeur

    On utilise maintenant les 7 canaux du transformateur. Ce jeu ne contient pas d'anomalies étiquetées :
    on en **injecte** donc dans la partie test, pour pouvoir mesurer ce que les détecteurs trouvent.
    Quatre types réalistes : un pic bref, un capteur figé, une dérive lente, un bruit anormal.
    On en injecte de la même façon dans la partie validation : elle servira à l'étape 6, pour choisir le
    seuil d'alerte sans regarder le test.

    On découpe la série en fenêtres de 48 heures. Une fenêtre de test est **anormale** si elle contient
    au moins une heure d'anomalie.
    """)
    return


@app.cell
def _(mo, np, plt, serie_te, serie_tr, serie_va, tp):
    FENETRE_ANOMALIE = 48
    serie_te_anomalies, etiquettes_te, evenements = tp.injecter_anomalies(serie_te, n_evenements=12, graine=0)
    A_tr = tp.fenetres_glissantes(serie_tr, FENETRE_ANOMALIE, pas=4).astype(np.float32)
    A_te = tp.fenetres_glissantes(serie_te_anomalies, FENETRE_ANOMALIE, pas=12).astype(np.float32)
    y_anomalie = (tp.fenetres_glissantes(etiquettes_te, FENETRE_ANOMALIE, pas=12).sum(axis=1) > 0).astype(int)
    # La période de validation reçoit elle aussi 12 anomalies (autres canaux) : elle sert à choisir le seuil, à l'étape 6.
    _serie_va_anomalies, _etiquettes_va, _ = tp.injecter_anomalies(serie_va, n_evenements=12, graine=1)
    A_va = tp.fenetres_glissantes(_serie_va_anomalies, FENETRE_ANOMALIE, pas=12).astype(np.float32)
    y_anomalie_va = (tp.fenetres_glissantes(_etiquettes_va, FENETRE_ANOMALIE, pas=12).sum(axis=1) > 0).astype(int)
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.plot(serie_te_anomalies[:, -1], lw=0.6, label="OT (test, après injection)")
    for _e in evenements:
        _ax.axvspan(_e["début"], _e["début"] + _e["durée"], color="tab:red", alpha=0.25)
    _ax.set_xlabel("heures")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"**{len(A_tr)}** fenêtres normales pour l'entraînement, **{len(A_te)}** fenêtres de test dont "
              f"**{int(y_anomalie.sum())}** anormales. En rouge : les {len(evenements)} anomalies injectées "
              "(chacune touche un canal tiré au hasard, pas forcément OT)."),
        _fig,
        tp.tableau(evenements, mo),
    ])
    return A_te, A_tr, A_va, y_anomalie, y_anomalie_va


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### L'autoencodeur convolutif (fourni)

    L'encodeur réduit la fenêtre `(48 pas, 7 canaux)` à un code de `(12 pas, 8 canaux)` : 96 nombres au lieu
    de 336. Le décodeur doit reconstruire la fenêtre à partir de ce résumé. Ce sont les convolutions 1D du TCN,
    sans la contrainte de causalité : pour reconstruire, on a le droit de regarder toute la fenêtre.
    """)
    return


@app.cell
def _(nn):
    class AutoencodeurConv(nn.Module):
        def __init__(self, canaux):
            super().__init__()
            self.encodeur = nn.Sequential(
                nn.Conv1d(canaux, 16, 5, stride=2, padding=2), nn.ReLU(),   # longueur divisée par 2
                nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),       # encore divisée par 2
                nn.Conv1d(32, 8, 3, padding=1))                             # le code : 8 canaux
            self.decodeur = nn.Sequential(
                nn.ConvTranspose1d(8, 32, 4, stride=2, padding=1), nn.ReLU(),   # longueur multipliée par 2
                nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv1d(16, canaux, 3, padding=1))

        def forward(self, x):                   # x : (lot, temps, canaux)
            code = self.encodeur(x.transpose(1, 2))
            return self.decodeur(code).transpose(1, 2)

    return (AutoencodeurConv,)


@app.cell(hide_code=True)
def _(mo):
    bouton_ae = mo.ui.run_button(label="Entraîner l'autoencodeur sur les données normales")
    bouton_ae
    return (bouton_ae,)


@app.cell
def _(A_tr, AutoencodeurConv, bouton_ae, mo, plt, torch, tp):
    _chemin = tp.SORTIES / "tp3_autoencodeur.pt"
    mo.stop(not (tp.lance(bouton_ae) or _chemin.exists()), mo.md("Cliquez pour lancer (moins de 30 secondes)."))
    tp.fixer_graine(0)   # avant la création du modèle : ses poids initiaux en dépendent
    autoencodeur = AutoencodeurConv(A_tr.shape[2])
    if tp.lance(bouton_ae):
        autoencodeur, _pertes = tp.entrainer_autoencodeur(
            autoencodeur, A_tr, epoques=30, progression=lambda it: mo.status.progress_bar(it, title="Autoencodeur"))
        torch.save({"etat": autoencodeur.state_dict(), "pertes": _pertes}, _chemin)
    else:
        _sauvegarde = torch.load(_chemin, map_location="cpu")
        autoencodeur.load_state_dict(_sauvegarde["etat"])
        _pertes = _sauvegarde["pertes"]
    autoencodeur = autoencodeur.to(tp.appareil()).eval()
    _fig, _ax = plt.subplots(figsize=(5, 2.4))
    _ax.plot(_pertes)
    _ax.set_xlabel("époque")
    _ax.set_ylabel("erreur de reconstruction")
    _fig.tight_layout()
    _fig
    return (autoencodeur,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 7 : le score d'anomalie

    Pour un tableau de fenêtres `X` de forme `(N, temps, canaux)`, renvoyez un tableau numpy `(N,)` :
    l'erreur quadratique moyenne entre chaque fenêtre et sa reconstruction (moyenne sur le temps et les canaux).
    Convertissez `X` en tenseur, placez-le sur le même appareil que le modèle, et calculez sans gradient
    (`torch.no_grad()`).
    """)
    return


@app.cell
def _(torch):
    def erreur_reconstruction(modele, X):
        """Une erreur quadratique moyenne par fenêtre : tableau numpy de forme (N,)."""
        # TODO 7
        raise NotImplementedError("À compléter")
    return (erreur_reconstruction,)


@app.cell(hide_code=True)
def _(erreur_reconstruction, mo, tests, tp):
    ok_todo7, _affichage = tp.verifier(tests.tests_erreur(erreur_reconstruction), mo)
    _affichage
    return (ok_todo7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 8 : précision et rappel à un seuil donné

    Une alerte est levée quand le score est **strictement supérieur** au seuil. Avec `etiquettes` à 0 (normal)
    ou 1 (anomalie), renvoyez le couple `(precision, rappel)` :

    - précision = part des alertes qui sont de vraies anomalies ;
    - rappel = part des anomalies qui ont déclenché une alerte.

    Renvoyez 0 quand un dénominateur est nul.
    """)
    return


@app.cell
def _(np):
    def precision_rappel(scores, etiquettes, seuil):
        """Renvoie (precision, rappel) pour les alertes scores > seuil."""
        # TODO 8
        raise NotImplementedError("À compléter")
    return (precision_rappel,)


@app.cell(hide_code=True)
def _(mo, precision_rappel, tests, tp):
    ok_todo8, _affichage = tp.verifier(tests.tests_precision_rappel(precision_rappel), mo)
    _affichage
    return (ok_todo8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparer l'autoencodeur aux détecteurs classiques

    Trois détecteurs reçoivent les mêmes fenêtres : votre autoencodeur, l'ACP et l'Isolation Forest.
    La **précision moyenne** (AP) résume la courbe précision/rappel sur tous les seuils : 1 est parfait,
    et un détecteur qui répond au hasard obtient la proportion d'anomalies du test.
    """)
    return


@app.cell
def _(
    A_te,
    A_tr,
    A_va,
    autoencodeur,
    erreur_reconstruction,
    mo,
    np,
    ok_todo7,
    ok_todo8,
    plt,
    precision_rappel,
    tp,
    y_anomalie,
):
    tp.bloquer_si(not (ok_todo7 and ok_todo8), mo, "Terminez les TODO 7 et 8 pour comparer les détecteurs.")
    scores_anomalie = {"Autoencodeur (deep learning)": erreur_reconstruction(autoencodeur, A_te)}
    scores_anomalie.update(tp.scores_classiques(A_tr, A_te))
    scores_validation = erreur_reconstruction(autoencodeur, A_va)     # pour choisir le seuil d'alerte, à l'étape 6
    _fig, _ax = plt.subplots(figsize=(4.6, 3.6))
    _table = []
    for _nom, _s in scores_anomalie.items():
        _seuils = np.quantile(_s, np.linspace(0, 0.99, 60))
        _pr = np.array([precision_rappel(_s, y_anomalie, t) for t in _seuils])
        _ax.plot(_pr[:, 1], _pr[:, 0], label=_nom)
        _table.append({"détecteur": _nom, **tp.bilan_detection(_s, y_anomalie)})
    _ax.axhline(y_anomalie.mean(), color="gray", ls=":", lw=0.8, label="hasard")
    _ax.set_xlabel("rappel")
    _ax.set_ylabel("précision")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    mo.hstack([_fig, tp.tableau(_table, mo)], justify="start", gap=2, widths=[1, 1.4])
    return scores_anomalie, scores_validation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 6 — Choisir un seuil d'alerte : le coût métier

    Un seuil bas détecte tout mais dérange l'opérateur pour rien ; un seuil haut laisse passer des pannes.
    Donnez un coût à chaque erreur : le seuil optimal en découle. Faites varier les coûts et observez
    comment le seuil optimal se déplace.

    En production, on ne connaît pas les scores à venir : le seuil se fixe donc sur une période passée.
    Ici, on le choisit sur la période de **validation** (le curseur l'exprime comme un centile des scores de
    validation), puis on l'applique tel quel à la période de **test**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cout_alerte = mo.ui.slider(1, 50, value=5, label="Coût d'une fausse alerte (déplacement d'un technicien)", show_value=True)
    cout_oubli = mo.ui.slider(1, 500, value=100, label="Coût d'une anomalie non détectée (panne)", show_value=True)
    seuil_centile = mo.ui.slider(50, 99, value=85, label="Seuil d'alerte (centile des scores de validation)", show_value=True)
    mo.vstack([cout_alerte, cout_oubli, seuil_centile])
    return cout_alerte, cout_oubli, seuil_centile


@app.cell(hide_code=True)
def _(
    cout_alerte,
    cout_oubli,
    mo,
    np,
    ok_todo8,
    plt,
    precision_rappel,
    scores_anomalie,
    scores_validation,
    seuil_centile,
    tp,
    y_anomalie,
    y_anomalie_va,
):
    tp.bloquer_si(not ok_todo8, mo, "Terminez le TODO 8 d'abord.")
    _s = scores_anomalie["Autoencodeur (deep learning)"]      # scores des fenêtres de test
    _sv = scores_validation                                    # scores des fenêtres de validation

    def _bilan(scores, etiquettes, seuil):
        """Fausses alertes, anomalies manquées et coût total pour un seuil donné."""
        alertes = scores > seuil
        fausses, manquees = int((alertes & (etiquettes == 0)).sum()), int((~alertes & (etiquettes == 1)).sum())
        return fausses, manquees, cout_alerte.value * fausses + cout_oubli.value * manquees

    # Le seuil est défini sur la validation, et le coût qui sert à le choisir aussi.
    _centiles = np.arange(50, 100)
    _couts_va = [_bilan(_sv, y_anomalie_va, np.percentile(_sv, _c))[2] for _c in _centiles]
    _seuil = np.percentile(_sv, seuil_centile.value)
    _fa_va, _ou_va, _cout_va = _bilan(_sv, y_anomalie_va, _seuil)
    # Puis on l'applique tel quel au test.
    _fa, _ou, _cout = _bilan(_s, y_anomalie, _seuil)
    _p, _r = precision_rappel(_s, y_anomalie, _seuil)
    _fig, _ax = plt.subplots(2, 1, figsize=(9, 5))
    _ax[0].bar(range(len(_s)), _s, color=["tab:red" if v else "tab:blue" for v in y_anomalie], width=1.0)
    _ax[0].axhline(_seuil, color="black", ls="--", lw=1)
    _ax[0].set_yscale("log")
    _ax[0].set_title("Score de chaque fenêtre de test (rouge : vraie anomalie ; pointillés : seuil fixé sur la validation)", fontsize=9)
    _ax[1].plot(_centiles, _couts_va)
    _ax[1].axvline(seuil_centile.value, color="black", ls="--", lw=1)
    _ax[1].set_xlabel("seuil (centile des scores de validation)")
    _ax[1].set_ylabel("coût total sur la validation")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"**Sur la validation**, au seuil choisi : {_fa_va} fausses alertes, {_ou_va} anomalies manquées, "
              f"coût total {_cout_va}. Le coût y est minimal au centile **{int(_centiles[int(np.argmin(_couts_va))])}**."),
        mo.md(f"**Sur le test**, avec ce même seuil : **{_fa}** fausses alertes, **{_ou}** anomalies manquées, "
              f"précision {_p:.0%}, rappel {_r:.0%}, coût total **{_cout}** "
              f"({len(_s)} fenêtres de test, {len(_sv)} de validation)."),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Un seuil se transporte-t-il d'une période à l'autre ?

    Fixer le seuil sur la validation puis l'appliquer au test suppose que les scores se répartissent de la même
    façon sur les deux périodes. La figure ci-dessous permet de le vérifier.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt, scores_anomalie, scores_validation, seuil_centile):
    _s, _sv = scores_anomalie["Autoencodeur (deep learning)"], scores_validation
    _bords = np.logspace(np.log10(min(_s.min(), _sv.min())), np.log10(max(_s.max(), _sv.max())), 40)
    _fig, _ax = plt.subplots(figsize=(7, 2.6))
    _ax.hist(_sv, bins=_bords, alpha=0.55, label="validation")
    _ax.hist(_s, bins=_bords, alpha=0.55, label="test")
    _ax.axvline(np.percentile(_sv, seuil_centile.value), color="black", ls="--", lw=1, label="seuil choisi")
    _ax.set_xscale("log")
    _ax.set_xlabel("score d'anomalie (erreur de reconstruction)")
    _ax.set_ylabel("nombre de fenêtres")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"Score médian des fenêtres : **{np.median(_sv):.3f}** sur la validation, **{np.median(_s):.3f}** sur le test. "
              f"Score dépassé par une fenêtre sur dix : **{np.quantile(_sv, 0.9):.3f}** sur la validation, "
              f"**{np.quantile(_s, 0.9):.3f}** sur le test."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :**

    1. Quel seuil retiendriez-vous si une panne de transformateur coûte 50 fois plus cher qu'un déplacement
       inutile ? Et si l'équipe ne peut traiter que deux alertes par semaine ?
    2. Le seuil choisi sur la validation donne-t-il, sur le test, les fausses alertes et les oublis que vous
       attendiez ? Les scores se répartissent-ils de la même façon sur les deux périodes ? Si ce n'est pas le
       cas, que faudrait-il prévoir pour un système qui reste en service plusieurs mois ?
    3. Quels types d'anomalies injectées sont les plus difficiles à détecter, et pourquoi ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 7 — Fiche de décision ingénieur
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Rubrique | Prévision de la température | Détection d'anomalies |
    |---|---|---|
    | Problème métier | Anticiper une surchauffe du transformateur | Alerter l'opérateur sur un comportement anormal |
    | Baseline classique et son score | ... | ... |
    | Modèle retenu et pourquoi | ... | ... |
    | Métrique principale et score obtenu | ... | ... |
    | Seuil ou réglage retenu, justification métier | ... | ... |
    | Coût (entraînement, inférence, matériel) | ... | ... |
    | Principaux cas d'échec | ... | ... |
    | Le deep learning est-il justifié ici ? | ... | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... | ... |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 8 — Bilan : deep learning ou classique ?

    Vous avez maintenant trois comparaisons chiffrées : la classification d'images (TD 1), la détection
    d'objets (TD 2) et les deux tâches d'aujourd'hui.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** dans quels cas le deep learning a-t-il nettement gagné, dans quels cas la méthode
    classique a-t-elle fait jeu égal ou mieux ? Proposez une règle pratique pour décider, au début d'un
    projet, s'il faut investir dans un réseau de neurones.

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 9 — Brique projet Usine 4.0 (volet au choix) : anticiper les pannes d'une pompe

    **SKAB** : les capteurs d'une pompe à eau d'un banc d'essai (vibrations, courant, pression, température,
    tension, débit), une mesure par seconde. Les expérimentateurs ont provoqué de vraies anomalies, par
    exemple en fermant progressivement une vanne, et les ont étiquetées. Contrairement à l'exercice précédent,
    les anomalies sont donc **réelles**.

    Dans le projet, c'est le volet « maintenance », l'un des deux volets au choix (voir `projet/SUJET.md`).

    Deux façons de définir le « normal » sont comparées :

    - **Protocole A** : entraîner sur un enregistrement séparé, sans aucune anomalie (`anomaly-free`) ;
    - **Protocole B** : entraîner sur les 400 premières secondes de chaque expérience, avant que l'anomalie
      ne soit provoquée, et tester sur la suite.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_skab = mo.ui.run_button(label="Charger SKAB et lancer la comparaison")
    bouton_skab
    return (bouton_skab,)


@app.cell
def _(
    AutoencodeurConv,
    bouton_skab,
    erreur_reconstruction,
    mo,
    np,
    ok_todo7,
    tp,
):
    tp.bloquer_si(not ok_todo7, mo, "Terminez le TODO 7 d'abord.")
    mo.stop(not tp.lance(bouton_skab), mo.md("Cliquez pour lancer (téléchargement de 4 Mo, puis environ 1 minute)."))
    skab = tp.charger_skab(progression=lambda it: mo.status.progress_bar(it, title="Téléchargement de SKAB"))
    _T, _pas = 32, 4

    def _fenetres(df, debut=0, fin=None, stats=None):
        _v = df[tp.SKAB_CAPTEURS].to_numpy(dtype=np.float32)
        _mu, _sd = stats
        _x = tp.fenetres_glissantes(((_v - _mu) / _sd)[debut:fin], _T, _pas)
        _y = tp.fenetres_glissantes(df["anomaly"].to_numpy()[debut:fin], _T, _pas).mean(axis=1) > 0.5
        return _x, _y.astype(int)

    def _comparer(X_train, X_test, y_test):
        tp.fixer_graine(0)
        _ae, _ = tp.entrainer_autoencodeur(AutoencodeurConv(X_train.shape[2]), X_train, epoques=30)
        _scores = {"Autoencodeur (deep learning)": erreur_reconstruction(_ae, X_test), **tp.scores_classiques(X_train, X_test)}
        return {n: tp.bilan_detection(s, y_test) for n, s in _scores.items()}

    # Protocole A : le « normal » vient d'un enregistrement séparé
    _ref = skab["anomaly-free"][tp.SKAB_CAPTEURS].to_numpy(dtype=np.float32)
    _stats = (_ref.mean(0), _ref.std(0) + 1e-8)
    _Xa, _ = _fenetres(skab["anomaly-free"].assign(anomaly=0), stats=_stats)
    _tests = [_fenetres(df, stats=_stats) for df in skab["valve1"]]
    _res_a = _comparer(_Xa, np.concatenate([x for x, _ in _tests]), np.concatenate([y for _, y in _tests]))
    # Protocole B : le « normal » est le début de chaque expérience
    _tr, _te, _ye = [], [], []
    for _df in skab["valve1"]:
        _v = _df[tp.SKAB_CAPTEURS].to_numpy(dtype=np.float32)
        _s = (_v[:400].mean(0), _v[:400].std(0) + 1e-8)
        _tr.append(_fenetres(_df, 0, 400, _s)[0])
        _x, _y = _fenetres(_df, 400, None, _s)
        _te.append(_x)
        _ye.append(_y)
    _res_b = _comparer(np.concatenate(_tr), np.concatenate(_te), np.concatenate(_ye))
    _lignes = [{"protocole": p, "détecteur": n, **r} for p, res in [("A : enregistrement sans anomalie", _res_a),
                                                                    ("B : début de chaque expérience", _res_b)]
               for n, r in res.items()]
    mo.vstack([mo.md(f"Expériences `valve1` : {len(skab['valve1'])} fichiers, fenêtres de {_T} secondes."),
               tp.tableau(_lignes, mo)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Si votre groupe choisit le volet maintenance**, poursuivez dans `projet/`, par exemple dans une copie de
    `projet/brique_modele.py` (l'organisation du dossier est libre).

    1. Lequel des deux protocoles donne les meilleurs résultats ? Le protocole A voit pourtant *plus* de
       données normales : comparez les distributions des capteurs entre `anomaly-free` et le début des
       expériences pour expliquer ce que vous observez. Le phénomène en jeu sera repris au TD 6.
    2. Refaites la comparaison sur les autres groupes d'expériences (`valve2` et `other`), déjà chargés dans `skab`.
    3. Choisissez un seuil d'alerte et justifiez-le par le coût d'une panne de pompe face à celui d'une
       intervention inutile.
    4. Quel capteur contribue le plus à l'erreur de reconstruction lors d'une anomalie ? C'est une
       information précieuse pour le technicien qui intervient.
    5. Remplissez la fiche de décision pour ce volet : l'autoencodeur vaut-il son coût face à
       l'Isolation Forest ?

    ### Pour aller plus loin (si vous avez terminé)

    - Remplacez l'autoencodeur convolutif par un autoencodeur LSTM (encodeur et décodeur `nn.LSTM`) et comparez.
    - Prévision : passez l'horizon de 24 à 96 heures. Le classement des modèles change-t-il ?
    - Prévision : donnez aux réseaux les 7 canaux en entrée au lieu de la seule température. Est-ce mieux ?
    """)
    return


if __name__ == "__main__":
    app.run()
