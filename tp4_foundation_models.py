import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="TD 4 - Foundation models et IA générative",
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
    from PIL import Image

    sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
    import tests_tp4 as tests
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return Image, mo, nn, np, plt, tests, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TD 4 — Foundation models et IA générative

    **Module Application au deep learning · ING3 IA · durée : 3h30 (l'étape 8 peut se terminer en autonomie)**

    Jusqu'ici, chaque modèle avait besoin d'exemples annotés de *votre* problème. Les **foundation models**
    changent la donne : entraînés une fois sur des volumes gigantesques, ils se réutilisent sur des tâches
    qu'ils n'ont jamais vues, parfois **sans aucun exemple**. Ce TD a deux parties :

    - **CLIP** : classer et rechercher des images à partir de simples phrases ;
    - **la diffusion** : comprendre de l'intérieur comment une machine génère une image, en codant les deux
      opérations clés d'un modèle de diffusion.

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
    | 2 | CLIP : classer sans aucun exemple | 40 min |
    | 3 | CLIP : un moteur de recherche d'images | 35 min |
    | 4 | Diffusion : comprendre sur des points 2D | 45 min |
    | 5 | Diffusion : générer des images, avec une condition | 25 min |
    | 6 | Fiche de décision ingénieur | 10 min |
    | 7 | Bilan : que change un foundation model ? | 10 min |
    | 8 | Brique projet Usine 4.0 (volet au choix) : les défauts rares | 30 min |
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch, tp):
    mo.vstack([mo.md("## Étape 0 — Vérification de l'installation"),
               mo.callout(mo.md(f"- Matériel utilisé : **{tp.appareil()}**\n- Threads CPU : {torch.get_num_threads()}\n"
                                "- CLIP pèse environ 600 Mo : il est téléchargé une seule fois, dans `modeles/`."
                                + ("\n- **Mode test actif** : CLIP et DINO sont chargés avec des poids aléatoires." if tp.MODE_TEST else "")),
                          kind="info")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 1 — Le cours en bref

    ### 1.1 Rappel : trois notions déjà rencontrées

    - **ViT** (TD 1) : une image découpée en patches, traitée par attention. C'est l'architecture qui sert
      d'encodeur d'images à CLIP, DINO et SAM.
    - **L'autoencodeur** (TD 3) : compresser puis reconstruire. Le VAE en est la version probabiliste.
    - **L'embedding** (TD 1) : un vecteur qui résume une image. Deux images qui se ressemblent ont des vecteurs
      proches. On mesure cette proximité par la **similarité cosinus**, le cosinus de l'angle entre deux vecteurs :
      1 s'ils pointent dans la même direction, 0 s'ils sont perpendiculaires.

    ### 1.2 Qu'est-ce qu'un foundation model ?

    Trois caractéristiques : un entraînement à **très grande échelle** sur des données peu ou pas annotées,
    des représentations **réutilisables** pour de nombreuses tâches, et une adaptation par **prompt** ou par un
    fine-tuning léger. On passe de « un modèle par problème » à « un socle, puis une adaptation ».

    | Modèle | Comment il apprend sans annotation manuelle | Ce qu'il permet |
    |---|---|---|
    | **CLIP** | rapproche chaque image de sa légende parmi 400 millions de paires trouvées sur le web (apprentissage contrastif) | classer et rechercher des images par le texte |
    | **DINO** | deux vues différentes de la même image doivent donner le même embedding (auto-distillation), sans aucun texte | recherche d'images similaires, features très fines |
    | **SAM** | entraîné sur plus d'un milliard de masques, guidé par un point ou une boîte | segmenter n'importe quel objet (TD 2) |

    **L'apprentissage contrastif de CLIP.** Deux encodeurs, un pour les images et un pour les textes, projettent
    dans le **même espace**. Pendant l'entraînement, dans un lot de $N$ paires, chaque image doit être plus proche
    de sa propre légende que des $N - 1$ autres. À l'usage, pour classer une image, on écrit une phrase par classe
    et on choisit la plus proche : c'est la classification **zéro-shot**.

    ### 1.3 Générer : VAE, GAN, diffusion

    - **VAE** : un autoencodeur dont le code suit une loi simple, qu'on peut échantillonner. Images souvent floues.
    - **GAN** : un générateur tente de tromper un discriminateur. Images nettes, mais entraînement instable.
    - **Diffusion** : on apprend à **retirer du bruit**. Partant d'un bruit pur, on débruite petit à petit jusqu'à
      obtenir une image. Entraînement stable, qualité élevée : c'est l'approche dominante aujourd'hui.

    **Stable Diffusion** combine les trois idées du jour : la diffusion se fait dans l'espace compressé d'un
    **VAE** (c'est ce qui la rend abordable), et elle est guidée par l'**encodeur de texte de CLIP**.

    Faites glisser le curseur : voici le processus de bruitage que le modèle apprendra à inverser.
    """)
    return


@app.cell(hide_code=True)
def _(mo, tp):
    demo_t = mo.ui.slider(0, tp.N_ETAPES_DIFFUSION - 1, value=0, label="Étape de bruitage t", show_value=True)
    demo_t
    return (demo_t,)


@app.cell(hide_code=True)
def _(Image, demo_t, mo, np, plt, torch, tp):
    from skimage import data as _donnees

    _x0 = torch.tensor(np.asarray(Image.fromarray(_donnees.chelsea()).resize((180, 120))), dtype=torch.float32) / 127.5 - 1
    _, _, _ac = tp.planning_diffusion()
    torch.manual_seed(0)
    _xt = _ac[demo_t.value].sqrt() * _x0 + (1 - _ac[demo_t.value]).sqrt() * torch.randn_like(_x0)
    _fig, _ax = plt.subplots(1, 2, figsize=(8, 2.6), gridspec_kw={"width_ratios": [1, 1.3]})
    _ax[0].imshow(((_xt + 1) / 2).clamp(0, 1).numpy())
    _ax[0].axis("off")
    _ax[1].plot(_ac.sqrt().numpy(), label="part du signal")
    _ax[1].plot((1 - _ac).sqrt().numpy(), label="part du bruit")
    _ax[1].axvline(demo_t.value, color="black", lw=0.8)
    _ax[1].set_xlabel("étape t")
    _ax[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"À l'étape {demo_t.value}, l'image contient {float(_ac[demo_t.value].sqrt()):.0%} de signal "
                           f"et {float((1 - _ac[demo_t.value]).sqrt()):.0%} de bruit (en amplitude).")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 2 — CLIP : classer sans aucun exemple

    On reprend les photos de chantier du TD 2, mais découpées en **vignettes** : chaque vignette contient un seul
    équipement (casque, gants, gilet, bottes, lunettes). Le problème devient une classification à 5 classes.
    Au TD 2, il a fallu 700 images annotées et 50 minutes d'entraînement. CLIP, lui, n'a reçu aucune de ces
    annotations. La tâche est toutefois plus simple qu'au TD 2 : chaque vignette est déjà découpée autour de
    l'objet, il ne reste qu'à le nommer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_clip = mo.ui.run_button(label="Charger CLIP et les vignettes")
    bouton_clip
    return (bouton_clip,)


@app.cell
def _(bouton_clip, mo, tp):
    _pret = ((tp.MODELES / "clip_vitb32_laion400m.pt").exists() and tp.ppe_pret()) or tp.MODE_TEST
    mo.stop(not (tp.lance(bouton_clip) or _pret),
            mo.md("Cliquez pour charger (au premier lancement : CLIP, 600 Mo, et Construction-PPE, 650 Mo, "
                  "s'il n'a pas été chargé au TD 2)."))
    with mo.status.spinner(title="Chargement de CLIP et préparation des vignettes..."):
        clip_modele, clip_pretraitement, clip_tokeniseur = tp.charger_clip()
        dossier_ppe = tp.preparer_construction_ppe().parent
        vignettes, y_vignettes, noms_vignettes = tp.vignettes_ppe(dossier_ppe)
    return (
        clip_modele,
        clip_pretraitement,
        clip_tokeniseur,
        dossier_ppe,
        noms_vignettes,
        vignettes,
        y_vignettes,
    )


@app.cell(hide_code=True)
def _(mo, noms_vignettes, np, tp, vignettes, y_vignettes):
    _rng = np.random.default_rng(3)
    _idx = _rng.choice(len(vignettes), 10, replace=False)
    mo.vstack([
        mo.md(f"**{len(vignettes)}** vignettes : " + ", ".join(f"{n} ({int((y_vignettes == k).sum())})" for k, n in enumerate(noms_vignettes)) + "."),
        tp.grille([vignettes[i] for i in _idx], [noms_vignettes[y_vignettes[i]] for i in _idx], n_col=10, taille=1.1),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 1 : la similarité cosinus

    Entrées : `A` de forme `(n, d)` et `B` de forme `(m, d)`. Sortie : un tableau `(n, m)` où la case $(i, j)$
    vaut le cosinus entre le vecteur $i$ de `A` et le vecteur $j$ de `B` :

    $$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

    Le plus simple : diviser chaque vecteur par sa norme (`np.linalg.norm(..., axis=1, keepdims=True)`),
    puis faire un produit matriciel.
    """)
    return


@app.cell
def _(np):
    def similarite_cosinus(A, B):
        """A : (n, d), B : (m, d) -> tableau (n, m) des similarités cosinus."""
        # TODO 1
        raise NotImplementedError("À compléter")
    return (similarite_cosinus,)


@app.cell(hide_code=True)
def _(mo, similarite_cosinus, tests, tp):
    ok_todo1, _affichage = tp.verifier(tests.tests_cosinus(similarite_cosinus), mo)
    _affichage
    return (ok_todo1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 2 : la classification zéro-shot

    Entrées : les embeddings des images `(n, d)` et ceux des phrases décrivant chaque classe `(c, d)`.
    Calculez les similarités cosinus, **multipliez-les par la température**, puis appliquez un softmax sur
    les classes (`axis=1`). Renvoyez le tableau `(n, c)` des probabilités.

    La température (100 dans CLIP) rend la décision plus tranchée : sans elle, des cosinus de 0,25 et 0,30
    donneraient des probabilités presque égales.
    """)
    return


@app.cell
def _(np, similarite_cosinus):
    def classer_zero_shot(emb_images, emb_textes, temperature=100):
        """Probabilité de chaque classe pour chaque image : softmax(temperature x cosinus)."""
        # TODO 2
        raise NotImplementedError("À compléter")
    return (classer_zero_shot,)


@app.cell(hide_code=True)
def _(classer_zero_shot, mo, ok_todo1, tests, tp):
    tp.bloquer_si(not ok_todo1, mo, "Terminez le TODO 1 d'abord.")
    ok_todo2, _affichage = tp.verifier(tests.tests_zero_shot(classer_zero_shot), mo)
    _affichage
    return (ok_todo2,)


@app.cell
def _(clip_modele, clip_pretraitement, mo, tp, vignettes):
    _cache = tp.charger_cache("tp4_emb_vignettes_ppe2026")
    if _cache is None or len(_cache["e"]) != len(vignettes):
        _e = tp.encoder_images(clip_modele, clip_pretraitement, vignettes,
                               progression=lambda it: mo.status.progress_bar(it, title="Encodage des vignettes par CLIP"))
        tp.sauver_cache("tp4_emb_vignettes_ppe2026", e=_e)
        _cache = {"e": _e}
    emb_vignettes = _cache["e"]
    mo.md(f"Embeddings des vignettes : {emb_vignettes.shape} (un vecteur de 512 nombres par image).")
    return (emb_vignettes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Écrivez vos classes

    Une phrase par ligne, dans l'ordre : casque, gants, gilet, bottes, lunettes. Essayez différentes
    formulations et observez l'accuracy : le mot seul, « a photo of a ... », une description plus précise
    (« a hard hat », « a high-visibility vest »). CLIP a été entraîné surtout sur des légendes en anglais :
    essayez aussi en français et comparez.

    Choisir ses phrases, c'est régler le système. Pour ne pas le juger sur les images qui ont servi à le régler,
    les vignettes sont séparées en deux moitiés : l'accuracy affichée sous la zone de texte est calculée sur la
    première. La seconde est mise de côté : elle donnera la mesure finale, quand vos phrases seront choisies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prompts = mo.ui.text_area(value="helmet\ngloves\nvest\nboots\ngoggles", rows=5, full_width=True, label="Une phrase par classe")
    prompts
    return (prompts,)


@app.cell
def _(
    classer_zero_shot,
    clip_modele,
    clip_tokeniseur,
    emb_vignettes,
    mo,
    noms_vignettes,
    np,
    ok_todo2,
    prompts,
    tp,
    y_vignettes,
):
    tp.bloquer_si(not ok_todo2, mo, "Terminez le TODO 2 pour classer les vignettes.")
    _phrases = [l.strip() for l in prompts.value.splitlines() if l.strip()]
    mo.stop(len(_phrases) != len(noms_vignettes), mo.callout(mo.md(f"Il faut exactement {len(noms_vignettes)} lignes."), kind="warn"))
    _probas = classer_zero_shot(emb_vignettes, tp.encoder_textes(clip_modele, clip_tokeniseur, _phrases))
    _pred = _probas.argmax(axis=1)
    # Une vignette sur deux sert à régler les phrases ; les autres sont mises de côté pour la mesure finale.
    _reglage = np.arange(len(y_vignettes)) % 2 == 0
    _juste = _pred == y_vignettes
    accuracy_zero_shot = float(_juste[~_reglage].mean())      # mesurée sur les vignettes mises de côté, affichée plus bas
    _par_classe = " · ".join(f"{n} {float(_juste[_reglage & (y_vignettes == k)].mean()):.0%}" for k, n in enumerate(noms_vignettes))
    mo.md(f"Accuracy zéro-shot sur les {int(_reglage.sum())} vignettes de réglage : **{float(_juste[_reglage].mean()):.1%}** "
          f"(hasard : 20 %). Par classe : {_par_classe}.")
    return (accuracy_zero_shot,)


@app.cell(hide_code=True)
def _(mo):
    bouton_mesure = mo.ui.run_button(label="Mesurer mes phrases sur les vignettes mises de côté")
    bouton_mesure
    return (bouton_mesure,)


@app.cell(hide_code=True)
def _(accuracy_zero_shot, bouton_mesure, mo, tp):
    mo.stop(not tp.lance(bouton_mesure), mo.md(
        "Quand vos phrases sont choisies, cliquez pour obtenir la mesure finale. Si vous retouchez ensuite vos "
        "phrases pour améliorer ce chiffre, il devient à son tour un chiffre de réglage."))
    mo.md(f"Accuracy zéro-shot sur les vignettes mises de côté : **{accuracy_zero_shot:.1%}**.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** notez les formulations essayées et leur score. Laquelle marche le mieux ? Pourquoi la
    formulation change-t-elle le résultat, alors que le modèle et les images sont identiques ?

    | Formulation | Accuracy sur les vignettes de réglage |
    |---|---|
    | mots seuls, en anglais | |
    | « a photo of a ... » | |
    | descriptions précises | |
    | en français | |

    Formulation retenue : ... Accuracy sur les vignettes mises de côté : ... Diffère-t-elle de l'accuracy de
    réglage ? Pour quelles raisons pourrait-elle en différer ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Avec quelques exemples : le foundation model face au classique

    Supposons maintenant qu'on dispose de $k$ images annotées par classe. Deux approches :

    - **classique** : un descripteur fait main (image réduite à 16 x 16 + histogrammes de couleur) et un SVM ;
    - **sonde linéaire** : une simple régression logistique sur les embeddings CLIP, sans toucher au modèle.
      C'est la façon la moins coûteuse d'adapter un foundation model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_fewshot = mo.ui.run_button(label="Comparer selon le nombre d'exemples")
    bouton_fewshot
    return (bouton_fewshot,)


@app.cell
def _(
    accuracy_zero_shot,
    bouton_fewshot,
    emb_vignettes,
    mo,
    np,
    plt,
    tp,
    vignettes,
    y_vignettes,
):
    mo.stop(not tp.lance(bouton_fewshot), mo.md("Cliquez pour lancer (quelques secondes)."))
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    _pixels = np.stack([tp.descripteur_pixels(v) for v in vignettes])
    _ks, _clip, _classique = [1, 2, 5, 10, 20], [], []
    for _k in _ks:
        _a, _b = [], []
        for _graine in range(5):  # 5 tirages différents des exemples, pour lisser le hasard
            _rng = np.random.default_rng(_graine)
            _tr = np.concatenate([_rng.choice(np.flatnonzero(y_vignettes == c), _k, replace=False) for c in range(5)])
            _te = np.setdiff1d(np.arange(len(y_vignettes)), _tr)
            _a.append((LogisticRegression(max_iter=2000, C=10).fit(emb_vignettes[_tr], y_vignettes[_tr]).predict(emb_vignettes[_te]) == y_vignettes[_te]).mean())
            _b.append((SVC(C=10).fit(_pixels[_tr], y_vignettes[_tr]).predict(_pixels[_te]) == y_vignettes[_te]).mean())
        _clip.append(float(np.mean(_a)))
        _classique.append(float(np.mean(_b)))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.plot(_ks, _clip, marker="o", label="CLIP + sonde linéaire")
    _ax.plot(_ks, _classique, marker="o", label="classique (pixels + couleur + SVM)")
    _ax.axhline(accuracy_zero_shot, color="tab:green", ls="--", label="CLIP zéro-shot (vos phrases, 0 exemple, vignettes mises de côté)")
    _ax.axhline(0.2, color="gray", ls=":", label="hasard")
    _ax.set_xscale("log")
    _ax.set_xticks(_ks, [str(k) for k in _ks])
    _ax.set_xlabel("exemples annotés par classe")
    _ax.set_ylabel("accuracy")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. La sonde linéaire dépasse-t-elle le zéro-shot ? Si oui, à partir de combien d'exemples par classe ?
    2. À nombre d'exemples égal, comparez l'approche classique et la sonde linéaire. L'écart se réduit-il quand
       le nombre d'exemples augmente ?
    3. Dans quel cas concret d'entreprise le zéro-shot est-il la seule option réaliste ?
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
    ## Étape 3 — CLIP : un moteur de recherche d'images

    Puisque textes et images vivent dans le même espace, on peut chercher des images **en décrivant ce qu'on veut**,
    sans aucun mot-clé associé aux images. On encode une fois un corpus de 500 images de chantier, puis chaque
    requête ne coûte qu'un produit matriciel.

    ### À compléter — TODO 3 : la recherche

    Entrées : l'embedding de la requête `(d,)`, ceux du corpus `(N, d)`, et `k`. Renvoyez les **indices** des `k`
    éléments du corpus les plus proches de la requête au sens de la similarité cosinus, du plus proche au moins
    proche. Aide : `np.argsort(-valeurs)` trie par ordre décroissant ; `requete[None, :]` donne la forme `(1, d)`.
    """)
    return


@app.cell
def _(np, similarite_cosinus):
    def rechercher(emb_requete, emb_corpus, k=5):
        """Indices des k éléments du corpus les plus proches de la requête (similarité cosinus décroissante)."""
        # TODO 3
        raise NotImplementedError("À compléter")
    return (rechercher,)


@app.cell(hide_code=True)
def _(mo, ok_todo1, rechercher, tests, tp):
    tp.bloquer_si(not ok_todo1, mo, "Terminez le TODO 1 d'abord.")
    ok_todo3, _affichage = tp.verifier(tests.tests_recherche(rechercher), mo)
    _affichage
    return (ok_todo3,)


@app.cell
def _(clip_modele, clip_pretraitement, dossier_ppe, mo, np, tp):
    corpus = tp.lister_images(dossier_ppe, "train")[:500]
    _cache = tp.charger_cache("tp4_emb_corpus_ppe2026")
    if _cache is None or len(_cache["e"]) != len(corpus):
        _e = tp.encoder_images(clip_modele, clip_pretraitement, corpus,
                               progression=lambda it: mo.status.progress_bar(it, title="Encodage du corpus par CLIP"))
        tp.sauver_cache("tp4_emb_corpus_ppe2026", e=_e)
        _cache = {"e": _e}
    emb_corpus = _cache["e"]
    # Pour évaluer : quelles classes sont présentes dans chaque image (d'après les annotations du TD 2) ?
    presence = np.zeros((len(corpus), len(tp.PPE_CLASSES)), bool)
    for _i, _p in enumerate(corpus):
        presence[_i, tp.lire_annotations(_p)[1]] = True
    mo.md(f"Corpus encodé : {emb_corpus.shape[0]} images.")
    return corpus, emb_corpus, presence


@app.cell(hide_code=True)
def _(mo):
    requete = mo.ui.text(value="a worker wearing an orange safety vest", label="Votre requête (en anglais)", full_width=True)
    requete
    return (requete,)


@app.cell
def _(
    clip_modele,
    clip_tokeniseur,
    corpus,
    emb_corpus,
    mo,
    ok_todo3,
    rechercher,
    requete,
    tp,
):
    tp.bloquer_si(not ok_todo3, mo, "Terminez le TODO 3 pour lancer une recherche.")
    _q = tp.encoder_textes(clip_modele, clip_tokeniseur, [requete.value])[0]
    _idx = rechercher(_q, emb_corpus, k=6)
    tp.grille([tp.lire_image_rgb(corpus[i], cote_max=320) for i in _idx], [f"résultat {r + 1}" for r in range(6)], n_col=6, taille=1.9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Essayez : `a person without a helmet`, `safety goggles`, `a worker on a roof`, `a group of workers`,
    `a blue helmet`, `someone holding a tool`. Certaines de ces notions n'ont **jamais été annotées** dans le jeu.

    ### À compléter — TODO 4 : mesurer la qualité de la recherche

    « Ça a l'air de marcher » n'est pas une évaluation. La **précision à k** est la proportion de résultats
    pertinents parmi les `k` premiers. Entrées : `indices`, la liste ordonnée des résultats ; `pertinents`, un
    tableau de booléens (une valeur par image du corpus) ; et `k`.
    """)
    return


@app.cell
def _(np):
    def precision_a_k(indices, pertinents, k):
        """Proportion d'images pertinentes parmi les k premiers résultats."""
        # TODO 4
        raise NotImplementedError("À compléter")
    return (precision_a_k,)


@app.cell(hide_code=True)
def _(mo, precision_a_k, tests, tp):
    ok_todo4, _affichage = tp.verifier(tests.tests_precision_k(precision_a_k), mo)
    _affichage
    return (ok_todo4,)


@app.cell
def _(
    clip_modele,
    clip_tokeniseur,
    emb_corpus,
    mo,
    ok_todo3,
    ok_todo4,
    precision_a_k,
    presence,
    rechercher,
    tp,
):
    tp.bloquer_si(not (ok_todo3 and ok_todo4), mo, "Terminez les TODO 3 et 4 pour évaluer le moteur.")
    _requetes = {
        "a person wearing a safety helmet": "helmet",
        "a person without a helmet": "no_helmet",
        "a person wearing a high-visibility vest": "vest",
        "a person wearing safety goggles": "goggles",
        "a person wearing work gloves": "gloves",
    }
    _lignes = []
    for _texte, _classe in _requetes.items():
        _pert = presence[:, tp.PPE_CLASSES.index(_classe)]
        _idx = rechercher(tp.encoder_textes(clip_modele, clip_tokeniseur, [_texte])[0], emb_corpus, k=10)
        _lignes.append({"requête": _texte, "précision à 10": round(precision_a_k(_idx, _pert, 10), 2),
                        "hasard (part du corpus concernée)": round(float(_pert.mean()), 2)})
    mo.vstack([mo.md("Une image est jugée pertinente si la classe correspondante y est annotée (annotations du TD 2)."),
               tp.tableau(_lignes, mo)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recherche par l'image : CLIP face à une méthode classique

    On peut aussi chercher les images qui ressemblent à une image donnée. La baseline classique compare les
    **histogrammes de couleur**. Choisissez une image requête et comparez les deux séries de résultats.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    image_requete = mo.ui.slider(0, 199, value=7, label="Image requête (jeu de test)", show_value=True)
    image_requete
    return (image_requete,)


@app.cell
def _(
    clip_modele,
    clip_pretraitement,
    corpus,
    dossier_ppe,
    emb_corpus,
    image_requete,
    mo,
    np,
    ok_todo3,
    rechercher,
    tp,
):
    tp.bloquer_si(not ok_todo3, mo, "Terminez le TODO 3 d'abord.")
    _chemin = tp.lister_images(dossier_ppe, "test")[image_requete.value]
    _img = tp.Image.open(_chemin).convert("RGB")
    _idx_clip = rechercher(tp.encoder_images(clip_modele, clip_pretraitement, [_img])[0], emb_corpus, k=5)
    _cache = tp.charger_cache("tp4_couleur_corpus_ppe2026")
    if _cache is None:
        _cache = {"h": np.stack([tp.descripteur_pixels(tp.Image.open(p))[-48:] for p in corpus])}
        tp.sauver_cache("tp4_couleur_corpus_ppe2026", h=_cache["h"])
    _idx_couleur = rechercher(tp.descripteur_pixels(_img)[-48:], _cache["h"], k=5)
    _lire = lambda i: tp.lire_image_rgb(corpus[i], cote_max=320)
    mo.vstack([
        tp.grille([tp.lire_image_rgb(_chemin, cote_max=320)], ["image requête"], n_col=1, taille=2.4),
        mo.md("**CLIP**"), tp.grille([_lire(i) for i in _idx_clip], n_col=5, taille=1.9),
        mo.md("**Classique (histogrammes de couleur)**"), tp.grille([_lire(i) for i in _idx_couleur], n_col=5, taille=1.9),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** sur quoi chaque méthode s'appuie-t-elle pour juger que deux images se ressemblent ?
    Laquelle retrouve le *contenu* (ce que font les personnes, ce qu'elles portent) ? Comparez la précision
    des différentes requêtes : lesquelles réussissent, lesquelles échouent, et comment l'expliquez-vous ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 4 — Diffusion : comprendre sur des points 2D

    Pour voir ce que fait un modèle de diffusion, on remplace les images par des **points du plan**.
    Le « jeu de données » est un nuage de points qui dessine une forme. Le modèle doit apprendre à transformer
    un nuage de bruit en cette forme. Tout tient en trois ingrédients.

    **1. Le bruitage** (rien à apprendre). On dégrade un point $x_0$ en $T = 200$ étapes. Grâce à une propriété
    des gaussiennes, on peut sauter directement à n'importe quelle étape $t$ :

    $$x_t = \sqrt{\bar\alpha_t}\; x_0 + \sqrt{1 - \bar\alpha_t}\; \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)$$

    où $\bar\alpha_t$ (`alphas_cumules[t]`) décroît de 1 vers 0 : c'est la part de signal restante.

    **2. L'entraînement.** On tire un point, une étape $t$ et un bruit $\varepsilon$, on calcule $x_t$, et on demande
    au réseau de **retrouver le bruit** $\varepsilon$ à partir de $x_t$ et $t$. La perte est une simple erreur quadratique.

    **3. La génération.** On part d'un bruit pur $x_T$ et on applique $T$ fois une étape de débruitage.

    ### À compléter — TODO 5 : le bruitage

    Entrées : `x0` de forme `(lot, ...)`, `t` un tenseur d'entiers `(lot,)` (une étape par exemple),
    `alphas_cumules` `(T,)` et `bruit`, de même forme que `x0`. Appliquez la formule ci-dessus.

    Attention aux formes : `alphas_cumules[t]` a la forme `(lot,)`. Pour le multiplier par `x0`, il faut lui
    ajouter des dimensions : `(lot, 1)` pour des points, `(lot, 1, 1, 1)` pour des images. Écrivez-le une fois
    pour toutes avec `coef.reshape(-1, *[1] * (x0.dim() - 1))`.
    """)
    return


@app.function
def bruiter(x0, t, alphas_cumules, bruit):
    """Saute directement à l'étape t du bruitage : x_t = racine(ac) x0 + racine(1 - ac) bruit."""
    # TODO 5
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo5, _affichage = tp.verifier(tests.tests_bruiter(bruiter), mo)
    _affichage
    return (ok_todo5,)


@app.cell(hide_code=True)
def _(mo):
    forme_2d = mo.ui.dropdown(["deux lunes", "spirale", "anneaux"], value="deux lunes", label="Forme à apprendre")
    etape_2d = mo.ui.slider(0, 199, value=0, label="Étape de bruitage t", show_value=True)
    mo.hstack([forme_2d, etape_2d], justify="start", gap=2)
    return etape_2d, forme_2d


@app.cell
def _(forme_2d, torch, tp):
    betas, alphas, alphas_cumules = tp.planning_diffusion()
    points = torch.tensor(tp.points_2d(3000, forme_2d.value))
    return alphas, alphas_cumules, betas, points


@app.cell
def _(alphas_cumules, etape_2d, mo, ok_todo5, plt, points, torch, tp):
    tp.bloquer_si(not ok_todo5, mo, "Terminez le TODO 5 pour voir le bruitage.")
    torch.manual_seed(0)
    _xt = bruiter(points, torch.full((len(points),), etape_2d.value), alphas_cumules, torch.randn_like(points))
    _fig, _ax = plt.subplots(figsize=(3.6, 3.6))
    _ax.scatter(_xt[:, 0], _xt[:, 1], s=2)
    _ax.set_xlim(-3, 3)
    _ax.set_ylim(-3, 3)
    _ax.set_title(f"Les données à l'étape t = {etape_2d.value}", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Le réseau débruiteur (fourni)

    Un petit réseau à couches denses. Il reçoit le point bruité $x_t$ et l'étape $t$, encodée par des sinus et
    cosinus (le même principe que l'encodage de position des Transformers), et prédit le bruit.
    """)
    return


@app.cell
def _(nn, torch, tp):
    class Debruiteur2D(nn.Module):
        def __init__(self, cache=128, dim_temps=32):
            super().__init__()
            self.dim_temps = dim_temps
            self.reseau = nn.Sequential(
                nn.Linear(2 + dim_temps, cache), nn.SiLU(),
                nn.Linear(cache, cache), nn.SiLU(),
                nn.Linear(cache, cache), nn.SiLU(),
                nn.Linear(cache, 2))

        def forward(self, x_t, t):                      # x_t : (lot, 2), t : (lot,)
            return self.reseau(torch.cat([x_t, tp.encodage_temps(t, self.dim_temps)], dim=1))

    return (Debruiteur2D,)


@app.cell(hide_code=True)
def _(mo):
    bouton_2d = mo.ui.run_button(label="Entraîner le débruiteur sur la forme choisie")
    bouton_2d
    return (bouton_2d,)


@app.cell
def _(
    Debruiteur2D,
    alphas_cumules,
    bouton_2d,
    mo,
    ok_todo5,
    plt,
    points,
    torch,
    tp,
):
    tp.bloquer_si(not ok_todo5, mo, "Terminez le TODO 5 d'abord.")
    mo.stop(not tp.lance(bouton_2d), mo.md("Cliquez pour lancer (environ 20 secondes)."))
    tp.fixer_graine(0)
    debruiteur_2d = Debruiteur2D()
    _opt = torch.optim.Adam(debruiteur_2d.parameters(), lr=2e-3)
    _pertes = []
    for _ in mo.status.progress_bar(range(3000), title="Entraînement du débruiteur"):
        _x0 = points[torch.randint(0, len(points), (256,))]            # 1. des points du jeu de données
        _t = torch.randint(0, len(alphas_cumules), (256,))             # 2. une étape au hasard pour chacun
        _bruit = torch.randn_like(_x0)                                 # 3. un bruit
        _perte = ((debruiteur_2d(bruiter(_x0, _t, alphas_cumules, _bruit), _t) - _bruit) ** 2).mean()   # 4. retrouver le bruit
        _opt.zero_grad()
        _perte.backward()
        _opt.step()
        _pertes.append(_perte.item())
    debruiteur_2d.eval()
    _fig, _ax = plt.subplots(figsize=(5, 2.2))
    _ax.plot(torch.tensor(_pertes).reshape(-1, 50).mean(1))
    _ax.set_xlabel("itérations (x 50)")
    _ax.set_ylabel("erreur sur le bruit")
    _fig.tight_layout()
    _fig
    return (debruiteur_2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 6 : une étape de débruitage

    C'est le cœur de la génération. À partir de $x_t$ et du bruit prédit $\hat\varepsilon$ par le réseau, on calcule
    $x_{t-1}$, un peu moins bruité :

    $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\; \hat\varepsilon \right) + \sqrt{\beta_t}\; z$$

    Le premier terme retire le bruit estimé ; le second rajoute un petit bruit frais $z$. **Exception** : à la toute dernière étape ($t = 0$), on n'ajoute
    pas de bruit.

    Ici `t` est un **entier** (toute l'image ou tout le nuage est à la même étape), donc `betas[t]`, `alphas[t]`
    et `alphas_cumules[t]` sont de simples nombres : aucun problème de forme.
    """)
    return


@app.function
def etape_debruitage(x_t, t, bruit_predit, betas, alphas, alphas_cumules, z):
    """Calcule x_(t-1) à partir de x_t et du bruit prédit. z : bruit frais, ignoré quand t == 0."""
    # TODO 6
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo6, _affichage = tp.verifier(tests.tests_debruitage(etape_debruitage), mo)
    _affichage
    return (ok_todo6,)


@app.cell
def _(
    alphas,
    alphas_cumules,
    betas,
    debruiteur_2d,
    mo,
    ok_todo6,
    plt,
    points,
    torch,
    tp,
):
    tp.bloquer_si(not ok_todo6, mo, "Terminez le TODO 6 pour générer des points.")
    torch.manual_seed(1)
    _x = torch.randn(2000, 2)                                   # on part d'un bruit pur
    _instantanes = {199: _x.clone()}
    with torch.no_grad():
        for _t in range(len(betas) - 1, -1, -1):                # T étapes de débruitage
            _eps = debruiteur_2d(_x, torch.full((len(_x),), _t))
            _x = etape_debruitage(_x, _t, _eps, betas, alphas, alphas_cumules, torch.randn_like(_x))
            if _t in (120, 60, 20, 0):
                _instantanes[_t] = _x.clone()
    _fig, _ax = plt.subplots(1, 6, figsize=(13, 2.4))
    for _a, (_t, _p) in zip(_ax, _instantanes.items()):
        _a.scatter(_p[:, 0], _p[:, 1], s=1)
        _a.set_title(f"génération, t = {_t}", fontsize=8)
    _ax[5].scatter(points[:, 0], points[:, 1], s=1, color="tab:green")
    _ax[5].set_title("vraies données", fontsize=8)
    for _a in _ax:
        _a.set_xlim(-3, 3)
        _a.set_ylim(-3, 3)
        _a.set_xticks([])
        _a.set_yticks([])
    _fig.tight_layout()
    mo.vstack([mo.md("De gauche à droite : le nuage de bruit devient progressivement la forme apprise."), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** les points générés sont-ils des copies des points d'entraînement, ou de nouveaux points
    qui suivent la même forme ? Que se passe-t-il si vous supprimez le bruit frais $z$ dans votre TODO 6
    (essayez, puis remettez-le) ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 5 — Diffusion : générer des images, avec une condition

    Passer des points aux images ne change **rien** aux deux fonctions que vous venez d'écrire. Seul le réseau
    change : c'est un **U-Net** (l'architecture du TD 2), qui reçoit une image bruitée et prédit une image de bruit.

    Le modèle fourni a été entraîné une quarantaine de minutes sur CPU, sur 60 000 petites images de vêtements (Fashion-MNIST,
    32 x 32 pixels). Il reçoit en plus une **condition** : la classe du vêtement à générer. C'est exactement le
    rôle que joue le texte dans Stable Diffusion, où la condition est l'embedding CLIP de votre phrase.

    **La force de la condition** (*guidance*) : à chaque étape, le réseau prédit le bruit deux fois, avec et sans
    la condition, et on amplifie l'écart :

    $$\hat\varepsilon = \hat\varepsilon_{\text{sans}} + g \times (\hat\varepsilon_{\text{avec}} - \hat\varepsilon_{\text{sans}})$$

    Avec $g = 0$ la condition est ignorée ; plus $g$ est grand, plus la condition pèse.
    C'est le paramètre *guidance scale* des générateurs d'images.
    """)
    return


@app.cell(hide_code=True)
def _(mo, tp):
    gen_classe = mo.ui.dropdown(tp.FASHION_CLASSES, value="basket", label="Vêtement à générer")
    gen_force = mo.ui.slider(0, 5, step=0.5, value=2, label="Force de la condition g", show_value=True)
    bouton_gen = mo.ui.run_button(label="Générer 12 images")
    mo.vstack([mo.hstack([gen_classe, gen_force], justify="start", gap=2), bouton_gen])
    return bouton_gen, gen_classe, gen_force


@app.cell
def _(
    alphas,
    alphas_cumules,
    betas,
    bouton_gen,
    gen_classe,
    gen_force,
    mo,
    ok_todo6,
    torch,
    tp,
):
    tp.bloquer_si(not ok_todo6, mo, "Terminez le TODO 6 pour générer des images.")
    mo.stop(not tp.lance(bouton_gen), mo.md("Cliquez pour générer (10 à 20 secondes)."))
    _unet = tp.charger_unet_fourni()
    _n = 12
    _classe = torch.full((_n,), tp.FASHION_CLASSES.index(gen_classe.value))
    _sans = torch.full((_n,), 10)                               # l'indice 10 signifie « pas de condition »
    _x = torch.randn(_n, 1, 32, 32)
    _suivi = []
    with torch.no_grad():
        for _t in mo.status.progress_bar(range(len(betas) - 1, -1, -1), title="Débruitage"):
            _tt = torch.full((_n,), _t)
            _eps_sans = _unet(_x, _tt, _sans)
            _eps = _eps_sans + gen_force.value * (_unet(_x, _tt, _classe) - _eps_sans)
            _x = etape_debruitage(_x, _t, _eps, betas, alphas, alphas_cumules, torch.randn_like(_x))
            if _t in (199, 150, 100, 50, 20, 0):
                _suivi.append((_t, _x[0, 0].clamp(-1, 1).clone()))
    mo.vstack([
        mo.md("**La première image au fil du débruitage**"),
        tp.grille([s.numpy() for _, s in _suivi], [f"t = {t}" for t, _ in _suivi], n_col=6, taille=1.5, cmap="gray"),
        mo.md(f"**12 images générées : {gen_classe.value}, g = {gen_force.value}**"),
        tp.grille([i[0].clamp(-1, 1).numpy() for i in _x], n_col=6, taille=1.5, cmap="gray"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** comparez $g = 0$, $g = 2$ et $g = 5$ pour la même classe. Que gagne-t-on et que
    perd-on quand $g$ augmente ? Les 12 images sont-elles identiques ? D'où vient leur diversité ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### De ce petit modèle à Stable Diffusion

    | | Votre modèle | Stable Diffusion |
    |---|---|---|
    | Ce qu'on débruite | une image 32 x 32 | un code compressé 64 x 64, décodé en image 512 x 512 par un **VAE** |
    | Réseau débruiteur | U-Net de 0,25 million de paramètres | U-Net d'environ 860 millions de paramètres |
    | Condition | une classe parmi 10 | une phrase, encodée par **CLIP** |
    | Données d'entraînement | 60 000 images | plusieurs milliards de paires image-texte |
    | Coût d'entraînement | 40 minutes sur un CPU | des centaines de milliers d'heures de GPU |

    Le principe est le même : bruiter, apprendre à prédire le bruit, débruiter pas à pas en amplifiant la condition.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 6 — Fiche de décision ingénieur
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | Retrouver dans une banque d'images de chantier les situations décrites par un responsable sécurité |
    | Baseline classique et son score | ... |
    | Modèle retenu et pourquoi | ... |
    | Métrique principale et score obtenu | ... |
    | Coût (annotation, entraînement, inférence, matériel) | ... |
    | Principaux cas d'échec | ... |
    | Le foundation model est-il justifié ici ? | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 7 — Bilan : que change un foundation model ?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** comparez le YOLO fine-tuné du TD 2 et CLIP utilisé aujourd'hui, selon trois critères :
    le nombre d'images annotées nécessaires, le temps de calcul avant d'obtenir un premier résultat, et la
    qualité du résultat. Attention : les deux ne résolvent pas le même problème. Le YOLO localise 11 classes dans
    l'image entière ; CLIP nomme ici, parmi 5 classes, un objet déjà découpé. Dites ce qui se compare directement
    et ce qui ne se compare pas. Dans quel cas partiriez-vous d'un foundation model, dans quel cas d'un modèle
    spécialisé ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 8 — Brique projet Usine 4.0 (volet au choix) : les défauts rares

    Dans une usine, certains défauts n'apparaissent que quelques fois par an : on n'a presque pas d'images pour
    entraîner un modèle. Pour le simuler, on choisit un type de défaut du jeu NEU, la **classe rare**, et on le
    réduit à 15 images dans l'entraînement. Dans le projet, c'est le volet « défauts rares », l'un des deux volets
    au choix (voir `projet/SUJET.md`). Deux pistes, au choix.

    **Piste A — Retrouver les défauts similaires.** Un opérateur photographie un défaut inconnu : le système
    lui montre les cas les plus proches de l'historique. La cellule ci-dessous pose la base avec CLIP.

    **Piste B — Générer des défauts synthétiques.** Entraîner un petit modèle de diffusion sur les images de la
    classe rare, générer de nouvelles images, et mesurer si elles améliorent le classifieur de type de défaut du projet.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_neu = mo.ui.run_button(label="Piste A : recherche de défauts similaires sur NEU")
    bouton_neu
    return (bouton_neu,)


@app.cell
def _(
    bouton_neu,
    clip_modele,
    clip_pretraitement,
    mo,
    np,
    ok_todo3,
    ok_todo4,
    precision_a_k,
    rechercher,
    tp,
):
    tp.bloquer_si(not (ok_todo3 and ok_todo4), mo, "Terminez les TODO 3 et 4 d'abord.")
    mo.stop(not tp.lance(bouton_neu), mo.md("Cliquez pour lancer (1 à 2 minutes). Les images NEU doivent être dans `donnees/NEU/`."))
    try:
        _hist, _y_hist, _req, _y_req, _noms = tp.charger_neu(n_train_par_classe=60, n_test_par_classe=10)
    except FileNotFoundError as _e:
        mo.stop(True, mo.callout(mo.md(str(_e)), kind="danger"))
    _e_hist = tp.encoder_images(clip_modele, clip_pretraitement, _hist)
    _e_req = tp.encoder_images(clip_modele, clip_pretraitement, _req)
    _h_hist = np.stack([tp.descripteur_pixels(tp.Image.open(p)) for p in _hist])
    _h_req = np.stack([tp.descripteur_pixels(tp.Image.open(p)) for p in _req])
    _lignes = []
    for _k, _nom in enumerate(_noms):
        _sel = np.flatnonzero(np.asarray(_y_req) == _k)
        _pert = np.asarray(_y_hist) == _k
        _lignes.append({
            "type de défaut": _nom,
            "précision à 5, CLIP": round(float(np.mean([precision_a_k(rechercher(_e_req[i], _e_hist, 5), _pert, 5) for i in _sel])), 2),
            "précision à 5, classique (pixels + histogrammes)": round(float(np.mean([precision_a_k(rechercher(_h_req[i], _h_hist, 5), _pert, 5) for i in _sel])), 2),
        })
    mo.vstack([mo.md("Historique : 60 images par type. Requêtes : 10 images par type. Une réponse est pertinente si elle montre le même type de défaut."),
               tp.tableau(_lignes, mo)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Si votre groupe choisit le volet défauts rares**, poursuivez dans `projet/`, par exemple dans une copie de
    `projet/brique_modele.py` (l'organisation du dossier est libre).

    *Piste A*

    1. CLIP a été entraîné sur des photos du web, très loin de textures d'acier. Fait-il mieux que la méthode
       classique sur tous les types de défauts ?
    2. Remplacez CLIP par **DINOv2** (`tp.charger_dino()`, puis passez les images dans le modèle pour obtenir les
       embeddings). DINO a été entraîné sans texte, sur des images seules : la précision change-t-elle ?
    3. Réduisez votre classe rare à 15 images dans l'historique : la retrouve-t-on encore ?

    *Piste B*

    1. Reprenez vos fonctions `bruiter` et `etape_debruitage` et le `PetitUNet` de `outils/tp_utils.py`.
       Entraînez-le sur les images de votre classe rare, réduites à 32 x 32 (prévoyez 20 à 30 minutes sur CPU).
    2. Générez 100 images. Sont-elles réalistes ? Sont-elles variées, ou des copies des 15 images d'origine ?
    3. Réentraînez votre classifieur de type de défaut avec et sans ces images, et comparez **sur de vraies images**.
       Comparez aussi à une augmentation classique (rotations, symétries, variations de contraste).
       Un résultat négatif, documenté honnêtement, est un résultat valable.

    ### Pour aller plus loin (si vous avez terminé)

    - **Stable Diffusion.** Si votre machine dispose de 8 Go de mémoire et d'une bonne connexion, installez
      `diffusers` (`uv pip install diffusers` : il restera jusqu'au prochain `uv sync`) et essayez le modèle
      `stabilityai/sd-turbo`, qui génère une image en 1 à 4 étapes. Retrouvez dans
      son code les trois composants du cours : le VAE, l'encodeur de texte CLIP et le U-Net débruiteur.
    - **Entraînez votre propre U-Net** sur Fashion-MNIST pendant 5 minutes (`tp.charger_fashion_mnist()`) et
      comparez ses images à celles du modèle fourni : que change la durée d'entraînement ?
    """)
    return


if __name__ == "__main__":
    app.run()
