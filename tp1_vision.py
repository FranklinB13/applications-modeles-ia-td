import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="medium",
    app_title="TD 1 - Vision avancée et transfer learning",
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
    from skimage import data as skdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
    import tests_tp1 as tests
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return (
        Image,
        LogisticRegression,
        SVC,
        StandardScaler,
        make_pipeline,
        mo,
        nn,
        np,
        plt,
        skdata,
        tests,
        time,
        torch,
        tp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TD 1 — Vision avancée et transfer learning

    **Module Applications et Modèles en IA · ING3 IA · durée : 3h30, de 13h30 à 17h (l'étape 11 se poursuit après la séance)**

    En entreprise, presque aucun projet de vision ne part de zéro : on prend un modèle pré-entraîné,
    on l'adapte à son problème avec quelques centaines d'images, et on passe l'essentiel de son temps
    à comprendre pourquoi il se trompe. C'est exactement le déroulé de cet après-midi.

    **Comment travailler dans ce notebook**

    - Pour exécuter une cellule : bouton de lecture à sa droite, ou Ctrl + Entrée (Cmd + Entrée sur Mac).
      Quand une cellule change, toutes celles qui utilisent ses résultats se relancent automatiquement.
    - Pour tester du code librement (par exemple afficher un modèle), ajoutez une cellule avec le bouton +
      qui apparaît entre deux cellules.
    - **Particularité de marimo** : une variable ne peut être définie que dans **une seule cellule**. Dans vos
      cellules de test, donnez à vos variables un nom qui commence par un tiret bas (`_x`, `_modele`) : elles
      restent alors locales à la cellule. Sinon, marimo affiche une erreur du type
      *« This cell redefines variables from other cells »* : il suffit de renommer la variable.
    - Les cellules marquées **À compléter** contiennent un `TODO`. Écrivez votre code, puis regardez la cellule
      de tests juste en dessous : elle indique ce qui est réussi et ce qui ne l'est pas.
    - Une étape ne se débloque que lorsque les tests de l'étape précédente sont réussis.
    - Le code de certaines cellules fournies est replié pour alléger la lecture : vous pouvez l'afficher
      depuis le menu de la cellule.
    - Les calculs longs ne partent que lorsque vous cliquez sur leur bouton. Leurs résultats sont mis en cache :
      si vous relancez le notebook, ils sont rechargés instantanément.
    - Les cellules **Votre réponse** sont à rédiger directement dans le notebook (double-cliquez sur le texte pour
      l'éditer) : elles font partie de ce que vous poussez sur GitHub.

    | Étape | Contenu | Durée indicative |
    |---|---|---|
    | 0 | Vérification de l'installation | 5 min |
    | 1 | Le cours en bref, avec démonstrations | 10 min |
    | 2 | Trois modèles pré-entraînés face à de vraies images | 15 min |
    | 3 | Le jeu de données Oxford Pets | 10 min |
    | 4 | Feature extraction : trois architectures comparées | 30 min |
    | 5 | Baseline de machine learning classique | 20 min |
    | 6 | Fine-tuning | 45 min |
    | 7 | Mesure finale sur le test, analyse des erreurs | 20 min |
    | 8 | Diagnostic avec Grad-CAM | 25 min |
    | 9 | Mini-application | 10 min |
    | 10 | Fiche de décision ingénieur | 10 min |
    | 11 | Brique projet Usine 4.0 (suite dans `projet/`) | 10 min |
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch, tp):
    _dev = tp.appareil()
    _lignes = [
        f"- Matériel utilisé : **{_dev}** (le TD est conçu pour fonctionner sur CPU)",
        f"- Threads CPU : {torch.get_num_threads()}",
        f"- PyTorch : {torch.__version__}",
        f"- Dossier de données : `{tp.DONNEES}`",
    ]
    if tp.MODE_TEST:
        _lignes.append("- **Mode test actif** : poids aléatoires et données synthétiques (vérification hors ligne).")
    mo.vstack([mo.md("## Étape 0 — Vérification de l'installation"), mo.callout(mo.md("\n".join(_lignes)), kind="info")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 1 — Le cours en bref

    ### 1.1 Rappel : ce qu'est un CNN, et ce que fait une convolution

    Un **réseau de neurones convolutif** (CNN) a deux parties. Le **backbone** transforme l'image en cartes de
    features : à chaque étage, des convolutions (avec BatchNorm et ReLU), puis la résolution est divisée par 2 et
    le nombre de canaux doublé (pour ResNet-18 : 224 × 224 × 3, puis 56 × 56 × 64, … , 7 × 7 × 512). Un pooling
    global résume chaque carte en une valeur ; la **tête**, une couche dense, en déduit un score par classe.
    ResNet-18 et EfficientNet-B0 sont des CNN ; DeiT-tiny, un Vision Transformer, n'en est pas un.

    Une couche de convolution fait glisser de petits filtres sur l'image. Chaque filtre répond à un motif local :
    un bord vertical, un contraste, une texture. En empilant les couches, le réseau combine ces motifs simples en
    motifs de plus en plus abstraits (un œil, une oreille, une tête de chat).

    En traitement d'image classique, ces filtres sont **choisis à la main** (Sobel, flou, etc.).
    Dans un CNN, ils sont **appris** à partir des données. C'est la différence essentielle avec le machine
    learning classique, et nous la mesurerons à l'étape 5.

    Choisissez un filtre fait main et observez ce qu'il fait ressortir.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    choix_filtre = mo.ui.dropdown(
        options=["Bords verticaux (Sobel x)", "Bords horizontaux (Sobel y)", "Flou (moyenne 5x5)", "Accentuation"],
        value="Bords verticaux (Sobel x)",
        label="Filtre",
    )
    choix_filtre
    return (choix_filtre,)


@app.cell(hide_code=True)
def _(choix_filtre, np, plt, skdata, torch):
    _img = skdata.chelsea().mean(axis=2) / 255.0
    _noyaux = {
        "Bords verticaux (Sobel x)": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "Bords horizontaux (Sobel y)": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "Flou (moyenne 5x5)": (np.ones((5, 5)) / 25).tolist(),
        "Accentuation": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    }
    _k = torch.tensor(_noyaux[choix_filtre.value], dtype=torch.float32)[None, None]
    _sortie = torch.nn.functional.conv2d(torch.tensor(_img, dtype=torch.float32)[None, None], _k, padding="same")[0, 0]
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3.4))
    _ax[0].imshow(_img, cmap="gray")
    _ax[0].set_title("Image d'entrée")
    _ax[1].imshow(_sortie.numpy(), cmap="gray")
    _ax[1].set_title(f"Carte de sortie : {choix_filtre.value}")
    for _a in _ax:
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Rappel : le mécanisme d'attention (vu en NLP)

    Chaque élément d'une séquence (un mot en NLP, un morceau d'image ici) produit trois vecteurs :
    une **requête** $q$, une **clé** $k$ et une **valeur** $v$. L'élément $i$ calcule une moyenne pondérée des valeurs de
    tous les éléments, lui compris, avec un poids d'autant plus fort que sa requête ressemble à leur clé :

    $$\text{poids}_{ij} = \text{softmax}_j\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right), \qquad
    \text{sortie}_i = \sum_j \text{poids}_{ij}\, v_j$$

    Contrairement à une convolution, qui ne regarde que le voisinage immédiat, l'attention permet à chaque
    élément de regarder **toute** l'image dès la première couche. Le curseur ci-dessous multiplie les scores avant le
    softmax : il règle la « netteté » de l'attention. Valeur faible : chaque élément répartit son attention sur
    tous les éléments. Valeur forte : il se concentre sur un seul.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    nettete = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="Netteté de l'attention", show_value=True)
    nettete
    return (nettete,)


@app.cell(hide_code=True)
def _(nettete, plt, torch):
    torch.manual_seed(3)
    _q, _k = torch.randn(6, 8), torch.randn(6, 8)
    _poids = torch.softmax(nettete.value * (_q @ _k.T) / 8**0.5, dim=1)
    _fig, _ax = plt.subplots(figsize=(4.2, 3.6))
    _im = _ax.imshow(_poids.numpy(), cmap="viridis", vmin=0, vmax=1)
    _ax.set_xlabel("élément regardé (clé)")
    _ax.set_ylabel("élément qui regarde (requête)")
    _ax.set_title("Matrice d'attention (chaque ligne somme à 1)", fontsize=9)
    _fig.colorbar(_im)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Les trois architectures du jour

    **ResNet (He et al., 2016).** Au-delà d'une certaine profondeur (sur CIFAR-10, 56 couches font moins bien que 20), les CNN classiques deviennent *moins* bons,
    même sur les données d'entraînement : ce n'est pas du surapprentissage, c'est l'optimisation qui ne trouve
    plus de bons poids. ResNet ajoute des **connexions résiduelles** : chaque bloc calcule $y = \text{ReLU}(F(x) + x)$.
    Le bloc n'a plus qu'à apprendre une *correction* de son entrée (l'identité est facile : il suffit que $F$ soit
    nulle), et le raccourci laisse passer l'entrée sans transformation. Chaque convolution y est suivie d'une
    **BatchNorm**, qui ne se comporte pas de la même façon en `train()` et en `eval()` : cela comptera à l'étape 6.
    On peut alors empiler plus de cent couches. ResNet reste la base de nombreux modèles en production.

    **EfficientNet (Tan et Le, 2019).** Pour rendre un réseau plus performant, on peut l'approfondir, l'élargir ou augmenter
    la résolution des images. EfficientNet montre qu'il faut faire **les trois ensemble, dans des proportions
    fixes** (*compound scaling*). Résultat : à budget de calcul égal, il est plus précis. C'est un très bon choix
    par défaut quand les ressources sont limitées, comme sur nos machines.

    **Vision Transformer, ViT (Dosovitskiy et al., 2021).** On découpe l'image en carrés (*patches*), on transforme chaque patch en
    vecteur, et on traite la suite de patches exactement comme une phrase : avec de l'attention. ViT a beaucoup
    moins d'a priori que les CNN (localité, équivariance à la translation) : il doit les apprendre, donc il lui faut beaucoup
    de données de pré-entraînement. Mais une fois pré-entraîné à grande échelle, il approche ou dépasse les meilleurs CNN.
    Nous utilisons **DeiT-tiny**, une petite version de ViT conçue pour être entraînée sur ImageNet seul.

    Faites varier la taille des patches : plus ils sont petits, plus la séquence est longue, et le coût de
    l'attention croît comme le **carré** du nombre de patches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    taille_patch = mo.ui.slider(steps=[8, 14, 16, 28, 32, 56], value=16, label="Taille des patches (pixels)", show_value=True)
    taille_patch
    return (taille_patch,)


@app.cell(hide_code=True)
def _(Image, mo, np, plt, skdata, taille_patch):
    _img = np.asarray(Image.fromarray(skdata.chelsea()).resize((224, 224)))
    _p = taille_patch.value
    _n = (224 // _p) ** 2
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.imshow(_img)
    for _x in range(0, 225, _p):
        _ax.axhline(_x, color="white", lw=0.6)
        _ax.axvline(_x, color="white", lw=0.6)
    _ax.axis("off")
    _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""
    - Image 224 x 224, patches de {_p} x {_p}
    - Nombre de patches (tokens) : **{_n}**
    - Taille de la matrice d'attention : {_n} x {_n} = **{_n * _n:,}** coefficients par tête et par couche
    """.replace(f"{_n * _n:,}", f"{_n * _n:,}".replace(",", " "))),
    ], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Réutiliser un modèle pré-entraîné : deux stratégies

    Un réseau pré-entraîné sur ImageNet a appris des features génériques dans ses premières couches (bords, couleurs,
    textures) et plus spécifiques dans les dernières (parties d'objets, classes d'ImageNet).
    Pour l'adapter à un nouveau problème, on remplace sa dernière couche (la *tête* de classification) par une
    nouvelle tête adaptée à nos classes, puis :

    | | Feature extraction | Fine-tuning |
    |---|---|---|
    | Ce qu'on entraîne | uniquement la nouvelle tête | la tête et tout ou partie du réseau (dans ce TD : les derniers étages) |
    | Coût | très faible (on calcule les features une fois) | plus élevé (passes avant et arrière) |
    | Quand l'utiliser | peu de données, domaine proche d'ImageNet | plus de données ; d'autant plus d'étages dégelés que le domaine est éloigné |
    | Risque | plafonne si le domaine est très différent | surapprentissage, oubli des features utiles |

    Peu de données et domaine éloigné : aucune stratégie n'est sûre à l'avance (CS231n conseille alors un classifieur
    linéaire sur des activations plus basses du réseau) ; on compare sur la validation. Le point de comparaison
    des deux stratégies est l'entraînement de zéro, sans transfert.

    Deux règles pratiques : avec peu d'images, on laisse les premiers étages gelés (c'est le choix de ce TD ; d2l
    réentraîne tout le réseau), et les étages dégelés reçoivent un **learning rate plus faible** que la tête,
    pour ne pas déformer trop vite les poids pré-entraînés.

    ### 1.5 Deep learning ou machine learning classique ?

    Avant le deep learning, on décrivait une image avec des **descripteurs construits à la main**
    (histogrammes de gradients orientés HOG, histogrammes de couleur, textures), puis on entraînait un classifieur
    classique (SVM, forêt aléatoire). C'est rapide, interprétable, et parfois suffisant. Le deep learning apprend
    lui-même ses descripteurs : il gagne quand les classes se distinguent par des détails que personne ne saurait
    décrire à la main. Un ingénieur commence toujours par se demander si le deep learning vaut son coût :
    c'est la question que vous trancherez avec des chiffres à l'étape 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 2 — Trois modèles pré-entraînés face à de vraies images

    On charge ResNet-18, EfficientNet-B0 et DeiT-tiny, tous pré-entraînés sur ImageNet (1000 classes),
    et on leur montre quatre images. Le chargement télécharge les poids la première fois (environ 90 Mo au total).
    Le temps affiché suit le même principe qu'en cours : une passe d'échauffement, puis la moyenne de 5 passes.
    """)
    return


@app.cell
def _(Image, skdata, tp):
    categories = tp.categories_imagenet()
    images_exemple = {
        "chat": Image.fromarray(skdata.chelsea()),
        "café": Image.fromarray(skdata.coffee()),
        "astronaute": Image.fromarray(skdata.astronaut()),
        "fusée": Image.fromarray(skdata.rocket()),
    }
    modeles_imagenet = {nom: tp.charger_modele(nom) for nom in ["resnet18", "efficientnet_b0", "deit_tiny"]}
    return categories, images_exemple, modeles_imagenet


@app.cell
def _(categories, images_exemple, mo, modeles_imagenet, time, torch, tp):
    _transfo = tp.transformation(224)
    _lignes = ["| Image | Modèle | Top 1 | Top 2 | Top 3 | Temps (ms) |", "|---|---|---|---|---|---|"]
    for _nom_img, _img in images_exemple.items():
        _x = _transfo(_img.convert("RGB"))[None]
        for _nom_m, _m in modeles_imagenet.items():
            with torch.no_grad():
                _m(_x)  # échauffement : la première passe est plus lente
                _t = time.perf_counter()
                for _ in range(5):
                    _p = _m(_x).softmax(-1)[0]
                _ms = (time.perf_counter() - _t) / 5 * 1000
            _top = _p.topk(3)
            _cases = [f"{categories[i]} ({v:.0%})" for v, i in zip(_top.values.tolist(), _top.indices.tolist())]
            _lignes.append(f"| {_nom_img} | {_nom_m} | " + " | ".join(_cases) + f" | {_ms:.0f} |")
    mo.vstack([(tp.grille([_i for _i in images_exemple.values()], list(images_exemple), n_col=4, taille=2.2)),
               mo.md("\n".join(_lignes))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Les trois modèles sont-ils d'accord ? Quand ils se trompent, l'erreur est-elle « raisonnable » ?
    2. Quelle confiance le modèle affiche-t-il pour un objet absent des 1000 classes d'ImageNet (l'astronaute) ?
       Qu'en déduisez-vous sur l'usage d'un score de confiance en production ?
    3. Comparez les temps d'inférence. Quel modèle déploieriez-vous sur une machine sans GPU ?
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
    ## Étape 3 — Le jeu de données Oxford Pets

    37 races de chats et de chiens. Pour que tout tourne sur CPU, on garde un sous-ensemble équilibré :
    30 images d'entraînement, 10 de validation et 15 de test par race. C'est aussi plus réaliste : en entreprise,
    on dispose rarement de milliers d'images annotées par classe.

    Les images de **validation** servent à comparer les méthodes et les configurations (étapes 4 à 6). Le jeu de
    **test** reste à l'écart jusqu'à l'étape 7 : il ne sert qu'à la mesure finale, une seule fois.

    Le premier lancement télécharge environ 800 Mo (une seule fois).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_donnees = mo.ui.run_button(label="Charger Oxford Pets")
    bouton_donnees
    return (bouton_donnees,)


@app.cell(hide_code=True)
def _(bouton_donnees, mo, tp):
    _deja_la = (tp.DONNEES / "oxford-iiit-pet" / "images").exists() or tp.MODE_TEST
    mo.stop(not (tp.lance(bouton_donnees) or _deja_la), mo.md("Cliquez sur le bouton pour charger les données."))
    img_tr, y_tr, img_va, y_va, img_te, y_te, classes = tp.charger_oxford_pets(n_train=30, n_test=15, n_val=10)
    return classes, img_te, img_tr, img_va, y_te, y_tr, y_va


@app.cell(hide_code=True)
def _(classes, img_tr, mo, np, tp, y_tr):
    _rng = np.random.default_rng(1)
    _idx = _rng.choice(len(img_tr), 10, replace=False)
    _ds = tp.DatasetImages([img_tr[i] for i in _idx], [y_tr[i] for i in _idx], transform=lambda x: x)
    mo.vstack([
        mo.md(f"**{len(img_tr)}** images d'entraînement, **{len(classes)}** classes. Quelques exemples :"),
        (tp.grille([_ds.image_brute(i) for i in range(10)], [classes[y_tr[i]] for i in _idx], n_col=5, taille=2.2)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 4 — Feature extraction : trois architectures comparées

    Idée : on retire la tête de classification d'un réseau pré-entraîné. Ce qui sort n'est plus une probabilité
    par classe ImageNet, mais la sortie du backbone : un **vecteur de features** (on dit aussi *embedding* de l'image,
    à ne pas confondre avec les embeddings de tokens du ViT) qui résume l'image, de 512 valeurs pour ResNet-18, 1280 pour EfficientNet-B0 et 192 pour DeiT-tiny. On calcule
    ces vecteurs une seule fois, puis on entraîne un simple classifieur linéaire dessus. Les trois architectures
    se comparent sur la **validation**.

    ### À compléter — TODO 1 : retirer la tête

    Les trois modèles ne nomment pas leur tête de la même façon. Affichez les noms de leurs modules dans une cellule de test
    (`[n for n, _ in tp.charger_modele("efficientnet_b0").named_children()]`) : le dernier module est la tête.
    Remplacez-le par `nn.Identity()`, une couche qui renvoie son entrée telle quelle. Indice : `hasattr(modele, "fc")`
    teste si le modèle a un attribut `fc`, ce qui permet d'écrire une seule fonction pour les trois modèles.

    - ResNet-18 (torchvision) : la tête s'appelle `fc`
    - EfficientNet-B0 (torchvision) : ?
    - DeiT-tiny (timm) : ?
    """)
    return


@app.cell
def _(nn):
    def retirer_tete(modele):
        """Remplace la couche de classification par nn.Identity() et renvoie le modèle."""
        # TODO 1 : trouvez la tête de classification et remplacez-la par nn.Identity().
        # La même fonction doit marcher pour les trois modèles.
        raise NotImplementedError("À compléter")
    return (retirer_tete,)


@app.cell(hide_code=True)
def _(mo, retirer_tete, tests, tp):
    ok_todo1, _affichage = tp.verifier(tests.tests_retirer_tete(retirer_tete), mo)
    _affichage
    return (ok_todo1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Une fois les tests réussis, lancez le calcul des embeddings pour les trois architectures. Sur un processeur
    à 2 cœurs, le calcul prend environ 5 minutes (moins sur une machine plus puissante). Pendant ce temps,
    lisez la suite.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_embeddings = mo.ui.run_button(label="Calculer les embeddings (3 modèles)")
    bouton_embeddings
    return (bouton_embeddings,)


@app.cell
def _(
    bouton_embeddings,
    img_tr,
    img_va,
    mo,
    ok_todo1,
    retirer_tete,
    time,
    tp,
    y_tr,
    y_va,
):
    tp.bloquer_si(not ok_todo1, mo, "Terminez le TODO 1 pour débloquer cette étape.")
    _noms = ["resnet18", "efficientnet_b0", "deit_tiny"]
    _caches = {n: tp.charger_cache(f"tp1_emb_va_{n}") for n in _noms}
    mo.stop(not (tp.lance(bouton_embeddings) or all(c is not None for c in _caches.values())),
            mo.md("Cliquez sur le bouton pour lancer le calcul."))
    _transfo = tp.transformation(224)
    _ds_tr = tp.DatasetImages(img_tr, y_tr, _transfo)
    _ds_va = tp.DatasetImages(img_va, y_va, _transfo)
    embeddings = {}
    for _n in _noms:
        if _caches[_n] is None:
            _m = retirer_tete(tp.charger_modele(_n))
            _t = time.perf_counter()
            _e_tr = tp.extraire_embeddings(_m, _ds_tr, progression=lambda it, n=_n: mo.status.progress_bar(it, title=f"{n} (entraînement)"))
            _e_va = tp.extraire_embeddings(_m, _ds_va, progression=lambda it, n=_n: mo.status.progress_bar(it, title=f"{n} (validation)"))
            _ms = (time.perf_counter() - _t) / (len(_ds_tr) + len(_ds_va)) * 1000
            tp.sauver_cache(f"tp1_emb_va_{_n}", tr=_e_tr, va=_e_va, ms=_ms, params=tp.nombre_parametres(_m))
            _caches[_n] = tp.charger_cache(f"tp1_emb_va_{_n}")
        embeddings[_n] = _caches[_n]
    mo.md("Embeddings calculés : " + ", ".join(f"{n} {embeddings[n]['tr'].shape}" for n in _noms))
    return (embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On entraîne maintenant une **régression logistique** (un classifieur linéaire) sur chaque jeu d'embeddings.
    Les embeddings sont d'abord standardisés (moyenne nulle, variance unité), ce qui aide l'optimisation.
    Le réglage C = 0,5 est fixé, le même pour les trois modèles, et non choisi sur la validation : la comparaison
    avec la baseline, dont le C est réglé à l'étape 5, est donc plutôt défavorable au deep learning.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    embeddings,
    make_pipeline,
    mo,
    np,
    tp,
    y_tr,
    y_va,
):

    resultats_fe = []
    for _n, _e in embeddings.items():
        _clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        _clf.fit(_e["tr"], y_tr)
        _acc = float(np.mean(_clf.predict(_e["va"]) == np.asarray(y_va)))
        resultats_fe.append({"méthode": f"{_n} + régression logistique", "accuracy validation": round(_acc, 3),
                             "dimension": _e["tr"].shape[1], "ms / image (lecture et prétraitement compris)": round(float(_e["ms"]), 1),
                             "paramètres du backbone (M)": round(float(_e["params"]) / 1e6, 1)})
    tp.tableau(resultats_fe, mo)
    return (resultats_fe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Sur la validation, quelle architecture donne les meilleurs embeddings ? Est-ce celle qui a le plus de paramètres ?
    2. Mettez en regard l'accuracy et le temps de calcul : quelle architecture offre le meilleur compromis
       pour une machine sans GPU ?
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
    ## Étape 5 — Baseline de machine learning classique

    Le deep learning vaut-il son coût ici ? Pour le savoir, il faut une **baseline** : une solution simple et
    raisonnable, sans réseau de neurones, à laquelle comparer le deep learning. On décrit chaque image avec deux descripteurs construits à la main :

    - **HOG** (histogrammes de gradients orientés) : décrit les contours et les formes locales ;
    - **histogrammes de couleur** : décrivent la distribution des couleurs, sans aucune information spatiale.

    ### À compléter — TODO 2 : descripteurs classiques

    1. Redimensionnez l'image en 128 x 128 (RGB).
    2. HOG sur l'image en niveaux de gris (valeurs entre 0 et 1), avec `skimage.feature.hog` :
       `orientations=9`, `pixels_per_cell=(16, 16)`, `cells_per_block=(2, 2)`. Vous obtenez 1764 valeurs.
    3. Pour chacun des 3 canaux R, G, B : histogramme de 16 intervalles sur [0, 256), **normalisé** (somme égale à 1).
    4. Concaténez le tout : un vecteur de 1764 + 48 = 1812 valeurs.
    """)
    return


@app.cell
def _(np):
    from skimage.feature import hog

    def descripteurs_classiques(image):
        """image : PIL.Image -> vecteur numpy de longueur 1812 (HOG + histogrammes de couleur)."""
        # TODO 2
        raise NotImplementedError("À compléter")
    return (descripteurs_classiques,)


@app.cell(hide_code=True)
def _(descripteurs_classiques, mo, tests, tp):
    ok_todo2, _affichage = tp.verifier(tests.tests_descripteurs(descripteurs_classiques), mo)
    _affichage
    return (ok_todo2,)


@app.cell
def _(
    SVC,
    StandardScaler,
    descripteurs_classiques,
    img_tr,
    img_va,
    make_pipeline,
    mo,
    np,
    ok_todo2,
    time,
    tp,
    y_tr,
    y_va,
):
    tp.bloquer_si(not ok_todo2, mo, "Terminez le TODO 2 pour débloquer cette étape.")

    _ds_tr = tp.DatasetImages(img_tr, y_tr, transform=lambda x: x)
    _ds_va = tp.DatasetImages(img_va, y_va, transform=lambda x: x)
    _t = time.perf_counter()
    _f_tr = np.stack([descripteurs_classiques(_ds_tr.image_brute(i)) for i in mo.status.progress_bar(range(len(_ds_tr)), title="Descripteurs (entraînement)")])
    _f_va = np.stack([descripteurs_classiques(_ds_va.image_brute(i)) for i in range(len(_ds_va))])
    _ms = (time.perf_counter() - _t) / (len(_f_tr) + len(_f_va)) * 1000
    # Le réglage C du SVM se choisit lui aussi sur la validation : une baseline équitable est une baseline réglée.
    _meilleur = None
    for _C in (1, 10, 100):
        _svm = make_pipeline(StandardScaler(), SVC(C=_C, kernel="rbf")).fit(_f_tr, y_tr)
        _acc = float(np.mean(_svm.predict(_f_va) == np.asarray(y_va)))
        if _meilleur is None or _acc > _meilleur[0]:
            _meilleur = (_acc, _C, _svm)
    _acc_va, _C, svm_classique = _meilleur
    resultat_classique = {"méthode": f"HOG + couleur + SVM (classique, C = {_C})",
                          "accuracy validation": round(_acc_va, 3),
                          "dimension": _f_tr.shape[1], "ms / image (lecture et prétraitement compris)": round(_ms, 1), "paramètres du backbone (M)": "—"}
    mo.md(f"Baseline classique : accuracy de validation = **{_acc_va:.1%}** (C = {_C}, choisi sur la validation parmi 1, 10 et 100).")
    return resultat_classique, svm_classique


@app.cell(hide_code=True)
def _(mo, resultat_classique, resultats_fe, tp):
    tableau_comparatif = [resultat_classique] + list(resultats_fe)
    mo.vstack([mo.md(f"Hasard : {1 / 37:.1%} (37 classes)."), tp.tableau(tableau_comparatif, mo)])
    return (tableau_comparatif,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Sur la validation, quel écart, et dans quel sens, entre le deep learning et la baseline ? Cet écart justifie-t-il le coût de calcul ?
    2. Des descripteurs construits à la main (HOG, couleur) peuvent-ils distinguer deux races proches ? Formulez une hypothèse ;
       vous la vérifierez à l'étape 7 avec la matrice de confusion.
    3. Imaginez un problème où la baseline classique serait suffisante. Décrivez-le.
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
    ## Étape 6 — Fine-tuning

    On va maintenant dégeler les derniers étages d'EfficientNet-B0 et les réentraîner avec la nouvelle tête.
    Le modèle est imposé, quel que soit le classement de l'étape 4 : des trois, c'est celui qui demande le moins
    de calcul par image, ce qui compte sur CPU. Vous situerez son résultat par rapport à l'étape 4 dans votre
    conclusion. Pour tenir sur CPU, on travaille en 160 x 160 pixels. Trois fonctions à écrire, chacune avec ses tests.

    Dans EfficientNet-B0 (torchvision), le réseau est découpé en `modele.features`, une suite de 9 éléments
    (`modele.features[0]` : la convolution d'entrée ; `[1]` à `[7]` : les sept étages de blocs MBConv ;
    `[8]` : une convolution 1 x 1 finale), puis `modele.classifier`, la tête. Dans ce TD, on appelle « étage »
    chacun de ces 9 éléments : dégeler 2 étages, c'est dégeler `features[7]` et `features[8]`. (Le nom `features`
    est celui du module dans torchvision ; il n'a rien à voir avec les vecteurs de features de l'étape 4.)

    ### À compléter — TODO 3 : geler le backbone

    Gelez **tous** les paramètres (`p.requires_grad = False`), puis dégelez les `n_etages_degeles` derniers étages
    de `modele.features`, et enfin la tête `modele.classifier`. Attention : `modele.features[-0:]` renvoie *tous*
    les étages ; traitez `n_etages_degeles = 0` à part.
    """)
    return


@app.function
def geler_backbone(modele, n_etages_degeles):
    """Gèle tout sauf les n derniers étages de modele.features et modele.classifier. Renvoie le modèle."""
    # TODO 3
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo3, _affichage = tp.verifier(tests.tests_geler(geler_backbone), mo)
    _affichage
    return (ok_todo3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 4 : un learning rate par partie du réseau

    Créez un optimiseur `torch.optim.AdamW` avec **deux groupes de paramètres** :

    - la tête (`modele.classifier`) avec `lr_tete` ;
    - les autres paramètres **entraînables** du backbone avec `lr_backbone`, plus faible.

    Un groupe s'écrit `{"params": liste_de_parametres, "lr": valeur}`. N'incluez pas les paramètres gelés,
    et si le backbone est entièrement gelé, ne créez que le groupe de la tête. Gardez `weight_decay=1e-4`, comme en cours.
    """)
    return


@app.cell
def _(torch):
    def creer_optimiseur(modele, lr_tete, lr_backbone):
        """Renvoie un AdamW avec un learning rate pour la tête et un autre pour le backbone dégelé."""
        # TODO 4
        raise NotImplementedError("À compléter")
    return (creer_optimiseur,)


@app.cell(hide_code=True)
def _(creer_optimiseur, mo, ok_todo3, tests, tp):
    tp.bloquer_si(not ok_todo3, mo, "Terminez le TODO 3 d'abord.")
    ok_todo4, _affichage = tp.verifier(tests.tests_optimiseur(creer_optimiseur, geler_backbone), mo)
    _affichage
    return (ok_todo4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 5 : une époque d'entraînement

    La boucle d'entraînement classique, vue en cours d'introduction au deep learning. Pour chaque lot `(x, y)` du chargeur :
    envoyer les données sur `appareil`, remettre les gradients à zéro, calculer la perte, rétropropager,
    mettre à jour les poids. Renvoyez la **perte moyenne sur l'ensemble des exemples** (un `float`) : pondérez la perte de chaque lot
    par sa taille, car le dernier lot peut être plus petit.
    N'oubliez pas de passer le modèle en mode entraînement au début.
    """)
    return


@app.function
def entrainer_une_epoque(modele, chargeur, optimiseur, critere, appareil):
    """Entraîne le modèle sur un passage complet du chargeur. Renvoie la perte moyenne (float)."""
    # TODO 5
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo5, _affichage = tp.verifier(tests.tests_entrainement(entrainer_une_epoque), mo)
    _affichage
    return (ok_todo5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Lancer le fine-tuning

    Choisissez une configuration et lancez. Avec 2 étages dégelés et 3 époques, comptez quelques minutes sur CPU : le temps dépend de la machine.
    Le modèle obtenu à la dernière époque est sauvegardé dans `sorties/` et rechargé automatiquement si vous
    relancez le notebook. Chaque nouvel entraînement remplace le précédent.

    Commencez par **0 étage dégelé** : seule la tête est entraînée, à la même résolution et avec la même
    augmentation que la suite. C'est la référence à laquelle comparer le fine-tuning. Ce n'est pas tout à fait
    la feature extraction de l'étape 4 : en mode entraînement, les BatchNorm du backbone normalisent avec les
    statistiques du lot et mettent à jour leurs statistiques mémorisées, même quand leurs poids sont gelés ; de plus,
    la *stochastic depth* d'EfficientNet annule au hasard la branche de certains blocs, pour certaines images. Essayez ensuite au moins une autre
    configuration (par exemple 2 étages) et notez les résultats. Chaque configuration repart d'une tête neuve.

    La perte utilisée est l'entropie croisée avec `label_smoothing=0.1` : la cible n'est plus 1 pour la bonne
    classe et 0 ailleurs, mais 0,1 / 37 ≈ 0,003 pour chacune des 36 autres classes et 0,9 + 0,003 ≈ 0,903 pour la bonne. Cela limite l'excès de confiance,
    utile avec 30 images par classe.

    Sur 370 images de validation, une accuracy comprise entre 70 % et 90 % est connue à ± 3 à 5 points près
    (intervalle de confiance à 95 %) : un écart de 1 ou 2 points entre deux configurations peut n'être que du bruit.

    Les configurations se comparent sur le jeu de **validation**. Le jeu de test n'intervient qu'à l'étape 7,
    une fois votre choix fait : terminez donc par la configuration que vous retenez, c'est elle qui sera analysée.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cfg_blocs = mo.ui.slider(0, 4, value=2, label="Étages dégelés", show_value=True)
    cfg_epoques = mo.ui.slider(1, 6, value=3, label="Époques", show_value=True)
    cfg_lr_tete = mo.ui.dropdown(["1e-2", "3e-3", "1e-3"], value="3e-3", label="LR tête")
    cfg_lr_backbone = mo.ui.dropdown(["1e-3", "3e-4", "1e-4", "3e-5"], value="3e-4", label="LR backbone")
    bouton_entrainement = mo.ui.run_button(label="Lancer le fine-tuning")
    mo.vstack([mo.hstack([cfg_blocs, cfg_epoques]), mo.hstack([cfg_lr_tete, cfg_lr_backbone]), bouton_entrainement])
    return (
        bouton_entrainement,
        cfg_blocs,
        cfg_epoques,
        cfg_lr_backbone,
        cfg_lr_tete,
    )


@app.cell
def _(
    bouton_entrainement,
    cfg_blocs,
    cfg_epoques,
    cfg_lr_backbone,
    cfg_lr_tete,
    classes,
    creer_optimiseur,
    img_tr,
    img_va,
    mo,
    nn,
    np,
    ok_todo4,
    ok_todo5,
    torch,
    tp,
    y_tr,
    y_va,
):
    tp.bloquer_si(not (ok_todo4 and ok_todo5), mo, "Terminez les TODO 3, 4 et 5 pour débloquer l'entraînement.")
    _chemin = tp.SORTIES / ("tp1_finetune_test.pt" if tp.MODE_TEST else "tp1_finetune.pt")
    mo.stop(not (tp.lance(bouton_entrainement) or _chemin.exists()), mo.md("Cliquez sur le bouton pour lancer l'entraînement."))

    def _construire(n_classes):
        _m = tp.charger_modele("efficientnet_b0")
        _m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, n_classes))
        return _m

    _dev = tp.appareil()
    if tp.lance(bouton_entrainement):
        tp.fixer_graine(0)
        _ds_tr = tp.DatasetImages(img_tr, y_tr, tp.transformation(160, augmentation=True), pre_redim=184)
        _ds_va = tp.DatasetImages(img_va, y_va, tp.transformation(160), pre_redim=184)
        _chargeur = torch.utils.data.DataLoader(_ds_tr, batch_size=32, shuffle=True, num_workers=0)
        modele_ft = geler_backbone(_construire(len(classes)), cfg_blocs.value).to(_dev)
        _opt = creer_optimiseur(modele_ft, float(cfg_lr_tete.value), float(cfg_lr_backbone.value))
        _crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        historique_ft = {"perte": [], "accuracy": [], "config": {
            "étages dégelés": cfg_blocs.value, "époques": cfg_epoques.value,
            "lr tête": cfg_lr_tete.value, "lr backbone": cfg_lr_backbone.value,
            "paramètres entraînables (M)": round(tp.nombre_parametres(modele_ft, entrainables=True) / 1e6, 2)}}
        with tp.Chrono() as _c:
            for _ep in mo.status.progress_bar(range(cfg_epoques.value), title="Fine-tuning", subtitle="une époque à la fois"):
                historique_ft["perte"].append(entrainer_une_epoque(modele_ft, _chargeur, _opt, _crit, _dev))
                _pred, _ = tp.predire(modele_ft, _ds_va)      # suivi sur la validation : le test reste intact
                historique_ft["accuracy"].append(float(np.mean(_pred == np.asarray(y_va))))
        historique_ft["durée (s)"] = round(_c.secondes)
        _etat = {k: v.detach().cpu() for k, v in modele_ft.state_dict().items()}
        torch.save({"etat": _etat, "historique": historique_ft, "n_classes": len(classes)}, _chemin)
    else:
        _sauvegarde = torch.load(_chemin, map_location="cpu")
        modele_ft = _construire(_sauvegarde["n_classes"])
        modele_ft.load_state_dict(_sauvegarde["etat"])
        historique_ft = _sauvegarde["historique"]
    modele_ft = modele_ft.to(_dev).eval()
    return historique_ft, modele_ft


@app.cell(hide_code=True)
def _(historique_ft, mo, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3))
    _ep = range(1, len(historique_ft["perte"]) + 1)
    _ax[0].plot(_ep, historique_ft["perte"], marker="o")
    _ax[0].set_title("Perte d'entraînement")
    _ax[1].plot(_ep, historique_ft["accuracy"], marker="o", color="tab:green")
    _ax[1].set_title("Accuracy de validation")
    for _a in _ax:
        _a.set_xlabel("époque")
    _fig.tight_layout()
    _cfg = ", ".join(f"{k} : {v}" for k, v in historique_ft["config"].items())
    mo.vstack([mo.md(f"**Configuration** : {_cfg} · durée : {historique_ft.get('durée (s)', '?')} s · "
                     f"accuracy de validation à la dernière époque : **{historique_ft['accuracy'][-1]:.1%}**"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** reportez ici chaque configuration testée et son résultat, puis concluez.

    | Étages dégelés | Époques | LR tête | LR backbone | Accuracy de validation | Durée |
    |---|---|---|---|---|---|
    | 0 | | | | | |
    | | | | | | |

    Conclusion (à résolution égale, dégeler des étages vaut-il son coût par rapport à la tête seule ? Les écarts
    dépassent-ils l'incertitude de mesure ? Comment le résultat se situe-t-il par rapport au meilleur score de
    l'étape 4 ? Quelle configuration retenez-vous ?) : ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 7 — Mesure finale sur le test et analyse des erreurs

    La configuration retenue est maintenant évaluée sur le jeu de **test**, resté à l'écart de vos essais :
    c'est la mesure finale. Ne cliquez qu'une fois votre configuration choisie sur la validation. On la met en face
    de la baseline classique, mesurée sur les mêmes images. Ce que vous observez ensuite (matrice de confusion,
    Grad-CAM) sert à comprendre le modèle, pas à le régler : toute modification décidée à partir du test
    demanderait un nouveau jeu de test.

    Sur ces 555 images, une accuracy entre 70 % et 90 % est connue à ± 2,5 à 4 points près (± 1,96 · √(p(1 − p) / n),
    comme à l'étape 6). Les classes du test sont équilibrées (15 images par race) : l'accuracy suffit ici. Le F1 macro
    servira dès que les classes seront déséquilibrées, dans les briques 3 et 4 du projet.

    Un score global ne dit pas *où* le modèle se trompe. La matrice de confusion croise classe réelle (lignes)
    et classe prédite (colonnes) : tout ce qui sort de la diagonale est une erreur.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_test = mo.ui.run_button(label="Évaluer la configuration retenue sur le test")
    bouton_test
    return (bouton_test,)


@app.cell
def _(bouton_test, classes, img_te, mo, modele_ft, np, tp, y_te):
    mo.stop(not tp.lance(bouton_test),
            mo.md("Une fois votre configuration choisie sur la validation, cliquez pour la mesure finale sur le test."))
    ds_test_ft = tp.DatasetImages(img_te, y_te, tp.transformation(160), pre_redim=184)
    pred_ft, probas_ft = tp.predire(modele_ft, ds_test_ft)
    matrice_ft, _fig = tp.matrice_confusion(y_te, pred_ft, classes)
    mo.vstack([mo.md(f"Accuracy du modèle retenu sur le jeu de test : **{float(np.mean(pred_ft == np.asarray(y_te))):.1%}** "
                     f"({len(y_te)} images)."), _fig])
    return ds_test_ft, matrice_ft, pred_ft


@app.cell(hide_code=True)
def _(descripteurs_classiques, historique_ft, img_te, mo, np, pred_ft, svm_classique, tp, y_te):
    _ds = tp.DatasetImages(img_te, y_te, transform=lambda x: x)
    _f_te = np.stack([descripteurs_classiques(_ds.image_brute(i)) for i in range(len(_ds))])
    _acc_classique = float(np.mean(svm_classique.predict(_f_te) == np.asarray(y_te)))
    _cfg = historique_ft["config"]
    _etages = _cfg.get("étages dégelés", _cfg.get("blocs dégelés"))
    _final = [
        {"méthode": "HOG + couleur + SVM (baseline classique)", "accuracy test": round(_acc_classique, 3)},
        {"méthode": f"EfficientNet-B0, {_etages} étage(s) dégelé(s) (configuration retenue)",
         "accuracy test": round(float(np.mean(pred_ft == np.asarray(y_te))), 3)},
    ]
    mo.vstack([mo.md(f"**Mesure finale sur le jeu de test** ({len(y_te)} images), à reporter dans la fiche de décision :"),
               tp.tableau(_final, mo)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 6 : les paires les plus confondues

    À partir de la matrice, renvoyez la liste des `k` paires `(classe réelle, classe prédite, nombre)` les plus
    fréquentes, triées par nombre décroissant, **hors diagonale** et sans les paires à zéro.
    """)
    return


@app.function
def paires_les_plus_confondues(matrice, noms, k=5):
    """matrice[i, j] = nombre d'images de classe i prédites j. Renvoie [(nom_i, nom_j, nombre), ...]."""
    # TODO 6
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo6, _affichage = tp.verifier(tests.tests_confusions(paires_les_plus_confondues), mo)
    _affichage
    return (ok_todo6,)


@app.cell(hide_code=True)
def _(classes, ds_test_ft, matrice_ft, mo, np, ok_todo6, pred_ft, tp, y_te):
    tp.bloquer_si(not ok_todo6, mo, "Terminez le TODO 6 pour voir les confusions.")
    _paires = paires_les_plus_confondues(matrice_ft, classes, k=5)
    _vrai, _predit, _ = _paires[0]
    _y, _p = np.asarray(y_te), np.asarray(pred_ft)
    _idx = np.flatnonzero((_y == classes.index(_vrai)) & (_p == classes.index(_predit)))[:4]
    _ref = np.flatnonzero(_y == classes.index(_predit))[:4]
    _imgs = [ds_test_ft.image_brute(i) for i in list(_idx) + list(_ref)]
    _titres = [f"réel {_vrai}\nprédit {_predit}" for _ in _idx] + [f"vrai {_predit}" for _ in _ref]
    mo.vstack([
        mo.md("| Classe réelle | Prédite | Nombre |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {n} |" for a, b, n in _paires)),
        mo.md(f"**Paire la plus confondue** : à gauche des « {_vrai} » prédits « {_predit} », à droite de vrais « {_predit} »."),
        (tp.grille(_imgs, _titres, n_col=4, taille=2.2)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** ces confusions sont-elles compréhensibles pour un humain ? Quelles pistes (données,
    résolution, modèle) proposeriez-vous pour les réduire ? Elles seraient à évaluer sur la validation, puis sur
    un nouveau jeu de test.

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 8 — Diagnostic avec Grad-CAM

    Grad-CAM répond à une question de débogage : *sur quelle zone de l'image le modèle s'est-il appuyé ?*
    On l'utilise ici comme un outil d'ingénieur, pour repérer un modèle qui s'appuierait sur un artefact
    (le fond, un objet présent par hasard). L'explicabilité en tant que discipline est traitée dans le module
    *Enjeux systémiques de l'IA*.

    Le principe, sur une couche convolutive dont la sortie a $K$ canaux de taille $h \times w$ :

    1. on fait une passe avant et on garde les **activations** $A_k$ de la couche ;
    2. on rétropropage le score $y^c$ de la classe visée $c$ (la sortie du réseau avant softmax) et on garde
       les **gradients** $\partial y^c / \partial A_k$ ;
    3. le poids de chaque canal est la **moyenne** de son gradient sur toutes les positions $(i, j)$ :
       $\alpha_k^c = \frac{1}{Z} \sum_{i,j} \partial y^c / \partial A_k^{ij}$, où $Z = h \times w$ ;
    4. la carte est $\text{ReLU}\left(\sum_k \alpha_k^c A_k\right)$, puis ramenée entre 0 et 1 :
       $(\text{carte} - \min) / (\max - \min + 10^{-8})$, le $10^{-8}$ évitant une division par zéro si la carte est nulle.

    À 160 pixels, la dernière couche d'EfficientNet-B0 a une sortie de 5 x 5 : la carte est grossière, puis
    agrandie à la taille de l'image.

    ### À compléter — TODO 7 : Grad-CAM

    Pour récupérer activations et gradients, utilisez un **crochet** (*hook*) sur la couche :
    `couche.register_forward_hook(f)` appelle `f(module, entree, sortie)` à chaque passe avant. Dans `f`,
    gardez `sortie` et appelez `sortie.register_hook(g)` : `g(gradient)` sera appelée pendant la rétropropagation.
    Pensez à retirer le crochet (`.remove()`) à la fin. N'utilisez ni `torch.no_grad()` ni `.detach()` avant la
    rétropropagation : Grad-CAM a besoin des gradients. La couche utilisée dans la suite est la dernière couche
    de convolution d'EfficientNet-B0, `modele.features[-1]`.
    """)
    return


@app.cell
def _(torch):
    def grad_cam(modele, couche, image, classe=None):
        """image : tenseur (1, 3, H, W). Renvoie une carte numpy (h, w) dans [0, 1] pour `classe`
        (par défaut, la classe prédite)."""
        # TODO 7
        raise NotImplementedError("À compléter")
    return (grad_cam,)


@app.cell(hide_code=True)
def _(grad_cam, mo, tests, tp):
    ok_todo7, _affichage = tp.verifier(tests.tests_gradcam(grad_cam), mo)
    _affichage
    return (ok_todo7,)


@app.cell
def _(
    classes,
    ds_test_ft,
    grad_cam,
    mo,
    modele_ft,
    np,
    ok_todo7,
    pred_ft,
    tp,
    y_te,
):
    tp.bloquer_si(not ok_todo7, mo, "Terminez le TODO 7 pour appliquer Grad-CAM.")
    for _p in modele_ft.parameters():
        _p.requires_grad_(True)  # nécessaire si une partie du réseau était gelée
    _y, _pr = np.asarray(y_te), np.asarray(pred_ft)
    _rng = np.random.default_rng(0)  # des exemples tirés au hasard, pas seulement les premières races
    _justes = _rng.permutation(np.flatnonzero(_y == _pr))[:4]
    _fausses = _rng.permutation(np.flatnonzero(_y != _pr))[:4]
    _dev = tp.appareil()
    _imgs, _titres = [], []
    for _i in list(_justes) + list(_fausses):
        _x, _ = ds_test_ft[_i]
        _carte = grad_cam(modele_ft, modele_ft.features[-1], _x[None].to(_dev))
        _imgs.append(tp.superposer(tp.denormaliser(_x), _carte))
        _etat = "juste" if _y[_i] == _pr[_i] else "erreur"
        _titres.append(f"{_etat}\nréel {classes[_y[_i]]}\nprédit {classes[_pr[_i]]}")
    mo.vstack([mo.md("Ligne du haut : prédictions justes. Ligne du bas : erreurs."),
               (tp.grille(_imgs, _titres, n_col=4, taille=2.4))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Pour une erreur, la carte explique la classe prédite, donc la mauvaise ; pour comparer, appelez aussi
    `grad_cam(..., classe=vraie_classe)`. Grad-CAM aide à repérer un indice parasite ; il ne prouve pas que le
    modèle est fiable.

    **Votre réponse :** le modèle regarde-t-il l'animal ou autre chose (fond, coussin, main) ?
    Les erreurs s'expliquent-elles par la zone regardée ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 9 — Mini-application

    Déposez une photo de chat ou de chien (la vôtre, ou trouvée en ligne) : le modèle fine-tuné renvoie
    ses trois races les plus probables et la carte Grad-CAM correspondante. Essayez aussi une image qui
    n'est ni un chat ni un chien : que se passe-t-il ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    televersement = mo.ui.file(filetypes=[".jpg", ".jpeg", ".png", ".webp"], label="Déposer une image", kind="area")
    televersement
    return (televersement,)


@app.cell(hide_code=True)
def _(
    Image,
    classes,
    grad_cam,
    mo,
    modele_ft,
    ok_todo7,
    plt,
    televersement,
    tp,
):
    import io as _io

    mo.stop(not televersement.value, mo.md("En attente d'une image."))
    tp.bloquer_si(not ok_todo7, mo, "Terminez le TODO 7 d'abord.")
    for _p in modele_ft.parameters():
        _p.requires_grad_(True)  # Grad-CAM a besoin des gradients, même si une partie du réseau était gelée
    _img = Image.open(_io.BytesIO(televersement.contents())).convert("RGB")
    _x = tp.transformation(160)(_img)[None].to(tp.appareil())
    _p = modele_ft(_x).softmax(-1)[0].detach().cpu()
    _top = _p.topk(3)
    _carte = grad_cam(modele_ft, modele_ft.features[-1], _x, classe=int(_top.indices[0]))
    _fig, _ax = plt.subplots(1, 2, figsize=(7, 3.4))
    _ax[0].imshow(tp.denormaliser(_x[0]))
    _ax[1].imshow(tp.superposer(tp.denormaliser(_x[0]), _carte))
    for _a in _ax:
        _a.axis("off")
    _fig.tight_layout()
    mo.hstack([_fig, mo.md("\n".join(f"{k + 1}. **{classes[i]}** : {v:.0%}" for k, (v, i) in
                                     enumerate(zip(_top.values.tolist(), _top.indices.tolist()))))])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Votre réponse :** sur une image qui n'est ni un chat ni un chien, que renvoie le modèle ? Ses probabilités,
    qui somment toujours à 100 % sur les 37 races, permettent-elles de le détecter ? En service, qu'est-ce qui
    différera des images de test (photos d'utilisateurs, appareils, cadrage) ?

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 10 — Fiche de décision ingénieur

    À remplir à la fin de chaque TD. Vous la réutiliserez directement dans le rapport du projet.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | Reconnaître la race d'un animal à partir d'une photo |
    | Baseline classique et son score (test) | ... |
    | Modèle retenu et pourquoi | ... |
    | Métrique principale et score obtenu (test, avec son intervalle à 95 %) | ... |
    | Coût (temps d'entraînement, temps d'inférence, matériel) | ... |
    | Principaux cas d'échec | ... |
    | Le deep learning est-il justifié ici ? | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 11 — Brique projet Usine 4.0 : de quel type de défaut s'agit-il ?

    On transpose la démarche aux défauts de surface d'acier (jeu NEU-DET, 6 types de défauts, images en niveaux de gris).
    Contrainte réaliste : **30 images annotées par type** seulement. Placez les images dans `donnees/NEU/`
    (voir le README à la racine du dépôt, section « Données ») avant de lancer.

    Le code ci-dessous pose une première comparaison, sur une **validation** de 20 images par type. Un jeu de test
    de 100 images par type est tiré à part et réservé à la mesure finale du projet : n'y touchez pas pendant vos
    essais. Il est tiré en premier, et ne change pas quand vous faites varier le nombre d'images d'entraînement. La suite est à développer dans votre dossier `projet/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_neu = mo.ui.run_button(label="Lancer la première comparaison sur NEU")
    bouton_neu
    return (bouton_neu,)


@app.cell
def _(
    LogisticRegression,
    SVC,
    StandardScaler,
    bouton_neu,
    descripteurs_classiques,
    make_pipeline,
    mo,
    np,
    ok_todo1,
    ok_todo2,
    retirer_tete,
    tp,
):
    tp.bloquer_si(not (ok_todo1 and ok_todo2), mo, "Terminez les TODO 1 et 2 d'abord.")
    mo.stop(not tp.lance(bouton_neu), mo.md("Cliquez sur le bouton pour lancer."))

    try:
        _a, _b, _c, _d, _images_test, _y_test, _noms = tp.charger_neu(n_train_par_classe=30, n_val_par_classe=20, n_test_par_classe=100)
    except FileNotFoundError as _e:
        mo.stop(True, mo.callout(mo.md(str(_e)), kind="danger"))
    _ds_tr = tp.DatasetImages(_a, _b, tp.transformation(224))
    _ds_va = tp.DatasetImages(_c, _d, tp.transformation(224))
    _m = retirer_tete(tp.charger_modele("efficientnet_b0"))
    _e_tr = tp.extraire_embeddings(_m, _ds_tr)
    _e_va = tp.extraire_embeddings(_m, _ds_va)
    _acc_deep = np.mean(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(_e_tr, _b).predict(_e_va) == np.asarray(_d))
    _f_tr = np.stack([descripteurs_classiques(_ds_tr.image_brute(i)) for i in range(len(_ds_tr))])
    _f_va = np.stack([descripteurs_classiques(_ds_va.image_brute(i)) for i in range(len(_ds_va))])
    # C = 10 fixé ici pour aller vite ; dans le projet, choisissez-le sur la validation comme à l'étape 5
    _acc_cl = np.mean(make_pipeline(StandardScaler(), SVC(C=10)).fit(_f_tr, _b).predict(_f_va) == np.asarray(_d))
    mo.md(f"""
    | Méthode (30 images par type) | Accuracy validation |
    |---|---|
    | HOG + couleur + SVM | {_acc_cl:.1%} |
    | EfficientNet-B0 (feature extraction) + régression logistique | {_acc_deep:.1%} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **À poursuivre dans `projet/brique1_classification.py`**, copie de `projet/brique_modele.py` (à rendre avec le projet)

    1. Remplacez les histogrammes de couleur, redondants sur des images en niveaux de gris (les trois canaux sont
       identiques : il ne reste qu'un histogramme d'intensité), par un descripteur de
       texture adapté (par exemple LBP, `skimage.feature.local_binary_pattern`). La baseline progresse-t-elle ?
    2. NEU est dans la case « peu de données, domaine éloigné » de la grille du cours. Fine-tunez EfficientNet-B0
       en ne dégelant que peu d'étages (0, 1, puis 2), et essayez aussi la piste de CS231n : un classifieur linéaire
       sur des activations plus basses du réseau (par exemple la sortie moyennée de `features[:6]`). Comme dans
       le TD : comparez et réglez sur la validation, face à la baseline, et gardez le test pour la mesure finale.
    3. Faites varier le nombre d'images par type (10, 30, 100) et tracez l'accuracy **de validation** des deux approches :
       comment l'écart entre les deux approches évolue-t-il avec le nombre d'images ?
    4. Appliquez Grad-CAM aux erreurs : le modèle regarde-t-il bien le défaut ?
    5. Remplissez la fiche de décision pour cette brique.

    ### Pour aller plus loin (si vous avez terminé)

    - Remplacez EfficientNet-B0 par DeiT-tiny dans le fine-tuning : le Transformer fait-il mieux avec si peu de données ?
    - Ajoutez de l'augmentation de données plus forte (rotations, variations de contraste) et mesurez son effet.
    """)
    return


if __name__ == "__main__":
    app.run()
