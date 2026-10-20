import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="TD 2 - Détection et segmentation",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _():
    import sys
    import time
    from pathlib import Path

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(mo.notebook_dir()) / "outils"))
    import tests_tp2 as tests
    import tp_utils as tp

    plt.rcParams["figure.dpi"] = 90
    return Image, cv2, mo, np, plt, tests, time, torch, tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TD 2 — Détection et segmentation

    **Module Applications et Modèles en IA · ING3 IA · durée : 3h30 (l'étape 11 peut se terminer en autonomie)**

    Classer une image ne suffit pas toujours : sur un chantier, on veut savoir **où** se trouve chaque personne
    et si elle porte son casque. C'est un problème de détection. Cet après-midi, vous allez coder vous-même les
    briques de base de la détection (IoU, NMS, évaluation), les comparer à une méthode classique de traitement
    d'image, puis analyser un détecteur YOLO fine-tuné sur des images de chantier.

    **Rappel du fonctionnement du notebook**

    - Exécuter une cellule : bouton de lecture à sa droite, ou Ctrl + Entrée (Cmd + Entrée sur Mac).
      Les cellules qui dépendent d'une cellule modifiée se relancent automatiquement.
    - Cellules **À compléter** : écrivez votre code à la place du `raise NotImplementedError`, puis lisez la
      cellule de tests juste en dessous. Une étape se débloque quand les tests de l'étape précédente sont réussis.
    - Les calculs longs ne partent qu'au clic sur leur bouton.
    - Le code de certaines cellules fournies est replié pour alléger la lecture : vous pouvez l'afficher
      depuis le menu de la cellule.
    - Cellules **Votre réponse** : double-cliquez sur le texte pour l'éditer.
    - Pour tester du code librement, ajoutez une cellule avec le bouton + qui apparaît entre deux cellules.
    - **Particularité de marimo** : une variable ne peut être définie que dans **une seule cellule**. Dans vos
      cellules de test, donnez à vos variables un nom qui commence par un tiret bas (`_x`, `_modele`) : elles
      restent alors locales à la cellule. Sinon, marimo affiche une erreur du type
      *« This cell redefines variables from other cells »* : il suffit de renommer la variable.

    | Étape | Contenu | Durée indicative |
    |---|---|---|
    | 0 | Vérification de l'installation | 5 min |
    | 1 | Le cours en bref, avec démonstrations | 15 min |
    | 2 | Coder l'IoU et la NMS | 30 min |
    | 3 | Le jeu de données Construction-PPE | 10 min |
    | 4 | Un détecteur pré-entraîné face à un nouveau problème | 15 min |
    | 5 | Baseline classique : détecter les casques par la couleur | 30 min |
    | 6 | Fine-tuning de YOLO | 30 min |
    | 7 | Choisir un seuil de confiance | 30 min |
    | 8 | Segmentation : DeepLab et SAM | 20 min |
    | 9 | Mini-application | 10 min |
    | 10 | Fiche de décision ingénieur | 10 min |
    | 11 | Brique 2 du projet Usine 4.0 : localiser les défauts d'acier | 30 min |
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch, tp):
    _lignes = [
        f"- Matériel utilisé : **{tp.appareil()}**",
        f"- Threads CPU : {torch.get_num_threads()}",
        f"- Dossier des modèles : `{tp.MODELES}`",
    ]
    if tp.MODE_TEST:
        _lignes.append("- **Mode test actif** : DeepLab est chargé avec des poids aléatoires.")
    mo.vstack([mo.md("## Étape 0 — Vérification de l'installation"), mo.callout(mo.md("\n".join(_lignes)), kind="info")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 1 — Le cours en bref

    ### 1.1 IoU, NMS et mesures de la détection (vues ce matin, parties 1 et 3 du cours ; précision et rappel viennent du TD 1)

    Une boîte est décrite par ses coins $[x_1, y_1, x_2, y_2]$, en pixels, l'origine en haut à gauche et
    l'axe $y$ vers le bas. Pour savoir si une boîte prédite correspond à
    une boîte réelle, on mesure leur recouvrement par l'**IoU** (*Intersection over Union*) :

    $$\text{IoU}(A, B) = \frac{\text{aire}(A \cap B)}{\text{aire}(A \cup B)} \in [0, 1]$$

    Critère classique : une détection est correcte si sa classe est la bonne et si son IoU avec une boîte
    réelle vaut au moins 0,5 (convention de COCO et d'Ultralytics, suivie dans ce TD ; Pascal VOC exige
    strictement plus de 0,5).

    Chaque boîte prédite est accompagnée d'un **score de confiance** entre 0 et 1 : l'estimation, par le
    modèle, que la boîte contient bien un objet de cette classe (chez YOLOv8, le plus élevé des scores de
    classe, chacun issu d'une sigmoïde ; ce n'est pas une probabilité calibrée). Un **seuil de confiance** écarte les boîtes de
    score trop faible.

    Un détecteur produit des milliers de boîtes candidates (YOLOv8 en propose 3 549 pour une image redimensionnée en 416 × 416),
    souvent plusieurs sur le même objet. La **NMS** (*Non-Maximum Suppression*) fait le ménage : on garde la
    boîte de meilleur score, on supprime toutes celles qui la recouvrent trop (IoU strictement au-dessus d'un
    seuil), et on recommence avec la meilleure boîte restante. Ultralytics, la bibliothèque qui fournit YOLO
    dans ce TD, l'applique classe par classe : deux boîtes de classes différentes ne se suppriment pas.

    **Les mesures de la détection.** Il n'y a pas de vrais négatifs (on ne compte pas les zones vides de
    l'image) : on mesure la **précision** (part des boîtes prédites qui sont correctes) et le **rappel** (part
    des objets réels trouvés), et on les résume par le $F_1 = 2PR/(P + R)$, élevé seulement si les deux le sont.
    Pour une classe, en abaissant le seuil de confiance on gagne en rappel et, en général, on perd en précision. En traçant
    la précision (en ordonnée) en fonction du rappel (en abscisse) pour tous les seuils, on obtient la courbe
    précision-rappel ; l'**AP** (*average precision*) est l'aire sous cette courbe rendue décroissante (pour chaque rappel, la
    meilleure précision atteignable à ce rappel ou au-delà), entre 0 et 1. Le **mAP** est la moyenne
    des AP sur les classes. Le **mAP@0.5** compte une détection correcte si son IoU vaut au moins 0,5 ; le
    **mAP@0.5:0.95**, l'indicateur de COCO, fait la moyenne des mAP obtenus avec dix seuils d'IoU, de 0,5 à 0,95 par pas de 0,05 :
    il récompense aussi l'exactitude du placement des boîtes.

    Déplacez la boîte orange pour sentir ce que vaut une IoU de 0,5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    demo_x = mo.ui.slider(0, 160, value=60, label="Position x", show_value=True)
    demo_y = mo.ui.slider(0, 160, value=40, label="Position y", show_value=True)
    demo_taille = mo.ui.slider(20, 140, value=100, label="Taille", show_value=True)
    mo.hstack([demo_x, demo_y, demo_taille])
    return demo_taille, demo_x, demo_y


@app.cell(hide_code=True)
def _(demo_taille, demo_x, demo_y, mo, plt, torch):
    import matplotlib.patches as _patches
    import torchvision as _tv

    _a = [40, 40, 140, 140]
    _b = [demo_x.value, demo_y.value, demo_x.value + demo_taille.value, demo_y.value + demo_taille.value]
    _v = float(_tv.ops.box_iou(torch.tensor([_a], dtype=torch.float32), torch.tensor([_b], dtype=torch.float32))[0, 0])
    _fig, _ax = plt.subplots(figsize=(3.6, 3.6))
    _ax.add_patch(_patches.Rectangle(_a[:2], 100, 100, fill=False, lw=2, color="tab:blue", label="vérité"))
    _ax.add_patch(_patches.Rectangle(_b[:2], demo_taille.value, demo_taille.value, fill=False, lw=2, color="tab:orange", label="prédiction"))
    _ax.set_xlim(0, 310)
    _ax.set_ylim(310, 0)
    _ax.set_aspect("equal")
    _ax.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    _verdict = "correcte (IoU ≥ 0,5)" if _v >= 0.5 else "incorrecte (IoU < 0,5)"
    mo.hstack([_fig, mo.md(f"**IoU = {_v:.2f}**\n\nAvec le seuil usuel de 0,5, cette détection serait {_verdict}.")],
              justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Deux familles de détecteurs

    | | One-stage (YOLO) | Two-stage (Faster R-CNN) |
    |---|---|---|
    | Principe | une seule passe prédit boîtes et classes en chaque point de trois grilles (pas 8, 16, 32) | 1) proposer des régions candidates, 2) classer et affiner chacune |
    | Vitesse | temps réel sur GPU ; proche du temps réel sur CPU pour les petites versions à résolution réduite | plus lent |
    | Qualité (mAP) | aujourd'hui équivalente ou supérieure (les premiers YOLO peinaient sur les petits objets groupés) | longtemps la référence |
    | Usage typique | la plupart des applications : vidéo, embarqué, contrôle sur ligne | quand la latence importe peu ; base de Mask R-CNN (segmentation d'instances) |

    Pour situer, sur COCO (mAP@0.5:0.95, en pourcentage ; les tableaux du TD donnent le même indicateur entre
    0 et 1) : YOLOv8n 37,3 et YOLOv8m 50,2 ; Faster R-CNN ResNet-50 FPN de torchvision 37,0 (46,7 pour sa
    version 2, avec 280 GFLOPs contre 79 pour YOLOv8m).

    **YOLO avancé : ce qu'un ingénieur règle vraiment**

    - **La taille du modèle** : n (nano), s, m, l, x. Chaque cran multiplie le calcul par 1,6 à 3,3 (de 8,7 GFLOPs
      pour n à 258 pour x, à 640 px) ; le temps mesuré augmente un peu moins vite.
      On choisit la plus petite version qui atteint le mAP visé dans le budget de latence.
    - **Le seuil de confiance** : en dessous, une boîte est ignorée. Il arbitre entre fausses alertes et oublis.
    - **Le seuil de la NMS** : trop bas, il supprime des objets voisins (deux casques côte à côte) ;
      trop haut, il laisse des doublons.
    - **La résolution d'entrée** : plus elle est grande, mieux on voit les petits objets, mais plus c'est lent.
    - YOLOv2 à YOLOv5 partaient de boîtes types de tailles fixées à l'avance, les *boîtes d'ancrage*, et
      prédisaient un ajustement de chacune. YOLOv8, comme YOLOX dès 2021, n'en a plus : en chaque point des
      trois grilles, le modèle prédit la distance aux quatre bords de la boîte. Moins
      d'hyperparamètres à régler.

    ### 1.3 Segmentation

    | Type | Ce qu'on prédit | Modèles du jour |
    |---|---|---|
    | Sémantique | une classe par pixel (toutes les personnes forment une seule région) | U-Net, DeepLab |
    | D'instances | un masque par objet (chaque personne séparément) | Mask R-CNN ; YOLO + SAM (SAM seul donne des masques sans classe) |
    | Panoptique | les deux : une classe par pixel, et un numéro par objet | Mask2Former, Panoptic FPN (pour mémoire) |

    **U-Net** : un encodeur réduit l'image pour en extraire le contexte, un décodeur la remet à pleine
    résolution, et des connexions directes entre les deux réinjectent les détails fins. Référence en imagerie
    médicale. **DeepLab** : des convolutions dilatées (*atrous*) agrandissent le champ de vision sans réduire
    davantage la résolution, et le module ASPP (*Atrous Spatial Pyramid Pooling*) applique en parallèle une
    convolution 1 × 1, trois convolutions dilatées de taux 12, 24 et 36, et un pooling global, puis combine ces
    échelles. **SAM** (*Segment Anything*) : un modèle géant,
    entraîné sur plus d'un milliard de masques, qui segmente un objet à partir d'une simple indication
    (un point, une boîte) sans entraînement spécifique. Nous utilisons MobileSAM, sa version légère.

    ### 1.4 Et sans deep learning ?

    Avant les CNN, on détectait des objets avec des caractéristiques conçues à la main : seuillage de couleur,
    contours, ou descripteurs HOG (histogrammes de gradients orientés, vus au TD 1) calculés sur une fenêtre glissante
    (un cadre qu'on déplace sur toute l'image, à plusieurs tailles), chaque position étant soumise à un classifieur. Une règle de couleur
    suppose un objet très typé (un casque jaune vif sur fond neutre) ; jusqu'où tient-elle sur de vraies photos
    de chantier, avec leurs éclairages, leurs angles et leurs fonds ? Vous le mesurerez à l'étape 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 2 — Coder l'IoU et la NMS

    ### À compléter — TODO 1 : l'IoU

    Les boîtes sont des listes ou tableaux `[x1, y1, x2, y2]` avec $x_1 < x_2$ et $y_1 < y_2$ ; l'aire d'une
    boîte vaut $(x_2 - x_1)(y_2 - y_1)$, sans $+1$.
    Calculez les coins de l'intersection (en haut à gauche, le max des $x_1$ et des $y_1$ ; en bas à droite,
    le min des $x_2$ et des $y_2$), son aire (attention : elle vaut 0 si les boîtes ne se touchent pas,
    une largeur ou une hauteur négative doit être ramenée à 0), puis l'aire de l'union.
    N'appelez pas `torchvision.ops.box_iou` : le but est de la calculer vous-même (un test le vérifie).
    """)
    return


@app.function
def iou(boite_a, boite_b):
    """IoU entre deux boîtes [x1, y1, x2, y2]. Renvoie un float dans [0, 1]."""
    # TODO 1
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, tests, tp):
    ok_todo1, _affichage = tp.verifier(tests.tests_iou(iou), mo)
    _affichage
    return (ok_todo1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 2 : la NMS

    Entrées : `boites`, un tableau `(N, 4)`, `scores`, un tableau `(N,)`, et `seuil_iou`.
    Sortie : la liste des **indices** des boîtes conservées, de la meilleure à la moins bonne.

    1. Triez les indices par score décroissant. `np.argsort` trie par ordre **croissant** : appliquez-le à
       `-scores`, ou inversez son résultat.
    2. Tant qu'il reste des indices : gardez le premier **et retirez-le de la liste** (`restants.pop(0)`), puis
       retirez aussi tous ceux dont l'IoU avec lui est **strictement supérieure** au seuil. (Ne comptez pas sur
       l'IoU pour le retirer : avec un seuil de 1, une boîte n'a pas une IoU supérieure à 1 avec elle-même, et
       la boucle ne s'arrêterait jamais.)

    Utilisez votre fonction `iou` du TODO 1, et non `torchvision.ops.nms`, qui ne sert ici que de référence aux
    tests (un test vérifie que votre fonction ne l'appelle pas).
    """)
    return


@app.cell
def _(np):
    def nms(boites, scores, seuil_iou):
        """Non-Maximum Suppression. Renvoie la liste des indices conservés, par score décroissant."""
        # TODO 2
        raise NotImplementedError("À compléter")
    return (nms,)


@app.cell(hide_code=True)
def _(mo, nms, ok_todo1, tests, tp):
    tp.bloquer_si(not ok_todo1, mo, "Terminez le TODO 1 d'abord : la NMS s'appuie sur votre IoU.")
    ok_todo2, _affichage = tp.verifier(tests.tests_nms(nms), mo)
    _affichage
    return (ok_todo2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 3 — Le jeu de données Construction-PPE

    Des photos de chantier annotées avec les équipements de protection individuelle (EPI) : casque, gilet,
    gants, bottes, lunettes, et l'**absence** de certains d'entre eux (`no_helmet`, `no_goggle`, `no_gloves`,
    `no_boots` ; rien pour le gilet). C'est un vrai cas
    de sécurité industrielle : détecter automatiquement une personne sans casque sur une zone à risque.

    1132 images d'entraînement, 143 de validation, 141 de test, 11 classes, numérotées comme dans `ppe.yaml` :
    0 `helmet` (casque), 1 `gloves` (gants), 2 `vest` (gilet), 3 `boots` (bottes), 4 `goggles` (lunettes),
    5 `none` (classe que la documentation qualifie seulement de « générique »), 6 `Person` (la personne entière),
    et les quatre absences 7 `no_helmet`, 8 `no_goggle`, 9 `no_gloves`, 10 `no_boots`. Environ 180 Mo,
    téléchargés une fois.

    Chaque image a un fichier `.txt` du même nom, fourni avec le jeu (format YOLO : une ligne `classe cx cy w h`
    par boîte, centre et taille divisés par la taille de l'image). `tp.preparer_construction_ppe` écrit
    `ppe.yaml` (dossier racine, parties, noms des classes) ; `tp.lire_annotations` convertit les lignes en coins
    `[x1, y1, x2, y2]` en pixels.

    Chaque partie a son rôle : on **entraîne** sur la première, on **règle** sur la validation (paramètres, seuils),
    et on ne **mesure** que sur le test. Un réglage choisi sur les images qui servent à le juger paraît toujours
    meilleur qu'il ne l'est.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_ppe = mo.ui.run_button(label="Charger Construction-PPE")
    bouton_ppe
    return (bouton_ppe,)


@app.cell(hide_code=True)
def _(bouton_ppe, mo, tp):
    _deja_la = (tp.DONNEES / "construction-ppe" / "images" / "train").exists()
    mo.stop(not (tp.lance(bouton_ppe) or _deja_la), mo.md("Cliquez sur le bouton pour charger les données."))
    with mo.status.spinner(title="Téléchargement et préparation..."):
        config_ppe = tp.preparer_construction_ppe()
    dossier_ppe = config_ppe.parent
    images_train = tp.lister_images(dossier_ppe, "train")
    images_val = tp.lister_images(dossier_ppe, "val")
    images_test = tp.lister_images(dossier_ppe, "test")
    return config_ppe, images_test, images_train, images_val


@app.cell(hide_code=True)
def _(images_test, images_train, images_val, mo, np, plt, tp):
    _rng = np.random.default_rng(2)
    _choix = [images_train[i] for i in _rng.choice(len(images_train), 6, replace=False)]
    _imgs = []
    for _p in _choix:
        _b, _c = tp.lire_annotations(_p)
        _imgs.append(tp.dessiner_boites(tp.Image.open(_p), _b, _c, tp.PPE_CLASSES))
    _comptes = np.zeros(len(tp.PPE_CLASSES), int)
    for _p in images_train:
        _, _c = tp.lire_annotations(_p)
        np.add.at(_comptes, _c, 1)
    _tailles = [[] for _ in tp.PPE_CLASSES]   # aire de chaque boîte, en % de l'image
    for _p in images_train:
        _b, _c = tp.lire_annotations(_p)
        _w, _h = tp.Image.open(_p).size
        for _bb, _cc in zip(_b, _c):
            _tailles[_cc].append(100 * (_bb[2] - _bb[0]) * (_bb[3] - _bb[1]) / (_w * _h))
    _fig, (_ax, _ax2) = plt.subplots(1, 2, figsize=(10, 2.8))
    _ax.bar(tp.PPE_CLASSES, _comptes, color="tab:blue")
    _ax.set_title("Nombre de boîtes par classe (entraînement)", fontsize=9)
    _ax2.bar(tp.PPE_CLASSES, [float(np.median(_x)) if _x else 0 for _x in _tailles], color="tab:green")
    _ax2.set_title("Taille médiane d'une boîte (% de la surface de l'image)", fontsize=9)
    for _a in (_ax, _ax2):
        _a.tick_params(axis="x", rotation=45, labelsize=8)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"**{len(images_train)}** images d'entraînement, **{len(images_val)}** de validation, **{len(images_test)}** de test, "
              f"**{_comptes.sum()}** boîtes annotées en entraînement."),
        tp.grille(_imgs, n_col=3, taille=3.4), _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Question** : les classes sont-elles équilibrées ? Laquelle risque d'être la plus mal détectée, et pourquoi
    est-ce embêtant pour un système de sécurité ?
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
    ## Étape 4 — Un détecteur pré-entraîné face à un nouveau problème

    YOLOv8n a été pré-entraîné sur COCO, 80 classes d'objets du quotidien (personne, voiture, chien...).
    Que voit-il sur nos images de chantier ? Faites varier le seuil de confiance (0,25 au départ, la valeur par défaut d'Ultralytics), et regardez quelles classes
    COCO apparaissent : l'une d'elles pourrait-elle servir à repérer un casque ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    seuil_coco = mo.ui.slider(0.05, 0.9, step=0.05, value=0.25, label="Seuil de confiance", show_value=True)
    seuil_coco
    return (seuil_coco,)


@app.cell
def _(images_val, mo, seuil_coco, time, tp):
    yolo_coco = tp.charger_yolo("yolov8n.pt")
    _imgs, _titres, _vus = [], [], set()
    for _p in images_val[:3]:
        _t = time.perf_counter()
        _r = yolo_coco.predict(str(_p), conf=seuil_coco.value, imgsz=416, verbose=False)[0]
        _ms = (time.perf_counter() - _t) * 1000
        _noms = [yolo_coco.names[int(c)] for c in _r.boxes.cls]
        _vus.update(_noms)
        _imgs.append(tp.dessiner_boites(tp.Image.open(_p), _r.boxes.xyxy.numpy(), _r.boxes.cls.numpy().astype(int),
                                        list(yolo_coco.names.values()), _r.boxes.conf.numpy()))
        _titres.append(f"{len(_noms)} boîtes, {_ms:.0f} ms")
    mo.vstack([tp.grille(_imgs, _titres, n_col=3, taille=3.4),
               mo.md("Classes COCO détectées : " + ", ".join(sorted(_vus)) if _vus else "Aucune détection.")])
    return (yolo_coco,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quel que soit le seuil, il ne peut pas détecter un casque ou son absence : ces classes n'existent pas
    dans COCO. Il faudra le **fine-tuner** (étape 6) : comme au TD 1, on garde le réseau pré-entraîné, mais
    Ultralytics remplace les dernières couches de classification de la tête par des couches à 11 sorties (la
    ligne de poids de `person`, dans COCO, est recopiée pour `Person` ; les 10 autres partent de valeurs neuves),
    puis réentraîne tout le réseau sur nos images. Les couches reprises partent de bons poids, ce qui va beaucoup
    plus vite qu'un entraînement depuis zéro.

    ### La NMS en action, avec votre fonction

    On demande à YOLO ses boîtes « personne » de score supérieur à 0,05, avec sa propre NMS presque désactivée
    (seuil 0,99), puis on applique **votre** NMS. Faites varier son seuil : à partir de quand des doublons
    réapparaissent-ils, et quand une vraie personne disparaît-elle ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    seuil_nms = mo.ui.slider(0.1, 1.0, step=0.05, value=0.5, label="Seuil d'IoU de la NMS", show_value=True)
    seuil_nms
    return (seuil_nms,)


@app.cell
def _(images_val, mo, nms, ok_todo2, seuil_nms, tp, yolo_coco):
    tp.bloquer_si(not ok_todo2, mo, "Terminez le TODO 2 pour voir votre NMS à l'œuvre.")
    _p = images_val[1]
    _r = yolo_coco.predict(str(_p), conf=0.05, iou=0.99, imgsz=416, classes=[0], verbose=False)[0]
    _b, _s = _r.boxes.xyxy.numpy(), _r.boxes.conf.numpy()
    _k = nms(_b, _s, seuil_nms.value)
    _avant = tp.dessiner_boites(tp.Image.open(_p), _b, epaisseur=1)
    _apres = tp.dessiner_boites(tp.Image.open(_p), _b[_k], scores=_s[_k], epaisseur=3)
    tp.grille([_avant, _apres], [f"avant NMS : {len(_b)} boîtes", f"après votre NMS : {len(_k)} boîtes"], n_col=2, taille=4.2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 5 — Baseline classique : détecter les casques par la couleur

    Les casques de chantier sont souvent jaunes, orange ou blancs. Le jaune et l'orange sont des couleurs vives et
    saturées ; le blanc, peu saturé, échappe à cette règle : la baseline ne vise que les casques jaunes et orange.
    Une méthode
    sans apprentissage : convertir l'image en **HSV** (teinte, saturation, valeur, à peu près la luminosité), garder les pixels dont
    la teinte est dans la plage, et la saturation et la valeur (luminosité) assez élevées, puis entourer chaque région de pixels retenus par une boîte.

    ### À compléter — TODO 3 : des boîtes à partir d'un masque

    Entrée : un masque booléen `(H, W)` (pixels retenus). Sortie : une liste de boîtes `[x1, y1, x2, y2]`,
    une par **composante connexe** (groupe de pixels voisins, diagonales comprises : la 8-connexité, le défaut d'OpenCV), en ignorant celles de moins de `aire_min` pixels.

    Utilisez `cv2.connectedComponentsWithStats(masque.astype(np.uint8))`. Elle renvoie
    `(n, etiquettes, stats, centres)`. La ligne 0 de `stats` est le fond, à ignorer. Pour chaque autre ligne,
    `stats[i]` vaut `[x, y, largeur, hauteur, aire]`, où `aire` est le nombre de pixels de la région. La boîte
    est `[x, y, x + largeur, y + hauteur]` : le second coin est exclu, comme dans une tranche NumPy
    `masque[y1:y2, x1:x2]` et dans les boîtes de YOLO.
    """)
    return


@app.cell
def _(cv2, np):
    def boites_depuis_masque(masque, aire_min):
        """Une boîte [x1, y1, x2, y2] par composante connexe d'au moins aire_min pixels."""
        # TODO 3
        raise NotImplementedError("À compléter")
    return (boites_depuis_masque,)


@app.cell(hide_code=True)
def _(boites_depuis_masque, mo, tests, tp):
    ok_todo3, _affichage = tp.verifier(tests.tests_boites_masque(boites_depuis_masque), mo)
    _affichage
    return (ok_todo3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### À compléter — TODO 4 : évaluer des détections

    Pour comparer la baseline et YOLO, il faut une évaluation commune. Entrées : `predictions`, une liste de
    `(boite, score)`, et `verites`, une liste de boîtes réelles.

    1. Parcourez les prédictions par **score décroissant**.
    2. Pour chacune, cherchez la vérité **pas encore associée** avec laquelle son IoU est la plus grande.
       Si cette IoU est au moins égale à `seuil_iou`, c'est un vrai positif (`tp`) et la vérité devient associée.
       Sinon, c'est un faux positif (`fp`).
    3. Les vérités jamais associées sont des faux négatifs (`fn`, les oublis).

    Les deux listes portent sur une seule classe : le filtrage par classe est fait avant l'appel.
    Renvoyez un dictionnaire `{"tp", "fp", "fn", "precision", "rappel"}` avec
    précision $= tp / (tp + fp)$ et rappel $= tp / (tp + fn)$, en renvoyant 0 quand le dénominateur est nul.
    """)
    return


@app.function
def evaluer_detections(predictions, verites, seuil_iou=0.5):
    """predictions : liste de (boite, score) ; verites : liste de boîtes. Renvoie tp, fp, fn, precision, rappel."""
    # TODO 4
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, ok_todo1, tests, tp):
    tp.bloquer_si(not ok_todo1, mo, "Terminez le TODO 1 d'abord : l'évaluation s'appuie sur votre IoU.")
    ok_todo4, _affichage = tp.verifier(tests.tests_evaluation(evaluer_detections), mo)
    _affichage
    return (ok_todo4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Réglez la baseline

    Ajustez les seuils pour que le masque (au centre) isole au mieux les casques de l'image d'exemple, puis
    mesurez vos réglages sur toute la **validation** avec le bouton : c'est ce score qui sert à comparer des
    réglages, et à comparer la baseline à YOLO à l'étape 7. Le test n'est utilisé qu'à la fin de l'étape 7, une
    seule mesure par classe, avec des réglages figés. Si vous touchez un curseur après avoir cliqué,
    recliquez : sinon l'étape 7 n'a plus de score de baseline.
    Dans OpenCV, la teinte va de 0 à 179 : environ 10 à 25 pour l'orange, 25 à 35 pour le jaune.
    Le masque est d'abord nettoyé par une ouverture morphologique, qui efface les petits amas de pixels isolés.
    En plus de la couleur, la baseline applique deux règles de forme écrites à la main : une boîte de casque
    a un rapport largeur / hauteur entre 0,6 et 2,2, et occupe moins de 3 % de l'image. L'aire minimale est
    comptée sur l'image dont le plus grand côté est ramené à 640 px au plus (les images plus petites ne sont pas
    agrandies). Les valeurs par défaut viennent d'une recherche en grille (on essaie toutes les combinaisons
    d'une liste de valeurs) sur la validation (environ 3 300 réglages, meilleur F1). Elles débordent un peu sur le rouge et
    coupent le haut du jaune : c'est ce qui marchait le mieux sur ces images. À vous de voir si vous faites mieux.
    """)
    return


@app.cell(hide_code=True)
def _(images_val, mo):
    hsv_teinte = mo.ui.range_slider(0, 179, value=[0, 30], label="Teinte", show_value=True)
    hsv_sat = mo.ui.slider(0, 255, value=70, label="Saturation minimale", show_value=True)
    hsv_val = mo.ui.slider(0, 255, value=150, label="Valeur (luminosité) minimale", show_value=True)
    hsv_aire = mo.ui.slider(20, 3000, step=20, value=1800, label="Aire minimale (pixels)", show_value=True)
    hsv_image = mo.ui.slider(0, len(images_val) - 1, value=min(10, len(images_val) - 1), label="Image de validation affichée", show_value=True)
    mo.vstack([mo.hstack([hsv_teinte, hsv_sat]), mo.hstack([hsv_val, hsv_aire]), hsv_image])
    return hsv_aire, hsv_image, hsv_sat, hsv_teinte, hsv_val


@app.cell
def _(boites_depuis_masque, cv2, hsv_aire, hsv_sat, hsv_teinte, hsv_val, np):
    def detecter_casques_couleur(image_rgb):
        """Baseline classique : renvoie (masque, liste de (boite, score)) pour une image RGB."""
        _hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        _bas = np.array([hsv_teinte.value[0], hsv_sat.value, hsv_val.value], np.uint8)
        _haut = np.array([hsv_teinte.value[1], 255, 255], np.uint8)
        _masque = cv2.inRange(_hsv, _bas, _haut) > 0
        _masque = cv2.morphologyEx(_masque.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        _boites = boites_depuis_masque(_masque, hsv_aire.value)
        # Règles de forme : un casque est à peu près aussi large que haut, et petit dans l'image.
        _aire_image = image_rgb.shape[0] * image_rgb.shape[1]
        _boites = [b for b in _boites if 0.6 < (b[2] - b[0]) / max(1, b[3] - b[1]) < 2.2
                   and (b[2] - b[0]) * (b[3] - b[1]) < 0.03 * _aire_image]
        return _masque, [(b, 1.0) for b in _boites]

    return (detecter_casques_couleur,)


@app.cell(hide_code=True)
def _(detecter_casques_couleur, hsv_image, images_val, mo, ok_todo3, tp):
    tp.bloquer_si(not ok_todo3, mo, "Terminez le TODO 3 pour voir la baseline.")
    _p = images_val[hsv_image.value]
    _img = tp.lire_image_rgb(_p, cote_max=640)
    _masque, _preds = detecter_casques_couleur(_img)
    _vb, _vc = tp.lire_annotations(_p)
    _echelle = _img.shape[1] / tp.Image.open(_p).size[0]
    _verites = _vb[_vc == 0] * _echelle
    tp.grille([tp.dessiner_boites(_img, _verites), _masque, tp.dessiner_boites(_img, [b for b, _ in _preds], classes=[3] * len(_preds))],
              ["casques annotés", "masque couleur", f"baseline : {len(_preds)} boîtes"], n_col=3, taille=3.4, cmap="gray")
    return


@app.cell
def _(detecter_casques_couleur, tp):
    def evaluer_baseline(images, barre=None):
        """Précision, rappel, F1 et comptes de la baseline couleur sur la classe casque, cumulés sur des images."""
        _tp, _fp, _fn = 0, 0, 0
        for _p in (barre(images) if barre else images):
            _img = tp.lire_image_rgb(_p, cote_max=640)
            _, _preds = detecter_casques_couleur(_img)
            _vb, _vc = tp.lire_annotations(_p)
            _echelle = _img.shape[1] / tp.Image.open(_p).size[0]
            _r = evaluer_detections(_preds, list(_vb[_vc == 0] * _echelle))
            _tp, _fp, _fn = _tp + _r["tp"], _fp + _r["fp"], _fn + _r["fn"]
        _p_, _r_ = _tp / max(1, _tp + _fp), _tp / max(1, _tp + _fn)
        return {"precision": _p_, "rappel": _r_, "f1": 2 * _p_ * _r_ / (_p_ + _r_) if _p_ + _r_ else 0.0,
                "tp": _tp, "fp": _fp, "fn": _fn}

    return (evaluer_baseline,)


@app.cell(hide_code=True)
def _(images_val, mo):
    bouton_baseline = mo.ui.run_button(label=f"Évaluer la baseline sur les {len(images_val)} images de validation")
    bouton_baseline
    return (bouton_baseline,)


@app.cell(hide_code=True)
def _(
    bouton_baseline,
    evaluer_baseline,
    images_val,
    mo,
    ok_todo3,
    ok_todo4,
    tp,
):
    score_baseline = None
    if not (ok_todo3 and ok_todo4):
        _sortie = mo.callout(mo.md("Terminez les TODO 3 et 4 d'abord."), kind="warn")
    elif not tp.lance(bouton_baseline):
        _sortie = mo.md("Cliquez pour évaluer vos réglages sur la validation (environ 10 secondes).")
    else:
        score_baseline = evaluer_baseline(images_val, lambda _i: mo.status.progress_bar(_i, title="Baseline couleur"))
        _sortie = mo.md(f"**Baseline couleur, classe casque, sur la validation** : précision = **{score_baseline['precision']:.1%}**, "
                        f"rappel = **{score_baseline['rappel']:.1%}**, F1 = **{score_baseline['f1']:.2f}** "
                        f"(vrais positifs {score_baseline['tp']}, fausses alertes {score_baseline['fp']}, oublis {score_baseline['fn']}).")
    _sortie
    return (score_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Quels types d'erreurs fait la baseline (regardez l'image d'exemple) ? Sur quels objets se trompe-t-elle ?
    2. Pourrait-elle détecter l'**absence** de casque (`no_helmet`) ? Pourquoi ?
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
    ## Étape 6 — Fine-tuning de YOLO

    Sur CPU, un fine-tuning complet de YOLOv8n sur ce jeu prend environ 5 minutes **par époque** (416 px, les
    1132 images) ; le modèle fourni en a reçu 20, soit 1 h 41 (Ultralytics en prévoit 100 par défaut). On procède
    donc en deux temps :

    1. **Vous lancez un fine-tuning court** (une partie des images, 2 époques, résolution réduite) pour voir
       le processus et mesurer son coût.
    2. **Vous analysez ensuite un modèle fourni**, fine-tuné dans les mêmes conditions mais pendant 20 époques
       sur toutes les images, à la résolution 416.

    Pendant l'entraînement, une barre indique les époques terminées ; la sortie de la cellule affiche aussi la
    barre d'Ultralytics avec les trois pertes (`box_loss`, `cls_loss`, `dfl_loss`), écrites à chaque époque dans
    `sorties/yolo/court/results.csv`. Les courbes ne sont pas tracées, pour gagner du temps. Les deux modèles sont ensuite évalués sur la **validation**, chacun à sa
    résolution d'entraînement. Un AP de 1 veut dire que tous les objets sont trouvés, et que toutes les bonnes
    boîtes ont un score supérieur à celui de chaque fausse alerte ; un AP proche de 0, que le modèle ne trouve
    presque rien de cette classe. Si vous changez la résolution après votre fine-tuning, relancez-le : le modèle
    court est évalué à la résolution choisie dans la liste.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cfg_fraction = mo.ui.slider(0.1, 1.0, step=0.1, value=0.3, label="Part des images d'entraînement", show_value=True)
    cfg_epoques = mo.ui.slider(1, 5, value=2, label="Époques", show_value=True)
    cfg_resolution = mo.ui.dropdown(["320", "416"], value="320", label="Résolution")
    bouton_yolo = mo.ui.run_button(label="Lancer le fine-tuning court")
    mo.vstack([mo.hstack([cfg_fraction, cfg_epoques, cfg_resolution]), bouton_yolo])
    return bouton_yolo, cfg_epoques, cfg_fraction, cfg_resolution


@app.cell
def _(
    bouton_yolo,
    cfg_epoques,
    cfg_fraction,
    cfg_resolution,
    config_ppe,
    mo,
    tp,
):
    _dossier = tp.SORTIES / "yolo" / "court"
    _poids = _dossier / "weights" / "best.pt"
    if tp.lance(bouton_yolo):
        _modele = tp.charger_yolo("yolov8n.pt")
        with tp.Chrono() as _c, mo.status.progress_bar(total=cfg_epoques.value, title="Fine-tuning YOLO") as _barre:
            _modele.add_callback("on_fit_epoch_end", lambda _tr: _barre.update())
            _modele.train(data=str(config_ppe), epochs=cfg_epoques.value, fraction=cfg_fraction.value,
                          imgsz=int(cfg_resolution.value), batch=16, workers=0, plots=False, amp=False, seed=0,
                          project=str(_dossier.parent), name="court", exist_ok=True, verbose=False)
        (_dossier / "duree.txt").write_text(f"{_c.secondes:.0f}")
    duree_court = float((_dossier / "duree.txt").read_text()) if (_dossier / "duree.txt").exists() else float("nan")
    modele_court = tp.charger_yolo(str(_poids)) if _poids.exists() else None
    mo.md("Cliquez pour lancer (2 à 4 minutes avec les réglages par défaut). La suite du TD fonctionne aussi sans."
          if modele_court is None else f"Fine-tuning court terminé en {duree_court:.0f} s.")
    return duree_court, modele_court


@app.cell(hide_code=True)
def _(cfg_resolution, config_ppe, duree_court, mo, modele_court, tp):
    _chemin_fourni = tp.MODELES / "tp2_yolov8n_ppe_20ep.pt"
    mo.stop(not _chemin_fourni.exists(), mo.callout(mo.md(f"Modèle fourni introuvable : `{_chemin_fourni}`."), kind="danger"))
    modele_fourni = tp.charger_yolo(_chemin_fourni.name)
    _lignes = ["| Modèle (sur la validation) | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|"]
    resultats_val = {}
    for _nom, _m in [("fine-tuning court (le vôtre)", modele_court), ("fourni : 20 époques, 416 px", modele_fourni)]:
        if _m is None:
            continue
        # Validation : le test est réservé à la mesure finale (étape 7). Chaque modèle est évalué à sa résolution d'entraînement.
        _taille = int(cfg_resolution.value) if _m is modele_court else 416
        _r = _m.val(data=str(config_ppe), split="val", imgsz=_taille, batch=16, workers=0, plots=False, verbose=False)
        resultats_val[_nom] = _r
        _lignes.append(f"| {_nom} | {_r.box.map50:.3f} | {_r.box.map:.3f} |")
    mo.vstack([mo.md("\n".join(_lignes)),
               mo.md(f"Durée de votre fine-tuning : {duree_court:.0f} s." if duree_court == duree_court else "")])
    return modele_fourni, resultats_val


@app.cell(hide_code=True)
def _(mo, resultats_val, tp):
    _r = resultats_val["fourni : 20 époques, 416 px"]
    _lignes = ["| Classe | AP@0.5 | AP@0.5:0.95 |", "|---|---|---|"]
    for _k, _c in enumerate(_r.box.ap_class_index):
        _lignes.append(f"| {tp.PPE_CLASSES[int(_c)]} | {_r.box.ap50[_k]:.3f} | {_r.box.ap[_k]:.3f} |")
    mo.vstack([mo.md("**Modèle fourni : précision moyenne (AP) par classe, sur la validation**"), mo.md("\n".join(_lignes))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Quelles classes sont les mieux et les moins bien détectées ? Reliez-le au nombre d'exemples et à la
       taille des boîtes (les deux graphiques de l'étape 3).
    2. Le mAP@0.5:0.95 est toujours inférieur ou égal au mAP@0.5. Pourquoi ? Dans le tableau par classe, pour
       quelles classes l'écart entre AP@0.5 et AP@0.5:0.95 est-il le plus grand, et pourquoi ?
       Indice : une boîte de 20 × 20 px décalée de 4 px en $x$ et en $y$ n'a plus qu'une IoU de 0,47 avec la
       vraie ; pour une boîte de 100 × 100 px, le même décalage laisse 0,85.
    3. Combien de temps faudrait-il pour reproduire le modèle fourni sur votre machine ? Partez de votre
       mesure : le temps d'entraînement croît à peu près comme le nombre d'images, le nombre d'époques et le
       carré de la résolution. Qu'en concluez-vous sur le matériel nécessaire pour ce type de projet ?
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
    ## Étape 7 — Choisir un seuil de confiance

    Le mAP résume la qualité sur tous les seuils de confiance (l'AP parcourt toute la courbe précision-rappel),
    mais en production il faut en **choisir un**. Pour une
    personne sans casque, un oubli peut coûter un accident ; une fausse alerte coûte quelques secondes à un
    superviseur. Le bon seuil n'est donc pas forcément celui qui maximise le F1.

    Ce seuil est un réglage : on le choisit sur le jeu de **validation**, puis on regarde sur le **test** ce
    qu'il donne réellement. Les prédictions sont calculées une fois pour toutes avec un seuil très bas (0,01) :
    votre fonction applique ensuite chaque seuil sans relancer le modèle. Votre évaluation suit la règle de
    COCO (TODO 4) ; `val` d'Ultralytics apparie un peu différemment : les chiffres de l'étape 6 et vos courbes
    peuvent différer légèrement.

    ### À compléter — TODO 5 : précision et rappel en fonction du seuil

    Entrées : `predictions_par_image`, une liste (une entrée par image) de listes de `(boite, score)` ;
    `verites_par_image`, une liste de listes de boîtes ; `seuils`, une liste de seuils.
    Pour chaque seuil : ne gardez que les prédictions de score **supérieur ou égal** au seuil (`predict`
    d'Ultralytics, lui, exige un score strictement supérieur à `conf` : sans effet pratique), appelez
    `evaluer_detections` sur chaque image, **additionnez** tp, fp et fn sur toutes les images, puis calculez
    précision et rappel (0 quand le dénominateur est nul : pour la précision, cela arrive aux seuils élevés,
    quand aucune boîte n'est gardée), puis
    $F_1 = 2PR / (P + R)$ (0 si $P + R = 0$).

    Renvoyez `{"seuil": [...], "precision": [...], "rappel": [...], "f1": [...]}`.
    """)
    return


@app.function
def courbe_seuils(predictions_par_image, verites_par_image, seuils):
    """Précision, rappel et F1 cumulés sur toutes les images, pour chaque seuil de confiance."""
    # TODO 5
    raise NotImplementedError("À compléter")


@app.cell(hide_code=True)
def _(mo, ok_todo4, tests, tp):
    tp.bloquer_si(not ok_todo4, mo, "Terminez le TODO 4 d'abord.")
    ok_todo5, _affichage = tp.verifier(tests.tests_courbe(courbe_seuils), mo)
    _affichage
    return (ok_todo5,)


@app.cell
def _(images_test, images_val, modele_fourni, tp):
    # Validation : pour choisir le seuil. Test : pour mesurer ce que donne le seuil choisi.
    predictions_val = tp.predictions_yolo(modele_fourni, images_val, conf=0.01, taille=416)
    verites_val = [tp.lire_annotations(p) for p in images_val]
    predictions_test = tp.predictions_yolo(modele_fourni, images_test, conf=0.01, taille=416)
    verites_test = [tp.lire_annotations(p) for p in images_test]
    return predictions_test, predictions_val, verites_test, verites_val


@app.cell(hide_code=True)
def _(mo):
    classe_analyse = mo.ui.dropdown(["helmet", "no_helmet", "Person", "vest", "gloves"], value="no_helmet", label="Classe analysée")
    classe_analyse
    return (classe_analyse,)


@app.cell(hide_code=True)
def _(
    classe_analyse,
    mo,
    np,
    ok_todo5,
    plt,
    predictions_val,
    score_baseline,
    tp,
    verites_val,
):
    tp.bloquer_si(not ok_todo5, mo, "Terminez le TODO 5 pour tracer les courbes.")
    _c = tp.PPE_CLASSES.index(classe_analyse.value)
    _preds = [[(b, s) for b, s, k in zip(*p) if k == _c] for p in predictions_val]
    _ver = [list(vb[vc == _c]) for vb, vc in verites_val]
    _seuils = list(np.round(np.arange(0.05, 0.96, 0.05), 2))
    courbe = courbe_seuils(_preds, _ver, _seuils)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.plot(courbe["seuil"], courbe["precision"], label="précision")
    _ax.plot(courbe["seuil"], courbe["rappel"], label="rappel")
    _ax.plot(courbe["seuil"], courbe["f1"], label="F1", ls="--")
    if classe_analyse.value == "helmet" and score_baseline is not None:
        _ax.axhline(score_baseline["precision"], color="tab:blue", lw=0.8, ls=":", label="précision de la baseline couleur (validation)")
        _ax.axhline(score_baseline["rappel"], color="tab:orange", lw=0.8, ls=":", label="rappel de la baseline couleur (validation)")
    _ax.set_xlabel("seuil de confiance")
    _ax.set_ylim(0, 1)
    _ax.legend(fontsize=8)
    _ax.set_title(f"Classe {classe_analyse.value}, jeu de validation ({sum(len(v) for v in _ver)} objets)", fontsize=9)
    _fig.tight_layout()
    _i = int(np.argmax(courbe["f1"]))
    mo.vstack([_fig, mo.md(f"Sur la validation, F1 maximal : **{courbe['f1'][_i]:.2f}** au seuil **{courbe['seuil'][_i]}** "
                           f"(précision {courbe['precision'][_i]:.0%}, rappel {courbe['rappel'][_i]:.0%}).")])
    return (courbe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choisir le seuil, puis le mesurer une fois sur le test

    Sur la courbe, la baseline couleur (classe `helmet` seulement, puisqu'elle ne détecte que des casques)
    apparaît en lignes horizontales : elle donne le même score à toutes ses boîtes et n'a donc pas de seuil.

    Réglez le curseur sur le seuil que vous retenez d'après la courbe de validation, et justifiez-le par le coût
    métier. Une fois le choix fait, mesurez-le sur le test avec le bouton, **une fois par classe analysée** :
    `no_helmet`, le cas métier, puis `helmet`, pour comparer YOLO et la baseline sur les mêmes images. Pour
    `helmet`, choisissez d'abord son propre seuil sur sa courbe de validation (par exemple le meilleur F1). La
    baseline n'apparaît que si vous avez cliqué sur « Évaluer la baseline » à l'étape 5 après votre dernier
    réglage. Ce qu'on observe sur le test sert à juger le modèle, pas à revenir sur le seuil.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    seuil_choisi = mo.ui.slider(0.05, 0.95, step=0.05, value=0.5, label="Seuil retenu", show_value=True)
    seuil_choisi
    return (seuil_choisi,)


@app.cell(hide_code=True)
def _(courbe, mo, seuil_choisi):
    _k = min(range(len(courbe["seuil"])), key=lambda _i: abs(courbe["seuil"][_i] - seuil_choisi.value))
    mo.md(f"Au seuil {courbe['seuil'][_k]:.2f}, sur la **validation** : précision {courbe['precision'][_k]:.0%}, "
          f"rappel {courbe['rappel'][_k]:.0%}, F1 {courbe['f1'][_k]:.2f}.")
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_test = mo.ui.run_button(label="Mesurer le seuil retenu sur le test (une fois)")
    bouton_test
    return (bouton_test,)


@app.cell(hide_code=True)
def _(
    bouton_test,
    classe_analyse,
    evaluer_baseline,
    images_test,
    mo,
    ok_todo5,
    predictions_test,
    score_baseline,
    seuil_choisi,
    tp,
    verites_test,
):
    tp.bloquer_si(not ok_todo5, mo, "Terminez le TODO 5 d'abord.")
    mo.stop(not tp.lance(bouton_test), mo.md("Choisissez le seuil sur la validation, puis cliquez pour le mesurer sur le test."))
    _c = tp.PPE_CLASSES.index(classe_analyse.value)
    _oublis, _alertes, _tp, _fp, _fn = [], [], 0, 0, 0
    for _p, (_b, _s, _k), (_vb, _vc) in zip(images_test, predictions_test, verites_test):
        _preds = [(b, s) for b, s, k in zip(_b, _s, _k) if k == _c and s >= seuil_choisi.value]
        _r = evaluer_detections(_preds, list(_vb[_vc == _c]))
        _tp, _fp, _fn = _tp + _r["tp"], _fp + _r["fp"], _fn + _r["fn"]
        if (_r["fn"] and len(_oublis) < 3) or (_r["fp"] and len(_alertes) < 3):
            # Sur chaque image : en vert les objets annotés, en rouge les prédictions retenues, avec leur score.
            _img = tp.dessiner_boites(tp.Image.open(_p), _vb[_vc == _c], classes=[1] * int((_vc == _c).sum()))
            _img = tp.dessiner_boites(_img, [b for b, _ in _preds], classes=[0] * len(_preds), scores=[s for _, s in _preds],
                                      epaisseur=2)
            if _r["fn"] and len(_oublis) < 3:
                _oublis.append(_img)
            if _r["fp"] and len(_alertes) < 3:
                _alertes.append(_img)
    _comparaison = mo.md("")
    if classe_analyse.value == "helmet" and score_baseline is not None:
        _b = evaluer_baseline(images_test)
        _comparaison = mo.md(f"Baseline couleur, avec vos réglages, sur le même test : précision **{_b['precision']:.0%}**, "
                             f"rappel **{_b['rappel']:.0%}** ({_b['tp']} trouvés, {_b['fn']} oubliés, {_b['fp']} fausses alertes).")
    mo.vstack([
        mo.md(f"**Sur le jeu de test, classe {classe_analyse.value}, seuil {seuil_choisi.value:.2f}** : "
              f"précision **{_tp / max(1, _tp + _fp):.0%}**, rappel **{_tp / max(1, _tp + _fn):.0%}** "
              f"({_tp + _fn} objets dans le test : {_tp} trouvés, {_fn} oubliés, {_fp} fausses alertes)."),
        _comparaison,
        mo.md("**Images avec des oublis** (en vert, les objets annotés ; en rouge, les prédictions retenues) : "
              "un objet vert sans boîte rouge qui le recouvre est un oubli"),
        tp.grille(_oublis, n_col=3, taille=3.2) if _oublis else mo.md("Aucun oubli."),
        mo.md("**Images avec des fausses alertes** (mêmes couleurs) : une boîte rouge sans objet vert en dessous "
              "est une fausse alerte"),
        tp.grille(_alertes, n_col=3, taille=3.2) if _alertes else mo.md("Aucune fausse alerte."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions**

    1. Quel seuil retiendriez-vous pour `no_helmet` dans un système d'alerte de chantier ? Justifiez avec la
       courbe de validation, en termes de coût métier.
    2. À ce seuil, la précision et le rappel mesurés sur le test sont-ils ceux que vous lisiez sur la courbe ?
       Combien d'objets de cette classe chaque jeu contient-il, et quelle confiance accordez-vous à ces chiffres ?
       (Rappel du TD 1 : une proportion $p$ mesurée sur $n$ cas est connue, avec un intervalle de confiance à
       95 %, à $\pm 1{,}96\sqrt{p(1 - p)/n}$ près, approximation valable si $n$ n'est pas trop petit ;
       $n = tp + fn$, le nombre d'objets, pour le rappel, et $n = tp + fp$, le nombre de boîtes retenues, pour la
       précision.)
    3. Quelles erreurs voyez-vous sur les images de test (oublis, fausses alertes) ?
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
    ## Étape 8 — Segmentation : DeepLab et SAM

    Deux façons d'obtenir des masques de personnes sur les mêmes images :

    - **DeepLabV3 MobileNetV3** de torchvision (sémantique, pré-entraîné sur un sous-ensemble de COCO avec les
      20 classes de Pascal VOC plus le fond, soit 21 sorties ; la classe 15 est la personne) :
      une classe par pixel, toutes les personnes dans une seule région ;
    - **YOLO puis SAM** (instances) : les boîtes « Person » du modèle fine-tuné servent d'indication à MobileSAM,
      qui renvoie un masque précis par personne. C'est un pipeline courant pour pré-annoter des données.

    Lecture de la figure : à gauche l'image ; au centre, en clair, les pixels classés « personne » par DeepLab ;
    à droite, une couleur par personne trouvée par YOLO puis SAM. Le temps de droite cumule YOLO et SAM. La
    première image inclut le temps de démarrage des modèles : comparez plutôt les temps des deux suivantes.
    Les images viennent de la validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_seg = mo.ui.run_button(label="Lancer la segmentation sur 3 images")
    bouton_seg
    return (bouton_seg,)


@app.cell
def _(bouton_seg, images_val, mo, modele_fourni, np, time, torch, tp):
    mo.stop(not tp.lance(bouton_seg), mo.md("Cliquez pour lancer (si verifier_installation.py n'a pas été lancé : téléchargement de MobileSAM, 40 Mo, et de DeepLab, 42 Mo)."))
    _deeplab = tp.charger_deeplab()
    _sam = tp.charger_sam()
    from torchvision import transforms as _T

    _transfo = _T.Compose([_T.ToTensor(), _T.Normalize(tp.MOYENNE, tp.ECART)])
    _imgs, _titres = [], []
    for _p in images_val[10:13]:
        _rgb = tp.lire_image_rgb(_p, cote_max=640)
        _pil = tp.Image.fromarray(_rgb)
        _t = time.perf_counter()
        with torch.no_grad():
            _x = _transfo(_pil)[None]
            _sem = _deeplab(_x)["out"][0].argmax(0).numpy() == 15
        _t_dl = (time.perf_counter() - _t) * 1000
        # Ultralytics lit un tableau NumPy comme une image BGR (la convention d'OpenCV) : on lui passe donc
        # l'image PIL, qui est en RGB. Avec le tableau `_rgb`, le rouge et le bleu seraient inversés.
        _t = time.perf_counter()
        _r = modele_fourni.predict(_pil, conf=0.3, imgsz=416, classes=[6], verbose=False)[0]
        _boites = _r.boxes.xyxy.numpy()
        _masques = _sam(_pil, bboxes=_boites.tolist(), verbose=False)[0].masks.data.numpy() if len(_boites) else np.zeros((0, *_rgb.shape[:2]))
        _t_sam = (time.perf_counter() - _t) * 1000
        _annotees = int((tp.lire_annotations(_p)[1] == 6).sum())
        _inst = np.zeros(_rgb.shape[:2])
        for _k, _m in enumerate(_masques):
            _inst[_m > 0] = _k + 1
        _imgs += [_rgb, _sem, _inst]
        _titres += ["image", f"DeepLab ({_t_dl:.0f} ms)", f"YOLO + SAM : {len(_masques)} personnes, {_annotees} annotées ({_t_sam:.0f} ms)"]
    tp.grille(_imgs, _titres, n_col=3, taille=3.2, cmap="tab20")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions** : quelle approche permet de compter les personnes ? Laquelle est la plus rapide ?
    Dans quel cas utiliseriez-vous chacune ?
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
    ## Étape 9 — Mini-application

    Déposez une photo (chantier, atelier, rue...) : le modèle fine-tuné détecte les EPI. Mettez le seuil de
    confiance sur celui que vous avez retenu pour `no_helmet` à l'étape 7 (il s'applique ici à toutes les classes) ; la NMS est réglée par défaut à 0,7, la valeur
    d'Ultralytics utilisée aux étapes 6 et 7.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    app_fichier = mo.ui.file(filetypes=[".jpg", ".jpeg", ".png", ".webp"], label="Déposer une image", kind="area")
    app_conf = mo.ui.slider(0.05, 0.9, step=0.05, value=0.3, label="Seuil de confiance", show_value=True)
    app_iou = mo.ui.slider(0.1, 0.9, step=0.05, value=0.7, label="Seuil de la NMS", show_value=True)
    mo.vstack([app_fichier, mo.hstack([app_conf, app_iou])])
    return app_conf, app_fichier, app_iou


@app.cell(hide_code=True)
def _(Image, app_conf, app_fichier, app_iou, mo, modele_fourni, tp):
    import io as _io

    mo.stop(not app_fichier.value, mo.md("En attente d'une image."))
    # Image PIL (RGB), et non tableau NumPy : Ultralytics lirait un tableau comme une image BGR.
    _img = Image.open(_io.BytesIO(app_fichier.contents())).convert("RGB")
    _r = modele_fourni.predict(_img, conf=app_conf.value, iou=app_iou.value, imgsz=416, verbose=False)[0]
    _cls = _r.boxes.cls.numpy().astype(int)
    _resume = ", ".join(f"{tp.PPE_CLASSES[c]} x{int((_cls == c).sum())}" for c in sorted(set(_cls))) or "rien"
    _alerte = "no_helmet" in [tp.PPE_CLASSES[c] for c in _cls]
    mo.vstack([
        mo.callout(mo.md("**Alerte : personne sans casque détectée.**" if _alerte else "Pas d'alerte casque."),
                   kind="danger" if _alerte else "success"),
        tp.dessiner_boites(_img, _r.boxes.xyxy.numpy(), _cls, tp.PPE_CLASSES, _r.boxes.conf.numpy()),
        mo.md(f"Détections : {_resume}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 10 — Fiche de décision ingénieur
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    | Rubrique | Votre réponse |
    |---|---|
    | Problème métier | Détecter les personnes sans casque sur un chantier |
    | Baseline classique et son score (test, classe casque : elle ne détecte pas l'absence de casque) | ... |
    | Modèle retenu et pourquoi | ... |
    | Métrique principale et score obtenu (test, classe `no_helmet`, au seuil retenu) | ... |
    | Seuil retenu et justification métier | ... |
    | Coût (entraînement, inférence, matériel) ; pour l'inférence, le modèle fourni a la même architecture que le YOLOv8n de l'étape 4 | ... |
    | Principaux cas d'échec | ... |
    | Le deep learning est-il justifié ici ? | ... |
    | Déployable en l'état ? Sinon, que manque-t-il ? | ... |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Étape 11 — Brique 2 du projet Usine 4.0 : localiser les défauts d'acier

    Le jeu NEU-DET fournit, pour chaque image de défaut d'acier (200 × 200 px, en niveaux de gris), des boîtes
    qui localisent le défaut, parmi six types : craquelures (`crazing`), inclusions (`inclusion`), plaques
    (`patches`), piqûres (`pitted_surface`), calamine incrustée (`rolled-in_scale`) et rayures (`scratches`).
    Les annotations sont au format Pascal VOC (fichiers `.xml`). Placez le dossier dans `donnees/NEU/` (voir le
    README). La cellule ci-dessous convertit les annotations au format YOLO (une ligne `classe cx cy w h` par
    boîte, centre et taille divisés par la taille de l'image), lance un fine-tuning court et le compare
    à une baseline classique : seuiller les pixels qui s'écartent du fond local de la tôle (l'image très floutée).
    Les réglages des deux méthodes se choisissent sur la validation : le seuil de confiance pour YOLO, le flou,
    le seuil et l'aire minimale pour la baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bouton_neu = mo.ui.run_button(label="Préparer NEU-DET et lancer la comparaison")
    bouton_neu
    return (bouton_neu,)


@app.cell
def _(boites_depuis_masque, bouton_neu, cv2, mo, np, ok_todo3, ok_todo4, tp):
    tp.bloquer_si(not (ok_todo3 and ok_todo4), mo, "Terminez les TODO 3 et 4 d'abord.")
    mo.stop(not tp.lance(bouton_neu), mo.md("Cliquez pour lancer (plusieurs minutes sur CPU : trois époques de fine-tuning)."))
    _cible = tp.DONNEES / "neu-yolo"
    try:
        _config = _cible / "config.yaml" if (_cible / "config.yaml").exists() else tp.voc_vers_yolo(tp.DONNEES / "NEU", _cible, tp.NEU_CLASSES)
    except FileNotFoundError as _e:
        mo.stop(True, mo.callout(mo.md(str(_e) + " Voir le README, section Données."), kind="danger"))
    _modele = tp.charger_yolo("yolov8n.pt")
    _modele.train(data=str(_config), epochs=3, imgsz=320, batch=16, workers=0, plots=False, amp=False, seed=0,
                  project=str(tp.SORTIES / "yolo"), name="neu", exist_ok=True, verbose=False)
    _r = _modele.val(data=str(_config), split="test", imgsz=320, workers=0, plots=False, verbose=False)

    # Même mesure pour les deux méthodes : une boîte est correcte si son IoU avec un défaut annoté atteint 0,3,
    # quel que soit le type de défaut (la baseline ne sait pas le nommer).
    def _compter(images, predictions):
        """Additionne vrais positifs, fausses alertes et oublis sur une liste d'images."""
        total = [0, 0, 0]
        for chemin, boites_scores in zip(images, predictions):
            res = evaluer_detections(boites_scores, list(tp.lire_annotations(chemin)[0]), seuil_iou=0.3)
            for k, cle in enumerate(("tp", "fp", "fn")):
                total[k] += res[cle]
        return total

    def _garder(brutes, seuil):
        """Ne garde que les boîtes de YOLO dont le score atteint le seuil."""
        return [[(b, s) for b, s in zip(bx, sc) if s >= seuil] for bx, sc, _ in brutes]

    def _baseline(images, sigma=25, k=2.0, aire=150):
        """Écart au fond local (flou gaussien d'écart-type sigma, en pixels), seuil à moyenne + k écarts-types,
        fermeture (dilatation puis érosion : bouche les petits trous, relie les morceaux proches), composantes."""
        sorties = []
        for chemin in images:
            gris = cv2.cvtColor(tp.lire_image_rgb(chemin), cv2.COLOR_RGB2GRAY).astype(np.float32)
            ecart = np.abs(gris - cv2.GaussianBlur(gris, (0, 0), sigma))
            masque = cv2.morphologyEx((ecart > ecart.mean() + k * ecart.std()).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            sorties.append([(b, 1.0) for b in boites_depuis_masque(masque, aire)])
        return sorties

    def _f1(c):
        return 2 * c[0] / max(1, 2 * c[0] + c[1] + c[2])

    def _pr(c):
        return f"{c[0] / max(1, c[0] + c[1]):.1%} | {c[0] / max(1, c[0] + c[2]):.1%}"

    # Les réglages se choisissent sur la validation (meilleur F1), puis on mesure une fois sur le test.
    # La NMS de YOLO est ici agnostique (toutes les boîtes comparées entre elles, quelle que soit leur classe) :
    # le comptage ignore le type de défaut, deux boîtes de types différents
    # sur le même défaut compteraient sinon comme une fausse alerte.
    _images_val, _images_neu = tp.lister_images(_cible, "val"), tp.lister_images(_cible, "test")
    _brutes_val = tp.predictions_yolo(_modele, _images_val, conf=0.01, taille=320, agnostique=True)
    _brutes_test = tp.predictions_yolo(_modele, _images_neu, conf=0.01, taille=320, agnostique=True)
    _seuil = max([round(0.05 * k, 2) for k in range(1, 20)], key=lambda s: _f1(_compter(_images_val, _garder(_brutes_val, s))))
    _reglages = [(s, k, a) for s in (5, 10, 25) for k in (1.5, 2.0, 2.5, 3.0) for a in (50, 150, 400)]
    _reglage = max(_reglages, key=lambda r: _f1(_compter(_images_val, _baseline(_images_val, *r))))
    _c_baseline = _compter(_images_neu, _baseline(_images_neu, *_reglage))
    _c_yolo = _compter(_images_neu, _garder(_brutes_test, _seuil))
    mo.md(f"""
    | Méthode | Précision | Rappel | mAP@0.5 |
    |---|---|---|---|
    | Baseline classique (écart au fond local, flou {_reglage[0]}, seuil {_reglage[1]} écarts-types, aire {_reglage[2]} px) | {_pr(_c_baseline)} | sans objet |
    | YOLOv8n, 3 époques, 320 px, seuil de confiance {_seuil:.2f} | {_pr(_c_yolo)} | {_r.box.map50:.3f} |

    Précision et rappel sont calculés de la même façon pour les deux méthodes : sur les {len(_images_neu)} images de test,
    une boîte est jugée correcte si son IoU avec un défaut annoté atteint 0,3, sans tenir compte du type de défaut.
    Le critère est assoupli (0,3 au lieu de 0,5) parce que les défauts ont des contours flous, que deux annotateurs
    ne placeraient pas au pixel près ; il est fixé à l'avance, pour les deux méthodes.
    Le seuil de confiance de YOLO et les réglages de la baseline sont ceux qui donnent le meilleur F1 sur les
    {len(_images_val)} images de validation. Le mAP@0.5 classe les boîtes par score de confiance et tient compte du
    type de défaut : la baseline n'ayant ni l'un ni l'autre, son mAP serait dégénéré et non comparable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **À poursuivre dans `projet/brique2_localisation.py`**, copie de `projet/brique_modele.py` (à rendre avec le projet)

    1. Entraînez YOLO plus longtemps (laissez tourner pendant une pause ou le soir) et suivez le mAP par classe.
       Quels défauts sont les plus difficiles à localiser ? Pourquoi (forme, contraste, contour flou) ?
    2. Améliorez la baseline classique (taille du flou, seuil, morphologie : la fermeture) : jusqu'où peut-elle aller ?
    3. Choisissez un seuil de confiance et justifiez-le par le coût d'un défaut non détecté sur la ligne.
    4. Testez SAM avec les boîtes de YOLO : les masques épousent-ils bien des défauts diffus comme les piqûres ?
    5. Remplissez la fiche de décision pour cette brique.

    ### Pour aller plus loin (si vous avez terminé)

    - **Le coût de l'annotation.** Ouvrez [makesense.ai](https://www.makesense.ai) (gratuit, sans compte),
      annotez les casques de 5 images de test en chronométrant, exportez au format YOLO, puis comparez vos boîtes
      aux annotations officielles avec votre fonction `iou`. Le format YOLO donne, par ligne, `classe cx cy w h`,
      le centre et la taille divisés par la taille de l'image : convertissez en `[x1, y1, x2, y2]` en pixels
      avant d'appeler `iou` (`tp.lire_annotations` le fait pour les fichiers du jeu). Combien de temps faudrait-il pour annoter les
      1132 images ? Vos boîtes sont-elles d'accord avec celles du jeu ?
    - **Faster R-CNN.** Comparez la vitesse de `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn`
      (pré-entraîné COCO) à celle de YOLOv8n sur la classe personne (classe 1 dans torchvision, où 0 est le fond ;
      classe 0 chez YOLO).
    """)
    return


if __name__ == "__main__":
    app.run()
