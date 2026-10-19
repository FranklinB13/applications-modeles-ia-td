"""Outils communs aux TD du module « Applications et Modèles en IA ».

Ce fichier est fourni : vous n'avez pas à le modifier. Il regroupe ce qui n'est
pas l'objet du TD (chargement des données, affichage, cache, exécution des tests)
pour que les notebooks restent centrés sur les notions du cours.
"""
from __future__ import annotations

import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# Chemins et modes d'exécution
# --------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parent.parent
DONNEES = RACINE / "donnees"
CACHE = RACINE / "cache"
SORTIES = RACINE / "sorties"
for _d in (DONNEES, CACHE, SORTIES):
    _d.mkdir(exist_ok=True)

# TP_MODE_TEST=1 : poids aléatoires et données synthétiques. Sert uniquement à
# vérifier qu'un notebook s'exécute de bout en bout sans réseau.
MODE_TEST = os.environ.get("TP_MODE_TEST") == "1"
# TP_AUTO=1 : les cellules lourdes se lancent sans cliquer sur leur bouton.
AUTO = os.environ.get("TP_AUTO") == "1"
# Sous Windows, le téléchargement des poids depuis Hugging Face affiche un long avertissement sans conséquence
# (le cache ne peut pas utiliser de liens symboliques). On le masque pour ne pas inquiéter inutilement.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def appareil() -> torch.device:
    """Choisit automatiquement le meilleur matériel disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def fixer_graine(graine: int = 0) -> None:
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)


def lance(bouton) -> bool:
    """Vrai si l'étudiant a cliqué sur le bouton (ou en exécution automatique)."""
    return AUTO or bool(getattr(bouton, "value", False))


class Chrono:
    """Mesure une durée : `with Chrono() as c: ...` puis `c.secondes`."""

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.secondes = time.perf_counter() - self._t
        return False


# --------------------------------------------------------------------------
# Exécution des tests
# --------------------------------------------------------------------------
def verifier(tests, mo):
    """Exécute une liste de (nom, fonction_de_test) et renvoie (tout_ok, affichage)."""
    lignes, reussis = [], 0
    for nom, test in tests:
        try:
            test()
            lignes.append(f"- **[réussi]** {nom}")
            reussis += 1
        except NotImplementedError:
            lignes.append(f"- **[à compléter]** {nom} : la fonction n'est pas encore écrite.")
        except AssertionError as e:
            lignes.append(f"- **[échec]** {nom} : {e}")
        except Exception as e:  # erreur dans le code de l'étudiant
            tb = traceback.extract_tb(e.__traceback__)
            ligne = f" (ligne {tb[-1].lineno})" if tb else ""
            lignes.append(f"- **[erreur]** {nom} : `{type(e).__name__}: {e}`{ligne}")
    total = len(tests)
    tout_ok = reussis == total
    kind = "success" if tout_ok else ("warn" if reussis else "danger")
    texte = f"**{reussis}/{total} tests réussis**\n\n" + "\n".join(lignes)
    if tout_ok:
        texte += "\n\nVous pouvez passer à la suite."
    return tout_ok, mo.callout(mo.md(texte), kind=kind)


def bloquer_si(condition: bool, mo, message: str) -> None:
    """Arrête la cellule courante avec un message tant qu'une étape n'est pas validée."""
    mo.stop(condition, mo.callout(mo.md(message), kind="warn"))


# --------------------------------------------------------------------------
# Cache disque
# --------------------------------------------------------------------------
def chemin_cache(nom: str) -> Path:
    suffixe = "_test" if MODE_TEST else ""
    return CACHE / f"{nom}{suffixe}.npz"


def charger_cache(nom: str):
    p = chemin_cache(nom)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def sauver_cache(nom: str, **tableaux) -> None:
    np.savez_compressed(chemin_cache(nom), **tableaux)


# --------------------------------------------------------------------------
# Modèles pré-entraînés
# --------------------------------------------------------------------------
def categories_imagenet() -> list[str]:
    import torchvision

    return list(torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"])


def charger_modele(nom: str):
    """Charge un modèle pré-entraîné sur ImageNet. Noms : resnet18, efficientnet_b0, deit_tiny."""
    import timm
    import torchvision

    pre = not MODE_TEST
    if nom == "resnet18":
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        modele = torchvision.models.resnet18(weights=w if pre else None)
    elif nom == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        modele = torchvision.models.efficientnet_b0(weights=w if pre else None)
    elif nom == "deit_tiny":
        modele = timm.create_model("deit_tiny_patch16_224", pretrained=pre)
    else:
        raise ValueError(f"Modèle inconnu : {nom}")
    return modele.eval()


def nombre_parametres(modele, entrainables: bool = False) -> int:
    return sum(p.numel() for p in modele.parameters() if (p.requires_grad or not entrainables))


MOYENNE = (0.485, 0.456, 0.406)
ECART = (0.229, 0.224, 0.225)


def transformation(taille: int = 224, augmentation: bool = False):
    from torchvision import transforms as T

    norm = T.Normalize(MOYENNE, ECART)
    if augmentation:
        return T.Compose([T.RandomResizedCrop(taille, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    return T.Compose([T.Resize(int(taille * 1.14)), T.CenterCrop(taille), T.ToTensor(), norm])


def denormaliser(t: torch.Tensor) -> np.ndarray:
    """Tenseur normalisé (3,H,W) -> image numpy (H,W,3) dans [0,1]."""
    x = t.detach().cpu().clone()
    for c in range(3):
        x[c] = x[c] * ECART[c] + MOYENNE[c]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------
# Jeux de données
# --------------------------------------------------------------------------
class DatasetImages(torch.utils.data.Dataset):
    """Images (chemins ou PIL) + étiquettes. `pre_redim` garde en mémoire des images réduites."""

    def __init__(self, images, etiquettes, transform, pre_redim: int | None = None):
        self.etiquettes = list(etiquettes)
        self.transform = transform
        self.images = list(images)
        if pre_redim:
            self.images = [self._ouvrir(im) for im in self.images]
            for im in self.images:
                im.thumbnail((pre_redim * 2, pre_redim * 2))
            self.images = [_redim_cote_court(im, pre_redim) for im in self.images]

    @staticmethod
    def _ouvrir(im):
        if isinstance(im, Image.Image):
            return im.convert("RGB")
        return Image.open(im).convert("RGB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self._ouvrir(self.images[i])), self.etiquettes[i]

    def image_brute(self, i) -> Image.Image:
        return self._ouvrir(self.images[i])


def _redim_cote_court(im: Image.Image, cote: int) -> Image.Image:
    w, h = im.size
    r = cote / min(w, h)
    return im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR)


def _echantillonner(etiquettes, n_par_classe, rng):
    etiquettes = np.asarray(etiquettes)
    idx = []
    for c in np.unique(etiquettes):
        ids = np.flatnonzero(etiquettes == c)
        idx.extend(rng.choice(ids, size=min(n_par_classe, len(ids)), replace=False).tolist())
    return sorted(idx)


def charger_oxford_pets(n_train: int = 30, n_test: int = 15, n_val: int = 10, graine: int = 0):
    """Sous-ensemble équilibré d'Oxford-IIIT Pet (37 races de chats et de chiens).

    Renvoie (images_train, y_train, images_val, y_val, images_test, y_test, noms_classes) ; les images sont des
    chemins. La validation est tirée parmi les images d'entraînement officielles non retenues pour l'entraînement :
    elle sert à comparer des réglages, le test ne sert qu'à la mesure finale.
    """
    if MODE_TEST:
        a, b, e, f, noms = _pets_synthetiques(n_train, n_test, graine)
        c, d = _pets_synthetiques(n_val, 0, graine + 1)[:2]
        return a, b, c, d, e, f, noms
    from torchvision.datasets import OxfordIIITPet

    tv = OxfordIIITPet(DONNEES, split="trainval", download=True)
    te = OxfordIIITPet(DONNEES, split="test", download=True)
    rng = np.random.default_rng(graine)
    i_tr = _echantillonner(tv._labels, n_train, rng)
    i_te = _echantillonner(te._labels, n_test, rng)
    pris = set(i_tr)
    reste = [i for i in range(len(tv._labels)) if i not in pris]
    i_va = [reste[i] for i in _echantillonner([tv._labels[i] for i in reste], n_val, rng)]
    return ([tv._images[i] for i in i_tr], [tv._labels[i] for i in i_tr],
            [tv._images[i] for i in i_va], [tv._labels[i] for i in i_va],
            [te._images[i] for i in i_te], [te._labels[i] for i in i_te], list(tv.classes))


def _image_synthetique(classe: int, rng, taille=200, n_classes=8) -> Image.Image:
    teinte = np.array([np.sin(classe * 0.9) + 1, np.cos(classe * 1.7) + 1, np.sin(classe * 2.3 + 1) + 1]) / 2
    img = rng.normal(0.5, 0.15, (taille, taille, 3)) * 0.4 + teinte * 0.6
    cx, cy, r = rng.integers(50, 150, 2).tolist() + [20 + 6 * (classe % n_classes)]
    yy, xx = np.ogrid[:taille, :taille]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = 1 - teinte
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def _pets_synthetiques(n_train, n_test, graine, n_classes=8):
    rng = np.random.default_rng(graine)
    noms = [f"classe_{k}" for k in range(n_classes)]

    def lot(n):
        ims, ys = [], []
        for c in range(n_classes):
            for _ in range(n):
                ims.append(_image_synthetique(c, rng, n_classes=n_classes))
                ys.append(c)
        return ims, ys

    a, b = lot(n_train)
    c, d = lot(n_test)
    return a, b, c, d, noms


NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
_NEU_PREFIXES = {"cr": "crazing", "in": "inclusion", "pa": "patches", "ps": "pitted_surface",
                 "rs": "rolled-in_scale", "sc": "scratches"}


def _classe_neu(p: Path):
    nom = p.stem.lower()
    for c in NEU_CLASSES:
        if nom.startswith(c):
            return c
    return _NEU_PREFIXES.get(nom.split("_")[0])


def charger_neu(n_train_par_classe: int = 30, n_test_par_classe: int = 100, graine: int = 0,
                n_val_par_classe: int | None = None):
    """Images de défauts d'acier NEU-DET (6 classes), à placer dans donnees/NEU/ (voir README).

    Renvoie (images_train, y_train, images_test, y_test, noms_classes).
    Avec n_val_par_classe, renvoie (images_train, y_train, images_val, y_val, images_test, y_test, noms_classes).
    Les images de chaque classe sont mélangées une seule fois (graine fixe) : le test est pris en premier, puis la
    validation, puis l'entraînement. Le test ne dépend donc ni de la taille de l'entraînement ni de la présence d'une
    validation, et l'entraînement à 10 images par classe est inclus dans celui à 30.
    """
    if MODE_TEST:
        a, b, c, d, _ = _pets_synthetiques(n_train_par_classe, 20, graine, n_classes=6)
        if n_val_par_classe is None:
            return a, b, c, d, NEU_CLASSES
        v, w, _, _, _ = _pets_synthetiques(min(n_val_par_classe, 10), 1, graine + 1, n_classes=6)
        return a, b, v, w, c, d, NEU_CLASSES
    dossier = DONNEES / "NEU"
    fichiers = sorted(p for p in dossier.rglob("*") if p.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"})
    fichiers = [p for p in fichiers if _classe_neu(p)]
    if not fichiers:
        raise FileNotFoundError(f"Aucune image NEU trouvée dans {dossier}. Voir le README, section « Données ».")
    vus, uniques = set(), []
    for p in fichiers:  # certaines archives contiennent des doublons (train/validation)
        if p.name not in vus:
            vus.add(p.name)
            uniques.append(p)
    y = [NEU_CLASSES.index(_classe_neu(p)) for p in uniques]
    rng = np.random.default_rng(graine)
    y_arr = np.asarray(y)
    nv = n_val_par_classe or 0
    i_te, i_va, i_tr = [], [], []
    for c in np.unique(y_arr):
        ids = rng.permutation(np.flatnonzero(y_arr == c)).tolist()
        i_te += ids[:n_test_par_classe]
        i_va += ids[n_test_par_classe:n_test_par_classe + nv]
        i_tr += ids[n_test_par_classe + nv:n_test_par_classe + nv + n_train_par_classe]
    jeu = lambda idx: ([uniques[i] for i in sorted(idx)], [y[i] for i in sorted(idx)])
    if n_val_par_classe is None:
        return jeu(i_tr) + jeu(i_te) + (NEU_CLASSES,)
    return jeu(i_tr) + jeu(i_va) + jeu(i_te) + (NEU_CLASSES,)


# --------------------------------------------------------------------------
# Calculs fournis
# --------------------------------------------------------------------------
@torch.no_grad()
def extraire_embeddings(modele, dataset, taille_lot: int = 32, progression=None) -> np.ndarray:
    """Passe tout le dataset dans `modele` et renvoie les vecteurs de sortie (N, D)."""
    dev = appareil()
    modele = modele.to(dev).eval()
    chargeur = torch.utils.data.DataLoader(dataset, batch_size=taille_lot, shuffle=False, num_workers=0)
    lots = chargeur if progression is None else progression(chargeur)
    sorties = [modele(x.to(dev)).float().cpu() for x, _ in lots]
    return torch.cat(sorties).numpy()


@torch.no_grad()
def predire(modele, dataset, taille_lot: int = 64):
    """Renvoie (classes prédites, probabilités) pour tout le dataset."""
    dev = appareil()
    modele = modele.to(dev).eval()
    chargeur = torch.utils.data.DataLoader(dataset, batch_size=taille_lot, shuffle=False, num_workers=0)
    probas = torch.cat([modele(x.to(dev)).softmax(-1).cpu() for x, _ in chargeur])
    return probas.argmax(1).numpy(), probas.numpy()


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------
def grille(images, titres=None, n_col: int = 5, taille: float = 2.6, cmap=None):
    import matplotlib.pyplot as plt

    n = len(images)
    n_lig = max(1, int(np.ceil(n / n_col)))
    fig, axes = plt.subplots(n_lig, n_col, figsize=(n_col * taille, n_lig * taille * 1.05))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap=cmap)
            if titres:
                ax.set_title(titres[i], fontsize=8)
    fig.tight_layout()
    return fig


def superposer(image: np.ndarray, carte: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose une carte de chaleur [0,1] (h,w) à une image (H,W,3) dans [0,1]."""
    import matplotlib

    carte_t = torch.tensor(carte)[None, None].float()
    carte_r = torch.nn.functional.interpolate(carte_t, size=image.shape[:2], mode="bilinear",
                                              align_corners=False)[0, 0].numpy()
    couleur = matplotlib.colormaps["jet"](carte_r)[..., :3]
    return np.clip((1 - alpha) * image + alpha * couleur, 0, 1)


def matrice_confusion(y_vrai, y_pred, noms, taille: float = 9):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    m = confusion_matrix(y_vrai, y_pred, labels=range(len(noms)))
    fig, ax = plt.subplots(figsize=(taille, taille))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(noms)), noms, rotation=90, fontsize=7)
    ax.set_yticks(range(len(noms)), noms, fontsize=7)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    fig.tight_layout()
    return m, fig


# ==========================================================================
# TD 2 : détection et segmentation
# ==========================================================================
MODELES = RACINE / "modeles"
MODELES.mkdir(exist_ok=True)

PPE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
PPE_CLASSES = ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
               "no_helmet", "no_goggle", "no_gloves", "no_boots"]


def _ecrire_config_yolo(config: Path, racine: Path, classes: list[str]) -> Path:
    """Écrit le fichier de configuration d'un jeu de données YOLO (chemins et noms des classes).

    Le fichier est écrit en UTF-8 et le chemin entre guillemets : un dossier dont le nom contient un accent,
    une espace ou un caractère spécial est ainsi relu correctement, sous Windows comme ailleurs."""
    import json

    noms = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    config.write_text(f"path: {json.dumps(racine.as_posix(), ensure_ascii=False)}\n"
                      f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{noms}\n", encoding="utf-8")
    return config


def preparer_construction_ppe() -> Path:
    """Télécharge (une fois) le jeu Construction-PPE et renvoie le chemin de son fichier de configuration YOLO."""
    import shutil
    import zipfile

    racine = DONNEES / "construction-ppe"
    if not (racine / "images" / "train").exists():
        archive = _telecharger(PPE_URL, DONNEES / "construction-ppe.zip")
        # On décompresse dans un dossier provisoire : le dossier définitif n'apparaît qu'une fois complet.
        provisoire = DONNEES / "construction-ppe.part"
        shutil.rmtree(provisoire, ignore_errors=True)
        try:
            with zipfile.ZipFile(archive) as z:
                z.extractall(provisoire)
        except zipfile.BadZipFile:
            archive.unlink()
            raise RuntimeError("L'archive Construction-PPE est illisible. Relancez la cellule : elle sera téléchargée de nouveau.")
        shutil.rmtree(racine, ignore_errors=True)
        provisoire.rename(racine)
        archive.unlink()
    return _ecrire_config_yolo(racine / "ppe.yaml", racine, PPE_CLASSES)


def lister_images(dossier_dataset: Path, partie: str) -> list[Path]:
    return sorted(p for p in (dossier_dataset / "images" / partie).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def lire_annotations(chemin_image: Path):
    """Lit les annotations YOLO d'une image. Renvoie (boîtes (n, 4) en pixels [x1, y1, x2, y2], classes (n,))."""
    chemin_image = Path(chemin_image)
    etiquettes = chemin_image.parent.parent.parent / "labels" / chemin_image.parent.name / (chemin_image.stem + ".txt")
    w, h = Image.open(chemin_image).size
    if not etiquettes.exists():
        return np.zeros((0, 4), np.float32), np.zeros(0, int)
    lignes = [l.split() for l in etiquettes.read_text().splitlines() if l.strip()]
    if not lignes:
        return np.zeros((0, 4), np.float32), np.zeros(0, int)
    t = np.array([[float(v) for v in l[:5]] for l in lignes])
    cx, cy, bw, bh = t[:, 1] * w, t[:, 2] * h, t[:, 3] * w, t[:, 4] * h
    boites = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)
    return boites, t[:, 0].astype(int)


_COULEURS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
             "#bcf60c", "#fabebe", "#008080"]


def dessiner_boites(image, boites, classes=None, noms=None, scores=None, epaisseur: int = 3):
    """Renvoie une copie PIL de l'image avec les boîtes dessinées."""
    from PIL import ImageDraw

    img = (Image.fromarray(image) if isinstance(image, np.ndarray) else image).convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for k, b in enumerate(np.asarray(boites).reshape(-1, 4)):
        c = int(classes[k]) if classes is not None else 0
        coul = _COULEURS[c % len(_COULEURS)]
        d.rectangle([float(v) for v in b], outline=coul, width=epaisseur)
        texte = (noms[c] if noms is not None else "") + (f" {scores[k]:.2f}" if scores is not None else "")
        if texte.strip():
            d.text((float(b[0]) + 3, float(b[1]) + 2), texte, fill=coul)
    return img


def charger_yolo(nom: str = "yolov8n.pt"):
    """Charge un modèle YOLO (Ultralytics). Les poids officiels sont téléchargés dans modeles/ au premier appel."""
    from ultralytics import YOLO

    chemin = MODELES / nom
    if chemin.exists():
        return YOLO(str(chemin))
    import os as _os

    ancien = Path.cwd()
    _os.chdir(MODELES)  # Ultralytics télécharge les poids officiels dans le dossier courant
    try:
        return YOLO(nom)
    finally:
        _os.chdir(ancien)


def predictions_yolo(modele, images, conf=0.01, iou=0.7, taille=416):
    """Liste, pour chaque image, de (boîtes (n,4), scores (n,), classes (n,))."""
    sorties = []
    for r in modele.predict([str(p) for p in images], conf=conf, iou=iou, imgsz=taille, verbose=False, stream=True):
        b = r.boxes
        sorties.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)))
    return sorties


def charger_deeplab():
    """DeepLabV3 (MobileNetV3) pré-entraîné sur un sous-ensemble de COCO, avec les 21 classes de Pascal VOC
    (classe 15 : personne)."""
    import torchvision

    w = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    m = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None if MODE_TEST else w,
                                                                      weights_backbone=None, num_classes=21)
    return m.eval()


def voc_vers_yolo(dossier_source: Path, dossier_cible: Path, classes: list[str], graine: int = 0,
                  proportions=(0.7, 0.15, 0.15)) -> Path:
    """Convertit des annotations Pascal VOC (.xml) en jeu de données YOLO (train/val/test). Renvoie la config."""
    import shutil
    import xml.etree.ElementTree as ET

    images = {p.stem: p for p in dossier_source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    xmls = sorted(dossier_source.rglob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"Aucune annotation .xml trouvée dans {dossier_source}.")
    rng = np.random.default_rng(graine)
    ordre = rng.permutation(len(xmls))
    n_tr, n_va = int(proportions[0] * len(xmls)), int(proportions[1] * len(xmls))
    parties = {i: ("train" if r < n_tr else "val" if r < n_tr + n_va else "test") for r, i in enumerate(ordre)}
    for i, x in enumerate(xmls):
        racine = ET.parse(x).getroot()
        nom = Path(racine.findtext("filename") or x.stem).stem
        if nom not in images:
            continue
        w = float(racine.findtext("size/width"))
        h = float(racine.findtext("size/height"))
        lignes = []
        for o in racine.iter("object"):
            c = o.findtext("name").strip()
            if c not in classes:
                continue
            bb = o.find("bndbox")
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            lignes.append(f"{classes.index(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        p = parties[i]
        (dossier_cible / "images" / p).mkdir(parents=True, exist_ok=True)
        (dossier_cible / "labels" / p).mkdir(parents=True, exist_ok=True)
        shutil.copy(images[nom], dossier_cible / "images" / p / images[nom].name)
        (dossier_cible / "labels" / p / f"{nom}.txt").write_text("\n".join(lignes))
    return _ecrire_config_yolo(dossier_cible / "config.yaml", dossier_cible, classes)


def charger_sam(nom: str = "mobile_sam.pt"):
    """MobileSAM (version légère de Segment Anything) via Ultralytics, téléchargé dans modeles/ au premier appel."""
    from ultralytics import SAM

    chemin = MODELES / nom
    if chemin.exists():
        return SAM(str(chemin))
    import os as _os

    ancien = Path.cwd()
    _os.chdir(MODELES)
    try:
        return SAM(nom)
    finally:
        _os.chdir(ancien)


def lire_image_rgb(chemin, cote_max: int | None = None) -> np.ndarray:
    img = Image.open(chemin).convert("RGB")
    if cote_max and max(img.size) > cote_max:
        img.thumbnail((cote_max, cote_max))
    return np.asarray(img)


# ==========================================================================
# TD 3 : séries temporelles
# ==========================================================================
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
SKAB_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data"
SKAB_FICHIERS = (["anomaly-free/anomaly-free.csv"] + [f"valve1/{i}.csv" for i in range(16)]
                 + [f"valve2/{i}.csv" for i in range(4)] + [f"other/{i}.csv" for i in range(1, 15)])
SKAB_CAPTEURS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple",
                 "Voltage", "Volume Flow RateRMS"]


def _telecharger(url: str, cible: Path) -> Path:
    """Télécharge `url` vers `cible`, une seule fois.

    Le fichier est d'abord écrit sous un nom provisoire, puis renommé : un téléchargement interrompu ne laisse
    donc jamais de fichier tronqué sous le nom définitif, et il suffit de relancer la cellule."""
    import urllib.request

    if not cible.exists():
        cible.parent.mkdir(parents=True, exist_ok=True)
        provisoire = cible.with_name(cible.name + ".part")
        try:
            urllib.request.urlretrieve(url, provisoire)
        except BaseException:
            provisoire.unlink(missing_ok=True)
            raise
        provisoire.replace(cible)
    return cible


def charger_ett():
    """ETTh1 : deux ans de relevés horaires d'un transformateur électrique (6 charges + température d'huile OT)."""
    import pandas as pd

    chemin = _telecharger(ETT_URL, DONNEES / "ett" / "ETTh1.csv")
    return pd.read_csv(chemin, parse_dates=["date"])


def charger_skab(progression=None) -> dict:
    """SKAB : capteurs d'une pompe (1 mesure par seconde). Renvoie {"anomaly-free": df, "valve1": [df, ...], ...}."""
    import pandas as pd

    sortie = {"anomaly-free": None, "valve1": [], "valve2": [], "other": []}
    fichiers = SKAB_FICHIERS if progression is None else progression(SKAB_FICHIERS)
    for rel in fichiers:
        chemin = _telecharger(f"{SKAB_URL}/{rel}", DONNEES / "skab" / rel)
        df = pd.read_csv(chemin, sep=";", parse_dates=["datetime"])
        groupe = rel.split("/")[0]
        if groupe == "anomaly-free":
            sortie[groupe] = df
        else:
            sortie[groupe].append(df)
    return sortie


def fenetres_glissantes(serie: np.ndarray, taille: int, pas: int) -> np.ndarray:
    """Découpe une série (T, ...) en fenêtres (N, taille, ...) espacées de `pas`."""
    return np.stack([serie[i:i + taille] for i in range(0, len(serie) - taille + 1, pas)])


def entrainer_prevision(modele, X_tr, Y_tr, X_va, Y_va, epoques: int = 8, lr: float = 1e-3, progression=None):
    """Entraîne un modèle de prévision (perte L1). Garde les poids de la meilleure époque sur la validation."""
    dev = appareil()
    modele = modele.to(dev)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)
    Xt, Yt = torch.as_tensor(X_tr, dtype=torch.float32), torch.as_tensor(Y_tr, dtype=torch.float32)
    Xv, Yv = torch.as_tensor(X_va, dtype=torch.float32).to(dev), torch.as_tensor(Y_va, dtype=torch.float32).to(dev)
    historique, meilleur, etat = {"train": [], "val": []}, float("inf"), None
    boucle = range(epoques) if progression is None else progression(range(epoques))
    debut = time.perf_counter()
    for _ in boucle:
        modele.train()
        ordre, total = torch.randperm(len(Xt)), 0.0
        for i in range(0, len(Xt), 64):
            idx = ordre[i:i + 64]
            opt.zero_grad()
            perte = torch.nn.functional.l1_loss(modele(Xt[idx].to(dev)), Yt[idx].to(dev))
            perte.backward()
            opt.step()
            total += perte.item() * len(idx)
        modele.eval()
        with torch.no_grad():
            val = torch.nn.functional.l1_loss(modele(Xv), Yv).item()
        historique["train"].append(total / len(Xt))
        historique["val"].append(val)
        if val < meilleur:
            meilleur, etat = val, {k: v.detach().clone() for k, v in modele.state_dict().items()}
    modele.load_state_dict(etat)
    historique["duree"] = time.perf_counter() - debut
    return modele.eval(), historique


def entrainer_autoencodeur(modele, X, epoques: int = 30, lr: float = 2e-3, progression=None):
    """Entraîne un autoencodeur à reconstruire ses entrées (erreur quadratique)."""
    dev = appareil()
    modele = modele.to(dev)
    opt = torch.optim.Adam(modele.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    pertes = []
    boucle = range(epoques) if progression is None else progression(range(epoques))
    for _ in boucle:
        modele.train()
        ordre, total = torch.randperm(len(Xt)), 0.0
        for i in range(0, len(Xt), 64):
            lot = Xt[ordre[i:i + 64]].to(dev)
            opt.zero_grad()
            perte = ((modele(lot) - lot) ** 2).mean()
            perte.backward()
            opt.step()
            total += perte.item() * len(lot)
        pertes.append(total / len(Xt))
    return modele.eval(), pertes


def injecter_anomalies(serie: np.ndarray, n_evenements: int = 12, graine: int = 0):
    """Injecte des anomalies réalistes dans une série normalisée (T, C).

    Quatre types : pic (2 h), capteur figé (18 h), dérive lente (24 h), bruit anormal (12 h).
    Renvoie (série modifiée, étiquettes (T,) à 0/1, liste des événements)."""
    rng = np.random.default_rng(graine)
    s, etiquettes, evenements = serie.copy(), np.zeros(len(serie), int), []
    positions = np.linspace(150, len(serie) - 150, n_evenements).astype(int)
    for k, t0 in enumerate(positions):
        c, genre = int(rng.integers(0, serie.shape[1])), ["pic", "capteur figé", "dérive", "bruit"][k % 4]
        if genre == "pic":
            duree = 2
            s[t0:t0 + duree, c] += rng.choice([-1, 1]) * 4
        elif genre == "capteur figé":
            duree = 18
            s[t0:t0 + duree, c] = s[t0, c]
        elif genre == "dérive":
            duree = 24
            s[t0:t0 + duree, c] += np.linspace(0, 3, duree)
        else:
            duree = 12
            s[t0:t0 + duree, c] += rng.normal(0, 1.5, duree)
        etiquettes[t0:t0 + duree] = 1
        evenements.append({"début": int(t0), "durée": duree, "type": genre, "canal": c})
    return s, etiquettes, evenements


def scores_classiques(X_train: np.ndarray, X_test: np.ndarray, graine: int = 0) -> dict:
    """Deux détecteurs d'anomalies classiques sur des fenêtres aplaties : PCA (erreur de reconstruction) et Isolation Forest."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest

    a, b = X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1)
    pca = PCA(0.9, random_state=graine).fit(a)
    s_pca = ((b - pca.inverse_transform(pca.transform(b))) ** 2).mean(1)
    s_if = -IsolationForest(n_estimators=200, random_state=graine).fit(a).score_samples(b)
    return {"PCA (classique)": s_pca, "Isolation Forest (classique)": s_if}


def bilan_detection(scores: np.ndarray, etiquettes: np.ndarray) -> dict:
    """Précision moyenne (aire sous la courbe précision/rappel) et meilleur F1 atteignable."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    p, r, _ = precision_recall_curve(etiquettes, scores)
    return {"précision moyenne (AP)": round(float(average_precision_score(etiquettes, scores)), 3),
            "meilleur F1": round(float((2 * p * r / (p + r + 1e-9)).max()), 3)}


def tableau(lignes: list[dict], mo):
    """Affiche une liste de dictionnaires sous forme de tableau."""
    if not lignes:
        return mo.md("Aucun résultat pour l'instant.")
    colonnes = list(lignes[0])
    texte = "| " + " | ".join(colonnes) + " |\n|" + "---|" * len(colonnes) + "\n"
    for l in lignes:
        texte += "| " + " | ".join(str(l.get(c, "")) for c in colonnes) + " |\n"
    return mo.md(texte)


# ==========================================================================
# TD 4 : foundation models et IA générative
# ==========================================================================
CLIP_URL = ("https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
            "vit_b_32-quickgelu-laion400m_e32-46683a32.pt")
FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FASHION_CLASSES = ["t-shirt", "pantalon", "pull", "robe", "manteau", "sandale", "chemise", "basket", "sac", "bottine"]
N_ETAPES_DIFFUSION = 200


def charger_clip():
    """CLIP ViT-B/32 (pré-entraîné sur 400 millions de paires image-texte). Renvoie (modèle, prétraitement, tokeniseur)."""
    import open_clip

    if MODE_TEST:
        modele, _, pre = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained=None)
    else:
        poids = _telecharger(CLIP_URL, MODELES / "clip_vitb32_laion400m.pt")
        modele, _, pre = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained=str(poids))
    return modele.eval(), pre, open_clip.get_tokenizer("ViT-B-32-quickgelu")


@torch.no_grad()
def encoder_images(modele, pretraitement, images, taille_lot: int = 32, progression=None) -> np.ndarray:
    """Embeddings CLIP (non normalisés) d'une liste d'images PIL ou de chemins. Forme (N, 512)."""
    lots = range(0, len(images), taille_lot)
    lots = lots if progression is None else progression(lots)
    sorties = []
    for i in lots:
        lot = [im if isinstance(im, Image.Image) else Image.open(im) for im in images[i:i + taille_lot]]
        sorties.append(modele.encode_image(torch.stack([pretraitement(im.convert("RGB")) for im in lot])).float())
    return torch.cat(sorties).numpy()


@torch.no_grad()
def encoder_textes(modele, tokeniseur, textes: list[str]) -> np.ndarray:
    """Embeddings CLIP (non normalisés) d'une liste de textes. Forme (N, 512)."""
    return modele.encode_text(tokeniseur(textes)).float().numpy()


def vignettes_ppe(dossier: Path, parties=("test", "val"), n_classes: int = 5, max_par_classe: int = 110,
                  cote_min: int = 32, graine: int = 0):
    """Découpe les objets annotés de Construction-PPE en vignettes. Renvoie (images PIL, étiquettes, noms)."""
    vignettes = {c: [] for c in range(n_classes)}
    for partie in parties:
        for chemin in lister_images(dossier, partie):
            boites, classes = lire_annotations(chemin)
            img = None
            for b, c in zip(boites, classes):
                if c < n_classes and b[2] - b[0] >= cote_min and b[3] - b[1] >= cote_min:
                    img = img or Image.open(chemin).convert("RGB")
                    vignettes[int(c)].append(img.crop(tuple(int(v) for v in b)))
    rng = np.random.default_rng(graine)
    images, etiquettes = [], []
    for c, liste in vignettes.items():
        for i in rng.permutation(len(liste))[:max_par_classe]:
            images.append(liste[i])
            etiquettes.append(c)
    return images, np.array(etiquettes), PPE_CLASSES[:n_classes]


def descripteur_pixels(image: Image.Image) -> np.ndarray:
    """Descripteur classique minimal : image réduite à 16 x 16 + histogrammes de couleur."""
    rgb = np.asarray(image.convert("RGB"))
    petit = np.asarray(image.convert("RGB").resize((16, 16)), dtype=np.float32).ravel() / 255
    histos = np.concatenate([np.histogram(rgb[..., k], bins=16, range=(0, 256))[0] for k in range(3)]).astype(np.float32)
    return np.concatenate([petit, histos / max(1.0, histos.sum())])


def charger_dino():
    """DINOv2 ViT-S/14 (auto-supervisé, sans texte) via timm. Renvoie (modèle, prétraitement)."""
    import timm

    modele = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=not MODE_TEST, num_classes=0, img_size=224)
    return modele.eval(), transformation(224)


def charger_fashion_mnist(n: int | None = None):
    """Fashion-MNIST : vêtements en niveaux de gris, agrandis à 32 x 32 et ramenés dans [-1, 1]."""
    import gzip

    def lire(nom, decalage):
        chemin = _telecharger(f"{FASHION_URL}/{nom}", DONNEES / "fashion-mnist" / nom)
        return np.frombuffer(gzip.open(chemin).read(), np.uint8, offset=decalage)

    X = lire("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).astype(np.float32) / 127.5 - 1
    y = lire("train-labels-idx1-ubyte.gz", 8).astype(np.int64)
    X = np.pad(X, ((0, 0), (0, 0), (2, 2), (2, 2)), constant_values=-1)
    return (X, y) if n is None else (X[:n], y[:n])


def planning_diffusion(n_etapes: int = N_ETAPES_DIFFUSION):
    """Renvoie (betas, alphas, alphas_cumules) : la quantité de bruit ajoutée à chaque étape."""
    betas = torch.linspace(1e-4, 0.04, n_etapes)
    alphas = 1 - betas
    return betas, alphas, torch.cumprod(alphas, dim=0)


def encodage_temps(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Encodage sinusoïdal de l'étape t (le même principe que l'encodage de position des Transformers)."""
    import math

    freq = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    angles = t[:, None].float() * freq[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class _BlocUNet(torch.nn.Module):
    def __init__(self, entree, sortie, dim):
        super().__init__()
        nn = torch.nn
        self.c1, self.c2 = nn.Conv2d(entree, sortie, 3, padding=1), nn.Conv2d(sortie, sortie, 3, padding=1)
        self.e = nn.Linear(dim, sortie)
        self.n1, self.n2 = nn.GroupNorm(8, sortie), nn.GroupNorm(8, sortie)
        self.r = nn.Conv2d(entree, sortie, 1) if entree != sortie else nn.Identity()

    def forward(self, x, e):
        f = torch.nn.functional
        h = f.silu(self.n1(self.c1(x))) + self.e(e)[:, :, None, None]
        return f.silu(self.n2(self.c2(h))) + self.r(x)


class PetitUNet(torch.nn.Module):
    """Petit U-Net conditionné par l'étape t et par une classe (indice 10 = « sans condition »)."""

    def __init__(self, canaux: int = 32, dim: int = 64, n_classes: int = 10):
        super().__init__()
        nn = torch.nn
        self.dim = dim
        self.te = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ce = nn.Embedding(n_classes + 1, dim)
        self.b1, self.b2, self.b3 = _BlocUNet(1, canaux, dim), _BlocUNet(canaux, 2 * canaux, dim), _BlocUNet(2 * canaux, 2 * canaux, dim)
        self.u2, self.u1 = _BlocUNet(4 * canaux, canaux, dim), _BlocUNet(2 * canaux, canaux, dim)
        self.out = nn.Conv2d(canaux, 1, 1)
        self.pool, self.up = nn.AvgPool2d(2), nn.Upsample(scale_factor=2)

    def forward(self, x, t, classe):
        e = self.te(encodage_temps(t, self.dim)) + self.ce(classe)
        h1 = self.b1(x, e)
        h2 = self.b2(self.pool(h1), e)
        h3 = self.b3(self.pool(h2), e)
        u = self.u2(torch.cat([self.up(h3), h2], 1), e)
        u = self.u1(torch.cat([self.up(u), h1], 1), e)
        return self.out(u)


def charger_unet_fourni():
    """U-Net de diffusion entraîné sur Fashion-MNIST, fourni dans modeles/."""
    modele = PetitUNet()
    chemin = MODELES / "tp4_ddpm_fashion.pt"
    if not chemin.exists():
        raise FileNotFoundError(f"Modèle fourni introuvable : {chemin}")
    modele.load_state_dict(torch.load(chemin, map_location="cpu"))
    return modele.eval()


def points_2d(n: int, forme: str = "deux lunes", graine: int = 0) -> np.ndarray:
    """Nuage de points 2D servant de « jeu de données » jouet pour la diffusion."""
    rng = np.random.default_rng(graine)
    if forme == "deux lunes":
        from sklearn.datasets import make_moons

        p = make_moons(n, noise=0.05, random_state=graine)[0]
        p = (p - [0.5, 0.25]) * 1.2
    elif forme == "spirale":
        r = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
        p = np.stack([r * np.cos(r), r * np.sin(r)], 1) / 7 + rng.normal(0, 0.03, (n, 2))
    else:  # anneaux
        angle = rng.uniform(0, 2 * np.pi, n)
        rayon = np.where(rng.uniform(size=n) < 0.5, 0.5, 1.2) + rng.normal(0, 0.04, n)
        p = np.stack([rayon * np.cos(angle), rayon * np.sin(angle)], 1)
    return p.astype(np.float32)


# ==========================================================================
# TD 5 : graphes et apprentissage par renforcement
# ==========================================================================
CORA_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora"
CORA_CLASSES = ["Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
                "Reinforcement_Learning", "Rule_Learning", "Theory"]


def charger_cora(graine: int = 0) -> dict:
    """Cora : 2 708 articles scientifiques reliés par leurs citations.

    Renvoie un dictionnaire : X (N, 1433) présence de mots, y (N,) thème, A (N, N) adjacence symétrique (0/1),
    aretes (E, 2), et trois ensembles d'indices : train (20 articles par thème), val (500), test (1000)."""
    contenu = np.genfromtxt(_telecharger(f"{CORA_URL}/cora.content", DONNEES / "cora" / "cora.content"), dtype=str)
    liens = np.genfromtxt(_telecharger(f"{CORA_URL}/cora.cites", DONNEES / "cora" / "cora.cites"), dtype=int)
    position = {int(i): k for k, i in enumerate(contenu[:, 0])}
    X = contenu[:, 1:-1].astype(np.float32)
    X = X / np.maximum(X.sum(axis=1, keepdims=True), 1)
    y = np.array([CORA_CLASSES.index(c) for c in contenu[:, -1]])
    aretes = np.array([[position[a], position[b]] for a, b in liens])
    n = len(y)
    A = torch.zeros(n, n)
    A[aretes[:, 0], aretes[:, 1]] = 1
    A = ((A + A.T) > 0).float()
    rng = np.random.default_rng(graine)
    train = np.concatenate([rng.choice(np.flatnonzero(y == c), 20, replace=False) for c in range(len(CORA_CLASSES))])
    reste = rng.permutation(np.setdiff1d(np.arange(n), train))
    return {"X": torch.tensor(X), "y": torch.tensor(y), "A": A, "aretes": aretes,
            "train": torch.tensor(train), "val": torch.tensor(reste[:500]), "test": torch.tensor(reste[500:1500])}


def entrainer_noeuds(modele, X, A_norm, y, train, val, test, epoques: int = 200, lr: float = 0.01, graine: int = 0):
    """Entraîne un modèle de classification de nœuds. Renvoie (accuracy test à la meilleure époque de validation, durée)."""
    torch.manual_seed(graine)
    for couche in modele.modules():
        if hasattr(couche, "reset_parameters"):
            couche.reset_parameters()
    opt = torch.optim.Adam(modele.parameters(), lr=lr, weight_decay=5e-4)
    meilleur_val, test_retenu, debut = -1.0, 0.0, time.perf_counter()
    for _ in range(epoques):
        modele.train()
        opt.zero_grad()
        torch.nn.functional.cross_entropy(modele(X, A_norm)[train], y[train]).backward()
        opt.step()
        modele.eval()
        with torch.no_grad():
            pred = modele(X, A_norm).argmax(dim=1)
        acc_val = (pred[val] == y[val]).float().mean().item()
        if acc_val >= meilleur_val:
            meilleur_val, test_retenu = acc_val, (pred[test] == y[test]).float().mean().item()
    return test_retenu, time.perf_counter() - debut


class CoucheGAT(torch.nn.Module):
    """Couche d'attention sur graphe (GAT) : chaque nœud pondère ses voisins par un score d'attention appris."""

    def __init__(self, entree: int, sortie: int, tetes: int, concatener: bool = True):
        super().__init__()
        self.tetes, self.sortie, self.concatener = tetes, sortie, concatener
        self.w = torch.nn.Linear(entree, sortie * tetes, bias=False)
        self.a_source = torch.nn.Parameter(torch.randn(tetes, sortie) * 0.1)
        self.a_cible = torch.nn.Parameter(torch.randn(tetes, sortie) * 0.1)
        self.dropout = torch.nn.Dropout(0.6)

    def reset_parameters(self):
        self.w.reset_parameters()
        torch.nn.init.normal_(self.a_source, std=0.1)
        torch.nn.init.normal_(self.a_cible, std=0.1)

    def forward(self, x, A_norm):
        source, cible = torch.nonzero(A_norm, as_tuple=True)        # les arêtes (boucles comprises)
        h = self.w(x).view(-1, self.tetes, self.sortie)
        score = torch.nn.functional.leaky_relu((h * self.a_source).sum(-1)[source] + (h * self.a_cible).sum(-1)[cible], 0.2)
        # softmax des scores sur les voisins de chaque nœud cible
        maxi = torch.full((x.shape[0], self.tetes), -1e9).scatter_reduce(0, cible[:, None].expand(-1, self.tetes), score, "amax")
        expo = torch.exp(score - maxi[cible])
        attention = self.dropout(expo / torch.zeros(x.shape[0], self.tetes).index_add_(0, cible, expo)[cible])
        sortie = torch.zeros(x.shape[0], self.tetes, self.sortie).index_add_(0, cible, attention[:, :, None] * h[source])
        return sortie.flatten(1) if self.concatener else sortie.mean(1)


# ---- Apprentissage par renforcement : CartPole -------------------------------------------------------
BORNES_CARTPOLE = [np.linspace(-2.4, 2.4, 7)[1:-1], np.linspace(-3, 3, 7)[1:-1],
                   np.linspace(-0.21, 0.21, 9)[1:-1], np.linspace(-3, 3, 9)[1:-1]]
FORME_TABLE_Q = tuple(len(b) + 1 for b in BORNES_CARTPOLE) + (2,)


def discretiser(observation) -> tuple:
    """Transforme les 4 nombres réels de CartPole en 4 indices de case (pour le Q-learning tabulaire)."""
    return tuple(int(np.digitize(observation[i], BORNES_CARTPOLE[i])) for i in range(4))


def creer_cartpole():
    import gymnasium as gym

    return gym.make("CartPole-v1")


def dessiner_cartpole(observation, texte: str = "") -> Image.Image:
    """Dessine l'état de CartPole (position du chariot, angle du mât)."""
    from PIL import ImageDraw

    largeur, hauteur = 360, 200
    img = Image.new("RGB", (largeur, hauteur), "white")
    d = ImageDraw.Draw(img)
    x, angle = float(observation[0]), float(observation[2])
    cx, sol = largeur / 2 + x * largeur / 4.8 * 0.9, 150
    d.line([(0, sol + 12), (largeur, sol + 12)], fill="#888888", width=2)
    d.rectangle([cx - 25, sol - 12, cx + 25, sol + 12], fill="#2c3e50")
    d.line([(cx, sol - 12), (cx + 90 * np.sin(angle), sol - 12 - 90 * np.cos(angle))], fill="#c0392b", width=6)
    if texte:
        d.text((8, 6), texte, fill="black")
    return img


def animation_gif(images: list, duree_ms: int = 40) -> bytes:
    import io

    tampon = io.BytesIO()
    images[0].save(tampon, format="GIF", save_all=True, append_images=images[1:], duration=duree_ms, loop=0)
    return tampon.getvalue()


def jouer_episode(politique, graine: int = 0, max_images: int = 250):
    """Joue un épisode avec `politique(observation) -> action`. Renvoie (récompense totale, images)."""
    env = creer_cartpole()
    obs, _ = env.reset(seed=graine)
    total, fini, images = 0.0, False, []
    while not fini:
        if len(images) < max_images:
            images.append(dessiner_cartpole(obs, f"pas {int(total)}"))
        obs, r, termine, tronque, _ = env.step(int(politique(obs)))
        total += r
        fini = termine or tronque
    return total, images


def evaluer_politique(politique, n_episodes: int = 10, premiere_graine: int = 1000) -> float:
    """Récompense moyenne de la politique sur n épisodes (sans exploration).

    Les épisodes sont tirés avec les graines premiere_graine, premiere_graine + 1, etc. : deux appels avec la même
    valeur évaluent donc exactement sur les mêmes situations de départ."""
    return float(np.mean([jouer_episode(politique, graine=premiere_graine + k, max_images=0)[0] for k in range(n_episodes)]))


def balise_gif(images: list, duree_ms: int = 40, largeur: int = 360) -> str:
    """Balise HTML d'une animation, à afficher avec mo.Html(...)."""
    import base64

    return f'<img src="data:image/gif;base64,{base64.b64encode(animation_gif(images, duree_ms)).decode()}" width="{largeur}">'


def trajectoire(environnement: str, politique, graine: int = 0):
    """Joue un épisode dans un environnement Gymnasium. Renvoie (récompense totale, tableau des observations)."""
    import gymnasium as gym

    env = gym.make(environnement)
    obs, _ = env.reset(seed=graine)
    total, fini, observations = 0.0, False, [obs]
    while not fini:
        obs, r, termine, tronque, _ = env.step(int(politique(obs)))
        observations.append(obs)
        total += r
        fini = termine or tronque
    return total, np.array(observations)


def lisser_par_pas(historique, n_pas: int, pas: int = 500) -> np.ndarray:
    """historique : liste de (numéro de pas de fin d'épisode, récompense). Renvoie la récompense moyenne des
    derniers épisodes terminés, échantillonnée tous les `pas` pas."""
    grille = np.arange(pas, n_pas + 1, pas)
    fins = np.array([h[0] for h in historique])
    recs = np.array([h[1] for h in historique])
    sortie = []
    for g in grille:
        sel = recs[(fins <= g)][-10:]
        sortie.append(sel.mean() if len(sel) else np.nan)
    return np.array(sortie)


class MemoireRejeu:
    """Mémoire circulaire de transitions (état, action, récompense, état suivant, terminé)."""

    def __init__(self, capacite: int = 20000, dim_etat: int = 4):
        self.capacite, self.n = capacite, 0
        self.etats = np.zeros((capacite, dim_etat), np.float32)
        self.suivants = np.zeros((capacite, dim_etat), np.float32)
        self.actions = np.zeros(capacite, np.int64)
        self.recompenses = np.zeros(capacite, np.float32)
        self.termines = np.zeros(capacite, np.float32)

    def ajouter(self, etat, action, recompense, suivant, termine):
        i = self.n % self.capacite
        self.etats[i], self.actions[i], self.recompenses[i], self.suivants[i], self.termines[i] = etat, action, recompense, suivant, float(termine)
        self.n += 1

    def tirer(self, taille: int, rng):
        idx = rng.integers(0, min(self.n, self.capacite), taille)
        return (torch.tensor(self.etats[idx]), torch.tensor(self.actions[idx]), torch.tensor(self.recompenses[idx]),
                torch.tensor(self.suivants[idx]), torch.tensor(self.termines[idx]))

    def __len__(self):
        return min(self.n, self.capacite)


def entrainer_ppo(n_pas: int, graine: int = 0, environnement: str = "CartPole-v1"):
    """Entraîne PPO (Stable-Baselines3). Renvoie (modèle, historique [(pas de fin d'épisode, récompense)])."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(gym.make(environnement))
    modele = PPO("MlpPolicy", env, seed=graine, verbose=0, device="cpu")
    modele.learn(n_pas)
    historique = list(zip(np.cumsum(env.get_episode_lengths()).tolist(), env.get_episode_rewards()))
    return modele, historique


def politique_ppo(modele):
    """La politique d'un modèle Stable-Baselines3, sans tirage aléatoire : fonction observation -> action."""
    return lambda observation: int(modele.predict(observation, deterministic=True)[0])


def charger_ppo(chemin):
    """Recharge un modèle PPO enregistré avec modele.save(chemin)."""
    from stable_baselines3 import PPO

    return PPO.load(str(chemin), device="cpu")
