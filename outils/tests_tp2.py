"""Tests automatiques du TD 2. Chaque fonction renvoie une liste (nom, test)."""
from __future__ import annotations

import numpy as np
import torch


def _appelle_torchvision(f, vus=None):
    """Vrai si la fonction f (ou une fonction qu'elle appelle, définie dans le notebook) utilise torchvision."""
    import types
    vus = vus if vus is not None else set()
    if id(f) in vus or not isinstance(f, types.FunctionType):
        return False
    vus.add(id(f))
    noms, pile = set(), [f.__code__]
    while pile:
        code = pile.pop()
        noms |= set(code.co_names)
        pile += [c for c in code.co_consts if isinstance(c, types.CodeType)]
    if {"torchvision", "box_iou", "batched_nms"} & noms:
        return True
    for n in noms:
        obj = f.__globals__.get(n)
        if isinstance(obj, types.ModuleType) and obj.__name__.startswith("torchvision"):
            return True
        if callable(obj) and str(getattr(obj, "__module__", "") or "").startswith("torchvision"):
            return True
        if isinstance(obj, types.FunctionType) and _appelle_torchvision(obj, vus):
            return True
    return False


# ---------------------------------------------------------------- TODO 1
def tests_iou(iou):
    def identiques():
        v = iou([10, 10, 50, 50], [10, 10, 50, 50])
        assert abs(v - 1) < 1e-6, f"deux boîtes identiques doivent donner 1, obtenu {v}"

    def disjointes():
        v = iou([0, 0, 10, 10], [20, 20, 30, 30])
        assert v == 0, f"deux boîtes disjointes doivent donner 0, obtenu {v} : l'intersection ne peut pas être négative"

    def recouvrement_partiel():
        v = iou([0, 0, 10, 10], [5, 0, 15, 10])
        assert abs(v - 50 / 150) < 1e-6, f"attendu 50 / 150 = 0.333, obtenu {v:.3f}"

    def incluse():
        v = iou([0, 0, 10, 10], [2, 2, 4, 4])
        assert abs(v - 4 / 100) < 1e-6, f"une petite boîte incluse dans une grande : attendu 0.04, obtenu {v:.3f}"

    def symetrique():
        a, b = [3, 1, 12, 9], [6, 4, 20, 7]
        assert abs(iou(a, b) - iou(b, a)) < 1e-9, "l'IoU doit être symétrique"

    def l_une_sous_l_autre():
        v = iou([0, 0, 10, 10], [0, 20, 10, 30])
        assert v == 0, (f"deux boîtes l'une au-dessus de l'autre (même largeur) doivent donner 0, obtenu {v} : "
                        "la hauteur de l'intersection aussi doit être ramenée à 0 si elle est négative")

    def coin_seulement():
        v = iou([0, 0, 10, 10], [10, 10, 20, 20])
        assert v == 0, f"deux boîtes qui se touchent par un coin ont une intersection d'aire nulle, obtenu {v}"

    def sans_torchvision():
        iou([0, 0, 10, 10], [5, 0, 15, 10])   # la fonction doit d'abord exister et tourner
        assert not _appelle_torchvision(iou), ("votre IoU appelle torchvision (box_iou) : le but est de la calculer "
                                                "vous-même, avec les coins et les aires")

    return [("boîtes identiques", identiques), ("boîtes disjointes", disjointes),
            ("recouvrement partiel", recouvrement_partiel), ("boîte incluse", incluse),
            ("symétrie", symetrique), ("boîtes l'une sous l'autre", l_une_sous_l_autre),
            ("contact par un coin", coin_seulement), ("calculée sans torchvision", sans_torchvision)]


# ---------------------------------------------------------------- TODO 2
def tests_nms(nms):
    import torchvision

    def cas_simple():
        b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
        s = np.array([0.9, 0.8, 0.7])
        r = list(nms(b, s, 0.5))
        assert r == [0, 2], f"attendu [0, 2] (la 2e boîte recouvre la 1re et a un score plus faible), obtenu {r}"

    def ordre_des_scores():
        b = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], float)
        s = np.array([0.3, 0.9])
        r = list(nms(b, s, 0.5))
        assert r == [1], f"c'est la boîte de meilleur score qui doit être gardée : attendu [1], obtenu {r}"

    def comme_torchvision():
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, 200, (150, 2))
        wh = rng.uniform(10, 60, (150, 2))
        b = np.concatenate([xy, xy + wh], 1)
        s = rng.uniform(0, 1, 150)
        for seuil in (0.3, 0.5, 0.7):
            ref = torchvision.ops.nms(torch.tensor(b), torch.tensor(s), seuil).tolist()
            r = [int(i) for i in nms(b, s, seuil)]
            assert r == ref, f"seuil {seuil} : {len(r)} boîtes gardées, la référence en garde {len(ref)}"

    def seuil_un():
        b = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
        r = list(nms(b, np.array([0.9, 0.8]), 1.0))
        assert len(r) == 2, "avec un seuil de 1, rien n'est supprimé (l'IoU doit dépasser strictement le seuil)"

    def sans_torchvision():
        nms(np.array([[0, 0, 10, 10], [1, 1, 11, 11]], float), np.array([0.9, 0.8]), 0.5)
        assert not _appelle_torchvision(nms), ("votre NMS appelle torchvision : le but est de la recoder avec votre "
                                                "fonction iou ; torchvision.ops.nms ne sert ici que de référence")

    return [("cas simple", cas_simple), ("garde la meilleure boîte", ordre_des_scores),
            ("identique à torchvision.ops.nms (150 boîtes, 3 seuils)", comme_torchvision), ("seuil égal à 1", seuil_un),
            ("recodée sans torchvision", sans_torchvision)]


# ---------------------------------------------------------------- TODO 3
def tests_boites_masque(boites_depuis_masque):
    def masque():
        m = np.zeros((100, 120), bool)
        m[10:30, 20:50] = True      # 20 x 30 = 600 pixels
        m[60:90, 70:110] = True     # 30 x 40 = 1200 pixels
        m[5:7, 100:102] = True      # 4 pixels : du bruit
        return m

    def deux_regions():
        b = boites_depuis_masque(masque(), aire_min=50)
        assert len(b) == 2, f"on attend 2 boîtes (le bruit de 4 pixels est filtré), obtenu {len(b)}"

    def coordonnees():
        b = sorted([list(map(float, x)) for x in boites_depuis_masque(masque(), aire_min=50)])
        assert b[0] == [20, 10, 50, 30], f"première boîte attendue [20, 10, 50, 30] (x1, y1, x2, y2), obtenu {b[0]}"
        assert b[1] == [70, 60, 110, 90], f"seconde boîte attendue [70, 60, 110, 90], obtenu {b[1]}"

    def filtre_aire():
        b = boites_depuis_masque(masque(), aire_min=1000)
        assert len(b) == 1, f"avec aire_min = 1000, seule la grande région reste, obtenu {len(b)} boîtes"

    def aire_limite():
        b = boites_depuis_masque(masque(), aire_min=600)
        assert len(b) == 2, ("une région de 600 pixels exactement doit être conservée avec aire_min = 600 "
                             f"(« au moins aire_min pixels »), obtenu {len(b)} boîtes")

    def region_en_l():
        m = np.zeros((40, 40), bool)
        m[0:20, 0:4] = True
        m[16:20, 0:20] = True      # une région en L : 144 pixels, dans une boîte de 20 x 20 = 400
        b = boites_depuis_masque(m, aire_min=200)
        assert len(b) == 0, ("aire_min porte sur le nombre de pixels de la région (stats[i][4]), pas sur l'aire de sa "
                             f"boîte : cette région en L a 144 pixels, elle doit être filtrée avec aire_min = 200, obtenu {len(b)} boîte(s)")

    def masque_vide():
        b = boites_depuis_masque(np.zeros((50, 50), bool), aire_min=10)
        assert len(b) == 0, "un masque vide ne donne aucune boîte"

    return [("deux régions détectées", deux_regions), ("coordonnées [x1, y1, x2, y2]", coordonnees),
            ("filtrage par aire", filtre_aire), ("aire égale à aire_min", aire_limite),
            ("aire en pixels, pas aire de la boîte", region_en_l), ("masque vide", masque_vide)]


# ---------------------------------------------------------------- TODO 4
def tests_evaluation(evaluer_detections):
    verites = [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]]

    def parfait():
        r = evaluer_detections([(v, 0.9) for v in verites], verites)
        assert (r["tp"], r["fp"], r["fn"]) == (3, 0, 0), f"détections parfaites : attendu tp=3, fp=0, fn=0, obtenu {r}"
        assert r["precision"] == 1 and r["rappel"] == 1, "précision et rappel doivent valoir 1"

    def doublon():
        preds = [([0, 0, 10, 10], 0.9), ([0, 0, 10, 10], 0.8), ([20, 20, 30, 30], 0.7)]
        r = evaluer_detections(preds, verites)
        assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1), (
            f"attendu tp=2, fp=1 (le doublon), fn=1 (la 3e vérité jamais trouvée), obtenu {r}. "
            "Une vérité ne peut être associée qu'à une seule prédiction.")

    def seuil():
        preds = [([2, 0, 12, 10], 0.9)]  # IoU = 80 / 120 = 0.67
        assert evaluer_detections(preds, verites, seuil_iou=0.5)["tp"] == 1, "IoU 0.67 > 0.5 : vrai positif"
        assert evaluer_detections(preds, verites, seuil_iou=0.75)["tp"] == 0, "IoU 0.67 < 0.75 : faux positif"

    def ordre_des_scores():
        # Deux objets voisins A et B. La prédiction à 0.9 recouvre surtout A ; celle à 0.3 recouvre A et B.
        # Parcourue en premier, la prédiction à 0.3 prendrait A, et celle à 0.9 ne trouverait plus de vérité libre.
        deux_objets = [[0, 0, 10, 10], [2, 0, 12, 10]]
        preds = [([0.5, 0, 10.5, 10], 0.3), ([-3, 0, 7, 10], 0.9)]
        r = evaluer_detections(preds, deux_objets, seuil_iou=0.5)
        assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0), (
            f"attendu tp=2, fp=0, fn=0, obtenu {r} : les prédictions doivent être parcourues par score décroissant")
        r2 = evaluer_detections(list(reversed(preds)), deux_objets, seuil_iou=0.5)
        assert r == r2, "le résultat ne doit pas dépendre de l'ordre dans lequel les prédictions sont fournies"

    def iou_egale_au_seuil():
        r = evaluer_detections([([0, 0, 10, 5], 1.0)], [[0, 0, 10, 10]], seuil_iou=0.5)
        assert r["tp"] == 1, ("une IoU exactement égale au seuil (ici 50 / 100 = 0.5) compte comme un vrai positif "
                              f"(« au moins seuil_iou »), obtenu {r}")

    def cas_vides():
        r = evaluer_detections([], verites)
        assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 3), f"aucune prédiction : attendu fn=3, obtenu {r}"
        assert r["precision"] == 0 and r["rappel"] == 0, "sans prédiction, précision et rappel valent 0 (pas d'erreur de division)"
        r = evaluer_detections([([0, 0, 5, 5], 0.5)], [])
        assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 0), f"aucune vérité : la prédiction est un faux positif, obtenu {r}"

    def meilleure_iou():
        # Deux objets qui se chevauchent. La première prédiction recouvre les deux vérités au-delà du seuil :
        # elle doit être associée à celle qu'elle recouvre le mieux, sinon la seconde prédiction n'a plus de vérité libre.
        deux_objets = [[0, 0, 10, 10], [2, 0, 12, 10]]
        preds = [([2, 0, 12, 10], 0.9), ([-3, 0, 7, 10], 0.8)]
        r = evaluer_detections(preds, deux_objets, seuil_iou=0.5)
        assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0), (
            f"attendu tp=2, fp=0, fn=0, obtenu {r}. Chaque prédiction doit être associée à la vérité libre "
            "avec laquelle son IoU est la plus grande, pas à la première qui dépasse le seuil.")

    return [("détections parfaites", parfait), ("doublon compté comme faux positif", doublon),
            ("respect du seuil d'IoU", seuil), ("indépendant de l'ordre des prédictions", ordre_des_scores),
            ("IoU égale au seuil", iou_egale_au_seuil), ("association à la vérité de plus grande IoU", meilleure_iou),
            ("cas vides", cas_vides)]


# ---------------------------------------------------------------- TODO 5
def tests_courbe(courbe_seuils):
    verites = [[[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]]]
    preds = [[([0, 0, 10, 10], 0.9), ([40, 40, 50, 50], 0.6)],
             [([0, 0, 10, 10], 0.8), ([20, 20, 30, 30], 0.3)]]

    def forme():
        r = courbe_seuils(preds, verites, [0.1, 0.5, 0.95])
        assert set(r) >= {"seuil", "precision", "rappel", "f1"}, "on attend un dictionnaire avec seuil, precision, rappel, f1"
        assert all(len(r[k]) == 3 for k in ("seuil", "precision", "rappel", "f1")), "une valeur par seuil"

    def valeurs():
        r = courbe_seuils(preds, verites, [0.1, 0.5, 0.95])
        # seuil 0.1 : tp=3 fp=1 fn=0 ; seuil 0.5 : tp=2 fp=1 fn=1 ; seuil 0.95 : tp=0 fp=0 fn=3
        attendu_p, attendu_r = [0.75, 2 / 3, 0], [1, 2 / 3, 0]
        assert np.allclose(r["precision"], attendu_p), f"précisions attendues {np.round(attendu_p, 3)}, obtenues {np.round(r['precision'], 3)}"
        assert np.allclose(r["rappel"], attendu_r), f"rappels attendus {np.round(attendu_r, 3)}, obtenus {np.round(r['rappel'], 3)}"

    def cumul():
        r = courbe_seuils(preds, verites, [0.5])
        assert np.isclose(r["rappel"][0], 2 / 3), ("les vrais positifs, faux positifs et faux négatifs doivent être "
                                                    "additionnés sur toutes les images avant de calculer précision et rappel")

    def f1():
        r = courbe_seuils(preds, verites, [0.1, 0.95])
        assert np.isclose(r["f1"][0], 2 * 0.75 * 1 / 1.75), "F1 = 2 P R / (P + R)"
        assert r["f1"][1] == 0, "F1 vaut 0 quand précision et rappel sont nuls"

    def seuil_inclus():
        r = courbe_seuils(preds, verites, [0.6])
        # au seuil 0.6, la prédiction de score 0.6 (une fausse alerte) est conservée : tp=2, fp=1, fn=1
        assert np.isclose(r["precision"][0], 2 / 3), (
            f"au seuil 0.6, une prédiction de score exactement 0.6 est gardée (score supérieur OU ÉGAL au seuil) : "
            f"précision attendue 0.667, obtenue {r['precision'][0]:.3f}")

    return [("format du résultat", forme), ("précision et rappel par seuil", valeurs),
            ("comptes cumulés sur toutes les images", cumul), ("score F1", f1), ("score égal au seuil", seuil_inclus)]
