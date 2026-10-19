"""Tests automatiques du TD 1. Chaque fonction renvoie une liste (nom, test)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _effnet(n_classes=5):
    import torchvision

    torch.manual_seed(0)
    return torchvision.models.efficientnet_b0(weights=None, num_classes=n_classes)


# ---------------------------------------------------------------- TODO 1
def tests_retirer_tete(retirer_tete):
    import timm
    import torchvision

    def dim_sortie(nom, constructeur, attendu):
        def test():
            torch.manual_seed(0)
            m = constructeur()
            out = retirer_tete(m)
            assert out is not None, "la fonction doit renvoyer le modèle modifié"
            with torch.no_grad():
                y = out.eval()(torch.randn(2, 3, 224, 224))
            assert y.ndim == 2, f"la sortie a la forme {tuple(y.shape)} : on attend 2 dimensions (taille du lot, taille du vecteur)"
            assert y.shape[1] == attendu, (
                f"{nom} : la sortie a {y.shape[1]} dimensions, on attend {attendu}. "
                "Avez-vous bien remplacé la couche de classification (et seulement elle) ?")
        return test

    return [
        ("ResNet18 renvoie des vecteurs de 512", dim_sortie("ResNet18", lambda: torchvision.models.resnet18(), 512)),
        ("EfficientNet-B0 renvoie des vecteurs de 1280",
         dim_sortie("EfficientNet-B0", lambda: torchvision.models.efficientnet_b0(), 1280)),
        ("DeiT-tiny renvoie des vecteurs de 192",
         dim_sortie("DeiT-tiny", lambda: timm.create_model("deit_tiny_patch16_224", pretrained=False), 192)),
    ]


# ---------------------------------------------------------------- TODO 2
def tests_descripteurs(descripteurs_classiques):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (180, 240, 3), dtype=np.uint8))

    def forme():
        v = descripteurs_classiques(img)
        assert isinstance(v, np.ndarray), "on attend un tableau numpy"
        assert v.ndim == 1, f"on attend un vecteur à 1 dimension, forme obtenue : {v.shape}"
        assert v.shape[0] == 1812, (f"longueur {v.shape[0]}, on attend 1764 (HOG) + 48 (histogrammes) = 1812. "
                                    "Vérifiez la taille 128x128 et les paramètres HOG.")

    def histogrammes():
        v = descripteurs_classiques(img)
        h = v[-48:]
        assert np.all(h >= 0), "les histogrammes de couleur ne peuvent pas être négatifs"
        for c in range(3):
            s = h[c * 16:(c + 1) * 16].sum()
            assert abs(s - 1) < 1e-3, f"l'histogramme du canal {c} doit être normalisé (somme 1), somme = {s:.3f}"

    def deterministe():
        assert np.allclose(descripteurs_classiques(img), descripteurs_classiques(img)), \
            "deux appels sur la même image doivent donner le même vecteur"

    def taille_independante():
        petite = img.resize((90, 120))
        assert descripteurs_classiques(petite).shape == (1812,), \
            "le vecteur doit avoir la même longueur quelle que soit la taille de l'image"

    def hog_sensible_aux_contours():
        bandes = np.tile(((np.arange(128) // 8) % 2 * 255).astype(np.uint8), (128, 1))  # bandes verticales de 8 pixels
        verticales = descripteurs_classiques(Image.fromarray(np.stack([bandes] * 3, axis=-1)))[:1764]
        horizontales = descripteurs_classiques(Image.fromarray(np.stack([bandes.T] * 3, axis=-1)))[:1764]
        assert np.all(np.isfinite(verticales)) and verticales.min() >= 0 and verticales.max() <= 1 + 1e-6, \
            "les 1764 premières valeurs doivent être le HOG : des valeurs comprises entre 0 et 1"
        assert verticales.max() - verticales.min() > 0.05, \
            "les 1764 premières valeurs sont presque constantes sur une image à bandes : est-ce bien le HOG de l'image ?"
        assert np.abs(verticales - horizontales).mean() > 0.01, \
            "des bandes verticales et des bandes horizontales doivent donner des HOG différents (le HOG décrit l'orientation des contours)"

    return [("vecteur 1D de longueur 1812", forme), ("histogrammes normalisés", histogrammes),
            ("résultat déterministe", deterministe), ("indépendant de la taille de l'image", taille_independante),
            ("le HOG réagit à l'orientation des contours", hog_sensible_aux_contours)]


# ---------------------------------------------------------------- TODO 3
def tests_geler(geler_backbone):
    def tete_seule():
        m = geler_backbone(_effnet(), 0)
        assert m is not None, "la fonction doit renvoyer le modèle"
        assert all(p.requires_grad for p in m.classifier.parameters()), "la tête (classifier) doit rester entraînable"
        assert not any(p.requires_grad for p in m.features.parameters()), \
            "avec n_etages_degeles = 0, aucun paramètre de `features` ne doit être entraînable"

    def deux_etages():
        m = geler_backbone(_effnet(), 2)
        assert all(p.requires_grad for p in m.features[-1].parameters()), "le dernier étage de `features` doit être dégelé"
        assert all(p.requires_grad for p in m.features[-2].parameters()), "l'avant-dernier étage doit être dégelé"
        assert not any(p.requires_grad for p in m.features[-3].parameters()), "le 3e étage en partant de la fin doit rester gelé"
        assert not any(p.requires_grad for p in m.features[0].parameters()), "le premier étage doit rester gelé"

    def idempotent():
        m = geler_backbone(_effnet(), 3)
        m = geler_backbone(m, 1)
        assert not any(p.requires_grad for p in m.features[-2].parameters()), \
            "regeler un modèle déjà dégelé doit fonctionner : commencez par tout geler"

    return [("seule la tête est entraînable (n = 0)", tete_seule), ("deux derniers étages dégelés (n = 2)", deux_etages),
            ("appels successifs cohérents", idempotent)]


# ---------------------------------------------------------------- TODO 4
def tests_optimiseur(creer_optimiseur, geler_backbone):
    def deux_groupes():
        m = geler_backbone(_effnet(), 2)
        opt = creer_optimiseur(m, lr_tete=1e-3, lr_backbone=1e-4)
        assert isinstance(opt, torch.optim.Optimizer), "on attend un optimiseur PyTorch"
        lrs = sorted(g["lr"] for g in opt.param_groups)
        assert lrs == [1e-4, 1e-3], f"on attend deux groupes de learning rates 1e-4 et 1e-3, obtenu {lrs}"
        g_tete = [g for g in opt.param_groups if g["lr"] == 1e-3][0]
        ids_tete = {id(p) for p in m.classifier.parameters()}
        assert {id(p) for p in g_tete["params"]} == ids_tete, "le groupe à 1e-3 doit contenir exactement les paramètres de la tête"

    def que_les_entrainables():
        m = geler_backbone(_effnet(), 2)
        opt = creer_optimiseur(m, lr_tete=1e-3, lr_backbone=1e-4)
        dans_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        entrainables = {id(p) for p in m.parameters() if p.requires_grad}
        assert dans_opt == entrainables, "l'optimiseur doit contenir tous les paramètres entraînables, et seulement eux"

    def tete_seule():
        m = geler_backbone(_effnet(), 0)
        opt = creer_optimiseur(m, lr_tete=1e-3, lr_backbone=1e-4)
        non_vides = [g for g in opt.param_groups if len(g["params"]) > 0]
        assert len(non_vides) == 1 and non_vides[0]["lr"] == 1e-3, \
            "si le backbone est entièrement gelé, il ne reste qu'un groupe (la tête)"

    return [("deux groupes avec les bons learning rates", deux_groupes),
            ("uniquement les paramètres entraînables", que_les_entrainables), ("cas backbone entièrement gelé", tete_seule)]


# ---------------------------------------------------------------- TODO 5
def tests_entrainement(entrainer_une_epoque):
    def donnees():
        g = torch.Generator().manual_seed(0)
        x = torch.randn(256, 10, generator=g)
        y = (x[:, 0] + x[:, 1] > 0).long()
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=False)

    def renvoie_un_float():
        torch.manual_seed(0)
        m = nn.Linear(10, 2)
        perte = entrainer_une_epoque(m, donnees(), torch.optim.SGD(m.parameters(), lr=0.1), nn.CrossEntropyLoss(),
                                     torch.device("cpu"))
        assert isinstance(perte, float), f"on attend un float (la perte moyenne), obtenu {type(perte).__name__}"

    def la_perte_baisse():
        torch.manual_seed(0)
        m = nn.Linear(10, 2)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        pertes = [entrainer_une_epoque(m, donnees(), opt, nn.CrossEntropyLoss(), torch.device("cpu")) for _ in range(5)]
        assert pertes[-1] < pertes[0] * 0.8, f"la perte devrait baisser nettement sur un problème simple : {np.round(pertes, 3)}"

    def les_poids_changent():
        torch.manual_seed(0)
        m = nn.Linear(10, 2)
        avant = copy.deepcopy(m.weight.data)
        entrainer_une_epoque(m, donnees(), torch.optim.SGD(m.parameters(), lr=0.1), nn.CrossEntropyLoss(), torch.device("cpu"))
        assert not torch.allclose(avant, m.weight.data), "les poids n'ont pas changé : appelez-vous optimiseur.step() ?"

    def mode_train():
        m = nn.Sequential(nn.Linear(10, 8), nn.Dropout(0.5), nn.Linear(8, 2))
        m.eval()
        entrainer_une_epoque(m, donnees(), torch.optim.SGD(m.parameters(), lr=0.1), nn.CrossEntropyLoss(), torch.device("cpu"))
        assert m.training, "le modèle doit être passé en mode entraînement (modele.train()) au début de l'époque"

    def pas_d_accumulation():
        torch.manual_seed(0)
        m = nn.Linear(10, 2)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)  # lr nul : les poids ne bougent pas
        chargeur = donnees()
        entrainer_une_epoque(m, chargeur, opt, nn.CrossEntropyLoss(), torch.device("cpu"))
        g = m.weight.grad
        if g is None or g.abs().sum() == 0:
            return  # zero_grad appelé après step : correct aussi
        x, y = list(chargeur)[-1]
        ref = copy.deepcopy(m)
        ref.zero_grad()
        nn.CrossEntropyLoss()(ref(x), y).backward()
        assert torch.allclose(g, ref.weight.grad, atol=1e-5), \
            "les gradients s'accumulent entre les lots : appelez optimiseur.zero_grad() à chaque lot"

    def moyenne_sur_les_exemples():
        # 40 exemples, lots de 32 : le dernier lot n'en compte que 8, et ils sont très mal classés.
        torch.manual_seed(0)
        x = torch.randn(40, 10)
        y = (x[:, 0] > 0).long()
        x[32:] *= 6
        y[32:] = 1 - y[32:]
        chargeur = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=False)
        m = nn.Linear(10, 2)
        with torch.no_grad():
            m.weight.zero_()
            m.bias.zero_()
            m.weight[1, 0] = 1.0
        critere = nn.CrossEntropyLoss()
        attendu = critere(m(x), y).item()
        # learning rate nul : les poids ne bougent pas, la perte de l'époque est donc celle de l'ensemble des exemples
        perte = entrainer_une_epoque(m, chargeur, torch.optim.SGD(m.parameters(), lr=0.0), critere, torch.device("cpu"))
        assert abs(perte - attendu) < 1e-3 * max(1.0, abs(attendu)), (
            f"perte renvoyée {perte:.3f}, perte moyenne sur les 40 exemples {attendu:.3f}. Pondérez la perte de chaque lot "
            "par sa taille avant de faire la moyenne : le dernier lot est plus petit que les autres.")

    return [("renvoie la perte moyenne (float)", renvoie_un_float), ("la perte diminue", la_perte_baisse),
            ("les poids sont mis à jour", les_poids_changent), ("mode entraînement activé", mode_train),
            ("gradients remis à zéro à chaque lot", pas_d_accumulation),
            ("moyenne sur les exemples, pas sur les lots", moyenne_sur_les_exemples)]


# ---------------------------------------------------------------- TODO 6
def tests_confusions(paires_les_plus_confondues):
    noms = ["a", "b", "c", "d"]
    m = np.array([[10, 3, 0, 1],
                  [7, 8, 0, 0],
                  [0, 2, 12, 5],
                  [0, 0, 1, 9]])

    def ordre():
        r = paires_les_plus_confondues(m, noms, k=3)
        assert isinstance(r, list) and len(r) == 3, "on attend une liste de k éléments"
        assert [tuple(x) for x in r] == [("b", "a", 7), ("c", "d", 5), ("a", "b", 3)], \
            f"résultat {r}, attendu [('b','a',7), ('c','d',5), ('a','b',3)] (vraie classe, classe prédite, nombre)"

    def pas_la_diagonale():
        r = paires_les_plus_confondues(m, noms, k=10)
        assert all(v != p for v, p, _ in r), "la diagonale (bonnes prédictions) ne doit pas apparaître"
        assert all(n > 0 for _, _, n in r), "n'incluez pas les paires jamais confondues (nombre = 0)"

    return [("paires triées par nombre décroissant", ordre), ("ni diagonale ni zéros", pas_la_diagonale)]


# ---------------------------------------------------------------- TODO 7
def _petit_reseau():
    """Réseau minuscule à poids fixés, sans pooling global : le gradient n'est pas le même d'une position à l'autre.

    La couche observée renvoie l'image telle quelle (2 canaux de 2 x 2), puis une couche linéaire donne 2 scores."""
    couche = nn.Conv2d(2, 2, kernel_size=1, bias=False)
    tete = nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        couche.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
        tete.weight.copy_(torch.tensor([[1.0, 0.0, 2.0, 1.0, -2.0, 0.0, -1.0, -1.0],
                                        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
    image = torch.tensor([[[[1.0, 3.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, 1.0]]]])
    return nn.Sequential(couche, nn.Flatten(), tete), couche, image


def tests_gradcam(grad_cam):
    import torchvision

    def contexte():
        torch.manual_seed(0)
        m = torchvision.models.resnet18(num_classes=10).eval()
        x = torch.randn(1, 3, 224, 224)
        return m, m.layer4, x

    def forme_et_bornes():
        m, couche, x = contexte()
        c = grad_cam(m, couche, x, classe=3)
        assert isinstance(c, np.ndarray), "on attend un tableau numpy"
        assert c.shape == (7, 7), f"forme {c.shape} : la carte doit avoir la taille spatiale de la couche (7, 7)"
        assert c.min() >= 0 and abs(c.max() - 1) < 1e-4, "la carte doit être normalisée entre 0 et 1 (ReLU, puis (carte - min) / (max - min))"

    def valeurs_classe_0():
        m, couche, x = _petit_reseau()
        c = np.asarray(grad_cam(m, couche, x, classe=0), dtype=float)
        attendu = np.array([[0.0, 1.0], [0.5, 0.0]])
        assert c.shape == (2, 2), f"forme {c.shape} : la carte doit avoir la taille spatiale de la couche (2, 2)"
        assert np.allclose(c, attendu, atol=1e-3), (
            f"sur un petit réseau dont on connaît les poids, la carte attendue pour la classe 0 est {attendu.tolist()}, "
            f"obtenu {np.round(c, 2).tolist()}. Vérifiez l'étape 3 (le poids d'un canal est la MOYENNE de son gradient "
            "sur toutes les positions) et l'étape 4 (ReLU avant la normalisation).")

    def valeurs_classe_1():
        m, couche, x = _petit_reseau()
        c = np.asarray(grad_cam(m, couche, x, classe=1), dtype=float)
        attendu = np.array([[1.0, 1.0], [0.5, 0.0]])
        assert np.allclose(c, attendu, atol=1e-3), (
            f"pour la classe 1, la carte attendue est {attendu.tolist()}, obtenu {np.round(c, 2).tolist()}. "
            "C'est le score de la classe demandée qu'il faut rétropropager.")

    def classe_par_defaut():
        m, couche, x = _petit_reseau()   # ce réseau prédit la classe 1 pour cette image
        c = np.asarray(grad_cam(m, couche, x), dtype=float)
        assert np.allclose(c, [[1.0, 1.0], [0.5, 0.0]], atol=1e-3), \
            "sans classe précisée, la carte doit expliquer la classe prédite (celle de plus grand score)"

    def pas_de_crochet_restant():
        m, couche, x = contexte()
        grad_cam(m, couche, x, classe=1)
        assert len(couche._forward_hooks) == 0 and len(couche._backward_hooks) == 0, \
            "pensez à retirer les crochets (hook.remove()) après usage"

    return [("forme (7, 7) et valeurs dans [0, 1]", forme_et_bornes),
            ("moyenne des gradients par canal, puis ReLU", valeurs_classe_0),
            ("carte de la classe demandée", valeurs_classe_1),
            ("classe prédite par défaut", classe_par_defaut), ("crochets retirés", pas_de_crochet_restant)]
