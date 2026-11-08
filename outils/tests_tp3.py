"""Tests automatiques du TD 3. Chaque fonction renvoie une liste (nom, test)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- TODO 1
def tests_decoupage(decouper_et_normaliser):
    rng = np.random.default_rng(0)
    v = rng.normal(5, 2, (100, 3)) + np.linspace(0, 10, 100)[:, None]  # série avec tendance

    def tailles():
        tr, va, te, mu, sd = decouper_et_normaliser(v, 60, 20, 20)
        assert tr.shape == (60, 3) and va.shape == (20, 3) and te.shape == (20, 3), \
            f"tailles attendues (60, 3), (20, 3), (20, 3), obtenues {tr.shape}, {va.shape}, {te.shape}"

    def chronologique():
        tr, va, te, mu, sd = decouper_et_normaliser(v, 60, 20, 20)
        assert np.allclose(tr * sd + mu, v[:60]), "l'entraînement doit être les 60 PREMIERS points, dans l'ordre (pas de mélange)"
        assert np.allclose(va * sd + mu, v[60:80]), "la validation doit suivre immédiatement l'entraînement"
        assert np.allclose(te * sd + mu, v[80:100]), "le test doit être la fin de la série"

    def stats_du_train():
        tr, va, te, mu, sd = decouper_et_normaliser(v, 60, 20, 20)
        assert np.allclose(mu, v[:60].mean(0)) and np.allclose(sd, v[:60].std(0)), \
            "moyenne et écart-type doivent être calculés sur l'entraînement uniquement, canal par canal"
        assert np.allclose(tr.mean(0), 0, atol=1e-6) and np.allclose(tr.std(0), 1, atol=1e-6), \
            "l'entraînement normalisé doit avoir une moyenne nulle et un écart-type de 1"

    def pas_de_fuite():
        tr, va, te, mu, sd = decouper_et_normaliser(v, 60, 20, 20)
        assert abs(te.mean()) > 0.5, ("le test semble normalisé avec ses propres statistiques : c'est une fuite "
                                       "d'information. Utilisez la moyenne et l'écart-type de l'entraînement.")

    return [("tailles des trois parties", tailles), ("découpage chronologique", chronologique),
            ("statistiques calculées sur l'entraînement", stats_du_train), ("pas de fuite vers le test", pas_de_fuite)]


# ---------------------------------------------------------------- TODO 2
def tests_fenetres(creer_fenetres):
    s = np.arange(20, dtype=np.float32)

    def formes():
        X, Y = creer_fenetres(s, 5, 3)
        assert X.shape == (13, 5) and Y.shape == (13, 3), \
            f"avec 20 points, entrée 5 et horizon 3 : 20 - 5 - 3 + 1 = 13 fenêtres. Obtenu X {X.shape}, Y {Y.shape}"

    def contenu():
        X, Y = creer_fenetres(s, 5, 3)
        assert np.array_equal(X[0], [0, 1, 2, 3, 4]) and np.array_equal(Y[0], [5, 6, 7]), \
            f"première fenêtre attendue X=[0..4], Y=[5, 6, 7], obtenu X={X[0]}, Y={Y[0]}"
        assert np.array_equal(X[-1], [12, 13, 14, 15, 16]) and np.array_equal(Y[-1], [17, 18, 19]), \
            f"dernière fenêtre attendue X=[12..16], Y=[17, 18, 19], obtenu X={X[-1]}, Y={Y[-1]}"

    def cible_apres_entree():
        X, Y = creer_fenetres(s, 5, 3)
        assert np.all(Y[:, 0] == X[:, -1] + 1), "la cible doit commencer juste après la fin de l'entrée (ni chevauchement, ni trou)"

    return [("nombre et forme des fenêtres", formes), ("contenu des fenêtres", contenu),
            ("la cible suit l'entrée", cible_apres_entree)]


# ---------------------------------------------------------------- TODO 3
def tests_baselines(prevision_naive, prevision_saisonniere):
    X = np.arange(60, dtype=np.float32).reshape(1, 60)

    def naive():
        p = prevision_naive(X, 4)
        assert p.shape == (1, 4), f"forme attendue (1, 4), obtenue {p.shape}"
        assert np.array_equal(p[0], [59, 59, 59, 59]), f"on répète la dernière valeur observée : attendu [59, 59, 59, 59], obtenu {p[0]}"

    def saison_courte():
        p = prevision_saisonniere(X, 4, periode=24)
        assert p.shape == (1, 4), f"forme attendue (1, 4), obtenue {p.shape}"
        assert np.array_equal(p[0], [36, 37, 38, 39]), \
            f"la prévision à t+h vaut la valeur observée à t+h-24 : attendu [36, 37, 38, 39], obtenu {p[0]}"

    def saison_complete():
        p = prevision_saisonniere(X, 24, periode=24)
        assert np.array_equal(p[0], np.arange(36, 60)), "avec horizon = période, on recopie les 24 dernières valeurs"

    def plusieurs_fenetres():
        X2 = np.stack([np.arange(60), np.arange(100, 160)]).astype(np.float32)
        assert np.array_equal(prevision_naive(X2, 2), [[59, 59], [159, 159]]), "la fonction doit traiter toutes les fenêtres du lot"

    return [("prévision naïve", naive), ("prévision saisonnière, horizon court", saison_courte),
            ("prévision saisonnière, horizon = période", saison_complete), ("plusieurs fenêtres à la fois", plusieurs_fenetres)]


# ---------------------------------------------------------------- TODO 4
def tests_conv_causale(ConvCausale):
    def longueur():
        torch.manual_seed(0)
        for k, d in [(3, 1), (3, 4), (5, 2)]:
            y = ConvCausale(2, 6, k, d)(torch.randn(4, 2, 50))
            assert y.shape == (4, 6, 50), f"noyau {k}, dilatation {d} : la sortie doit garder la longueur 50, obtenu {tuple(y.shape)}"

    def causalite():
        torch.manual_seed(0)
        c = ConvCausale(1, 1, 3, 2).eval()
        x = torch.randn(1, 1, 40)
        x2 = x.clone()
        x2[:, :, 25:] += 10  # on modifie le futur à partir de t = 25
        y, y2 = c(x), c(x2)
        assert torch.allclose(y[:, :, :25], y2[:, :, :25]), \
            "modifier le futur change les sorties passées : la convolution n'est pas causale. Complétez à GAUCHE uniquement."
        assert not torch.allclose(y[:, :, 25:], y2[:, :, 25:]), "les sorties à partir de t = 25 devraient changer"

    def champ_receptif():
        torch.manual_seed(0)
        c = ConvCausale(1, 1, 3, 4).eval()
        x = torch.zeros(1, 1, 40)
        x2 = x.clone()
        x2[0, 0, 10] = 1
        diff = (c(x2) - c(x)).abs()[0, 0] > 1e-9
        assert set(torch.nonzero(diff).flatten().tolist()) <= {10, 14, 18}, \
            "avec un noyau de 3 et une dilatation de 4, une entrée à t = 10 n'influence que les sorties t = 10, 14 et 18"

    return [("la longueur est conservée", longueur), ("aucune fuite du futur vers le passé", causalite),
            ("dilatation respectée", champ_receptif)]


# ---------------------------------------------------------------- TODO 5
def tests_patches(decouper_en_patches):
    x = torch.arange(20, dtype=torch.float32).reshape(1, 20).repeat(3, 1)

    def forme():
        p = decouper_en_patches(x, 8, 4)
        assert tuple(p.shape) == (3, 4, 8), f"(20 - 8) / 4 + 1 = 4 patches de 8 valeurs : attendu (3, 4, 8), obtenu {tuple(p.shape)}"

    def contenu():
        p = decouper_en_patches(x, 8, 4)
        assert torch.equal(p[0, 0], torch.arange(0, 8.0)), f"premier patch attendu [0..7], obtenu {p[0, 0].tolist()}"
        assert torch.equal(p[0, 1], torch.arange(4, 12.0)), f"deuxième patch attendu [4..11] (décalage de 4), obtenu {p[0, 1].tolist()}"

    def sans_recouvrement():
        p = decouper_en_patches(x, 5, 5)
        assert tuple(p.shape) == (3, 4, 5) and torch.equal(p[0].flatten(), torch.arange(20.0)), \
            "avec taille = pas, les patches ne se recouvrent pas et couvrent toute la série"

    return [("nombre et taille des patches", forme), ("contenu et décalage", contenu), ("cas sans recouvrement", sans_recouvrement)]


# ---------------------------------------------------------------- TODO 6
def tests_norm_fenetre(PrevisionNormalisee):
    class Moyenne(nn.Module):
        def forward(self, x):
            return x.mean(dim=1, keepdim=True).repeat(1, 3)

    def definition():
        torch.manual_seed(0)
        x = torch.randn(5, 12)
        m = PrevisionNormalisee(Moyenne())
        attendu = Moyenne()(x - x[:, -1:]) + x[:, -1:]
        assert torch.allclose(m(x), attendu), "sortie = modele(x - derniere_valeur) + derniere_valeur"

    def invariance():
        torch.manual_seed(0)
        lin = nn.Linear(12, 3)
        m = PrevisionNormalisee(lin)
        x = torch.randn(5, 12)
        assert torch.allclose(m(x + 7.0), m(x) + 7.0, atol=1e-5), \
            "si toute la fenêtre est décalée de +7, la prévision doit être décalée de +7 exactement"

    def forme():
        m = PrevisionNormalisee(nn.Linear(12, 3))
        assert tuple(m(torch.randn(4, 12)).shape) == (4, 3), "la forme de sortie du modèle interne doit être conservée"

    def derniere_valeur():
        class Somme(nn.Module):
            def forward(self, x):
                return x.sum(dim=1, keepdim=True).repeat(1, 2)

        y = PrevisionNormalisee(Somme())(torch.tensor([[1.0, 2.0, 3.0, 6.0]]))
        # fenêtre [1, 2, 3, 6] : on retire 6, le modèle interne voit [-5, -4, -3, 0] (somme -12), puis on rajoute 6
        assert torch.allclose(y, torch.tensor([[-6.0, -6.0]])), (
            f"avec la fenêtre [1, 2, 3, 6] et un modèle interne qui fait la somme, on attend -12 + 6 = -6, obtenu {y[0, 0].item():.1f}. "
            "C'est la DERNIÈRE valeur de la fenêtre qu'on retire puis qu'on rajoute (pas sa moyenne).")

    return [("définition", definition), ("invariance au niveau de la série", invariance), ("forme de sortie", forme),
            ("référence : la dernière valeur observée", derniere_valeur)]


# ---------------------------------------------------------------- TODO 7
def tests_erreur(erreur_reconstruction):
    class Identite(nn.Module):
        def forward(self, x):
            return x

    class Zero(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    X = np.stack([np.ones((6, 2)), 2 * np.ones((6, 2)), np.zeros((6, 2))]).astype(np.float32)

    def parfaite():
        e = erreur_reconstruction(Identite(), X)
        assert isinstance(e, np.ndarray) and e.shape == (3,), f"on attend un tableau numpy de forme (3,), obtenu {type(e).__name__} {getattr(e, 'shape', '')}"
        assert np.allclose(e, 0), "une reconstruction parfaite donne une erreur nulle"

    def valeurs():
        e = erreur_reconstruction(Zero(), X)
        assert np.allclose(e, [1, 4, 0]), f"erreur quadratique moyenne par fenêtre attendue [1, 4, 0], obtenu {np.round(e, 3)}"

    def entree_numpy():
        e = erreur_reconstruction(Zero(), X[:1])
        assert e.shape == (1,) and np.isclose(e[0], 1), "la fonction doit accepter un tableau numpy et le convertir en tenseur"

    def sans_gradient():
        class Sonde(nn.Module):
            def __init__(self):
                super().__init__()
                self.poids = nn.Parameter(torch.ones(1))
                self.gradient_actif = None

            def forward(self, x):
                self.gradient_actif = torch.is_grad_enabled()
                return x * self.poids

        sonde = Sonde()
        e = erreur_reconstruction(sonde, X)
        assert isinstance(e, np.ndarray), "on attend un tableau numpy"
        assert sonde.gradient_actif is False, ("le modèle a été appelé avec le calcul des gradients activé : placez l'appel "
                                               "dans un bloc `with torch.no_grad():` (inutile de garder les gradients pour évaluer)")

    return [("erreur nulle si reconstruction parfaite", parfaite), ("une erreur par fenêtre", valeurs),
            ("entrée numpy acceptée", entree_numpy), ("calcul sans gradient", sans_gradient)]


# ---------------------------------------------------------------- TODO 8
def tests_precision_rappel(precision_rappel):
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    y = np.array([0, 1, 1, 0, 0, 1])

    def cas_general():
        p, r = precision_rappel(scores, y, 0.5)
        assert np.isclose(p, 2 / 3) and np.isclose(r, 2 / 3), \
            f"au seuil 0.5 : 3 alertes dont 2 justes, 3 anomalies dont 2 trouvées. Attendu (0.667, 0.667), obtenu ({p:.3f}, {r:.3f})"

    def seuil_bas():
        p, r = precision_rappel(scores, y, 0.0)
        assert np.isclose(p, 0.5) and np.isclose(r, 1.0), f"seuil très bas : tout est alerte. Attendu (0.5, 1.0), obtenu ({p:.3f}, {r:.3f})"

    def seuil_haut():
        p, r = precision_rappel(scores, y, 0.95)
        assert p == 0 and r == 0, f"aucune alerte : précision et rappel valent 0 (pas d'erreur de division), obtenu ({p}, {r})"

    def strict():
        p, r = precision_rappel(scores, y, 0.7)
        assert np.isclose(p, 1.0) and np.isclose(r, 2 / 3), "une alerte est levée quand le score est STRICTEMENT supérieur au seuil"

    return [("cas général", cas_general), ("seuil très bas", seuil_bas), ("seuil très haut", seuil_haut),
            ("inégalité stricte", strict)]
