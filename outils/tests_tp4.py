"""Tests automatiques du TD 4. Chaque fonction renvoie une liste (nom, test)."""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------- TODO 1
def tests_cosinus(similarite_cosinus):
    A = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    B = np.array([[5.0, 0.0], [0.0, -1.0]])

    def forme():
        S = similarite_cosinus(A, B)
        assert isinstance(S, np.ndarray) and S.shape == (3, 2), f"on attend un tableau (3, 2) : une ligne par vecteur de A, une colonne par vecteur de B. Obtenu {getattr(S, 'shape', type(S))}"

    def valeurs():
        S = similarite_cosinus(A, B)
        attendu = np.array([[1, 0], [0, -1], [np.sqrt(0.5), -np.sqrt(0.5)]])
        assert np.allclose(S, attendu, atol=1e-6), f"valeurs attendues\n{np.round(attendu, 3)}\nobtenues\n{np.round(S, 3)}"

    def echelle():
        assert np.allclose(similarite_cosinus(A, B), similarite_cosinus(10 * A, 0.1 * B), atol=1e-6), \
            "la similarité cosinus ne dépend pas de la longueur des vecteurs : pensez à les normaliser"

    def bornes():
        rng = np.random.default_rng(0)
        S = similarite_cosinus(rng.normal(size=(20, 8)), rng.normal(size=(15, 8)))
        assert S.min() >= -1 - 1e-6 and S.max() <= 1 + 1e-6, "une similarité cosinus est toujours comprise entre -1 et 1"

    return [("forme (n, m)", forme), ("valeurs sur un exemple", valeurs), ("insensible à la longueur des vecteurs", echelle),
            ("valeurs entre -1 et 1", bornes)]


# ---------------------------------------------------------------- TODO 2
def tests_zero_shot(classer_zero_shot):
    images = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 3.0]])
    textes = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])

    def probabilites():
        P = classer_zero_shot(images, textes)
        assert P.shape == (3, 3), f"forme attendue (3 images, 3 classes), obtenue {P.shape}"
        assert np.allclose(P.sum(axis=1), 1, atol=1e-5), "chaque ligne doit sommer à 1 (softmax sur les classes)"
        assert (P >= 0).all(), "des probabilités ne peuvent pas être négatives"

    def bonne_classe():
        P = classer_zero_shot(images, textes)
        assert list(P.argmax(axis=1)) == [0, 1, 2], f"chaque image doit être associée au texte le plus proche : attendu [0, 1, 2], obtenu {list(P.argmax(axis=1))}"

    def temperature():
        doux = classer_zero_shot(images, textes, temperature=1)
        net = classer_zero_shot(images, textes, temperature=100)
        assert net.max(axis=1).mean() > doux.max(axis=1).mean() + 0.2, \
            "les similarités doivent être multipliées par la température AVANT le softmax : plus elle est grande, plus la décision est tranchée"

    def valeur_exacte():
        P = classer_zero_shot(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), temperature=2)
        attendu = np.exp(2) / (np.exp(2) + 1)
        assert np.isclose(P[0, 0], attendu, atol=1e-5), f"softmax(2 x [1, 0]) : attendu {attendu:.4f} pour la première classe, obtenu {P[0, 0]:.4f}"

    def insensible_aux_longueurs():
        # Le second texte est un vecteur très long, mais il pointe dans une autre direction que l'image.
        P = classer_zero_shot(np.array([[3.0, 0.5]]), np.array([[1.0, 0.0], [20.0, 20.0]]))
        assert P.argmax(axis=1)[0] == 0, ("un texte dont l'embedding est très long ne doit pas gagner d'office : "
                                          "utilisez la similarité COSINUS (votre TODO 1), pas le produit scalaire brut")
        Q = classer_zero_shot(np.array([[30.0, 5.0]]), np.array([[0.1, 0.0], [2.0, 2.0]]))
        assert np.allclose(P, Q, atol=1e-6), "multiplier un embedding par une constante ne doit rien changer aux probabilités"

    return [("des probabilités par classe", probabilites), ("la classe la plus proche l'emporte", bonne_classe),
            ("effet de la température", temperature), ("valeur exacte sur un exemple", valeur_exacte),
            ("insensible à la longueur des embeddings", insensible_aux_longueurs)]


# ---------------------------------------------------------------- TODO 3
def tests_recherche(rechercher):
    corpus = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]])

    def ordre():
        r = [int(i) for i in rechercher(np.array([2.0, 0.0]), corpus, k=3)]
        assert r == [0, 1, 4], f"les 3 plus proches de [2, 0] sont, dans l'ordre, les indices [0, 1, 4]. Obtenu {r}"

    def nombre():
        assert len(rechercher(np.array([0.0, 1.0]), corpus, k=2)) == 2, "la fonction doit renvoyer exactement k indices"
        assert len(rechercher(np.array([0.0, 1.0]), corpus, k=5)) == 5, "avec k = taille du corpus, tous les indices sont renvoyés"

    def cosinus_et_non_produit():
        c = np.array([[100.0, 100.0], [1.0, 0.0]])
        r = [int(i) for i in rechercher(np.array([1.0, 0.0]), c, k=1)]
        assert r == [1], "le classement doit utiliser la similarité cosinus, pas le produit scalaire brut (un long vecteur ne doit pas gagner d'office)"

    return [("résultats triés du plus proche au plus lointain", ordre), ("k résultats", nombre),
            ("classement par similarité cosinus", cosinus_et_non_produit)]


# ---------------------------------------------------------------- TODO 4
def tests_precision_k(precision_a_k):
    pertinents = np.array([True, False, True, True, False, False])

    def cas_general():
        v = precision_a_k([0, 1, 2, 4], pertinents, k=4)
        assert np.isclose(v, 0.5), f"parmi les 4 premiers résultats [0, 1, 2, 4], deux sont pertinents : attendu 0.5, obtenu {v}"

    def seulement_les_k_premiers():
        v = precision_a_k([1, 4, 0, 2, 3], pertinents, k=2)
        assert v == 0, f"seuls les k premiers résultats comptent : attendu 0, obtenu {v}"

    def parfait():
        assert precision_a_k([0, 2, 3], pertinents, k=3) == 1, "trois résultats pertinents sur trois : attendu 1"

    return [("cas général", cas_general), ("seuls les k premiers comptent", seulement_les_k_premiers), ("cas parfait", parfait)]


# ---------------------------------------------------------------- TODO 5
def _planning():
    betas = torch.linspace(1e-4, 0.04, 200)
    alphas = 1 - betas
    return betas, alphas, torch.cumprod(alphas, 0)


def tests_bruiter(bruiter):
    betas, alphas, ac = _planning()

    def formule():
        torch.manual_seed(0)
        x0, bruit, t = torch.randn(4, 2), torch.randn(4, 2), torch.tensor([0, 50, 120, 199])
        attendu = ac[t].sqrt()[:, None] * x0 + (1 - ac[t]).sqrt()[:, None] * bruit
        assert torch.allclose(bruiter(x0, t, ac, bruit), attendu, atol=1e-6), \
            "x_t = racine(alpha_cumule[t]) * x0 + racine(1 - alpha_cumule[t]) * bruit"

    def images():
        torch.manual_seed(0)
        x0, bruit, t = torch.randn(3, 1, 8, 8), torch.randn(3, 1, 8, 8), torch.tensor([10, 100, 190])
        xt = bruiter(x0, t, ac, bruit)
        assert xt.shape == x0.shape, f"la forme doit être conservée : attendu {tuple(x0.shape)}, obtenu {tuple(xt.shape)}"
        attendu = ac[t].sqrt()[:, None, None, None] * x0 + (1 - ac[t]).sqrt()[:, None, None, None] * bruit
        assert torch.allclose(xt, attendu, atol=1e-6), ("la fonction doit marcher pour des points (lot, 2) comme pour des images "
                                                        "(lot, 1, H, W) : adaptez la forme du coefficient au nombre de dimensions de x0")

    def extremes():
        x0, bruit = torch.ones(2, 2), torch.zeros(2, 2)
        debut = bruiter(x0, torch.tensor([0, 0]), ac, bruit)
        fin = bruiter(x0, torch.tensor([199, 199]), ac, bruit)
        assert debut.mean() > 0.99, "à t = 0, il reste presque tout le signal"
        assert fin.mean() < 0.2, "à t = 199, il ne reste presque plus de signal"

    return [("formule sur des points 2D", formule), ("fonctionne aussi sur des images", images), ("début et fin du processus", extremes)]


# ---------------------------------------------------------------- TODO 6
def tests_debruitage(etape_debruitage):
    betas, alphas, ac = _planning()

    def formule():
        torch.manual_seed(0)
        xt, eps, z, t = torch.randn(5, 2), torch.randn(5, 2), torch.randn(5, 2), 77
        moyenne = (xt - betas[t] / (1 - ac[t]).sqrt() * eps) / alphas[t].sqrt()
        attendu = moyenne + betas[t].sqrt() * z
        assert torch.allclose(etape_debruitage(xt, t, eps, betas, alphas, ac, z), attendu, atol=1e-6), \
            "x_(t-1) = (x_t - beta_t / racine(1 - alpha_cumule_t) * bruit_predit) / racine(alpha_t) + racine(beta_t) * z"

    def derniere_etape():
        torch.manual_seed(0)
        xt, eps = torch.randn(5, 2), torch.randn(5, 2)
        a = etape_debruitage(xt, 0, eps, betas, alphas, ac, torch.randn(5, 2))
        b = etape_debruitage(xt, 0, eps, betas, alphas, ac, torch.randn(5, 2))
        assert torch.allclose(a, b), "à la dernière étape (t = 0), on n'ajoute plus de bruit : le résultat ne doit pas dépendre de z"

    def images():
        torch.manual_seed(0)
        xt, eps, z = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
        assert etape_debruitage(xt, 30, eps, betas, alphas, ac, z).shape == xt.shape, "la forme doit être conservée"

    def retire_le_bruit():
        torch.manual_seed(0)
        x0, bruit, t = torch.zeros(2000, 2), torch.randn(2000, 2), 150
        xt = ac[t].sqrt() * x0 + (1 - ac[t]).sqrt() * bruit
        suivant = etape_debruitage(xt, t, bruit, betas, alphas, ac, torch.zeros_like(xt))
        assert suivant.std() < xt.std(), "avec le vrai bruit comme prédiction, une étape doit rapprocher les points des données (ici l'origine)"

    return [("formule de l'étape", formule), ("pas de bruit ajouté à t = 0", derniere_etape), ("fonctionne sur des images", images),
            ("l'étape retire bien du bruit", retire_le_bruit)]
