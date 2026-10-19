"""Vérifie l'installation et télécharge à l'avance les poids et les données du module.

Usage :
    uv run python verifier_installation.py            # vérifie et télécharge les poids des modèles
    uv run python verifier_installation.py --donnees  # télécharge aussi les jeux des TD (environ 1 Go)
"""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "outils"))

MODULES = ["torch", "torchvision", "timm", "marimo", "numpy", "sklearn", "skimage", "matplotlib", "PIL", "cv2",
           "ultralytics", "pandas", "open_clip", "networkx", "gymnasium", "stable_baselines3"]


def etape(titre):
    print(f"\n== {titre}")


def main():
    erreurs = 0
    etape("Bibliothèques")
    for nom in MODULES:
        try:
            m = importlib.import_module(nom)
            print(f"  [ok] {nom} {getattr(m, '__version__', '')}")
        except ImportError as e:
            print(f"  [manquant] {nom} : {e}")
            erreurs += 1
    if erreurs:
        print("\nLancez `uv sync` dans le dossier du dépôt (voir README), puis relancez avec `uv run`.")
        return 1

    import torch
    import tp_utils as tp

    etape("Matériel")
    print(f"  Python {sys.version.split()[0]}  |  appareil utilisé : {tp.appareil()}  |  threads CPU : {torch.get_num_threads()}")

    etape("Poids pré-entraînés (téléchargés une seule fois)")
    for nom in ["resnet18", "efficientnet_b0", "deit_tiny"]:
        try:
            t = time.perf_counter()
            m = tp.charger_modele(nom)
            with torch.no_grad():
                m(torch.randn(8, 3, 224, 224))
            ms = (time.perf_counter() - t) * 1000
            print(f"  [ok] {nom} ({tp.nombre_parametres(m) / 1e6:.1f} M paramètres, chargement + 8 images : {ms:.0f} ms)")
        except Exception as e:
            print(f"  [échec] {nom} : {e}")
            erreurs += 1

    a_charger = [("YOLOv8n", lambda: tp.charger_yolo("yolov8n.pt")), ("MobileSAM", tp.charger_sam),
                 ("DeepLabV3", tp.charger_deeplab), ("CLIP ViT-B/32 (600 Mo)", tp.charger_clip)]
    # Le U-Net est fourni dans modeles/ avec le TD 4 : avant sa mise en ligne, son absence est normale.
    unet_present = (tp.MODELES / "tp4_ddpm_fashion.pt").exists()
    if unet_present:
        a_charger.append(("U-Net de diffusion fourni", tp.charger_unet_fourni))
    for nom, charge in a_charger:
        try:
            charge()
            print(f"  [ok] {nom}")
        except Exception as e:
            print(f"  [échec] {nom} : {e}")
            erreurs += 1
    if not unet_present:
        print("  [plus tard] U-Net de diffusion fourni : il arrivera avec le TD 4")

    etape("Environnement de simulation (TD 5b)")
    try:
        score = tp.evaluer_politique(lambda obs: int(obs[2] + 0.25 * obs[3] > 0), 2)
        print(f"  [ok] CartPole (score d'une règle simple : {score:.0f})")
    except Exception as e:
        print(f"  [échec] CartPole : {e}")
        erreurs += 1

    etape("Vitesse de la machine")
    try:
        m = tp.charger_modele("efficientnet_b0")
        x = torch.randn(32, 3, 160, 160)
        with torch.no_grad():
            m(x)
            t = time.perf_counter()
            m(x)
        ms = (time.perf_counter() - t) / 32 * 1000
        print(f"  EfficientNet-B0, 160 px : {ms:.1f} ms par image en inférence")
        print("  Repère : en dessous de 25 ms, les TD tiennent confortablement dans les temps.")
    except Exception as e:
        print(f"  [non mesurée] {e}")

    if "--donnees" in sys.argv:
        etape("Données des TD")
        try:
            tp.charger_oxford_pets()
            print("  [ok] Oxford Pets prêt")
            tp.preparer_construction_ppe()
            print("  [ok] Construction-PPE prêt")
            tp.charger_ett()
            print("  [ok] ETTh1 prêt")
            tp.charger_skab()
            print("  [ok] SKAB prêt")
            tp.charger_fashion_mnist()
            print("  [ok] Fashion-MNIST prêt")
            tp.charger_cora()
            print("  [ok] Cora prêt")
        except Exception as e:
            print(f"  [échec] {e}")
            erreurs += 1
    neu = tp.DONNEES / "NEU"
    etape("Données du projet (NEU)")
    print(f"  {'[ok]' if neu.exists() and any(neu.rglob('*.*')) else '[à faire]'} dossier {neu}")

    print("\nTout est prêt." if not erreurs else f"\n{erreurs} problème(s) à corriger.")
    return 1 if erreurs else 0


if __name__ == "__main__":
    sys.exit(main())
