"""Récupère le nouveau TD (ou une mise à jour) depuis le dépôt de l'enseignant.

À lancer dans le dossier du dépôt, à chaque nouvelle séance :

    uv run python recuperer_td.py

Le script :
1. enregistre votre travail en cours (git add puis git commit), pour ne rien perdre ;
2. récupère les nouveaux fichiers du dépôt de l'enseignant (git fetch puis git merge) ;
3. pousse le tout sur votre dépôt de rendu (git push), si vous en avez un.

Votre travail n'est jamais écrasé : si une mise à jour touche un fichier que vous avez modifié, le script
annule la récupération, laisse vos fichiers tels quels et vous dit quoi faire.
"""
import subprocess
import sys

DEPOT_PROF = "FranklinB13/applications-modeles-ia-td"
# Fichiers fournis, que les étudiants ne modifient pas : en cas de conflit, on reprend la version de l'enseignant.
FOURNIS = ("outils/", "modeles/", "pyproject.toml", "uv.lock", "README.md", "verifier_installation.py",
           "recuperer_td.py", ".gitignore", ".python-version", "projet/brique_modele.py")


def git(*args, verifier=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if verifier and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} a échoué :\n{(r.stderr or r.stdout).strip()}")
    return r


def arreter(message):
    print("\n" + message)
    sys.exit(1)


def distants():
    """Dictionnaire nom -> URL des dépôts distants (en lecture)."""
    res = {}
    for ligne in git("remote", "-v").stdout.splitlines():
        parties = ligne.split()
        if len(parties) >= 2 and parties[-1:] != ["(push)"]:
            res[parties[0]] = parties[1]
    return res


def main():
    try:
        git("rev-parse", "--is-inside-work-tree")
    except (RuntimeError, FileNotFoundError):
        arreter("Lancez ce script dans le dossier du dépôt cloné (là où se trouve README.md), avec git installé.")

    d = distants()
    prof = next((nom for nom, url in d.items() if DEPOT_PROF in url), None)
    if prof is None:
        arreter("Le dépôt de l'enseignant n'est pas relié à ce dossier. Ajoutez-le une fois pour toutes :\n"
                f"    git remote add prof https://github.com/{DEPOT_PROF}\npuis relancez le script.")
    rendu = next((nom for nom in d if nom != prof), None)

    # 1. Enregistrer le travail en cours
    if git("status", "--porcelain").stdout.strip():
        git("add", "-A")
        r = git("commit", "-m", "Travail enregistré avant de récupérer le TD suivant", verifier=False)
        if r.returncode != 0:
            if "user.email" in (r.stderr + r.stdout) or "identity" in (r.stderr + r.stdout):
                arreter("Git ne connaît pas encore votre nom. Faites-le une fois pour toutes, puis relancez :\n"
                        '    git config --global user.name "Prénom Nom"\n'
                        '    git config --global user.email "votre.adresse@cyu.fr"')
            arreter("L'enregistrement de votre travail a échoué :\n" + (r.stderr or r.stdout).strip())
        print("Votre travail en cours est enregistré (commit).")

    # 2. Récupérer les nouveautés de l'enseignant
    print("Récupération des nouveautés de l'enseignant...")
    try:
        git("fetch", prof)
    except RuntimeError as e:
        arreter(f"Impossible de joindre le dépôt de l'enseignant (connexion internet ?).\n{e}")
    if git("rev-list", "--count", f"HEAD..{prof}/main").stdout.strip() == "0":
        print("Rien de nouveau : vous avez déjà tous les TD publiés.")
    else:
        avant = git("rev-parse", "HEAD").stdout.strip()
        r = git("merge", "--no-edit", f"{prof}/main", verifier=False)
        if r.returncode != 0:
            conflits = git("diff", "--name-only", "--diff-filter=U").stdout.split()
            if not conflits:
                git("merge", "--abort", verifier=False)
                arreter("La récupération a échoué, vos fichiers n'ont pas changé :\n" + (r.stderr or r.stdout).strip())
            a_vous = [f for f in conflits if not f.startswith(FOURNIS)]
            if a_vous:
                git("merge", "--abort")
                arreter("L'enseignant a modifié un fichier que vous avez aussi modifié :\n    " + "\n    ".join(a_vous)
                        + "\nRien n'a changé dans votre dossier et votre travail est intact. Prévenez l'enseignant.")
            for f in conflits:   # fichiers fournis : on reprend la version de l'enseignant
                git("checkout", f"{prof}/main", "--", f)
                git("add", f)
            git("commit", "--no-edit")
            print("Fichiers fournis remis dans la version de l'enseignant : " + ", ".join(conflits))
        nouveaux = git("diff", "--name-only", "--diff-filter=A", avant, "HEAD").stdout.split()
        modifies = git("diff", "--name-only", "--diff-filter=M", avant, "HEAD").stdout.split()
        if nouveaux:
            print("Nouveaux fichiers :\n    " + "\n    ".join(nouveaux))
        if modifies:
            print("Fichiers mis à jour :\n    " + "\n    ".join(modifies))
        notebooks = [f for f in nouveaux if f.startswith("tp") and f.endswith(".py")]
        for f in notebooks:
            print(f"\nPour ouvrir le nouveau TD :\n    uv run marimo edit {f}")

    # 3. Pousser sur le dépôt de rendu
    if rendu is None:
        print("\nAucun dépôt de rendu n'est configuré : votre travail reste sur cet ordinateur.\n"
              "Créez-le (README, Installation, étape 3) pour qu'il soit sauvegardé et évalué.")
        return
    r = git("push", rendu, "HEAD", verifier=False)
    if r.returncode == 0:
        print(f"\nTout est envoyé sur votre dépôt de rendu ({rendu}).")
    else:
        print(f"\nLe TD est récupéré, mais l'envoi sur votre dépôt de rendu a échoué :\n{(r.stderr or r.stdout).strip()}\n"
              "Réessayez plus tard avec : git push")


if __name__ == "__main__":
    main()
