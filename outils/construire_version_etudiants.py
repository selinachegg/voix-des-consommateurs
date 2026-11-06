"""Construit notebooks/analyse_plaintes_consommateurs.ipynb, la version remise aux étudiants, à partir de la
version complète notebooks/analyse_plaintes_consommateurs_bis.ipynb (écrite par construire_notebook.py).

    python construire_notebook.py && python construire_version_etudiants.py

La version étudiants ne lit que la base réelle : plus d'interrupteur BASE_FICTIVE, plus de vérifications propres
à la base fictive, plus de partie 5 (incidents cachés) ni de partie 8 (changer de base). Les règles de ce qui peut
sortir de Kaggle sont gardées, dans une courte section finale. La version bis garde tout, pour réutiliser un jour
la base fictive.
"""
import json
import pathlib
import re

ICI = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
SOURCE = ICI / "analyse_plaintes_consommateurs_bis.ipynb"
SORTIE = ICI / "analyse_plaintes_consommateurs.ipynb"
VALEURS = {"BASE_FICTIVE": False, "AFFICHER_TEXTES": False}


def resoudre_si(code):
    """Remplace chaque bloc « if BASE_FICTIVE: ... [else: ...] » par la branche qui s'exécute sur la base réelle."""
    lignes, sortie, i = code.split("\n"), [], 0
    motif = re.compile(r"^(\s*)if (not )?(BASE_FICTIVE|AFFICHER_TEXTES):\s*$")
    while i < len(lignes):
        m = motif.match(lignes[i])
        if not m:
            sortie.append(lignes[i]); i += 1; continue
        marge, vrai = m.group(1), VALEURS[m.group(3)] != bool(m.group(2))
        corps, i = [], i + 1
        while i < len(lignes) and (lignes[i].strip() == "" or len(lignes[i]) - len(lignes[i].lstrip()) > len(marge)):
            corps.append(lignes[i]); i += 1
        sinon = []
        if i < len(lignes) and lignes[i] == marge + "else:":
            i += 1
            while i < len(lignes) and (lignes[i].strip() == "" or len(lignes[i]) - len(lignes[i].lstrip()) > len(marge)):
                sinon.append(lignes[i]); i += 1
        garde = corps if vrai else sinon
        retrait = min((len(l) - len(l.lstrip()) for l in garde if l.strip()), default=len(marge))
        sortie += [marge + l[retrait:] if l.strip() else "" for l in garde]
    return "\n".join(sortie)


REMPLACEMENTS_CODE = [
    ("""BASE_FICTIVE = True                       # ce dépôt ne contient que la base fictive
NOM_FICHIER = "plaintes_fictives.xlsx"

# Sur la base réelle, aucun verbatim ne s'affiche : seulement des chiffres d'ensemble.
AFFICHER_TEXTES = BASE_FICTIVE""",
     """NOM_FICHIER = "plaintes_fictives.xlsx"

# Base fictive, de même forme que l'export réel.
AFFICHER_TEXTES = False"""),
    ("# 0.1  Choisir la base : la seule cellule à modifier pour changer de base", "# 0.1  La base des plaintes"),
    (', Path.cwd().parent / "donnees_fictives"]', "]"),
    ('print("Base :", "fictive" if BASE_FICTIVE else "RÉELLE (aucun verbatim ne sera affiché)")',
     'print("Base : fictive")'),
    ('"""Affiche quelques verbatims, seulement sur la base fictive."""', '"""N\'affiche jamais de verbatim : la base est confidentielle."""'),
    ("EXPORTER_MOTS = BASE_FICTIVE   # base réelle : True seulement avec l'accord écrit de l'entreprise",
     "EXPORTER_MOTS = False   # True seulement avec l'accord écrit de l'entreprise"),
    ('"base": "fictive" if BASE_FICTIVE else "réelle",', '"base": "réelle",'),
]
REMPLACEMENTS_REGEX = [
    (r'\(f"rappel incidents[^\n]*\n\s*if BASE_FICTIVE else (f"stabilité \{m\[\'stabilite\'\]:\.2f\}")\)', r"(\1)"),
    (r' \+ \\\n\s*\(\["rappel_incidents", "alertes_retrouvees", "fausses_alertes", "ari_verite"\] if BASE_FICTIVE else \[\]\)', ""),
]
REMPLACEMENTS_TEXTE = [
    ("""**Les données.** Le notebook lit par défaut la **base réelle** des plaintes, couverte par un accord de confidentialité que chaque participant a signé. Elle ne quitte jamais Kaggle : sur elle, le notebook n'affiche aucun verbatim, seulement des chiffres d'ensemble. Une **base fictive** de même forme (mêmes colonnes, même disposition du fichier, textes inventés, marques numérotées) sert à s'entraîner : des incidents y sont cachés volontairement, ce qui permet de mesurer, en partie 5, si la méthode les retrouve.""",
     """**Les données.** Le notebook lit la **base réelle** des plaintes, couverte par un accord de confidentialité que chaque participant a signé. Elle ne quitte jamais Kaggle, et le notebook n'en affiche aucun verbatim : seulement des chiffres d'ensemble."""),
    ("| | 5 | vérifier, sur la base fictive seulement, que la méthode retrouve les incidents cachés |\n", ""),
    ("Parties 0 à 6. Un seul outil", "Parties 0 à 4, puis 6. Un seul outil"),
    ("| 8 | changer de base : réelle ou fictive |", "| 8 | ce qui peut sortir de Kaggle |"),
    ("La cellule 0.1 est la seule à modifier, et seulement pour changer de base.",
     "La cellule 0.1 donne le nom du fichier de la base ; il n'y a rien à modifier."),
    ("- l'échantillon compte 1 301 plaintes, mais la livraison complète en comptera beaucoup plus.",
     "- l'échantillon compte environ 1 300 plaintes, mais la livraison complète en comptera beaucoup plus."),
    ("et s'arrêterait à 1 301 lignes, sans aucun message.", "et s'arrêterait à un nombre de lignes fixé d'avance, sans aucun message."),
    ("Le graphique par période montre des bosses, l'été 2025 et au printemps 2026 dans la base fictive. Est-ce un incident ?",
     "Le graphique par période montre des bosses. Est-ce un incident ?"),
    ("ont été choisis sur la base fictive.", "ont été choisis sur une base d'entraînement."),
    (" Sur la base fictive, on compare aussi aux incidents cachés : **rappel** de chaque groupe caché (la part de ses plaintes réunies dans un même problème), alertes retrouvées et fausses alertes.", ""),
]


def main():
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cellules = []
    saute = False
    for c in nb["cells"]:
        s = "".join(c["source"])
        tete = s.lstrip()
        if c["cell_type"] == "markdown" and tete.startswith("## Partie 5"):
            saute = True
        elif c["cell_type"] == "markdown" and tete.startswith("## Partie 6"):
            saute = False
        if saute:
            continue
        if c["cell_type"] == "markdown" and tete.startswith("## Partie 8"):
            # on ne garde que les règles de sortie de Kaggle
            regles = s[s.index("**Ce qu'on peut sortir de Kaggle**"):]
            s = "## Partie 8 : ce qui peut sortir de Kaggle\n\n" + regles.replace(" C'est pourquoi `EXPORTER_MOTS` vaut `False` sur la base réelle.", " C'est pourquoi `EXPORTER_MOTS` vaut `False`.")
        if c["cell_type"] == "code":
            for a, b in REMPLACEMENTS_CODE:
                s = s.replace(a, b)
            for a, b in REMPLACEMENTS_REGEX:
                s = re.sub(a, b, s)
            s = resoudre_si(s)
        else:
            for a, b in REMPLACEMENTS_TEXTE:
                s = s.replace(a, b)
        c = dict(c)
        c["source"] = s.splitlines(keepends=True)
        cellules.append(c)
    nb["cells"] = cellules
    texte = json.dumps(nb, ensure_ascii=False, indent=1)
    for interdit in ("BASE_FICTIVE", "verite_incidents", "changer de base"):
        assert interdit not in texte, interdit
    for c in cellules:                         # chaque cellule de code doit rester du Python valide
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            compile("\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!"))), "cellule", "exec")
    SORTIE.write_text(texte, encoding="utf-8")
    print(SORTIE, len(cellules), "cellules")


if __name__ == "__main__":
    main()
