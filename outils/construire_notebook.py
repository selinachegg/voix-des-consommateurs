# Copyright (c) 2026 Selina Cheggour. Tous droits réservés. Voir le fichier LICENSE.
"""Construit notebooks/analyse_plaintes_consommateurs_bis.ipynb, la version complète (base réelle ou fictive),
cellule par cellule. La version remise aux étudiants en est tirée par construire_version_etudiants.py.

    python construire_notebook.py

Le notebook est fait pour Kaggle (carte graphique et accès internet activés). Il trouve le fichier de
données tout seul dans /kaggle/input. Il lit la base réelle par défaut ; pour s'entraîner sur la base fictive,
on ne change que NOM_FICHIER et BASE_FICTIVE dans la cellule 0.1.
"""
import json
import pathlib

CELLULES = []


def md(texte):
    CELLULES.append({"cell_type": "markdown", "metadata": {}, "source": texte.strip("\n")})


def code(texte):
    CELLULES.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": texte.strip("\n")})


# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
# Analyse des plaintes consommateurs : regrouper des plaintes multilingues pour repérer les incidents

Une entreprise agroalimentaire reçoit chaque mois des plaintes de consommateurs, dans de nombreux pays et plusieurs langues. Chaque plainte est un court texte libre, le **verbatim**, accompagné de la marque, du segment d'activité et de la période. Personne n'a classé ces textes.

La question posée : **quand plusieurs consommateurs décrivent le même problème, peut-on les regrouper automatiquement, et voir si ce problème se concentre sur une marque et une période ?** C'est ainsi qu'on repère un incident, par exemple un lot de chocolat qui a un goût de savon, avant qu'il ne prenne de l'ampleur.

Il n'y a pas d'étiquette : on ne prédit rien, on **regroupe**. C'est de l'apprentissage non supervisé, et toute la difficulté est de montrer que les groupes trouvés veulent dire quelque chose.

**Les données.** Le notebook lit par défaut la **base réelle** des plaintes, couverte par un accord de confidentialité que chaque participant a signé. Elle ne quitte jamais Kaggle : sur elle, le notebook n'affiche aucun verbatim, seulement des chiffres d'ensemble. Une **base fictive** de même forme (mêmes colonnes, même disposition du fichier, textes inventés, marques numérotées) sert à s'entraîner : des incidents y sont cachés volontairement, ce qui permet de mesurer, en partie 5, si la méthode les retrouve.

Le travail se fait en **deux grandes parties**, dans cet ordre.

**Première grande partie : la chaîne principale.** Un seul modèle de plongement (Qwen3-Embedding-0.6B), de la lecture du fichier jusqu'au tableau de bord. C'est elle qui produit les résultats.

**Deuxième grande partie : la chaîne de comparaison.** Une fois la chaîne principale validée (ses vérifications passées, `resultats.js` écrit), on la rejoue telle quelle avec huit outils, du TF-IDF à Qwen3-Embedding-4B, et on compare temps, mémoire, coût et qualité.

| grande partie | partie | contenu |
|---|---|---|
| I. chaîne principale | 0 | mise en route |
| | 1 | lire le fichier sans le connaître à l'avance, et le contrôler |
| | 2 | transformer chaque plainte en vecteur : TF-IDF, plongements multilingues, retrait de la signature de la langue |
| | 3 | regrouper les plaintes, fusionner les groupes jumeaux d'une langue à l'autre, régler, nommer |
| | 4 | repérer les problèmes en excès sur une marque et une période ; écarter saisons et campagnes |
| | 5 | vérifier, sur la base fictive seulement, que la méthode retrouve les incidents cachés |
| | 6 | exporter les résultats chiffrés pour le tableau de bord |
| II. chaîne de comparaison | 7 | rejouer la chaîne avec huit outils et les comparer en temps, mémoire, coût et qualité |
| pour finir | 8 | changer de base : réelle ou fictive |

© 2026 Selina Cheggour. Tous droits réservés. Voir le fichier LICENSE.
""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
# Première grande partie : la chaîne principale

Parties 0 à 6. Un seul outil de plongement, Qwen3-Embedding-0.6B, et une seule chaîne : lire, nettoyer, détecter la langue, transformer en vecteurs, retirer la signature de la langue, regrouper, fusionner les jumeaux, régler, nommer, repérer les incidents, exporter pour le tableau de bord. On ne passe à la deuxième grande partie qu'une fois cette chaîne validée.
""")

md(r"""
## Partie 0 : mise en route

Active la carte graphique (**Settings > Accelerator > GPU T4**) et l'accès internet (**Settings > Internet on**), puis exécute les trois cellules. La cellule 0.1 est la seule à modifier, et seulement pour changer de base.
""")

code(r'''
# 0.1  Choisir la base : la seule cellule à modifier pour changer de base
from pathlib import Path

BASE_FICTIVE = True                       # ce dépôt ne contient que la base fictive
NOM_FICHIER = "plaintes_fictives.xlsx"

# Sur la base réelle, aucun verbatim ne s'affiche : seulement des chiffres d'ensemble.
AFFICHER_TEXTES = BASE_FICTIVE

def trouver(nom):
    """Cherche le fichier dans les données Kaggle attachées, puis dans le dossier courant et ses voisins."""
    for racine in [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent / "donnees_fictives"]:
        if racine.exists():
            trouves = sorted(racine.rglob(nom))
            if trouves:
                return trouves[0]
    raise FileNotFoundError(f"{nom} introuvable : ajoute le jeu de données au notebook (Add Input).")

CHEMIN = trouver(NOM_FICHIER)
print("Base :", "fictive" if BASE_FICTIVE else "RÉELLE (aucun verbatim ne sera affiché)")
print("Fichier :", CHEMIN.name)
''')

code(r'''
# 0.2  Une bibliothèque absente de Kaggle par défaut : la détection de langue
%pip install -q py3langid babel
''')

code(r'''
# 0.3  Imports et outils du TP (à exécuter sans modifier)
import csv, io, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
try:
    from sklearn.cluster import HDBSCAN                 # scikit-learn 1.3 et plus
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hdbscan"], check=True)
    from hdbscan import HDBSCAN

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)
CACHE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

# ---------------------------------------------------------------- vérifications
VERIFICATIONS = []   # (nom, réussi) de chaque vérification, pour le bilan avant la comparaison

def verifier(nom, test, indice=""):
    """Affiche [OK] si test() renvoie vrai, sinon [À revoir] et un indice. Ne lève jamais d'erreur."""
    try:
        reussi, detail = bool(test()), ""
    except Exception as erreur:
        reussi, detail = False, f" ({type(erreur).__name__} : {str(erreur)[:90]})"
    VERIFICATIONS.append((nom, reussi))
    if reussi:
        print(f"[OK]        {nom}")
    else:
        print(f"[À revoir]  {nom}{detail}" + (f"\n            Indice : {indice}" if indice else ""))

def proche(valeur, cible, tolerance):
    return abs(float(valeur) - cible) <= tolerance

def montrer_textes(textes, n=5):
    """Affiche quelques verbatims, seulement sur la base fictive."""
    if not AFFICHER_TEXTES:
        print(f"({min(n, len(textes))} textes non affichés : base réelle)")
        return
    for t in list(textes)[:n]:
        print("  -", t[:160])

# ---------------------------------------------------------------- graphiques
# Palette du chocolat au rouge : chocolat noir, chocolat, cacao, taupe, ambre, corail, rouge, bordeaux
CHOCOLAT_NOIR, CHOCOLAT, CACAO, TAUPE = "#3B2219", "#5C3A2E", "#A9705A", "#8C776C"
AMBRE, CORAIL, ROUGE, BORDEAUX = "#E4A857", "#E8765C", "#C0392B", "#7B1E16"
CREME, LAIT = "#FBF7F2", "#E9DCCF"
PALETTE = [ROUGE, CHOCOLAT, AMBRE, TAUPE, CORAIL, BORDEAUX, CACAO, "#D35400", CHOCOLAT_NOIR, "#F0B27A",
           "#922B21", "#B9770E", "#6E4B3A", "#EC7063", "#A04000", "#C9A68A", "#641E16"]
COULEURS_LANGUES = {"en": CHOCOLAT, "fr": ROUGE, "de": AMBRE, "nl": CORAIL}
from matplotlib.colors import LinearSegmentedColormap
# carte de chaleur : un déficit reste discret (taupe pâle), un excès monte du crème au bordeaux
CHOCOLAT_ROUGE = LinearSegmentedColormap.from_list("chocolat_rouge",
    [(0.0, "#CDB9A8"), (0.4, CREME), (0.6, "#F3C9B4"), (0.8, CORAIL), (1.0, BORDEAUX)])
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": CREME, "axes.facecolor": CREME, "savefig.facecolor": CREME,
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10.5,
    "text.color": CHOCOLAT_NOIR, "axes.labelcolor": CHOCOLAT, "xtick.color": TAUPE, "ytick.color": TAUPE,
    "axes.edgecolor": LAIT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LAIT, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": CHOCOLAT_NOIR,
    "axes.titlelocation": "left", "axes.titlepad": 26, "legend.frameon": False,
})

def sous_titre(ax, texte):
    """Une ligne d'explication sous le titre, en plus petit."""
    ax.text(0, 1.012, texte, transform=ax.transAxes, fontsize=9, color=TAUPE, style="italic", va="bottom")
FIGURES = CACHE / "figures"
FIGURES.mkdir(exist_ok=True)

def montrer(fig, nom):
    """Affiche la figure et la range dans figures/ (récupérable dans les sorties du notebook Kaggle)."""
    fig.savefig(FIGURES / f"{nom}.png", dpi=150, bbox_inches="tight")
    plt.show()

print("Outils prêts.")
''')

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
## Partie 1 : lire le fichier sans le connaître à l'avance

L'export que fournit l'entreprise est un tableau Excel, et il n'est pas « propre » :

- les deux premières lignes sont vides, et les noms de colonnes sont en **ligne 3** ;
- une colonne s'appelle `Blank` et ne contient qu'un caractère `■` ;
- des lignes vides peuvent traîner à la fin du tableau ;
- l'échantillon compte 1 301 plaintes, mais la livraison complète en comptera beaucoup plus.

La tentation est d'écrire `pd.read_excel(chemin, skiprows=2, nrows=1301)`. Ce code marche aujourd'hui et cassera en silence demain : si l'en-tête descend d'une ligne, ou si le fichier grossit, on lit de travers ou on tronque sans aucun message d'erreur. On écrit plutôt un lecteur qui **trouve lui-même** où commencent et où finissent les données.
""")

md(r"""
### Exercice 1.1 : un lecteur qui s'adapte au fichier

Écris `charger_plaintes(chemin)`, qui renvoie un tableau avec les colonnes `Case Number`, `Verbatim`, `Brand`, `Business Segment` et `Year Period`, une ligne par plainte. Elle doit :

1. lire le fichier **sans en-tête** et en texte : `pd.read_excel(chemin, sheet_name=None, header=None, dtype=str)` pour un tableau Excel (toutes les feuilles), ou le module `csv` pour un CSV, en devinant encodage et séparateur ;
2. trouver la **ligne d'en-tête** : la première ligne qui contient à la fois `Case Number` et `Verbatim`, sans tenir compte des majuscules ni des espaces en trop ;
3. garder les colonnes attendues et ignorer les autres (`Blank`, colonnes sans nom) ;
4. retirer les lignes entièrement vides, où qu'elles soient ;
5. renvoyer aussi un petit **rapport** : feuille, numéro de la ligne d'en-tête, nombre de lignes vides retirées, colonnes ignorées.
""")

code(r'''
COLONNES = ["Case Number", "Verbatim", "Brand", "Business Segment", "Year Period"]

def normaliser(nom):
    return re.sub(r"\s+", " ", str(nom)).strip().lower()

def lire_brut(chemin):
    """Renvoie {nom de feuille : liste de lignes}, chaque ligne étant une liste de textes ('' pour une case vide)."""
    chemin = Path(chemin)
    if chemin.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        feuilles = pd.read_excel(chemin, sheet_name=None, header=None, dtype=str)
        return {nom: f.fillna("").values.tolist() for nom, f in feuilles.items()}
    brut = chemin.read_bytes()
    for encodage in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            texte = brut.decode(encodage)
            break
        except UnicodeDecodeError:
            continue
    try:
        separateur = csv.Sniffer().sniff(texte[:20000], delimiters=",;\t").delimiter
    except csv.Error:
        separateur = ","
    return {chemin.name: list(csv.reader(io.StringIO(texte), delimiter=separateur))}

def charger_plaintes(chemin):
    attendues = {normaliser(c): c for c in COLONNES}
    for feuille, lignes in lire_brut(chemin).items():
        i_entete = next((i for i, l in enumerate(lignes)
                         if {"case number", "verbatim"} <= {normaliser(c) for c in l}), None)
        if i_entete is not None:
            break
    else:
        raise ValueError("Aucune ligne ne contient à la fois « Case Number » et « Verbatim ».")
    entete = [normaliser(c) for c in lignes[i_entete]]
    positions = {attendues[c]: j for j, c in enumerate(entete) if c in attendues}
    manquantes = [c for c in COLONNES if c not in positions]
    if manquantes:
        raise ValueError(f"Colonnes absentes de l'en-tête : {manquantes}")

    donnees, vides = [], 0
    for l in lignes[i_entete + 1:]:
        l = [str(case).strip() for case in l]
        if not any(l):
            vides += 1
            continue
        l = l + [""] * (len(entete) - len(l))
        donnees.append({c: l[j] for c, j in positions.items()})

    rapport = {
        "feuille": feuille,
        "ligne d'en-tête (numérotée à partir de 1)": i_entete + 1,
        "lignes vides retirées": vides,
        "colonnes ignorées": [str(lignes[i_entete][j]) or "(sans nom)" for j, c in enumerate(entete) if c not in attendues],
    }
    tableau = pd.DataFrame(donnees, columns=COLONNES)
    tableau["Case Number"] = tableau["Case Number"].str.replace(r"\.0$", "", regex=True)   # 10063926.0 -> 10063926
    return tableau, rapport

plaintes, rapport = charger_plaintes(CHEMIN)
for cle, valeur in rapport.items():
    print(f"{cle:45s} {valeur}")
print(f"\n{len(plaintes)} plaintes lues")
''')

code(r'''
# Vérification 1.1
verifier("les cinq colonnes attendues, et seulement elles", lambda: list(plaintes.columns) == COLONNES)
verifier("aucune ligne entièrement vide", lambda: (plaintes.apply(lambda c: c.str.len()).sum(axis=1) > 0).all())
if BASE_FICTIVE:
    verifier("en-tête trouvé en ligne 3", lambda: rapport["ligne d'en-tête (numérotée à partir de 1)"] == 3)
    verifier("1 301 plaintes", lambda: len(plaintes) == 1301, "les lignes vides de fin de fichier ne comptent pas")
    verifier("la colonne Blank est ignorée", lambda: "Blank" in rapport["colonnes ignorées"])
''')

md(r"""
**Réflexion 1.1.** Si la prochaine livraison compte 40 000 lignes, avec une ligne de titre en plus avant l'en-tête, ou arrive en CSV avec un séparateur `;`, que faut-il changer dans `charger_plaintes` ? Et dans `pd.read_excel(chemin, skiprows=2, nrows=1301)` ?
""")

md(r"""
**Réponse.** Rien dans `charger_plaintes` : elle cherche l'en-tête par son contenu, sur toutes les feuilles, devine le séparateur d'un CSV, et lit jusqu'à la dernière ligne non vide. La version `skiprows=2, nrows=1301` lirait la ligne de titre comme en-tête et s'arrêterait à 1 301 lignes, sans aucun message. Le rapport affiché sert justement à voir, à chaque livraison, que le fichier a la forme attendue.
""")

md(r"""
### Exercice 1.2 : nettoyer les verbatims

Les verbatims sortent d'un outil de gestion des plaintes, et ça se voit :

- chacun commence par le mot `EXTERNAL`, collé au premier mot (`EXTERNALBonjour`) ;
- les retours à la ligne ont disparu, et les phrases sont collées (`Bonjour,J'ai`, `organisme.Lien`) ;
- les produits cités sont masqués par des étoiles (`*******`) ;
- certains messages contiennent un lien web.

Pour un modèle, `EXTERNALBonjour` est un mot inconnu, et `organisme.Lien` aussi. Écris `nettoyer(texte)` qui :

1. retire `EXTERNAL` en tête de texte ;
2. remplace les liens web (`https?://\S+`) par une espace, **avant** l'étape suivante, pour ne pas couper une adresse ;
3. remplace chaque suite d'au moins deux étoiles par une espace ;
4. ajoute une espace après une ponctuation suivie directement d'une lettre (`([.!?;:,،])(?=[^\W\d_])`) ;
5. réduit les espaces multiples à une seule.

Range le résultat dans `plaintes["texte"]`. Le verbatim d'origine reste dans `plaintes["Verbatim"]`, pour les contrôles de l'exercice 1.5.
""")

code(r'''
def nettoyer(texte):
    texte = re.sub(r"^\s*EXTERNAL", "", texte)
    texte = re.sub(r"https?://\S+", " ", texte)
    texte = re.sub(r"\*{2,}", " ", texte)
    texte = re.sub(r"([.!?;:,،])(?=[^\W\d_])", r"\1 ", texte)
    return re.sub(r"\s+", " ", texte).strip()

plaintes["texte"] = plaintes.Verbatim.map(nettoyer)
exemple = "EXTERNALBonjour,J'ai acheté ******* hier.Il avait un goût bizarre.Lien : https://exemple.io/aB3dE5Merci."
print(nettoyer(exemple))
''')

code(r'''
# Vérification 1.2
verifier("plus aucun texte ne commence par EXTERNAL", lambda: not plaintes.texte.str.startswith("EXTERNAL").any())
verifier("plus d'étoiles ni de liens", lambda: not plaintes.texte.str.contains(r"\*\*|https?://", regex=True).any())
verifier("les phrases sont séparées", lambda: nettoyer("Bonjour,J'ai faim.Merci") == "Bonjour, J'ai faim. Merci")
verifier("une adresse web n'est pas coupée en morceaux", lambda: "exemple" not in nettoyer("voir https://exemple.io/x.Merci"))
''')

md(r"""
**Réflexion 1.2.** Pourquoi retirer le lien **avant** de remettre des espaces après la ponctuation ?
""")

md(r"""
**Réponse.** Une adresse web contient des points suivis de lettres (`app.exemple.io`). Si l'on ajoute d'abord une espace après chaque point, l'adresse est coupée en morceaux (`app. exemple. io/...`) que l'expression `https?://\S+` ne reconnaît plus en entier : des bouts d'adresse resteraient dans le texte, et le modèle les prendrait pour des mots. L'ordre des étapes d'un nettoyage compte autant que les étapes elles-mêmes.
""")

md(r"""
### Exercice 1.3 : contrôler ce qu'on a reçu

Avant tout modèle, on vérifie. Calcule :

- `n_doublons` : le nombre de `Case Number` qui apparaissent plus d'une fois ;
- `n_vides` : le nombre de plaintes sans verbatim ;
- la longueur des textes nettoyés en mots (`plaintes["n_mots"]`), et sa médiane ;
- le nombre de plaintes par marque, par segment et par période.

Trace ensuite le nombre de plaintes par période.
""")

code(r'''
n_doublons = int(plaintes["Case Number"].duplicated(keep=False).sum())
n_vides = int((plaintes.Verbatim == "").sum())
plaintes["n_mots"] = plaintes.texte.str.split().str.len()
print(f"{n_doublons} lignes dont le numéro de cas apparaît plusieurs fois ; {n_vides} verbatims vides")
print(f"longueur médiane : {plaintes.n_mots.median():.0f} mots (de {plaintes.n_mots.min()} à {plaintes.n_mots.max()})")
print(f"{plaintes.Brand.nunique()} marques, {plaintes['Business Segment'].nunique()} segments, "
      f"{plaintes['Year Period'].nunique()} périodes")
print("\nPlaintes par segment :")
print(plaintes["Business Segment"].value_counts().to_string())

par_periode = plaintes["Year Period"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(11, 3.6))
moyenne = par_periode.mean()
couleurs = [ROUGE if v > 1.25 * moyenne else CACAO for v in par_periode.values]
ax.bar(par_periode.index, par_periode.values, color=couleurs, width=0.75, edgecolor=CREME, linewidth=0.8)
ax.axhline(moyenne, color=CHOCOLAT, lw=1, ls="--")
ax.text(-0.6, moyenne + 1, f"moyenne {moyenne:.0f}", va="bottom", ha="left", fontsize=8.5, color=CHOCOLAT,
        bbox=dict(fc=CREME, ec="none", pad=1))
ax.set_title("Nombre de plaintes par période")
sous_titre(ax, "en rouge, les périodes à plus de 25 % au-dessus de la moyenne")
ax.tick_params(axis="x", rotation=90, labelsize=8)
ax.grid(axis="x", visible=False)
plt.tight_layout(); montrer(fig, "1_plaintes_par_periode")
''')

md(r"""
**Comment lire ce graphique.** Une barre par période (« 26 P05 » : année 2026, cinquième période). Sa hauteur est le nombre de plaintes reçues pendant cette période. La ligne pointillée est la moyenne, et les barres rouges dépassent la moyenne de plus d'un quart. Ce graphique dit *quand* il y a eu beaucoup de plaintes, jamais *pourquoi* : une barre rouge peut venir d'un incident, d'une saison, d'une campagne ou d'un simple afflux de courrier. On le regarde pour situer les bosses, puis on cherche leur contenu dans les parties 3 et 4.
""")

code(r'''
# Vérification 1.3
verifier("longueurs calculées", lambda: plaintes.n_mots.notna().all() and plaintes.n_mots.min() >= 0)
if BASE_FICTIVE:
    verifier("aucun doublon, aucun verbatim vide", lambda: n_doublons == 0 and n_vides == 0)
    verifier("20 marques, 4 segments, 23 périodes", lambda: plaintes.Brand.nunique() == 20
             and plaintes["Business Segment"].nunique() == 4 and plaintes["Year Period"].nunique() == 23)
''')

md(r"""
**Réflexion 1.3.** Le graphique par période montre des bosses, l'été 2025 et au printemps 2026 dans la base fictive. Est-ce un incident ?
""")

md(r"""
**Réponse.** Pas forcément. Un volume qui monte sur une période peut venir d'un incident, d'une saison (du chocolat qui fond l'été), d'une campagne de communication, ou simplement d'un changement dans la façon d'enregistrer les plaintes. Le volume total ne dit pas *de quoi* les gens se plaignent : il faut d'abord regrouper les textes par problème (partie 3), puis regarder où chaque groupe se concentre (partie 4).
""")

md(r"""
### Exercice 1.4 : quelles langues ?

La base couvre plusieurs pays, sans colonne pays ni langue. Détecte la langue de chaque texte nettoyé avec `py3langid` (`py3langid.classify(texte)` renvoie le code de la langue et un score) et range-la dans `plaintes["langue"]`. Affiche la répartition.

La langue n'est **pas** le pays : l'anglais, le français et l'allemand couvrent chacun plusieurs pays. Elle dit en revanche quelle représentation des textes pourra rapprocher des plaintes écrites dans des langues différentes.
""")

code(r'''
import py3langid
plaintes["langue"] = [py3langid.classify(t)[0] if t else "?" for t in plaintes.texte]
repartition = plaintes.langue.value_counts()

# le détecteur renvoie un code ISO 639 (« fr », « nl », « ary »...). La liste de ses langues est dans son modèle ;
# les noms français viennent de la base standard CLDR (paquet babel), complétée pour quelques codes rares.
import py3langid.langid as langid_interne
try:
    LANGUES_POSSIBLES = sorted(set((langid_interne.IDENTIFIER or langid_interne._get_identifier()).nb_classes))
except Exception:
    LANGUES_POSSIBLES = []
try:
    from babel import Locale
    NOMS_CLDR = Locale("fr").languages
except ImportError:
    NOMS_CLDR = {}
COMPLEMENT_LANGUES = {'gcf': 'créole guadeloupéen', 'hbo': 'hébreu ancien', 'fuv': 'peul du Nigeria', 'gcr': 'créole guyanais', 'guw': 'goun', 'uzs': 'ouzbek du Sud', 'zxx': 'sans contenu linguistique', '?': 'texte vide'}
nom_langue = lambda code: COMPLEMENT_LANGUES.get(code) or NOMS_CLDR.get(code) or code
print(f"le détecteur choisit parmi {len(LANGUES_POSSIBLES)} langues ; noms trouvés pour "
      f"{sum(nom_langue(c) != c for c in LANGUES_POSSIBLES)} d'entre elles")
print(repartition.rename(index=lambda c: f"{nom_langue(c)} ({c})").to_string())
''')

code(r'''
# Vérification 1.4
verifier("une langue pour chaque plainte", lambda: plaintes.langue.notna().all())
if BASE_FICTIVE:
    verifier("les trois langues de la base fictive", lambda: set(repartition.index[:3]) == {"en", "fr", "de"})
    verifier("l'anglais est la langue la plus fréquente", lambda: repartition.index[0] == "en")
''')

md(r"""
### Exercice 1.5 : l'anonymisation a-t-elle laissé passer quelque chose ?

L'entreprise a anonymisé les verbatims selon ses propres règles : les produits cités sont remplacés par des étoiles. On vérifie sur les verbatims d'origine, sans lire les textes, en **comptant** les plaintes qui contiennent encore une forme reconnaissable de donnée personnelle :

| motif | expression régulière |
|---|---|
| adresse électronique | `[\w.+-]+@[\w-]+\.[\w.]+` |
| numéro de téléphone | `\+?\d[\d .-]{7,}\d` |
| suite d'au moins six chiffres | `\b\d{6,}\b` |
| adresse web | `https?://\S+` |

Range les comptes dans un dictionnaire `residus`, et compte aussi les plaintes qui contiennent un masque d'étoiles (`n_masques`).
""")

code(r'''
MOTIFS = {
    "adresse électronique": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "numéro de téléphone": r"\+?\d[\d .-]{7,}\d",
    "suite d'au moins six chiffres": r"\b\d{6,}\b",
    "adresse web": r"https?://\S+",
}
residus = {nom: int(plaintes.Verbatim.str.contains(motif, regex=True).sum()) for nom, motif in MOTIFS.items()}
n_masques = int(plaintes.Verbatim.str.contains(r"\*{2,}", regex=True).sum())
for nom, n in residus.items():
    print(f"{nom:32s} {n} plaintes")
print(f"\n{n_masques} plaintes contiennent un produit masqué par des étoiles")
''')

code(r'''
# Vérification 1.5
verifier("quatre motifs comptés", lambda: len(residus) == 4)
if BASE_FICTIVE:
    verifier("ni courriel, ni téléphone, ni longue suite de chiffres", lambda: residus["adresse électronique"] == 0
             and residus["numéro de téléphone"] == 0 and residus["suite d'au moins six chiffres"] == 0)
    verifier("40 plaintes contiennent un lien web", lambda: residus["adresse web"] == 40,
             "cherche dans plaintes.Verbatim, pas dans le texte nettoyé")
    verifier("toutes les plaintes fictives citent un produit masqué", lambda: n_masques == len(plaintes))
''')

md(r"""
**Réflexion 1.5.** Sur la base réelle, un compte non nul ne prouve pas une fuite, et un compte nul ne prouve pas l'absence de fuite. Pourquoi ?
""")

md(r"""
**Réponse.** Une suite de chiffres peut être un numéro de lot ou un poids, et un lien web pointe souvent vers une fiche produit publique, pas vers une personne : un compte non nul se vérifie avec l'entreprise, sans recopier les textes. À l'inverse, les expressions régulières ne voient ni un nom de famille, ni une adresse écrite en toutes lettres : un compte nul dit seulement que les formes les plus faciles ont été retirées. Ce contrôle complète l'anonymisation de l'entreprise ; il ne la remplace pas.
""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
## Partie 2 : transformer chaque plainte en vecteur

Pour regrouper des textes, il faut d'abord les transformer en nombres. On compare deux représentations.

- **TF-IDF** : une coordonnée par mot ; chaque mot pèse d'autant plus qu'il est fréquent dans la plainte et rare dans les autres. C'est simple, rapide et lisible, mais chaque langue a ses propres mots : « savon », « Seife » et « soap » sont trois coordonnées sans rapport.
- **Plongement multilingue** : un modèle pré-entraîné sur des textes de nombreuses langues transforme chaque plainte en un vecteur de sens, où deux phrases qui disent la même chose dans deux langues différentes sont proches.

Les deux représentations partent du texte nettoyé de l'exercice 1.2 : sans le préfixe `EXTERNAL`, ni étoiles, ni liens, avec des phrases séparées.
""")

md(r"""
### Exercice 2.1 : TF-IDF, et à quoi ressemblent les plus proches voisins

1. Construis `X_tfidf` avec `TfidfVectorizer(sublinear_tf=True, min_df=2)` sur `plaintes.texte`.
2. Pour chaque plainte, trouve sa plus proche voisine (similarité cosinus la plus forte, elle-même exclue) et calcule `meme_langue_tfidf` : la part des plaintes dont la plus proche voisine est **dans la même langue**.

Ce diagnostic ne demande aucune étiquette : il marche aussi sur la base réelle.
""")

code(r'''
textes = plaintes.texte
vectoriseur = TfidfVectorizer(sublinear_tf=True, min_df=2)
X_tfidf = vectoriseur.fit_transform(textes)
print(f"{X_tfidf.shape[0]} plaintes, {X_tfidf.shape[1]} mots dans le vocabulaire")

def part_voisin_meme_langue(X):
    """Part des plaintes dont la plus proche voisine (cosinus) est dans la même langue."""
    S = (X @ X.T)
    S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    np.fill_diagonal(S, -np.inf)
    voisin = S.argmax(axis=1)
    return float((plaintes.langue.values == plaintes.langue.values[voisin]).mean())

meme_langue_tfidf = part_voisin_meme_langue(normalize(X_tfidf))
print(f"plus proche voisine dans la même langue : {meme_langue_tfidf:.0%}")
''')

code(r'''
# Vérification 2.1
verifier("une ligne par plainte", lambda: X_tfidf.shape[0] == len(plaintes))
verifier("TF-IDF : la voisine est presque toujours dans la même langue", lambda: meme_langue_tfidf > 0.95,
         "normalize(X_tfidf) puis produit scalaire")
''')

md(r"""
### Exercice 2.2 : plongements multilingues

On utilise `Qwen/Qwen3-Embedding-0.6B` (licence Apache 2.0, plus de cent langues) avec `sentence-transformers`. Si ce modèle ne se charge pas sur la version installée, la cellule se rabat sur `intfloat/multilingual-e5-base` (licence MIT). Les vecteurs sont rangés dans un fichier pour ne pas les recalculer.

Calcule ensuite `meme_langue_plongement`, la même mesure qu'en 2.1.
""")

code(r'''
import hashlib, torch
from sentence_transformers import SentenceTransformer

MODELES = [
    ("Qwen/Qwen3-Embedding-0.6B", "Instruct: Identify the product problem described in this consumer complaint\nQuery: "),
    ("intfloat/multilingual-e5-base", "query: "),
]
appareil = "cuda" if torch.cuda.is_available() else "cpu"
empreinte = hashlib.md5("".join(textes).encode()).hexdigest()[:10]

for nom_modele, consigne in MODELES:
    fichier = CACHE / f"plongements_{nom_modele.split('/')[-1]}_{empreinte}.npy"
    try:
        if fichier.exists():
            E = np.load(fichier)
        else:
            modele = SentenceTransformer(nom_modele, device=appareil)
            modele.max_seq_length = 512          # au-delà, une plainte coûte beaucoup plus pour peu d'information en plus
            E = modele.encode([consigne + t for t in textes], batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False)
            np.save(fichier, E)
        break
    except Exception as erreur:
        print(f"{nom_modele} indisponible ({type(erreur).__name__}), modèle suivant.")
print(f"modèle : {nom_modele} sur {appareil} ; {E.shape[0]} vecteurs de dimension {E.shape[1]}")

meme_langue_plongement = part_voisin_meme_langue(E)
print(f"plus proche voisine dans la même langue : TF-IDF {meme_langue_tfidf:.0%}, plongement {meme_langue_plongement:.0%}")
''')

code(r'''
# Vérification 2.2
verifier("un vecteur normé par plainte", lambda: E.shape[0] == len(plaintes) and proche(np.linalg.norm(E[0].astype("float64")), 1, 0.01))
verifier("le plongement rapproche des plaintes de langues différentes",
         lambda: meme_langue_plongement < meme_langue_tfidf)
''')

md(r"""
**Réflexion 2.2.** Le plongement multilingue rapproche-t-il vraiment des plaintes de langues différentes ? Regarde le pourcentage.
""")

md(r"""
**Réponse.** À peine. Même avec un modèle multilingue, la plus proche voisine d'une plainte est presque toujours dans sa langue : deux plaintes allemandes sur des problèmes différents partagent des tournures, des formules de politesse, une syntaxe, et le modèle le voit. Le sens commun à une plainte française et à une plainte allemande sur le même problème existe dans les vecteurs, mais il est masqué par cette « signature » de la langue. Pour la recherche d'incidents, c'est grave : un incident signalé dans quatre pays serait éclaté en quatre petits groupes. L'exercice suivant retire cette signature.
""")

md(r"""
### Exercice 2.3 : retirer la signature de la langue

Chaque langue déplace tous ses vecteurs dans une même direction. On l'estime par le **vecteur moyen de la langue**, et on le retire :

$$\tilde e_i = \frac{e_i - \mu_{\ell(i)}}{\lVert e_i - \mu_{\ell(i)} \rVert}$$

où $e_i$ est le plongement de la plainte $i$, $\ell(i)$ sa langue et $\mu_\ell$ la moyenne des plongements des plaintes de la langue $\ell$. Après ce centrage, deux plaintes sont proches parce qu'elles s'écartent de la même façon de la plainte « moyenne » de leur langue, c'est-à-dire par leur contenu.

Une langue représentée par moins de 10 plaintes donne une moyenne peu fiable : on lui retire alors la moyenne de toutes les plaintes.

Écris `centrer_par_langue(E, langues, n_min=10)`, calcule `E_c`, puis `meme_langue_centre` avec la mesure de l'exercice 2.1.
""")

code(r'''
def centrer_par_langue(E, langues, n_min=10):
    C = E.astype("float64").copy()
    moyenne_globale = C.mean(axis=0)
    for l in np.unique(langues):
        m = langues == l
        C[m] -= C[m].mean(axis=0) if m.sum() >= n_min else moyenne_globale
    return normalize(C)

E_c = centrer_par_langue(E, plaintes.langue.values)
meme_langue_centre = part_voisin_meme_langue(E_c)
print(f"plus proche voisine dans la même langue : TF-IDF {meme_langue_tfidf:.0%}, plongement {meme_langue_plongement:.0%}, "
      f"plongement centré {meme_langue_centre:.0%}")

# Carte des plaintes en deux dimensions (t-SNE), colorée par langue, avant et après le centrage
from sklearn.manifold import TSNE
carte_brute = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(E)
carte_centree = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(E_c)
langues_principales = plaintes.langue.value_counts().index[:6]
couleur_langue = {l: COULEURS_LANGUES.get(l, c) for l, c in zip(langues_principales, [CORAIL, TAUPE, BORDEAUX] * 2)}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, carte, titre, texte in zip(axes, [carte_brute, carte_centree],
                                   ["Plongement brut", "Plongement centré par langue"],
                                   [f"plus proche voisine dans la même langue : {meme_langue_plongement:.0%}",
                                    f"plus proche voisine dans la même langue : {meme_langue_centre:.0%}"]):
    for l in langues_principales:
        m = (plaintes.langue == l).values
        ax.scatter(carte[m, 0], carte[m, 1], s=11, alpha=0.75, color=couleur_langue[l], edgecolor=CREME,
                   linewidth=0.3, label=nom_langue(l))
    ax.set_title(titre); sous_titre(ax, texte)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_visible(False)
axes[0].legend(markerscale=2.5, loc="lower left", fontsize=9)
plt.tight_layout(); montrer(fig, "2_carte_langues_avant_apres")
''')

md(r"""
**Comment lire ce graphique.** Chaque point est une plainte. La méthode t-SNE place les points en deux dimensions de façon que deux plaintes proches dans l'espace des vecteurs (1 024 dimensions) restent proches sur la carte ; les axes eux-mêmes n'ont pas de sens, seules les distances entre points en ont. La couleur est la langue de la plainte.

- **À gauche**, les couleurs forment des îlots séparés : une plainte est d'abord proche des plaintes de sa langue, quel que soit son sujet. Un même incident signalé en français et en allemand tombe dans deux îlots différents.
- **À droite**, après avoir retiré à chaque plainte la moyenne de sa langue, les trois couleurs se mélangent dans les mêmes paquets : chaque paquet est un sujet, écrit dans plusieurs langues.

Si les couleurs restaient séparées à droite, le centrage n'aurait pas suffi, et il faudrait traduire les plaintes avant de les comparer.
""")

code(r'''
# Vérification 2.3
verifier("un vecteur centré et normé par plainte", lambda: E_c.shape == E.shape and proche(np.linalg.norm(E_c[0]), 1, 1e-6))
verifier("le centrage mélange davantage les langues", lambda: meme_langue_centre < meme_langue_plongement)
''')

md(r"""
**Réflexion 2.3.** Le centrage retire-t-il seulement la langue ?
""")

md(r"""
**Réponse.** Non : il retire tout ce qui est en moyenne propre à une langue. Si les plaintes polonaises portent surtout sur la nourriture pour animaux, une partie de « nourriture pour animaux » est attribuée au polonais et retirée de ces vecteurs. Le centrage reste utile parce que la recherche d'incidents compare des plaintes *entre elles* à l'intérieur d'un même problème, mais il faut le savoir, et il faut regarder les langues de chaque groupe trouvé (partie 3). Une autre voie, plus lourde, consiste à traduire toutes les plaintes dans une seule langue avant de les représenter.
""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
## Partie 3 : regrouper les plaintes

### Exercice 3.1 : HDBSCAN sur les plongements centrés

**HDBSCAN** cherche les zones denses : il trouve lui-même le nombre de groupes, et laisse de côté (étiquette `-1`) les plaintes isolées qui ne ressemblent à aucune autre. C'est ce qu'on veut : un incident est une zone dense au milieu de plaintes variées. HDBSCAN fonctionne mal en grande dimension ; on réduit d'abord les vecteurs centrés à 20 dimensions par une **analyse en composantes principales** (ACP), qui garde les directions où les plaintes diffèrent le plus.

1. `PCA(20, random_state=0)` sur `E_c`, puis normalisation.
2. `HDBSCAN(min_cluster_size=8)` sur le résultat.
3. Range les groupes dans `plaintes["groupe"]`, compte les groupes, la part de plaintes laissées de côté, et la part de la langue majoritaire dans chaque groupe.
""")

code(r'''
from sklearn.decomposition import PCA

# Réglages par défaut ; l'exercice 3.3 les choisit ensuite sur les données elles-mêmes
REGLAGES = {"taille_min": 8, "methode": "eom", "cosinus_min": 0.4, "k": None}

def regrouper(E_c, taille_min=None, methode=None, k=None):
    Z = normalize(PCA(20, random_state=0).fit_transform(E_c))
    methode = methode or REGLAGES["methode"]
    if methode == "ward":      # plan B de l'exercice 3.3 : k groupes, chaque plainte rangée
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=k or REGLAGES["k"], linkage="ward").fit_predict(Z)
    return HDBSCAN(min_cluster_size=taille_min or REGLAGES["taille_min"], cluster_selection_method=methode).fit_predict(Z)

plaintes["groupe"] = regrouper(E_c)
n_groupes = plaintes.groupe[plaintes.groupe >= 0].nunique()
part_bruit = float((plaintes.groupe == -1).mean())
purete_langue_groupes = (plaintes[plaintes.groupe >= 0].groupby("groupe").langue
                         .agg(lambda l: l.value_counts(normalize=True).iat[0]))
print(f"{n_groupes} groupes ; {part_bruit:.0%} des plaintes laissées de côté")
print(f"part de la langue majoritaire dans un groupe, en médiane : {purete_langue_groupes.median():.0%}")
''')

code(r'''
# Vérification 3.1
verifier("au moins cinq groupes", lambda: n_groupes >= 5)
verifier("moins de la moitié des plaintes laissées de côté", lambda: part_bruit < 0.5)
''')

md(r"""
### Exercice 3.2 : fusionner les groupes jumeaux d'une langue à l'autre

Même après centrage, beaucoup de groupes restent dominés par une langue : HDBSCAN trouve le groupe « goût de savon » en français, puis le même en allemand, à côté. Ce sont des **groupes jumeaux**. On les fusionne en comparant leurs **centres** : le centre d'un groupe est la moyenne des vecteurs centrés de ses plaintes, et deux groupes dont les centres ont un cosinus supérieur à 0,4 parlent du même problème.

On le fait par une **classification ascendante hiérarchique** sur les centres : `AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.6)`. La distance cosinus vaut 1 moins le cosinus : un seuil de 0,6 sur la distance correspond à un cosinus de 0,4.

Range le résultat dans `plaintes["probleme"]` (les plaintes laissées de côté gardent `-1`), et compte les problèmes et le nombre de langues de chacun.
""")

code(r'''
from sklearn.cluster import AgglomerativeClustering

def fusionner(groupes, E_c, cosinus_min=None):
    ids = np.sort(np.unique(groupes[groupes >= 0]))
    if len(ids) < 2:
        return groupes.copy()
    centres = normalize(np.vstack([E_c[groupes == g].mean(axis=0) for g in ids]))
    fusion = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - (cosinus_min or REGLAGES["cosinus_min"])).fit_predict(centres)
    correspondance = dict(zip(ids, fusion))
    return np.array([correspondance.get(g, -1) for g in groupes])

plaintes["probleme"] = fusionner(plaintes.groupe.values, E_c)
n_problemes = plaintes.probleme[plaintes.probleme >= 0].nunique()
langues_par_probleme = plaintes[plaintes.probleme >= 0].groupby("probleme").langue.nunique()
print(f"{n_groupes} groupes fusionnés en {n_problemes} problèmes")
print(f"nombre de langues par problème, en médiane : {langues_par_probleme.median():.0f}")
''')

code(r'''
# Vérification 3.2
verifier("moins de problèmes que de groupes", lambda: 0 < n_problemes < n_groupes)
verifier("les problèmes réunissent plusieurs langues", lambda: langues_par_probleme.median() >= 2)
''')

md(r"""
### Exercice 3.3 : régler le regroupement sur les données

Les réglages des exercices 3.1 et 3.2 (groupes d'au moins 8 plaintes, fusion au-delà d'un cosinus de 0,4) ont été choisis sur la base fictive. Sur d'autres plaintes, plus longues ou plus homogènes, ils peuvent tout réunir en un ou deux paquets géants : le regroupement est alors parfaitement stable, et parfaitement inutile.

On essaie donc plusieurs réglages et on choisit sur les données elles-mêmes, sans étiquette :

- la **taille minimale** d'un groupe : 5, 8 ou 12 plaintes ;
- la **méthode de sélection** de HDBSCAN : `eom` garde les zones denses les plus durables, souvent grandes ; `leaf` garde les plus fines ;
- le **seuil de fusion** des jumeaux : un cosinus de 0,4, 0,55 ou 0,7 (plus il est haut, moins on fusionne).

Une **campagne** (beaucoup de messages presque identiques, voir l'exercice 4.2) a le droit d'être grosse : c'est un vrai phénomène, pas un défaut du regroupement. On repère donc d'abord les plaintes **quasi doublons** (une voisine TF-IDF au-delà d'un cosinus de 0,9), et un problème dont plus de la moitié des plaintes sont des quasi doublons est compté comme campagne.

Un réglage est **admis** si son plus gros problème, campagnes exclues, contient au plus 30 % des plaintes, s'il laisse de côté au plus la moitié des plaintes hors quasi doublons, et s'il trouve au moins 5 problèmes. Parmi les réglages admis, on garde le plus **stable** (l'accord moyen avec deux regroupements refaits sur 90 % des plaintes).

**Plan B.** Si aucun réglage de HDBSCAN n'est admis, les plaintes ne forment pas de zones denses nettes. On passe alors à une **classification ascendante hiérarchique de Ward**, qui range chaque plainte dans l'un de $k$ groupes en minimisant la dispersion à l'intérieur des groupes ; on essaie $k$ = 8, 12, 16, 20 et 30, avec les mêmes règles d'admission.
""")

code(r'''
# quasi doublons : une voisine TF-IDF au-delà d'un cosinus de 0,9 (la signature des campagnes, exercice 4.2)
S_tfidf = (normalize(X_tfidf) @ normalize(X_tfidf).T).toarray()
np.fill_diagonal(S_tfidf, 0)
quasi_doublon = S_tfidf.max(axis=1) > 0.9
del S_tfidf
print(f"quasi doublons : {quasi_doublon.mean():.0%} des plaintes")

def evaluer_reglage(E_c, taille_min, methode, cosinus_min, k=None):
    groupes = regrouper(E_c, taille_min, methode, k)
    probleme = fusionner(groupes, E_c, cosinus_min)
    ranges = probleme[probleme >= 0]
    tailles = pd.Series(ranges).value_counts()
    part_doublons = pd.Series(quasi_doublon[probleme >= 0]).groupby(ranges).mean()
    hors_campagnes = tailles[part_doublons.reindex(tailles.index).fillna(0) <= 0.5]
    stabilite = []
    for graine in (1, 2):
        t = np.sort(np.random.default_rng(graine).choice(len(E_c), int(0.9 * len(E_c)), replace=False))
        stabilite.append(adjusted_rand_score(probleme[t], fusionner(regrouper(E_c[t], taille_min, methode, k), E_c[t], cosinus_min)))
    return {"methode": methode, "taille_min": taille_min if methode != "ward" else None, "k": k, "cosinus_min": cosinus_min,
            "groupes": int(len(set(groupes) - {-1})), "problemes": int(len(tailles)),
            "campagnes": int(len(tailles) - len(hors_campagnes)),
            "part_cote": float(((probleme == -1) & ~quasi_doublon).sum() / max((~quasi_doublon).sum(), 1)),
            "part_max": float(hors_campagnes.max() / len(probleme)) if len(hors_campagnes) else 1.0,
            "stabilite": float(np.mean(stabilite))}

def admettre(essais):
    return essais[(essais.part_max <= 0.3) & (essais.part_cote <= 0.5) & (essais.problemes >= 5)]

def choisir_reglages(E_c):
    essais = pd.DataFrame([evaluer_reglage(E_c, t, m, c) for t in (5, 8, 12) for m in ("eom", "leaf") for c in (0.4, 0.55, 0.7)])
    if not len(admettre(essais)):          # plan B : Ward, chaque plainte rangée
        essais = pd.concat([essais, pd.DataFrame([evaluer_reglage(E_c, None, "ward", c, k)
                                                  for k in (8, 12, 16, 20, 30) for c in (0.55, 0.7)])], ignore_index=True)
    admis = admettre(essais)
    choix = (admis.sort_values(["stabilite", "problemes"], ascending=False).iloc[0] if len(admis)
             else essais.sort_values("part_max").iloc[0])
    essais["admis"] = essais.index.isin(admis.index)
    reglages = {"methode": choix.methode, "cosinus_min": float(choix.cosinus_min),
                "taille_min": int(choix.taille_min) if choix.methode != "ward" else REGLAGES["taille_min"],
                "k": int(choix.k) if choix.methode == "ward" else None}
    return reglages, essais

REGLAGES, essais_reglages = choisir_reglages(E_c)
print(essais_reglages.sort_values(["admis", "stabilite"], ascending=False).head(12).to_string(
    index=False, formatters={"part_cote": "{:.0%}".format, "part_max": "{:.0%}".format, "stabilite": "{:.2f}".format}))
print("\nréglages retenus :", REGLAGES)

plaintes["groupe"] = regrouper(E_c)
plaintes["probleme"] = fusionner(plaintes.groupe.values, E_c)
n_groupes = plaintes.groupe[plaintes.groupe >= 0].nunique()
n_problemes = plaintes.probleme[plaintes.probleme >= 0].nunique()
part_bruit = float((plaintes.probleme == -1).mean())
langues_par_probleme = plaintes[plaintes.probleme >= 0].groupby("probleme").langue.nunique()
print(f"{n_groupes} groupes, {n_problemes} problèmes ; laissées de côté : {part_bruit:.0%}")
''')

code(r'''
# Vérification 3.3
verifier("au moins dix-huit réglages essayés", lambda: len(essais_reglages) >= 18)
verifier("un réglage admis a été trouvé", lambda: essais_reglages.admis.any())
''')

md(r"""
**À retenir.** On ne prend pas simplement le réglage le plus stable : tout mettre dans un seul paquet serait parfaitement stable et parfaitement inutile. C'est pourquoi les conditions d'admission (pas de problème géant, pas trop de plaintes de côté, assez de problèmes) passent avant la stabilité.
""")

md(r"""
### Exercice 3.4 : nommer chaque problème

Un groupe ne sert à rien tant qu'on ne sait pas de quoi il parle. On le nomme par ses mots les plus **caractéristiques**, avec un TF-IDF calculé au niveau des groupes (c-TF-IDF) : on concatène les plaintes de chaque groupe en un seul document, puis on pondère chaque mot par sa fréquence dans le groupe et sa rareté dans les autres groupes. Comme un problème réunit plusieurs langues, ses mots caractéristiques en mélangent plusieurs.

Écris `mots_caracteristiques(groupes, n=8)`, puis affiche un tableau par problème : taille, nombre de langues, marque la plus fréquente et sa part, mots.
""")

code(r'''
# Mots vides (articles, pronoms, auxiliaires) en anglais, français et allemand : ils ne disent rien du problème
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
MOTS_VIDES = sorted(set(ENGLISH_STOP_WORDS) | set("""
le la les un une des de du au aux et en est était été être avoir ai avait a ce cet cette ces ça je j il elle on
nous vous ils elles mon ma mes ton ta tes son sa ses notre votre leur leurs pour par pas ne plus que qui dans sur
avec très tout tous toute fait comme mais ou si me se sont suis bonjour merci madame monsieur cordialement vos
der die das den dem des ein eine einen einem einer und ist war ich sie er es wir ihr mit für auf im an am zu zum
zur von vom nicht hat habe hatte sehr auch aber oder wie so da dass mein meine meinen meiner mir mich bei aus nach
noch nur schon bitte hallo guten tag danke grüße freundlichen sehr geehrte damen herren
hello hi dear thanks regards best kind
""".split()))

def mots_caracteristiques(groupes, n=8):
    garde = groupes >= 0
    compteur = CountVectorizer(min_df=2, stop_words=MOTS_VIDES)
    C = compteur.fit_transform(textes[garde])
    ids = np.sort(np.unique(groupes[garde]))
    par_groupe = np.vstack([np.asarray(C[groupes[garde] == g].sum(axis=0)).ravel() for g in ids])
    tf = par_groupe / par_groupe.sum(axis=1, keepdims=True)
    idf = np.log(1 + par_groupe.sum(axis=1).mean() / np.maximum(par_groupe.sum(axis=0), 1))
    poids = tf * idf
    vocab = np.array(compteur.get_feature_names_out())
    return {g: ", ".join(vocab[np.argsort(poids[i])[::-1][:n]]) for i, g in enumerate(ids)}

mots = mots_caracteristiques(plaintes.probleme.values)
resume = (plaintes[plaintes.probleme >= 0].groupby("probleme")
          .agg(taille=("Verbatim", "size"), langues=("langue", "nunique"),
               marque=("Brand", lambda b: b.value_counts().index[0]),
               part_marque=("Brand", lambda b: b.value_counts(normalize=True).iat[0])))
resume["mots"] = resume.index.map(mots)
print(resume.sort_values("taille", ascending=False).to_string(formatters={"part_marque": "{:.0%}".format}))

# Carte des problèmes : la carte centrée de l'exercice 2.3, colorée par problème ; un numéro sur la carte,
# la légende à droite, pour que les étiquettes ne se chevauchent pas
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(13, 7.5))
bruit = (plaintes.probleme == -1).values
ax.scatter(carte_centree[bruit, 0], carte_centree[bruit, 1], s=6, color=LAIT, edgecolor="none")
ordre_taille = list(resume.sort_values("taille", ascending=False).index)
couleur_probleme = dict(zip(ordre_taille, PALETTE * 3))
for g in ordre_taille:
    m = (plaintes.probleme == g).values
    ax.scatter(carte_centree[m, 0], carte_centree[m, 1], s=13, alpha=0.8, color=couleur_probleme[g],
               edgecolor=CREME, linewidth=0.3)
    cx, cy = np.median(carte_centree[m], axis=0)
    ax.text(cx, cy, str(g), fontsize=8, weight="bold", ha="center", va="center", color="white",
            bbox=dict(boxstyle="circle,pad=0.25", fc=couleur_probleme[g], ec="white", lw=1))
poignees = [Line2D([0], [0], marker="o", ls="", ms=7, color=couleur_probleme[g],
                   label=f"{g} : {', '.join(mots[g].split(', ')[:4])} ({resume.taille[g]})") for g in ordre_taille]
poignees.append(Line2D([0], [0], marker="o", ls="", ms=7, color=LAIT, label="plaintes laissées de côté"))
ax.legend(handles=poignees, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, title="problème : mots (taille)")
ax.set_title("Les problèmes trouvés, toutes langues confondues")
sous_titre(ax, "même carte que la partie droite de l'exercice 2.3, colorée cette fois par problème")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for cote in ("left", "bottom"):
    ax.spines[cote].set_visible(False)
plt.tight_layout(); montrer(fig, "3_carte_problemes")
''')

md(r"""
**Comment lire ce graphique.** C'est la carte de droite de l'exercice 2.3, mais la couleur est maintenant le **problème** trouvé par la méthode, et non plus la langue. Chaque pastille numérotée est posée au centre d'un problème ; la légende donne, pour chaque numéro, ses mots les plus caractéristiques et son nombre de plaintes. Les points pâles sont les plaintes que HDBSCAN a laissées de côté parce qu'elles ne ressemblent assez à aucune autre.

Un bon regroupement donne des paquets compacts d'une seule couleur. Un problème dont les points sont dispersés sur toute la carte est un groupe fourre-tout, à regarder de près. Les mots de la légende permettent de nommer chaque problème sans lire les plaintes, ce qui compte sur la base réelle.
""")

code(r'''
# Vérification 3.4
verifier("un nom pour chaque problème", lambda: len(mots) == n_problemes and all(mots.values()))
''')

md(r"""
### Exercice 3.4 bis : une définition pour chaque problème

Des mots comme « savon, Seife, soap, goût » disent de quoi parle un groupe, mais il faut encore le formuler. On le fait faire par un **modèle de langue génératif** (un LLM), qui tourne **dans le notebook**, sur la carte graphique : `Qwen/Qwen3-4B-Instruct-2507` (licence Apache 2.0), ou `Qwen/Qwen2.5-1.5B-Instruct` s'il ne se charge pas. Aucune plainte n'est envoyée à un service extérieur.

Pour chaque problème, le modèle reçoit ses mots caractéristiques et ses cinq plaintes les plus **représentatives** (les plus proches du centre du groupe, coupées à 350 caractères). Il rend un **titre** de 3 à 6 mots, une **définition** d'une phrase et une **catégorie** prise dans une liste fixe.

Ce sont des propositions : le modèle peut se tromper ou généraliser à partir de cinq exemples. Elles servent à lire le tableau de bord plus vite, pas à remplacer la lecture de quelques plaintes par une personne habilitée.
""")

code(r'''
import gc, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_CANDIDATS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen2.5-1.5B-Instruct"]
CATEGORIES = ["goût ou odeur", "texture ou aspect", "corps étranger", "emballage ou quantité", "santé ou allergie",
              "date ou fraîcheur", "composition ou ingrédients", "prix ou promotion", "service ou livraison",
              "compliment", "autre"]
CONSIGNE_LLM = """Tu analyses des plaintes de consommateurs adressées à une entreprise agroalimentaire. Elles sont écrites en plusieurs langues et ont été regroupées automatiquement par sujet.

Mots les plus caractéristiques du groupe : {mots}

Plaintes représentatives du groupe :
{exemples}

Réponds uniquement par un objet JSON, en français, avec trois clés :
"titre" : le problème commun, en 3 à 6 mots ;
"definition" : une phrase qui décrit ce dont se plaignent les consommateurs de ce groupe ;
"categorie" : une seule valeur parmi {categories}."""

globals().pop("modele", None); gc.collect()        # le modèle de plongement n'est plus utile ici
if torch.cuda.is_available(): torch.cuda.empty_cache()

for nom_llm in LLM_CANDIDATS:
    try:
        tokeniseur = AutoTokenizer.from_pretrained(nom_llm)
        llm = AutoModelForCausalLM.from_pretrained(nom_llm, torch_dtype=torch.float16, device_map="auto")
        break
    except Exception as erreur:
        print(f"{nom_llm} indisponible ({type(erreur).__name__}), modèle suivant.")
print("modèle de langue :", nom_llm)

def representants(g, n=5):
    membres = np.where(plaintes.probleme.values == g)[0]
    centre = normalize(E_c[membres].mean(axis=0, keepdims=True))[0]
    proches = membres[np.argsort(-(E_c[membres] @ centre))[:n]]
    return [plaintes.texte.iat[i][:350] for i in proches]

def definir(g):
    exemples = "\n".join(f"- {t}" for t in representants(g))
    message = [{"role": "user", "content": CONSIGNE_LLM.format(mots=mots[g], exemples=exemples,
                                                              categories=", ".join(f'"{c}"' for c in CATEGORIES))}]
    # return_dict=True : les versions récentes de transformers renvoient un dictionnaire (input_ids, attention_mask)
    entree = tokeniseur.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True).to(llm.device)
    with torch.no_grad():
        sortie = llm.generate(**entree, max_new_tokens=180, do_sample=False)
    reponse = tokeniseur.decode(sortie[0, entree["input_ids"].shape[1]:], skip_special_tokens=True)
    trouve = re.search(r"\{.*\}", reponse, re.S)
    try:
        d = json.loads(trouve.group(0)) if trouve else {}
    except json.JSONDecodeError:
        d = {}
    return {"titre": str(d.get("titre") or ", ".join(mots[g].split(", ")[:3]))[:70],
            "definition": str(d.get("definition") or "")[:300],
            "categorie": d.get("categorie") if d.get("categorie") in CATEGORIES else "autre",
            "par_llm": bool(d.get("titre"))}

definitions = {int(g): definir(g) for g in resume.index}
resume["titre"] = resume.index.map(lambda g: definitions[int(g)]["titre"])
resume["categorie"] = resume.index.map(lambda g: definitions[int(g)]["categorie"])
del llm; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

if AFFICHER_TEXTES:
    for g in resume.sort_values("taille", ascending=False).index:
        print(f"{g:3d} ({resume.taille[g]:3d}) [{definitions[int(g)]['categorie']}] {definitions[int(g)]['titre']} : {definitions[int(g)]['definition']}")
else:
    print("Base réelle : les définitions ne s'affichent pas ici ; elles vont dans tableau_de_bord/noms_problemes.js.")
    print("problèmes par catégorie :", resume.categorie.value_counts().to_dict())
''')

code(r'''
# Vérification 3.4 bis
verifier("une définition pour chaque problème", lambda: len(definitions) == n_problemes)
verifier("la plupart des titres viennent du modèle de langue", lambda: np.mean([d["par_llm"] for d in definitions.values()]) >= 0.8)
verifier("une catégorie de la liste pour chaque problème", lambda: all(d["categorie"] in CATEGORIES for d in definitions.values()))
''')

md(r"""
### Exercice 3.5 : les problèmes sont-ils stables ?

Sans étiquette, une façon de juger un regroupement est de vérifier qu'il ne tient pas à quelques plaintes. Tire trois fois 90 % des plaintes au hasard (graines 1, 2, 3), refais tout le regroupement sur ce sous-ensemble (`regrouper` puis `fusionner`), et mesure l'accord avec le regroupement complet sur les plaintes communes, par l'**indice de Rand ajusté** (ARI : 1 pour un accord parfait, 0 pour un accord dû au hasard).
""")

code(r'''
stabilite = []
for graine in (1, 2, 3):
    tirage = np.sort(np.random.default_rng(graine).choice(len(plaintes), int(0.9 * len(plaintes)), replace=False))
    sous_groupes = regrouper(E_c[tirage])
    sous_problemes = fusionner(sous_groupes, E_c[tirage])
    stabilite.append(adjusted_rand_score(plaintes.probleme.values[tirage], sous_problemes))
print("ARI entre le regroupement complet et trois sous-échantillons à 90 % :", [round(a, 2) for a in stabilite])
''')

code(r'''
# Vérification 3.5
verifier("trois mesures d'accord", lambda: len(stabilite) == 3)
verifier("regroupement assez stable (ARI > 0,5)", lambda: min(stabilite) > 0.5)
''')

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
## Partie 4 : repérer les incidents

Un incident, c'est un problème qui a **plus de plaintes qu'attendu** sur une marque et une période. Pour un problème $g$ et une case (marque $b$, période $t$), si les problèmes se répartissaient au hasard, on attendrait

$$e_{g,b,t} = \frac{n_g \times n_{b,t}}{N}$$

plaintes, où $n_g$ est la taille du problème, $n_{b,t}$ le nombre de plaintes de la marque $b$ à la période $t$, et $N$ le nombre total de plaintes. On compare le nombre observé $o_{g,b,t}$ à ce nombre attendu par une **loi de Poisson** : la p-valeur $P(X \ge o)$, avec $X$ de loi de Poisson de paramètre $e$, dit à quel point un tel excès serait improbable par hasard.

On teste des centaines de cases : certaines sortiront par hasard. On corrige par la procédure de **Benjamini-Hochberg**, qui contrôle la proportion de fausses découvertes parmi les cases retenues.

Un excès sur une case ne suffit pas : l'été, le chocolat fond chez toutes les marques, et chaque marque a sa petite bosse. On calcule donc aussi la **part de la marque dans le pic** :

$$\pi_{g,b,t} = \frac{o_{g,b,t}}{o_{g,t}}$$

où $o_{g,t}$ est le nombre de plaintes du problème $g$ à la période $t$, toutes marques confondues. Un incident de production touche une marque : $\pi$ est grand. Une saison touche tout le monde : $\pi$ est petit.
""")

md(r"""
### Exercice 4.1 : les cases anormales

Écris `cases_anormales(cles, seuil=0.01, o_min=5)` qui, pour les colonnes `cles`, calcule pour chaque problème et chaque case : observé, attendu, p-valeur, valeur corrigée (`q`), et renvoie les cases avec `q < seuil` et au moins `o_min` plaintes. Applique-la à `["Brand", "Year Period"]`, en ajoutant la colonne `part_pic`, puis à `["Year Period"]` seul.
""")

code(r'''
def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    ordre = np.argsort(p)
    q = p[ordre] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    sortie = np.empty_like(q)
    sortie[ordre] = np.minimum(q, 1)
    return sortie

def cases_anormales(cles, seuil=0.01, o_min=5):
    d = plaintes[plaintes.probleme >= 0]
    N = len(plaintes)
    obs = d.groupby(["probleme"] + cles).size().rename("observe").reset_index()
    n_g = d.groupby("probleme").size()
    n_case = plaintes.groupby(cles).size().rename("n_case").reset_index()
    obs = obs.merge(n_case, on=cles)
    obs["attendu"] = obs.probleme.map(n_g) * obs.n_case / N
    obs["p"] = poisson.sf(obs.observe - 1, obs.attendu)
    obs["q"] = benjamini_hochberg(obs.p)
    return obs[(obs.q < seuil) & (obs.observe >= o_min)].sort_values("q")

anomalies_marque = cases_anormales(["Brand", "Year Period"])
par_periode = plaintes[plaintes.probleme >= 0].groupby(["probleme", "Year Period"]).size()
anomalies_marque["part_pic"] = [o / par_periode[(g, t)] for g, t, o in
                                zip(anomalies_marque.probleme, anomalies_marque["Year Period"], anomalies_marque.observe)]
anomalies_periode = cases_anormales(["Year Period"])

fmt = {"attendu": "{:.1f}".format, "q": "{:.1e}".format, "part_pic": "{:.0%}".format}
print("Problèmes en excès sur une marque et une période :")
print(anomalies_marque[["probleme", "Brand", "Year Period", "observe", "attendu", "q", "part_pic"]]
      .to_string(index=False, formatters=fmt))
print("\nProblèmes en excès sur une période, toutes marques confondues :")
print(anomalies_periode[["probleme", "Year Period", "observe", "attendu", "q"]].to_string(index=False, formatters=fmt))
''')

code(r'''
# Vérification 4.1
verifier("des cases anormales sur marque et période", lambda: len(anomalies_marque) >= 1)
verifier("valeurs q et parts entre 0 et 1", lambda: anomalies_marque.q.between(0, 1).all()
         and anomalies_marque.part_pic.between(0, 1).all())
''')

md(r"""
### Exercice 4.2 : incident, pic partagé ou campagne ?

Deux autres cas que l'incident reviennent souvent :

- le **pic partagé** : un problème monte à une période chez beaucoup de marques à la fois, comme le chocolat qui fond l'été ;
- la **campagne** : beaucoup de consommateurs envoient presque le même message, par exemple après avoir scanné un produit dans une application de notation. Le groupe est gros et très homogène, mais il mesure une mobilisation, pas un défaut de production.

Une campagne se reconnaît à ses **messages quasi identiques**. Pour chaque plainte, calcule la similarité cosinus TF-IDF avec sa plus proche voisine (`plaintes["sim_voisine"]`) : un message de campagne a presque toujours une voisine au-dessus de 0,9, une plainte ordinaire rarement.

Classe ensuite chaque problème, dans cet ordre :

- **campagne** si plus de la moitié de ses plaintes ont une voisine au-dessus de 0,9 ;
- **incident candidat** s'il a une case anormale où la marque porte au moins la moitié du pic ($\pi \ge 0{,}5$) ;
- **pic partagé** s'il a une case ou une période anormale, sans marque qui porte le pic ;
- **fond** sinon.

Range les cases des incidents candidats dans `alertes`, et trace pour chacun l'évolution des plaintes du problème, pour la marque signalée et pour les autres.
""")

code(r'''
S = (normalize(X_tfidf) @ normalize(X_tfidf).T).toarray()
np.fill_diagonal(S, 0)
plaintes["sim_voisine"] = S.max(axis=1)
resume["quasi_doublons"] = plaintes[plaintes.probleme >= 0].groupby("probleme").sim_voisine.agg(lambda x: (x > 0.9).mean())

campagnes = set(resume.index[resume.quasi_doublons > 0.5])
alertes = anomalies_marque[(anomalies_marque.part_pic >= 0.5) & ~anomalies_marque.probleme.isin(campagnes)]
incidents = set(alertes.probleme)
pics = (set(anomalies_marque.probleme) | set(anomalies_periode.probleme)) - incidents - campagnes
resume["statut"] = ["campagne" if g in campagnes else "incident candidat" if g in incidents
                    else "pic partagé" if g in pics else "fond" for g in resume.index]
print(resume.statut.value_counts().to_string())
print("\nAlertes (incidents candidats) :")
print(alertes[["probleme", "Brand", "Year Period", "observe", "attendu", "q", "part_pic"]].to_string(index=False, formatters=fmt))
print()
print(resume[resume.statut != "fond"].sort_values("statut")[["taille", "langues", "statut", "quasi_doublons", "mots"]]
      .to_string(formatters={"quasi_doublons": "{:.0%}".format}))

periodes = sorted(plaintes["Year Period"].unique())
fig, axes = plt.subplots(max(len(incidents), 1), 1, figsize=(11, 2.9 * max(len(incidents), 1)), squeeze=False)
for ax, g in zip(axes[:, 0], sorted(incidents)):
    alertes_g = alertes[alertes.probleme == g]
    marque_g = alertes_g.Brand.iat[0]
    d = plaintes[plaintes.probleme == g]
    de_la_marque = d[d.Brand == marque_g]["Year Period"].value_counts().reindex(periodes, fill_value=0)
    des_autres = d[d.Brand != marque_g]["Year Period"].value_counts().reindex(periodes, fill_value=0)
    ax.bar(periodes, de_la_marque, color=ROUGE, width=0.75, label=marque_g, edgecolor=CREME, linewidth=0.6)
    ax.bar(periodes, des_autres, bottom=de_la_marque, color=LAIT, width=0.75, label="autres marques",
           edgecolor=CREME, linewidth=0.6)
    # les étiquettes se posent en haut du cadre, du côté où il y a de la place ; la légende va de l'autre côté
    ax.set_ylim(0, (de_la_marque + des_autres).max() * 1.9)
    a_droite = np.mean([periodes.index(t) for t in alertes_g["Year Period"]]) > len(periodes) / 2
    x_min, x_max = ax.get_xlim()
    for k, (_, a) in enumerate(alertes_g.sort_values("Year Period", ascending=not a_droite).iterrows()):
        x = periodes.index(a["Year Period"])
        fx = (x - x_min) / (x_max - x_min)
        ax.annotate(f"{a['Year Period']} : {a.observe} plaintes de {marque_g}\nau lieu de {a.attendu:.1f} attendues",
                    xy=(x, de_la_marque.iloc[x] + des_autres.iloc[x]),
                    xytext=(fx - 0.04 if a_droite else fx + 0.04, 0.95 - 0.2 * k), textcoords="axes fraction",
                    ha="right" if a_droite else "left", va="top", fontsize=8, color=BORDEAUX, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=BORDEAUX, lw=0.8, shrinkA=0, shrinkB=2))
    ax.set_title(f"Alerte sur le problème {g} : {', '.join(mots[g].split(', ')[:5])}")
    sous_titre(ax, f"en rouge, les plaintes de {marque_g} ; en beige, celles des autres marques pour le même problème")
    ax.tick_params(axis="x", rotation=90, labelsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="upper left" if a_droite else "upper right")
plt.tight_layout(); montrer(fig, "5_incidents_par_periode")

# Carte de chaleur : pour chaque problème et chaque période, plaintes observées / attendues
d = plaintes[plaintes.probleme >= 0]
obs = d.groupby(["probleme", "Year Period"]).size().unstack(fill_value=0).reindex(columns=periodes, fill_value=0)
attendu = np.outer(obs.sum(axis=1), plaintes["Year Period"].value_counts().reindex(periodes).values) / len(plaintes)
ratio = obs / attendu
ordre = resume.sort_values(["statut", "taille"], ascending=[True, False]).index
fig, ax = plt.subplots(figsize=(13, 0.42 * len(ordre) + 1.8))
im = ax.imshow(np.log2(ratio.loc[ordre].clip(lower=0.25)), cmap=CHOCOLAT_ROUGE, vmin=-2, vmax=3, aspect="auto")
ax.set_yticks(range(len(ordre)))
ax.set_yticklabels([f"{g}  {', '.join(mots[g].split(', ')[:3])}" for g in ordre], fontsize=8)
for etiquette, g in zip(ax.get_yticklabels(), ordre):
    etiquette.set_color({"incident candidat": ROUGE, "campagne": CACAO, "pic partagé": AMBRE}.get(resume.statut[g], TAUPE))
    etiquette.set_fontweight("bold" if resume.statut[g] != "fond" else "normal")
ax.set_xticks(range(len(periodes))); ax.set_xticklabels(periodes, rotation=90, fontsize=8); ax.grid(False)
ax.set_xticks(np.arange(-0.5, len(periodes)), minor=True); ax.set_yticks(np.arange(-0.5, len(ordre)), minor=True)
ax.grid(which="minor", color=CREME, lw=1.2); ax.tick_params(which="minor", length=0)
for _, a in alertes.iterrows():
    ax.add_patch(plt.Rectangle((periodes.index(a["Year Period"]) - 0.5, list(ordre).index(a.probleme) - 0.5), 1, 1,
                               fill=False, ec=CHOCOLAT_NOIR, lw=2.4))
barre = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, ticks=[-2, -1, 0, 1, 2, 3])
barre.ax.set_yticklabels(["÷4 ou aucune", "÷2", "comme attendu", "×2", "×4", "×8"], fontsize=8)
barre.set_label("plaintes observées / attendues", color=CHOCOLAT)
ax.set_title("Chaque problème, période par période")
sous_titre(ax, "rouge : plus de plaintes qu'attendu ; taupe pâle : moins ; cadres foncés : alertes. "
               "Noms en rouge : incidents, en ambre : pics partagés, en cacao : campagnes")
plt.tight_layout(); montrer(fig, "4_carte_chaleur_problemes_periodes")
''')

md(r"""
**Comment lire ces deux graphiques.**

*La carte de chaleur.* Une ligne par problème, une colonne par période. La couleur d'une case compare le nombre de plaintes de ce problème à cette période à ce qu'on attendrait si le problème se répartissait au hasard dans le temps : crème quand c'est comme attendu, rouge de plus en plus foncé quand il y en a deux, quatre, huit fois plus, taupe pâle quand il y en a moins ou aucune. Une ligne foncée sur une ou deux cases seulement est un problème qui surgit puis disparaît : c'est la signature d'un incident, d'une saison ou d'une campagne. Les cases encadrées sont les alertes : là, en plus, une seule marque porte l'essentiel du pic. La couleur du nom de chaque ligne donne le statut décidé à l'exercice 4.2.

*Les barres par alerte.* Pour chaque problème en alerte, le nombre de plaintes par période, en rouge pour la marque signalée, en beige pour toutes les autres marques. Un incident de production se lit d'un coup d'œil : une barre haute, presque entièrement rouge, au milieu de barres basses. Si la barre haute était surtout beige, ce serait un problème partagé par toutes les marques, et non un incident.
""")

code(r'''
# Vérification 4.2
verifier("chaque problème a un statut", lambda: resume.statut.notna().all())
verifier("au moins un incident candidat", lambda: len(alertes) >= 1)
verifier("similarité de chaque plainte à sa voisine calculée", lambda: plaintes.sim_voisine.between(0, 1.0001).all())
''')

md(r"""
**Réflexion 4.2.** Une alerte est levée sur une marque et une période. Que reste-t-il à faire avant de prévenir l'entreprise ?
""")

md(r"""
**Réponse.** Le test dit qu'un excès est improbable par hasard, pas qu'il y a un défaut de production. Il faut lire quelques plaintes du problème (ce que fera une personne habilitée, sur la base réelle), vérifier qu'elles décrivent bien la même chose, regarder si la date de l'événement confirme la période (la date de la plainte peut la suivre de plusieurs semaines), et confronter avec les informations de l'entreprise : lots, usines, changements de recette. L'outil trie et signale ; la décision reste humaine.
""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
## Partie 5 : la méthode retrouve-t-elle les incidents cachés ?

Sur la base fictive, on connaît la vérité : trois incidents ont été cachés (un goût de savon, des morceaux de plastique bleu, des chats malades après une nouvelle recette), chacun sur une marque, une ou deux périodes et plusieurs langues, plus deux groupes qui ne sont pas des incidents : un pic saisonnier (du chocolat fondu l'été 2025, sur toutes les marques Chocolate) et une campagne (le même message sur un additif, envoyé en plusieurs langues au printemps 2026, sur plusieurs marques). Leur liste est dans `verite/verite_incidents.csv`, rangé à part : ce fichier ne fait pas partie de la base, il sert seulement à cette vérification.

On mesure deux choses.

- **Le regroupement** : pour chaque groupe caché, le problème trouvé qui le recouvre le mieux, avec le **rappel** (part des plaintes du groupe caché qui y tombent) et la **précision** (part du problème trouvé qui appartient au groupe caché). On fait la même mesure sur les groupes d'avant la fusion des jumeaux (exercice 3.1), pour voir ce que la fusion apporte.
- **Les alertes** : chaque incident caché a-t-il déclenché une alerte sur la bonne marque et la bonne période, et combien d'alertes ne correspondent à aucun incident ?
""")

code(r'''
if BASE_FICTIVE:
    verite = pd.read_csv(trouver("verite_incidents.csv"), dtype=str)
    plaintes["cache"] = plaintes["Case Number"].map(dict(zip(verite["Case Number"], verite.groupe_cache))).fillna("")

    def evaluer(colonne):
        lignes = []
        for cache in sorted(plaintes.cache[plaintes.cache != ""].unique()):
            membres = plaintes[plaintes.cache == cache]
            candidats = membres[colonne][membres[colonne] >= 0].value_counts()
            if candidats.empty:
                lignes.append({"groupe caché": cache, "trouvé": None, "rappel": 0.0, "précision": 0.0})
                continue
            g = candidats.index[0]
            lignes.append({"groupe caché": cache, "trouvé": g, "rappel": candidats.iat[0] / len(membres),
                           "précision": candidats.iat[0] / int((plaintes[colonne] == g).sum())})
        return pd.DataFrame(lignes)

    eval_plongement = evaluer("probleme")
    eval_plongement["statut"] = eval_plongement["trouvé"].map(resume.statut)
    eval_avant = evaluer("groupe")
    fmt_eval = {"rappel": "{:.0%}".format, "précision": "{:.0%}".format}
    print("Après la fusion des jumeaux (problèmes) :")
    print(eval_plongement.to_string(index=False, formatters=fmt_eval))
    print("\nAvant la fusion des jumeaux (groupes de HDBSCAN) :")
    print(eval_avant.to_string(index=False, formatters=fmt_eval))

    attendues = {(l.marque, p) for _, l in verite[verite.type == "incident"].drop_duplicates("groupe_cache").iterrows()
                 for p in re.findall(r"\d{2} P\d{2}", l.periodes)}
    trouvees = set(zip(alertes.Brand, alertes["Year Period"]))
    print(f"\nAlertes attendues : {sorted(attendues)}")
    print(f"Alertes levées    : {sorted(trouvees)}")
    print(f"retrouvées : {len(attendues & trouvees)} sur {len(attendues)} ; fausses alertes : {len(trouvees - attendues)}")

    noms = {"C1_campagne": "campagne\nadditif", "I1_savon": "incident\ngoût de savon", "I2_plastique": "incident\nplastique bleu",
            "I3_recette": "incident\nrecette chat", "S1_fondu": "saison\nchocolat fondu"}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(eval_plongement))
    b1 = ax.bar(x - 0.2, eval_avant.rappel, 0.38, color=LAIT, label="avant la fusion des jumeaux", edgecolor=TAUPE, linewidth=0.6)
    b2 = ax.bar(x + 0.2, eval_plongement.rappel, 0.38, color=ROUGE, label="après la fusion des jumeaux")
    for barres, couleur in ((b1, TAUPE), (b2, BORDEAUX)):
        for b in barres:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.0%}", ha="center",
                    fontsize=8.5, color=couleur, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([noms.get(c, c) for c in eval_plongement["groupe caché"]], fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("rappel"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_title("Quelle part de chaque groupe caché finit dans un même groupe trouvé ?")
    sous_titre(ax, "plus la barre est haute, moins le groupe caché a été éparpillé")
    plt.tight_layout(); montrer(fig, "6_rappel_avant_apres_fusion")
else:
    print("Base réelle : pas de vérité connue, cette partie ne s'exécute pas.")
''')

code(r'''
# Vérification 5
if BASE_FICTIVE:
    par_cache = eval_plongement.set_index("groupe caché")
    verifier("chaque incident caché déclenche une alerte sur la bonne marque et la bonne période",
             lambda: attendues <= trouvees)
    verifier("au plus une fausse alerte", lambda: len(trouvees - attendues) <= 1)
    verifier("le pic saisonnier n'est pas pris pour un incident", lambda: par_cache.loc["S1_fondu", "statut"] != "incident candidat")
    verifier("la campagne est reconnue comme telle", lambda: par_cache.loc["C1_campagne", "statut"] == "campagne")
    verifier("la fusion des jumeaux ne disperse aucun incident",
             lambda: (par_cache.loc[["I1_savon", "I2_plastique", "I3_recette"], "rappel"]
                      >= eval_avant.set_index("groupe caché").loc[["I1_savon", "I2_plastique", "I3_recette"], "rappel"]).all())
''')

md(r"""
**Comment lire ce graphique.** Une paire de barres par groupe caché dans la base fictive. Chaque barre est le **rappel** : la part des plaintes de ce groupe caché qui se retrouvent ensemble dans un même groupe trouvé. 100 % veut dire que le groupe caché est entier dans un seul groupe trouvé ; 33 % qu'il est éparpillé, par exemple coupé en trois langues. La barre beige donne les groupes de HDBSCAN avant la fusion des jumeaux, la barre rouge les problèmes après la fusion. Quand les deux barres sont égales, le centrage par langue avait déjà réuni les langues de ce groupe caché, et la fusion n'avait rien à ajouter ; quand la rouge est plus haute, la fusion a recollé des morceaux. Le rappel ne dit pas si le groupe trouvé contient aussi d'autres plaintes : c'est la précision, dans le tableau au-dessus.
""")

md(r"""
**Réflexion 5.** Les problèmes trouvés pour les incidents ont une précision modeste : le problème qui contient le goût de savon contient aussi d'autres goûts bizarres. Pourquoi les alertes sont-elles quand même justes ?
""")

md(r"""
**Réponse.** Parce que l'alerte ne repose pas sur la pureté du groupe, mais sur un excès localisé. Le problème « goût étrange » regroupe des plaintes de toutes les marques et de toutes les périodes ; à l'intérieur, la case (marque 003, période 26 P05) en compte beaucoup plus qu'attendu, et cette marque porte presque tout le pic de la période. Un groupe un peu large dilue l'incident sans l'effacer. Si un incident était coupé en morceaux, par exemple un par langue, chaque morceau serait trop petit pour faire un excès significatif : c'est ce qu'évitent le centrage par langue et la fusion des jumeaux.
""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
## Partie 6 : exporter les résultats pour le tableau de bord

Le tableau de bord (`TP_plaintes/tableau_de_bord/tableau_de_bord.html`) est une page fixe : il affiche tout ce qu'on lui donne dans un seul fichier, `resultats.js`, écrit par cette cellule. On le télécharge depuis les sorties du notebook Kaggle (dossier `tableau_de_bord/`), on le pose à côté de la page, et on ouvre la page.

Le fichier ne contient que des **résultats** : des comptes par période, langue, segment, marque et problème ; les cases anormales ; une position en trois dimensions par plainte, avec sa langue, sa période et son problème. Il ne contient **ni verbatim, ni numéro de cas** : avant d'écrire, la cellule cherche chaque numéro de cas et chaque début de verbatim dans le texte de l'export. S'il en trouve un, **le fichier n'est pas écrit**, et la cellule affiche seulement le nom des champs en cause, jamais leur valeur.

La carte en trois dimensions est un t-SNE à trois composantes sur les plongements centrés : la même carte qu'à l'exercice 2.3, avec un axe de plus, que l'on peut faire tourner.

Les noms des problèmes sont tirés des verbatims. Sur la base réelle, `EXPORTER_MOTS` vaut `False` : dans `resultats.js`, les problèmes s'appellent « problème 5 », « problème 9 »… On ne le passe à `True` qu'avec l'accord écrit de l'entreprise.

Les mots sont écrits à part, dans `tableau_de_bord/noms_problemes.js`. Sur la base réelle, ce fichier est pour la personne habilitée seule : elle le télécharge elle-même et le glisse dans son tableau de bord, qui remplace alors les numéros par les mots, sur sa machine seulement. Il ne se joint à rien et ne se partage pas.
""")

code(r'''
import json, datetime

EXPORTER_MOTS = BASE_FICTIVE   # base réelle : True seulement avec l'accord écrit de l'entreprise

def nom_export(g):
    return ", ".join(mots[g].split(", ")[:5]) if EXPORTER_MOTS else f"problème {g}"

# en double précision : un nombre en 32 bits arrondi à 0,072 s'écrirait 0.07199999690055847, une longue suite
# de chiffres où un numéro de cas pourrait apparaître par hasard
carte_3d = TSNE(n_components=3, perplexity=30, init="pca", random_state=0).fit_transform(E_c).astype("float64")
carte_3d = (carte_3d - carte_3d.mean(axis=0)) / carte_3d.std()

langues_ordre = list(plaintes.langue.value_counts().index)
d = plaintes[plaintes.probleme >= 0]
obs_pp = d.groupby(["probleme", "Year Period"]).size().unstack(fill_value=0).reindex(columns=periodes, fill_value=0)
par_periode_total = plaintes["Year Period"].value_counts().reindex(periodes, fill_value=0)
seg_pp = (plaintes.groupby(["Business Segment", "Year Period"]).size().unstack(fill_value=0)
          .reindex(columns=periodes, fill_value=0))

def series_alerte(a):
    dg = plaintes[plaintes.probleme == a.probleme]
    marque = dg[dg.Brand == a.Brand]["Year Period"].value_counts().reindex(periodes, fill_value=0)
    autres = dg[dg.Brand != a.Brand]["Year Period"].value_counts().reindex(periodes, fill_value=0)
    return marque.astype(int).tolist(), autres.astype(int).tolist()

alertes_export = []
for _, a in alertes.iterrows():
    serie_marque, serie_autres = series_alerte(a)
    alertes_export.append({"probleme": int(a.probleme), "nom": nom_export(a.probleme), "marque": a.Brand,
                           "periode": a["Year Period"], "observe": int(a.observe), "attendu": round(float(a.attendu), 2),
                           "q": float(f"{a.q:.3g}"), "part_pic": round(float(a.part_pic), 3),
                           "serie_marque": serie_marque, "serie_autres": serie_autres})

evaluation = None
if BASE_FICTIVE:
    av = eval_avant.set_index("groupe caché")
    evaluation = {"groupes": [{"groupe": r["groupe caché"], "trouve": None if pd.isna(r["trouvé"]) else int(r["trouvé"]),
                               "statut": r.statut if isinstance(r.statut, str) else None,
                               "rappel": round(float(r.rappel), 3), "precision": round(float(r["précision"]), 3),
                               "rappel_avant_fusion": round(float(av.loc[r["groupe caché"], "rappel"]), 3)}
                              for _, r in eval_plongement.iterrows()],
                  "alertes_attendues": len(attendues), "alertes_retrouvees": len(attendues & trouvees),
                  "fausses_alertes": len(trouvees - attendues)}

tableau = {
    "format": "tableau_de_bord_plaintes", "version": 1,
    "base": "fictive" if BASE_FICTIVE else "réelle",
    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "modele": nom_modele,
    "chiffres": {"plaintes": int(len(plaintes)), "marques": int(plaintes.Brand.nunique()),
                 "segments": int(seg_pp.shape[0]), "periodes": len(periodes), "langues": len(langues_ordre),
                 "groupes": int(n_groupes), "problemes": int(n_problemes), "part_cote": round(part_bruit, 4),
                 "alertes": int(len(alertes)), "incidents": len(incidents), "campagnes": len(campagnes),
                 "pics": len(pics), "stabilite": [round(float(a), 3) for a in stabilite],
                 "meme_langue": {"tfidf": round(float(meme_langue_tfidf), 3), "brut": round(float(meme_langue_plongement), 3),
                                 "centre": round(float(meme_langue_centre), 3)}},
    "periodes": periodes,
    "plaintes_par_periode": par_periode_total.astype(int).tolist(),
    "langues": {l: int(n) for l, n in plaintes.langue.value_counts().items()},
    "segments": {s: seg_pp.loc[s].astype(int).tolist() for s in seg_pp.index},
    "problemes": [{"id": int(g), "nom": nom_export(g), "taille": int(r.taille), "statut": r.statut,
                   "marque": r.marque, "part_marque": round(float(r.part_marque), 3),
                   "quasi_doublons": round(float(r.quasi_doublons), 3),
                   "langues": {l: int(n) for l, n in d[d.probleme == g].langue.value_counts().items()},
                   "observe": obs_pp.loc[g].astype(int).tolist(),
                   "attendu": [round(float(v), 3) for v in obs_pp.loc[g].sum() * par_periode_total.values / len(plaintes)]}
                  for g, r in resume.iterrows()],
    "alertes": alertes_export,
    "pics_periode": [{"probleme": int(a.probleme), "periode": a["Year Period"], "observe": int(a.observe),
                      "attendu": round(float(a.attendu), 2), "q": float(f"{a.q:.3g}")} for _, a in anomalies_periode.iterrows()],
    "points": {"langues": langues_ordre,
               "x": np.round(carte_3d[:, 0], 3).tolist(), "y": np.round(carte_3d[:, 1], 3).tolist(),
               "z": np.round(carte_3d[:, 2], 3).tolist(),
               "probleme": plaintes.probleme.astype(int).tolist(),
               "langue": [langues_ordre.index(l) for l in plaintes.langue],
               "periode": [periodes.index(t) for t in plaintes["Year Period"]]},
    "evaluation": evaluation,
}

def chemins_de_fuite(objet, contenu):
    """Champs de l'export (chemins JSON, jamais de valeur) où apparaît un numéro de cas ou un début de verbatim."""
    sensibles = ({str(c) for c in plaintes["Case Number"].astype(str) if len(str(c)) >= 5}
                 | {t[:40] for t in plaintes.Verbatim.astype(str) if len(t) >= 40})
    if not any(s in contenu for s in sensibles):
        return []
    chemins = set()
    def parcourir(x, chemin):
        if isinstance(x, dict):
            for k, v in x.items():
                if any(s in str(k) for s in sensibles):
                    chemins.add(chemin + ".<clé>")
                parcourir(v, f"{chemin}.{k}")
        elif isinstance(x, list):
            for v in x:
                parcourir(v, chemin + "[]")
        elif any(s in json.dumps(x, ensure_ascii=False) for s in sensibles):
            chemins.add(chemin)
    parcourir(objet, "")
    return sorted(chemins) or ["<à cheval entre deux valeurs voisines>"]

def arrondir(x):
    """Arrondit tous les nombres à virgule à 4 décimales : un nombre court ne peut pas contenir un numéro de cas par hasard."""
    if isinstance(x, dict):
        return {k: arrondir(v) for k, v in x.items()}
    if isinstance(x, list):
        return [arrondir(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.4g}") if abs(x) < 1e-3 else round(float(x), 4)
    return x

def ecrire_export(objet, nom, variable):
    """Écrit tableau_de_bord/<nom>.js et .json seulement si aucun numéro de cas ni verbatim n'y figure."""
    DOSSIER_TABLEAU.mkdir(exist_ok=True)
    objet = arrondir(objet)
    texte = json.dumps(objet, ensure_ascii=False, separators=(",", ":"))
    fuites = chemins_de_fuite(objet, texte)
    for f in (DOSSIER_TABLEAU / f"{nom}.js", DOSSIER_TABLEAU / f"{nom}.json"):
        f.unlink(missing_ok=True)
    if fuites:
        # aucune valeur n'est affichée : seulement le nom des champs en cause et la forme des numéros de cas
        longueurs = plaintes["Case Number"].astype(str).str.len()
        print(f"DIAGNOSTIC : {nom} NON écrit ; champs en cause : {fuites}")
        print(f"DIAGNOSTIC : numéros de cas, longueur min {longueurs.min()}, médiane {longueurs.median():.0f}, max {longueurs.max()} ; "
              f"part entièrement en chiffres {plaintes['Case Number'].astype(str).str.fullmatch(r'[0-9]+').mean():.0%}")
        return texte, fuites
    (DOSSIER_TABLEAU / f"{nom}.json").write_text(texte, encoding="utf-8")
    (DOSSIER_TABLEAU / f"{nom}.js").write_text(f"window.{variable} = " + texte + ";\n", encoding="utf-8")
    print(f"tableau_de_bord/{nom}.js écrit : {len(texte) / 1024:.0f} ko")
    return texte, []

DOSSIER_TABLEAU = CACHE / "tableau_de_bord"
contenu, fuites_resultats = ecrire_export(tableau, "resultats", "RESULTATS")

# Les mots de chaque problème, à part : sur la base réelle, ce fichier est pour l'encadrante seule. Elle le
# télécharge elle-même et le glisse dans son tableau de bord ; il ne se joint à rien et ne se partage pas.
noms_problemes = {"format": "noms_problemes", "date": tableau["date"], "modele_langue": nom_llm,
                  "noms": {int(g): ", ".join(mots[g].split(", ")[:6]) for g in resume.index},
                  "definitions": definitions}
(DOSSIER_TABLEAU / "noms_problemes.js").write_text(
    "window.NOMS_PROBLEMES = " + json.dumps(noms_problemes, ensure_ascii=False) + ";\n", encoding="utf-8")
print("tableau_de_bord/noms_problemes.js écrit (à garder pour soi sur la base réelle)")
print(f"{len(tableau['problemes'])} problèmes, {len(alertes_export)} alertes, {len(carte_3d)} points")
''')

code(r'''
# Vérification 6
verifier("aucun numéro de cas ni verbatim dans l'export", lambda: not fuites_resultats)
verifier("le fichier du tableau de bord est écrit", lambda: (DOSSIER_TABLEAU / "resultats.js").stat().st_size > 1000)
verifier("une position 3D par plainte", lambda: len(tableau["points"]["x"]) == len(plaintes))
''')

# ════════════════════════════════════════════════════════════════════════════════════════════════
md(r"""
## Partie 8 : changer de base

Le notebook lit la base réelle par défaut : il faut que le jeu de données **privé** qui la contient soit attaché (**Add Input**). Sur elle, les verbatims ne s'affichent pas, les vérifications propres à la base fictive sont sautées et la partie 5 ne s'exécute pas.

Pour s'entraîner sur la base fictive : attache le jeu de la base fictive, mets `BASE_FICTIVE = True` et `NOM_FICHIER = "plaintes_fictives.xlsx"` dans la cellule 0.1, puis exécute tout. Les textes s'affichent alors, et la partie 5 mesure si la méthode retrouve les incidents cachés.

**Ce qu'on peut sortir de Kaggle** : le rapport de lecture (partie 1), les comptes par langue, marque et période, le nombre et la taille des groupes, la stabilité, la liste des cases anormales (groupe, marque, période, observé, attendu, q), et les fichiers `tableau_de_bord/resultats.js` (partie 6) et `tableau_de_bord/comparaison.js` (partie 7). **Ce qui ne sort pas** : les verbatims, les numéros de cas, les plongements (`plongements_*.npy`), et tout fichier qui en contient.

Les mots caractéristiques des groupes (exercice 3.4) sont tirés des verbatims : on les consulte dans Kaggle, et on ne les recopie à l'extérieur qu'avec l'accord de l'entreprise. C'est pourquoi `EXPORTER_MOTS` vaut `False` sur la base réelle.
""")


def main():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    import cellules_comparaison
    partie8 = next(i for i, c in enumerate(CELLULES) if c["cell_type"] == "markdown" and c["source"].lstrip().startswith("## Partie 8"))
    toutes = CELLULES[:partie8] + cellules_comparaison.CELLULES + CELLULES[partie8:]
    for i, c in enumerate(toutes):
        c = dict(c)
        c["id"] = f"c{i:03d}"
        c["source"] = c["source"].splitlines(keepends=True)
        nb["cells"].append(c)
    sortie = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "analyse_plaintes_consommateurs_bis.ipynb"
    sortie.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(sortie, len(toutes), "cellules")


if __name__ == "__main__":
    main()
