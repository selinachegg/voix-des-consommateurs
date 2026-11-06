# Cellules de la partie 7 du TP plaintes : comparer les représentations (TF-IDF et sept transformers) à chaîne égale.
# Branchées dans le notebook par construire_notebook.py, juste avant la partie 8 (base réelle).
CELLULES = []


def md(texte):
    CELLULES.append({"cell_type": "markdown", "metadata": {}, "source": texte.strip("\n")})


def code(texte):
    CELLULES.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": texte.strip("\n")})


md(r"""
# Deuxième grande partie : la chaîne de comparaison

La chaîne principale est faite et validée : elle a produit `resultats.js` pour le tableau de bord. On la rejoue maintenant, étape par étape, avec huit outils différents. Seul l'outil qui transforme chaque plainte en vecteur change ; tout le reste (centrage, réglage, ACP, HDBSCAN, fusion, cases anormales) est identique. Le résultat est `comparaison.js`, la rubrique « Comparer les outils » du tableau de bord.
""")

code(r'''
# 7.0  Bilan de la chaîne principale, avant de lancer la comparaison
a_revoir = [nom for nom, reussi in VERIFICATIONS if not reussi]
print(f"Chaîne principale : {len(VERIFICATIONS) - len(a_revoir)} vérifications [OK] sur {len(VERIFICATIONS)}")
for nom in a_revoir:
    print(f"  [À revoir]  {nom}")
print("resultats.js écrit :", (DOSSIER_TABLEAU / "resultats.js").exists())
''')

code(r'''
# Vérification 7.0
verifier("la chaîne principale a écrit resultats.js", lambda: (DOSSIER_TABLEAU / "resultats.js").exists())
''')

md(r"""
**Avant d'aller plus loin.** Lis chaque ligne [À revoir] ci-dessus. Une vérification qui ne passe pas n'est pas forcément une erreur (par exemple, aucune alerte sur une base où il n'y a pas d'incident), mais il faut savoir l'expliquer. Si `resultats.js` n'est pas écrit, reviens à la chaîne principale : comparer des outils sur une chaîne qui ne marche pas ne dit rien.
""")

md(r"""
## Partie 7 : comparer les outils, du comptage de mots aux transformers

La chaîne des parties 2 à 4 a été construite avec un seul modèle de plongement. Cette partie la reprend telle quelle et ne change qu'une chose : la façon de transformer chaque plainte en vecteur. On compare ainsi, à chaîne égale, le TF-IDF et sept **transformers** multilingues, de 118 millions à 4 milliards de paramètres, en temps, en mémoire, en coût et en qualité. Aucun de ces modèles ne génère de texte, et tous tournent dans le notebook : aucune plainte n'est envoyée à un service extérieur.

### Exercice 7.1 : la même chaîne pour chaque représentation

La fonction `chaine(E)` reprend les exercices 2.3 à 4.2 du TP, sans les graphiques : centrage par langue, ACP à 20 dimensions, HDBSCAN, fusion des jumeaux, cases anormales par marque et période, campagnes, alertes. Elle renvoie les problèmes et les alertes.
""")

code(r'''
import time, gc, json, datetime
import torch
# le modèle de la partie 2 occupe encore la carte graphique : on le libère avant la comparaison
globals().pop("modele", None); gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
from scipy.stats import poisson
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

langues = plaintes.langue.values
X_tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2).fit_transform(plaintes.texte)
S = (normalize(X_tfidf) @ normalize(X_tfidf).T).toarray(); np.fill_diagonal(S, 0)
sim_voisine = S.max(axis=1)          # pour reconnaître les campagnes (messages quasi identiques)
del S

def part_voisin_meme_langue(X):
    S = X @ X.T; np.fill_diagonal(S, -np.inf)
    return float((langues == langues[S.argmax(axis=1)]).mean())

def centrer_par_langue(E, n_min=10):
    C = E.astype("float64").copy(); moyenne = C.mean(axis=0)
    for l in np.unique(langues):
        m = langues == l
        C[m] -= C[m].mean(axis=0) if m.sum() >= n_min else moyenne
    return normalize(C)

# regrouper, fusionner et choisir_reglages sont ceux de la partie 3 : chaque modèle reçoit son propre réglage

def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    s = np.empty_like(q); s[o] = np.minimum(q, 1); return s

def alertes_de(probleme):
    d = plaintes.assign(probleme=probleme)
    g = d[d.probleme >= 0]
    obs = g.groupby(["probleme", "Brand", "Year Period"]).size().rename("observe").reset_index()
    n_case = d.groupby(["Brand", "Year Period"]).size().rename("n_case").reset_index()
    obs = obs.merge(n_case, on=["Brand", "Year Period"])
    obs["attendu"] = obs.probleme.map(g.groupby("probleme").size()) * obs.n_case / len(d)
    obs["q"] = benjamini_hochberg(poisson.sf(obs.observe - 1, obs.attendu))
    obs = obs[(obs.q < 0.01) & (obs.observe >= 5)]
    par_periode = g.groupby(["probleme", "Year Period"]).size()
    obs["part_pic"] = [o / par_periode[(p, t)] for p, t, o in zip(obs.probleme, obs["Year Period"], obs.observe)]
    quasi = pd.Series(sim_voisine[probleme >= 0], index=g.index).groupby(g.probleme).agg(lambda x: (x > 0.9).mean())
    campagnes = set(quasi.index[quasi > 0.5])
    return obs[(obs.part_pic >= 0.5) & ~obs.probleme.isin(campagnes)], campagnes

def chaine(E):
    E_c = centrer_par_langue(E)
    reglages, essais = choisir_reglages(E_c)             # le réglage est choisi pour ce modèle, comme en 3.3
    groupes = regrouper(E_c, reglages["taille_min"], reglages["methode"], reglages["k"])
    probleme = fusionner(groupes, E_c, reglages["cosinus_min"])
    alertes, campagnes = alertes_de(probleme)
    retenu = evaluer_reglage(E_c, reglages["taille_min"], reglages["methode"], reglages["cosinus_min"], reglages["k"])
    return {"E_c": E_c, "groupes": groupes, "probleme": probleme, "alertes": alertes, "campagnes": campagnes,
            "stabilite": float(retenu["stabilite"]), "reglages": reglages, "reglage_admis": bool(essais.admis.any())}
print("chaîne prête")
''')

md(r"""
### Exercice 7.2 : les modèles comparés

| représentation | type | paramètres | licence |
|---|---|---|---|
| TF-IDF + SVD à 100 dimensions | comptage de mots, sans apprentissage | aucun | |
| paraphrase-multilingual-MiniLM-L12-v2 | transformer (sentence-transformers) | 118 M | Apache 2.0 |
| multilingual-e5-small | transformer | 118 M | MIT |
| multilingual-e5-base | transformer | 278 M | MIT |
| multilingual-e5-large | transformer | 560 M | MIT |
| BGE-M3 | transformer | 568 M | MIT |
| Qwen3-Embedding-0.6B | transformer dérivé d'un LLM | 0,6 G | Apache 2.0 |
| Qwen3-Embedding-4B | transformer dérivé d'un LLM | 4 G | Apache 2.0 |

Le nombre de paramètres affiché dans les résultats est compté sur le modèle chargé. Le **temps de chargement** comprend le téléchargement du modèle la première fois ; le **temps de calcul** est celui des vecteurs seuls, carte graphique synchronisée.
""")

code(r'''
CONSIGNE_QWEN = "Instruct: Identify the product problem described in this consumer complaint\nQuery: "
MODELES_COMPARES = [
    # nom affiché, identifiant Hugging Face, préfixe, taille de lot, demi-précision
    ("TF-IDF", None, "", 0, False),
    ("MiniLM-L12 multilingue", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", 64, False),
    ("e5-small multilingue", "intfloat/multilingual-e5-small", "query: ", 64, False),
    ("e5-base multilingue", "intfloat/multilingual-e5-base", "query: ", 64, False),
    ("e5-large multilingue", "intfloat/multilingual-e5-large", "query: ", 32, True),
    ("BGE-M3", "BAAI/bge-m3", "", 32, True),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", CONSIGNE_QWEN, 32, True),
    ("Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-4B", CONSIGNE_QWEN, 8, True),
]
LICENCES = {"TF-IDF": "", "MiniLM-L12 multilingue": "Apache 2.0", "e5-small multilingue": "MIT", "e5-base multilingue": "MIT",
            "e5-large multilingue": "MIT", "BGE-M3": "MIT", "Qwen3-Embedding-0.6B": "Apache 2.0", "Qwen3-Embedding-4B": "Apache 2.0"}

from sentence_transformers import SentenceTransformer
appareil = "cuda" if torch.cuda.is_available() else "cpu"
carte = torch.cuda.get_device_name(0) if appareil == "cuda" else "processeur"
print("carte graphique :", carte)

def vecteurs(nom, ident, prefixe, lot, demi):
    """Renvoie les vecteurs et les mesures (temps, mémoire, paramètres) ; None si le modèle ne se charge pas."""
    if ident is None:
        t0 = time.perf_counter()
        E = normalize(TruncatedSVD(100, random_state=0).fit_transform(X_tfidf))
        return E, {"t_chargement": 0.0, "t_calcul": time.perf_counter() - t0, "memoire_go": 0.0, "parametres_millions": 0.0,
                   "dimension": E.shape[1], "appareil": "processeur"}
    if appareil == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    options = {"torch_dtype": torch.float16} if (demi and appareil == "cuda") else {}
    modele = SentenceTransformer(ident, device=appareil, model_kwargs=options)
    modele.max_seq_length = min(modele.max_seq_length or 512, 512)   # mêmes textes coupés à 512 jetons pour tous
    t_chargement = time.perf_counter() - t0
    textes = [prefixe + t for t in plaintes.texte]
    modele.encode(textes[:16], batch_size=lot)                    # chauffe : la première passe est toujours plus lente
    if appareil == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    E = modele.encode(textes, batch_size=lot, normalize_embeddings=True, show_progress_bar=False)
    if appareil == "cuda": torch.cuda.synchronize()
    t_calcul = time.perf_counter() - t1
    mesures = {"t_chargement": t_chargement, "t_calcul": t_calcul,
               "memoire_go": torch.cuda.max_memory_allocated() / 1e9 if appareil == "cuda" else 0.0,
               "parametres_millions": round(sum(p.numel() for p in modele.parameters()) / 1e6, 1), "dimension": int(E.shape[1]),
               "appareil": carte}
    del modele; gc.collect()
    if appareil == "cuda": torch.cuda.empty_cache()
    return np.asarray(E, dtype="float32"), mesures
''')

md(r"""
### Exercice 7.3 : mesurer

Pour chaque modèle : les vecteurs, la part des plus proches voisines dans la même langue (avant et après centrage), puis la chaîne complète. Sur la base fictive, on compare aussi aux incidents cachés : **rappel** de chaque groupe caché (la part de ses plaintes réunies dans un même problème), alertes retrouvées et fausses alertes.
""")

code(r'''
if BASE_FICTIVE:
    verite = pd.read_csv(trouver("verite_incidents.csv"), dtype=str)
    cache = plaintes["Case Number"].map(dict(zip(verite["Case Number"], verite.groupe_cache))).fillna("").values
    attendues = {(l.marque, p) for _, l in verite[verite.type == "incident"].drop_duplicates("groupe_cache").iterrows()
                 for p in re.findall(r"\d{2} P\d{2}", l.periodes)}

resultats = []
for nom, ident, prefixe, lot, demi in MODELES_COMPARES:
    try:
        E, m = vecteurs(nom, ident, prefixe, lot, demi)
    except Exception as erreur:
        print(f"{nom} : non chargé ({type(erreur).__name__} : {str(erreur)[:120]})")
        gc.collect(); torch.cuda.empty_cache() if appareil == "cuda" else None
        continue
    t0 = time.perf_counter()
    r = chaine(E)
    m["t_chaine"] = time.perf_counter() - t0
    m.update({"nom": nom, "identifiant": ident or "", "licence": LICENCES[nom], "transformer": ident is not None,
              "meme_langue_brut": part_voisin_meme_langue(normalize(E)), "meme_langue_centre": part_voisin_meme_langue(r["E_c"]),
              "groupes": int(len(set(r["groupes"]) - {-1})), "problemes": int(len(set(r["probleme"]) - {-1})),
              "part_cote": float((r["probleme"] == -1).mean()), "stabilite": r["stabilite"], "alertes": int(len(r["alertes"])),
              "reglages": r["reglages"], "reglage_admis": r["reglage_admis"],
              "part_max": float(np.bincount(r["probleme"][r["probleme"] >= 0]).max() / len(E)) if (r["probleme"] >= 0).any() else 1.0})
    if BASE_FICTIVE:
        rappels = {}
        for g in sorted(set(cache) - {""}):
            p = r["probleme"][cache == g]; p = p[p >= 0]
            rappels[g] = float(np.bincount(p).max() / (cache == g).sum()) if len(p) else 0.0
        trouvees = set(zip(r["alertes"].Brand, r["alertes"]["Year Period"]))
        membres = cache != ""
        m.update({"rappels": rappels, "rappel_incidents": float(np.mean([v for g, v in rappels.items() if g.startswith("I")])),
                  "alertes_retrouvees": len(attendues & trouvees), "alertes_attendues": len(attendues),
                  "fausses_alertes": len(trouvees - attendues),
                  "ari_verite": float(adjusted_rand_score(cache[membres], r["probleme"][membres]))})
    resultats.append(m)
    print(f"{nom:24s} calcul {m['t_calcul']:6.1f} s  mémoire {m['memoire_go']:4.1f} Go  problèmes {m['problemes']:3d}  "
          + (f"rappel incidents {m['rappel_incidents']:.0%}  alertes {m['alertes_retrouvees']}/{m['alertes_attendues']}  fausses {m['fausses_alertes']}"
             if BASE_FICTIVE else f"stabilité {m['stabilite']:.2f}"))
    del E, r; gc.collect()
''')

code(r'''
# Vérification 7.3
verifier("au moins cinq représentations mesurées", lambda: len(resultats) >= 5)
verifier("le TF-IDF est mesuré", lambda: any(not m["transformer"] for m in resultats))
verifier("des temps positifs", lambda: all(m["t_calcul"] > 0 for m in resultats))
if BASE_FICTIVE:
    verifier("le modèle du TP retrouve les quatre alertes",
             lambda: any(m["nom"] == "Qwen3-Embedding-0.6B" and m["alertes_retrouvees"] == m["alertes_attendues"] for m in resultats))
''')

md(r"""
### Exercice 7.4 : le coût, et l'export pour le tableau de bord

Le coût d'un modèle est son temps de carte graphique (chargement compris), multiplié par un tarif horaire. Le tarif écrit ici est indicatif ; le tableau de bord permet de le changer. On extrapole aussi le temps de calcul des vecteurs à 10 000 et 100 000 plaintes, en supposant qu'il croît comme le nombre de plaintes.
""")

code(r'''
TARIF_HEURE = 0.35   # dollars par heure de carte graphique T4 chez un loueur en ligne : valeur indicative, modifiable dans le tableau de bord

for m in resultats:
    m["secondes_par_1000"] = 1000 * m["t_calcul"] / len(plaintes)

comparaison = {"format": "comparaison_modeles", "version": 1, "base": "fictive" if BASE_FICTIVE else "réelle",
               "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "plaintes": int(len(plaintes)),
               "carte": carte, "tarif_heure": TARIF_HEURE, "modeles": resultats}
contenu, fuites_comparaison = ecrire_export(comparaison, "comparaison", "COMPARAISON")   # même garde qu'en partie 6

tableau = pd.DataFrame(resultats).set_index("nom")
colonnes = ["parametres_millions", "t_chargement", "t_calcul", "memoire_go", "problemes", "stabilite"] + \
           (["rappel_incidents", "alertes_retrouvees", "fausses_alertes", "ari_verite"] if BASE_FICTIVE else [])
print(tableau[colonnes].to_string(float_format=lambda v: f"{v:.2f}"))
''')

code(r'''
# Vérification 7.4
verifier("aucun numéro de cas ni verbatim dans l'export", lambda: not fuites_comparaison)
verifier("le fichier de comparaison est écrit", lambda: (DOSSIER_TABLEAU / "comparaison.js").stat().st_size > 500)
''')
