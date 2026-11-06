# Voix des consommateurs

Regrouper des plaintes multilingues, nommer les problèmes et repérer les incidents. Projet
conçu par Selina Cheggour pour l'École d'automne IA 2026. Le dépôt ne contient que des données
fictives : `donnees_fictives/plaintes_fictives.xlsx`.

*Unsupervised analysis of multilingual consumer complaints: language-centred embeddings,
HDBSCAN clustering, local-LLM problem definitions, and incident detection with a Poisson test
and Benjamini-Hochberg correction. Designed by Selina Cheggour for the 2026 AI Fall School.
This repository contains fictional data only.*

## Chaîne de traitement

1. Lecture et nettoyage de l'export, détection des langues.
2. Plongements multilingues, centrage par langue, réduction par ACP.
3. Regroupement HDBSCAN, fusion des groupes jumeaux, réglage automatique, stabilité (ARI).
4. Définition des problèmes par un modèle de langue local.
5. Détection des incidents : loi de Poisson, correction de Benjamini-Hochberg, part du pic.
6. Export vers le tableau de bord, puis comparaison de plusieurs modèles de plongement.

Les notebooks s'exécutent sur GPU, par exemple sur Kaggle avec une carte T4 : attacher la base
fictive au notebook, puis tout exécuter. Le tableau de bord s'ouvre avec
`tableau_de_bord/tableau_de_bord.html`.

| Dossier | Contenu |
|---|---|
| `notebooks/` | notebook étudiants et notebook complet |
| `outils/` | générateur de la base fictive et construction des notebooks |
| `docs_etudiants/` | fiche des étudiants (PDF et sources LaTeX) |
| `donnees_fictives/` | base fictive et incidents cachés |
| `tableau_de_bord/` | tableau de bord, avec les résultats de la base fictive |

Pour reconstruire les notebooks, depuis `outils/` : `construire_notebook.py`, puis
`construire_version_etudiants.py`.

## Licence

Tous droits réservés. Voir [LICENSE](LICENSE) et, pour les éléments de tiers, [NOTICE](NOTICE).
