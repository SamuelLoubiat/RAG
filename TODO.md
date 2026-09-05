# RAG against the machine — TODO

> Sujet : `en.subject.pdf` — **Version 2.0**

## Résumé du projet

Construire un système **Retrieval-Augmented Generation (RAG)** en Python capable de répondre à des questions sur le dépôt vLLM (fourni en pièce jointe, `data/raw/vllm-0.10.1/`). Le pipeline complet : ingestion → indexation → recherche → génération de réponse → évaluation.

---

## 1. Setup du projet

- [x] Initialiser le projet avec `uv` (`pyproject.toml` présent)
- [x] Créer `src/` directory
- [x] Créer `.gitignore` (partiel)
- [x] Créer le `Makefile` avec les règles `install`, `run`, `debug`, `clean`, `lint`
- [ ] Ajouter la règle `lint-strict` au Makefile (optionnel, recommandé : `flake8 .` + `mypy . --strict`)
- [x] Ajouter les dépendances dans `pyproject.toml` (`transformers`, `fire`, `tqdm`, `pydantic`, `flake8`, `mypy` — `bm25s` en attente du choix BM25 vs TF-IDF)
- [x] S'assurer que `uv sync` seul suffit à installer le projet (le correcteur et la moulinette ne lancent que `uv sync`, rien d'autre)
- [ ] Créer la structure de dossiers data attendue par la moulinette (voir "Structure de dépôt attendue" plus bas)
- [ ] Compléter `.gitignore` (ajouter `data/`, poids de modèles, outputs générés — rien de tout ça ne doit être commit, ça peut peser plusieurs Go)

---

## 2. Data Models (pydantic)

- [x] Implémenter `MinimalSource` (`file_path: str`, `first_character_index: int`, `last_character_index: int`)
- [x] Implémenter `UnansweredQuestion` (`question_id: str` avec `default_factory=lambda: str(uuid.uuid4())`, `question: str`)
- [x] Implémenter `AnsweredQuestion(UnansweredQuestion)` (`sources: List[MinimalSource]`, `answer: str`)
- [x] Implémenter `RagDataset` (`rag_questions: List[AnsweredQuestion | UnansweredQuestion]`)
- [x] Implémenter `MinimalSearchResults` (`question_id: str`, `question: str`, `retrieved_sources: List[MinimalSource]`)
- [x] Implémenter `MinimalAnswer(MinimalSearchResults)` (`answer: str`)
- [x] Implémenter `StudentSearchResults` (`search_results: List[MinimalSearchResults]`, `k: int`)
- [x] Implémenter `StudentSearchResultsAndAnswer` (`search_results: List[MinimalAnswer]`, `k: int`)
- [x] Tous les modèles ont des docstrings PEP 257 et passent `flake8`/`mypy` (`src/models.py`)
- [ ] Ces modèles sont une base : on peut les étendre (champs/modèles supplémentaires) si besoin, mais pas casser la base

---

## 3. Knowledge Base Ingestion System

- [ ] Lire et parcourir tous les fichiers utiles du dépôt vLLM (`data/raw/vllm-0.10.1/`)
- [ ] Implémenter le **chunking Python** (par fonctions/classes avec `ast`)
- [ ] Implémenter le **chunking texte/Markdown** (par paragraphes/sections)
- [ ] Taille de chunk max : 2000 caractères par défaut, **configurable via `--max_chunk_size`**
- [ ] Ne jamais dépasser `--max_chunk_size` : la moulinette rejette toute source récupérée de plus de 2000 caractères (`max_context_length`), et une seule source trop longue invalide toute la sortie
- [ ] Stocker les chunks avec leurs `file_path` (chemin exact, relatif à la racine du projet, tel qu'ingéré — ex. `data/raw/vllm-0.10.1/docs/features/lora.md`), `first_character_index`, `last_character_index`
- [ ] Persister l'index sous `data/processed/`
- [ ] Indexation complète en **moins de 5 minutes**

---

## 4. Retrieval System

- [ ] Implémenter **au moins une** méthode lexicale classique : **BM25** (via `bm25s`) ou **TF-IDF** (les deux méthodes d'indexation/recherche restent au choix, on peut ajouter d'autres méthodes en plus)
- [ ] Sauvegarder l'index dans `data/processed/` pour éviter de recalculer
- [ ] Retourner les top-k chunks les plus pertinents pour une requête
- [ ] Supporter le **batch processing** de datasets JSON (`search_dataset`)
- [ ] Throughput : **200 questions traitées en < 90 secondes** *(⚠️ changé : c'était 1000 questions dans l'ancien TODO, le sujet v2.0 dit 200)*
- [ ] Atteindre **Recall@5 ≥ 80%** sur les questions "docs" et **≥ 50%** sur les questions "code"
- [ ] `file_path` doit matcher **exactement** le chemin du corpus ingéré (comparaison verbatim par le correcteur)

> ℹ️ La latence "cold start < 60s" qui figurait dans l'ancien TODO n'apparaît plus dans les critères de performance du sujet v2.0 (section VII.1.2 : seulement indexing time, throughput, recall@5).

---

## 5. Answer Generation System

- [ ] Charger le modèle **Qwen/Qwen3-0.6B** (via `transformers`, modèle par défaut obligatoire)
- [ ] Autres modèles autorisés en plus, tant que tout fonctionne toujours avec Qwen/Qwen3-0.6B
- [ ] Passer le contexte récupéré au LLM dans les limites de tokens
- [ ] Générer des réponses en JSON structuré conforme à `StudentSearchResultsAndAnswer` / `MinimalAnswer`
- [ ] La réponse doit être : cohérente, compréhensible, groundée dans les sources récupérées (pas d'hallucination majeure), pertinente par rapport à la question posée
- [ ] Ajouter une barre de progression `tqdm` pour le traitement des datasets

---

## 6. Evaluation System

- [ ] Implémenter la commande `evaluate` (calcul de **recall@k**, k = 1, 3, 5, 10) — **pour son propre usage/itération uniquement**
- [ ] Comparer les sources récupérées aux annotations ground truth (dataset `AnsweredQuestions`)
- [ ] Une source est "trouvée" si même `file_path` **et** chevauchement (IoU) ≥ 5% avec la plage de référence
- [ ] Score par question : `nombre_trouvées / total_sources_correctes`
- [ ] Afficher un rapport de performance complet
- [ ] **Ne jamais importer ni appeler la moulinette** dans le code du projet : le recall@k officiel de la soutenance est calculé par l'exécutable `moulinette` fourni (`evaluate_student_search_results`), pas par notre `evaluate`
- [ ] Penser à renommer `moulinette-ubuntu`/`moulinette-fedora` en `moulinette` avant de l'exécuter localement pour tester

---

## 7. Command-Line Interface (Python Fire)

- [ ] `index --max_chunk_size <int>` — ingérer `data/raw/` et construire l'index sous `data/processed/`
- [ ] `search <query> -k <int>` — recherche simple, top-k
- [ ] `search_dataset --dataset_path <path> -k <int> --save_directory <dir>` — batch search → JSON `StudentSearchResults`
- [ ] `answer <query> -k <int>` — répondre à une question
- [ ] `answer_dataset --student_search_results_path <path> --save_directory <dir>` — batch answer → JSON `StudentSearchResultsAndAnswer`
- [ ] `evaluate --student_search_results_path <path> --dataset_path <path>` — évaluation perso (recall@k)
- [ ] Toutes les commandes invoquées via `uv run python -m src <command> [options]`
- [ ] Tous les chemins d'entrée/sortie doivent être des arguments CLI configurables, jamais hardcodés
- [ ] Gérer les erreurs gracieusement (query vide, query absurde, `k=0`, fichiers manquants, JSON malformé — jamais de crash avec traceback non géré)
- [ ] Ajouter des barres de progression `tqdm` sur les opérations longues

---

## 8. README.md

> `README.md` existe mais est **vide** — tout est à faire. Rédigé **en anglais**.

- [ ] Première ligne en italique : *"This project has been created as part of the 42 curriculum by \<login1>, \<login2>, \<login3>[...]."*
- [ ] Section **Description** : objectif et aperçu du projet
- [ ] Section **Instructions** : installation et exécution
- [ ] Section **Resources** : références classiques sur le sujet + description de l'usage de l'IA (pour quelles tâches, quelles parties du projet)
- [ ] Section **System architecture** : composants du pipeline RAG et leurs interactions
- [ ] Section **Chunking strategy** : approche de segmentation
- [ ] Section **Retrieval method** : algorithme et mécanisme de ranking
- [ ] Section **Performance analysis** : discussion des scores recall@k et des performances
- [ ] Section **Design decisions** : choix d'implémentation clés
- [ ] Section **Challenges faced** : difficultés rencontrées et solutions
- [ ] Section **Example usage** : exemples concrets de commandes

---

## 9. Qualité du code

- [ ] Python 3.10+, type hints partout (`typing`)
- [ ] Toutes les fonctions passent `mypy` sans erreurs
- [ ] Code conforme `flake8` (y compris les fichiers bonus)
- [ ] Docstrings PEP 257 (Google ou NumPy style) sur toutes les fonctions et classes
- [ ] Gestion des exceptions avec `try-except` (pas de crash non géré → considéré non fonctionnel)
- [ ] Utiliser des context managers pour fichiers et connexions (pas de leaks de ressources)
- [ ] (Recommandé) écrire des tests `pytest`/`unittest` couvrant les cas limites — non soumis/noté mais conseillé

---

## 10. Bonus (optionnel — 5 points, un par bonus)

> ⚠️ Sujet v2.0 : liste de bonus **différente** de l'ancienne version. Ne pas implémenter "query expansion" ou "local LLM inference via vLLM" (retirés) — remplacés par incremental indexing et local HTTP API.
> Le bonus n'est évalué **que si la totalité de la partie obligatoire est validée**, et compte seulement si implémenté et fonctionnel (pas juste décrit dans le README).

- [ ] **Semantic embeddings** : ajouter un index vectoriel avec un modèle CPU léger (ex. `all-MiniLM-L6-v2`) en plus de l'index lexical
- [ ] **Hybrid retrieval** : combiner les rankings lexical et sémantique en une seule liste de résultats
- [ ] **Incremental indexing** : quand un fichier change, ré-indexer uniquement ce fichier au lieu de tout reconstruire
- [ ] **Caching** : mettre en cache l'index et les résultats de requêtes pour accélérer le cold start et les requêtes répétées
- [ ] **Local HTTP API** : exposer la recherche/génération de réponses via une petite API HTTP locale, pilotable autrement que par la CLI

---

## Structure de dépôt attendue

Layout exact requis pour que les scripts de référence (`index` → `search_dataset` → moulinette `evaluate_student_search_results` → `answer_dataset`) tournent en soutenance sans restructuration manuelle :

```
src/                                                  # module Python, exécutable via `uv run python -m src <command>`
pyproject.toml
uv.lock
README.md
Makefile
.gitignore
data/raw/                                             # sources ingérées (dépôt vLLM fourni)
data/processed/                                       # index produit par `index`
data/datasets/UnansweredQuestions/                    # datasets de questions
data/datasets/AnsweredQuestions/                      # datasets ground-truth
data/output/search_results/<DatasetScope>/            # sortie de `search_dataset`, scopée par dataset
data/output/search_results_and_answer/<DatasetScope>/ # sortie de `answer_dataset`, scopée par dataset
```

> **Ne pas commit** : `data/`, poids de modèles, fichiers générés — le stack deep-learning + poids peuvent peser plusieurs Go, le correcteur les régénère lui-même.

---

## Notes soutenance (Chapter X)

- [ ] Prévoir un "recode" possible en soutenance : modification mineure demandée à la volée (quelques lignes, comportement, structure de données) pour vérifier la compréhension réelle du projet — pas forcément dans l'environnement habituel
