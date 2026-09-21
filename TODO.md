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
- [x] Ajouter les dépendances dans `pyproject.toml` (`transformers`, `torch`, `bm25s`, `fire`, `tqdm`, `pydantic`, `flake8`, `mypy`)
- [x] S'assurer que `uv sync` seul suffit à installer le projet (le correcteur et la moulinette ne lancent que `uv sync`, rien d'autre)
- [x] Créer la structure de dossiers data attendue par la moulinette : `data/raw/vllm-0.10.1/` peuplé depuis l'archive fournie (3226 fichiers, 1763 `.py`, 178 `.md`)
- [x] Compléter `.gitignore` (ajouté `data/`, poids de modèles ; **retiré `uv.lock` qui était incorrectement ignoré alors que le sujet l'exige dans le dépôt**)

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

- [x] Lire et parcourir tous les fichiers utiles du dépôt vLLM (`src/indexer.py::_iter_corpus_files`, filtré sur `.py`/`.md`/`.rst`/`.txt`)
- [x] Implémenter le **chunking Python** (`src/chunking.py::chunk_python_source`, découpage par fonctions/classes avec `ast`, decorateurs inclus, hard-split si trop gros, fallback gracieux si syntaxe invalide)
- [x] Implémenter le **chunking texte/Markdown** (`src/chunking.py::chunk_markdown_source`, découpage par paragraphes séparés par lignes vides, un titre `#`..`######` force une nouvelle section, hard-split si un paragraphe est trop gros)
- [x] Taille de chunk max : 2000 caractères par défaut, **configurable via `--max_chunk_size`** (`build_index(max_chunk_size=...)`)
- [x] Ne jamais dépasser `--max_chunk_size` : chaque chunk hérite de la contrainte des chunkers (voir section chunking) qui hard-split tout ce qui dépasse
- [x] Stocker les chunks avec leurs `file_path` (relatif, préservé tel que passé à `--raw_dir`), `first_character_index`, `last_character_index` — via `MinimalSource`, testé avec chemins relatifs
- [x] Persister l'index sous `data/processed/` (`src/indexer.py::build_index`, `bm25s.BM25.save`, corpus metadata en `corpus.jsonl`)
- [x] Indexation complète en **moins de 5 minutes** — mesuré à 3.4s sur le vrai corpus vLLM (1969 fichiers, 18733 chunks)

---

## 4. Retrieval System

- [x] Implémenter **BM25** (via `bm25s`) — choisi plutôt que TF-IDF pour mieux gérer l'hétérogénéité code/doc du corpus vLLM (on peut ajouter d'autres méthodes en plus). Indexation faite dans `src/indexer.py::build_index`, testée avec un round-trip index→recherche (`bm25s.BM25.load` + `.retrieve`)
- [x] Sauvegarder l'index dans `data/processed/` pour éviter de recalculer
- [x] Retourner les top-k chunks les plus pertinents pour une requête (`src/cli.py::RagCLI.search`)
- [x] Supporter le **batch processing** de datasets JSON (`search_dataset`)
- [x] Throughput : **200 questions traitées en < 90 secondes** — mesuré à 0.38s réel sur le corpus vLLM indexé, largement dans la limite
- [x] Atteindre **Recall@5 ≥ 80%** sur les questions "docs" et **≥ 50%** sur les questions "code" — mesuré avec `evaluate` sur les vrais datasets fournis (`data/datasets/AnsweredQuestions/dataset_docs_public.json` — 100 questions — et `dataset_code_public.json` — 99 questions —, `search_dataset --k 10`) : **docs 82.0%** (recall@1 63.0%, @3 79.0%, @10 89.0%), **code 50.5%** (recall@1 26.3%, @3 41.4%). Les deux passent le seuil, mais le code est très juste (+0.5 pt) — à re-vérifier si le chunking Python change
- [ ] `file_path` doit matcher **exactement** le chemin du corpus ingéré (comparaison verbatim par le correcteur)

> ℹ️ La latence "cold start < 60s" qui figurait dans l'ancien TODO n'apparaît plus dans les critères de performance du sujet v2.0 (section VII.1.2 : seulement indexing time, throughput, recall@5).

---

## 5. Answer Generation System

- [x] Charger le modèle **Qwen/Qwen3-0.6B** (via `transformers`, modèle par défaut obligatoire) — `src/generation.py::load_generator`, `model_name` configurable via `--model_name` sur `answer`/`answer_dataset` pour tester d'autres modèles
- [x] Passer le contexte récupéré au LLM dans les limites de tokens (`src/generation.py::_build_prompt`, budget de caractères `MAX_CONTEXT_CHARS` réparti entre les sources, tronque plutôt que dépasser)
- [x] Générer des réponses en JSON structuré conforme à `StudentSearchResultsAndAnswer` / `MinimalAnswer` (`src/cli.py::RagCLI.answer_dataset`)
- [x] La réponse doit être cohérente/groundée : prompt qui force le modèle à répondre uniquement à partir du contexte fourni ; `enable_thinking=False` pour éviter le bruit du raisonnement Qwen3 dans la réponse finale ; testé manuellement sur le vrai corpus vLLM (ex. "What is a KV cache?" → réponse correcte et sourcée)
- [x] Ajouter une barre de progression `tqdm` pour le traitement des datasets (`answer_dataset`)

---

## 6. Evaluation System

- [x] Implémenter la commande `evaluate` (calcul de **recall@k**, k = 1, 3, 5, 10) — **pour son propre usage/itération uniquement** (`src/evaluation.py::evaluate_recall`, `src/cli.py::RagCLI.evaluate`)
- [x] Comparer les sources récupérées aux annotations ground truth (dataset `AnsweredQuestions`) — matching par `question_id`, sources ground truth extraites des `AnsweredQuestion` du `RagDataset`
- [x] Une source est "trouvée" si même `file_path` **et** chevauchement (IoU) ≥ 5% avec la plage de référence (`src/evaluation.py::_overlap_iou`/`_is_match`, `MIN_IOU = 0.05`)
- [x] Score par question : `nombre_trouvées / total_sources_correctes` (`src/evaluation.py::recall_at_k`)
- [x] Afficher un rapport de performance complet (`src/evaluation.py::format_report`, moyenne par k + avertissement si k demandé > k de récupération réel)
- [x] **Ne jamais importer ni appeler la moulinette** dans le code du projet — `src/evaluation.py` ne dépend que de `src/models.py`, aucune référence à la moulinette
- [x] Testé avec un dataset synthétique (source exacte → recall 100%, source à un rang tardif → recall partiel selon k, source inexistante → recall 0%) et les cas d'erreur (fichier manquant, JSON malformé, dataset sans `AnsweredQuestion`)
- [ ] Penser à renommer `moulinette-ubuntu`/`moulinette-fedora` en `moulinette` avant de l'exécuter localement pour tester

---

## 7. Command-Line Interface (Python Fire)

- [x] `index --max_chunk_size <int>` — ingérer `data/raw/` et construire l'index sous `data/processed/` (`src/cli.py::RagCLI.index`)
- [x] `search <query> -k <int>` — recherche simple, top-k (`src/cli.py::RagCLI.search`)
- [x] `search_dataset --dataset_path <path> -k <int> --save_directory <dir>` — batch search → JSON `StudentSearchResults` (`src/cli.py::RagCLI.search_dataset`), testé avec un dataset factice + cas d'erreur
- [x] `answer <query> -k <int>` — répondre à une question (`src/cli.py::RagCLI.answer`), testé sur le vrai corpus vLLM
- [x] `answer_dataset --student_search_results_path <path> --save_directory <dir>` — batch answer → JSON `StudentSearchResultsAndAnswer` (`src/cli.py::RagCLI.answer_dataset`), testé en chaîne après `search_dataset`
- [x] `evaluate --student_search_results_path <path> --dataset_path <path>` — évaluation perso (recall@k) (`src/cli.py::RagCLI.evaluate`)
- [x] Commandes `index`/`search`/`search_dataset`/`answer`/`answer_dataset`/`evaluate` invoquées via `uv run python -m src <command> [options]` (`src/__main__.py` + `src/cli.py`)
- [x] Chemins d'entrée/sortie configurables via arguments CLI (`raw_dir`, `processed_dir`, `dataset_path`, `save_directory`, `index_dir`, `student_search_results_path`, `model_name`), jamais hardcodés
- [x] Gestion gracieuse testée pour `index`/`search`/`search_dataset`/`answer`/`answer_dataset`/`evaluate` : query vide, `k=0`, `k` > taille du corpus, index absent, fichier dataset manquant, JSON malformé, JSON de mauvaise forme, modèle introuvable, dataset sans `AnsweredQuestion` — aucun crash avec traceback non géré
- [x] Barres de progression `tqdm` sur `index` (chunking), `search_dataset` (par question) et `answer_dataset` (par question)

---

## 8. README.md

- [x] Première ligne en italique : *"This project has been created as part of the 42 curriculum by sloubiat."* (login récupéré depuis `git log`, projet solo)
- [x] Section **Description** : objectif et aperçu du projet
- [x] Section **Instructions** : installation et exécution
- [x] Section **Resources** : références classiques (BM25, RAG, bm25s, Qwen3, ast, pydantic, Fire) + description honnête de l'usage de l'IA (Claude Code, tâches précises listées)
- [x] Section **System architecture** : composants du pipeline RAG et leurs interactions (schéma ASCII)
- [x] Section **Chunking strategy** : approche de segmentation (AST Python + paragraphes Markdown)
- [x] Section **Retrieval method** : BM25 via `bm25s`, justification vs TF-IDF
- [x] Section **Performance analysis** : chiffres réels (indexation 3.4s, throughput 0.38s, recall@5 docs 82.0%/code 50.5%)
- [x] Section **Design decisions** : BM25, chunking AST, torch CPU pin, re-lecture du texte depuis disque, `enable_thinking=False`, gestion gracieuse des erreurs
- [x] Section **Challenges faced** : bugs `bm25s`, gotcha `BatchEncoding` de transformers, régression torch/CUDA, marge serrée sur recall@5 code
- [x] Section **Example usage** : toutes les commandes CLI avec des exemples concrets sur les vrais datasets

---

## 9. Qualité du code

- [x] Python 3.10+, type hints partout (`typing`) — projet en 3.13, tous les fichiers `src/` et `main.py` typés
- [x] Toutes les fonctions passent `mypy` sans erreurs — vérifié projet entier via `make lint` (`mypy src main.py ...`, `data/` exclu car ce n'est pas notre code)
- [x] Code conforme `flake8` (y compris les fichiers bonus) — vérifié projet entier via `make lint` (`flake8 . --exclude=.venv,data`)
- [x] Docstrings PEP 257 (Google ou NumPy style) sur toutes les fonctions et classes
- [x] Gestion des exceptions avec `try-except` (pas de crash non géré → considéré non fonctionnel) — testé sur toutes les commandes CLI (index/search/search_dataset/answer/answer_dataset)
- [x] Utiliser des context managers pour fichiers et connexions (pas de leaks de ressources) — `open(...) as ...` / `Path.read_text`/`write_text` partout, pas de fichier ouvert sans fermeture
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

