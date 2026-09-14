# Docling Hybrid RAG

Le code à exécuter pour accompagner l'article **Construire un RAG documentaire pour l’assurance** de TRA Bi Néné Othniel. L'article explique les choix ; ce dépôt permet de les essayer sur un PDF avec sa propre clé API.

Les fonctions reprennent son pipeline : Docling conserve la structure, les parents regroupent les sections, les enfants servent à la recherche dense. BM25 recherche les parents et BGE-M3 les enfants. RRF combine les classements au niveau des parents. Le modèle reçoit les trois parents retenus avec leurs images et répond avec des citations lisibles.

## Démarrer

Python **3.12 ou supérieur**, Git et un environnement virtuel sont nécessaires. Le notebook se lance depuis la racine du dépôt.

```bash
git clone https://github.com/FranckTbn/docling-hybrid-rag.git
cd docling-hybrid-rag
python -m venv .venv
```

Activer l'environnement avec `source .venv/bin/activate` sous macOS/Linux ou `.venv\Scripts\Activate.ps1` sous Windows. Ensuite :

```bash
python -m pip install -e ".[notebook,test]"
python -c "from shutil import copyfile; copyfile('.env.example', '.env')"
python -m jupyterlab demo.ipynb
```

Renseigner `OPENAI_API_KEY` dans `.env`. Le modèle multimodal est configurable avec `OPENAI_MODEL` selon l'accès du compte. Le code utilise l'API Responses, la sortie JSON structurée et le paramètre de raisonnement de l'article. Choisir un modèle compatible avec ces options.

Ne jamais mettre sa clé dans le notebook. `.env`, les documents et les résultats de `data` sont exclus de Git. Les appels OpenAI transmettent le texte et les images des parents sélectionnés et sont facturés sur le compte du lecteur. Le parsing et les embeddings s'exécutent localement.

Le premier parsing enrichi peut être long, particulièrement sur le guide de 87 pages. Il télécharge aussi les modèles Docling et BGE-M3. Commencer avec un petit PDF numérique permet de vérifier l'installation. Prévoir plusieurs Go libres ; sur une machine avec 8 Go de RAM, exécuter une ingestion à la fois. Le profil reprend l'article et désactive l'OCR : les PDF scannés ne sont pas la cible de cette première version.

## 1. Enrichir la base

```python
from lib import ingest_document

record = ingest_document("mon_document.pdf", data_dir="data", title="Mon document")
```

Le premier argument accepte aussi une URL HTTP(S) renvoyant un PDF, notamment le guide de l'article :

```python
record = ingest_document(
    "https://www.institutdesactuaires.com/global/gene/link.php?doc_id=17739&fg=1",
    title="Guide de provisionnement des sinistres en assurance non-vie",
)
```

Une autre ingestion ajoute un document sans remplacer les précédents. Le SHA-256 du PDF distingue les versions. Réingérer exactement le même contenu reprend les étapes sauvegardées sans relancer le parsing ou l'encodeur. Un échec ne rend pas un document partiel cherchable. Si le code de parsing, le chunking ou leurs dépendances changent, utiliser un nouveau dossier `data` pour reconstruire la base.

Le dossier `data/documents/<sha256>/` conserve le PDF source, `document.json`, ses images dans `artifacts`, les parents/enfants dans `chunks.json`, les vecteurs dans `children-embeddings.npz` et les informations de reprise dans `record.json`. L'index BM25 de l'ensemble des parents est sauvegardé dans `data/indexes/bm25`. `data/manifest.json` liste les documents et l'index disponibles. Les chemins internes sont relatifs pour déplacer cette base avec le projet.

## 2. Poser une question avec LangGraph

```python
from lib import create_workflow

rag = create_workflow(data_dir="data", env_path=".env")
result = rag.invoke({"question": "Comment la méthode de Mack mesure-t-elle l'incertitude des provisions ?"})
print(result["answer"])
```

Le graphe suit deux branches :

- Une salutation explicite reçoit une réponse directe, sans clé API, recherche ou modèle.
- Toute question documentaire ou ambiguë déclenche la recherche, la construction du contexte et la réponse sourcée.

Ce routeur reprend volontairement la règle simple de l'article. Ce n'est ni un classificateur universel d'intention ni un assistant conversationnel avec mémoire. Chaque `invoke` traite une question indépendante.

La base et l'index BM25 sont chargés à la première question documentaire, puis réutilisés. Recréer `rag` après une nouvelle ingestion pour prendre en compte les nouveaux documents. `result["context"]` permet de voir les parents transmis ; `result["sources"]` contient les références effectivement citées. Le notebook affiche le Markdown, les tableaux, les formules et les images du contexte.

## 3. Examiner la recherche indépendamment du modèle

```python
from lib import load_knowledge_base, hybrid_retrieval, build_context

base = load_knowledge_base("data")
hits = hybrid_retrieval("Comment calculer l'incertitude avec Mack ?", base)
context = build_context(hits["parent_ids"], base)
```

BM25 est construit pendant l'ingestion sur l'ensemble des parents, avec les stopwords français pour le corpus et les questions, puis sauvegardé. Les deux index sont rechargés sans réencodage ; seule une nouvelle question est encodée. Les vingt candidats de chaque canal sont ramenés à des parents distincts avant RRF (constante 60). Un parent dense garde son meilleur enfant. Le contexte contient au plus trois parents, avec toutes leurs figures conservées.

## Correspondance avec l'article

| Partie de l'article | Code du dépôt |
|---|---|
| Configurer, parser, sauvegarder et recharger | `lib/parsing.py` |
| Construire les parents et les enfants | `lib/chunking.py` |
| Enrichir la base et reprendre les calculs | `lib/ingestion.py`, `lib/storage.py` |
| BM25, dense, meilleurs enfants et RRF | `lib/retrieval.py` |
| Parents et images, registre des sources | `lib/context.py` |
| Prompt, réponse structurée, citations | `lib/answer.py` |
| Preuve nécessaire, branches et `invoke` | `lib/workflow.py` |

Les adaptations concernent les arguments des fonctions, les identifiants distincts entre PDF, les chemins portables et la persistance. Les budgets de l'article restent inchangés : enfants autour de 400 tokens, sections entières comme parents, 20 candidats par canal et 3 parents restitués. Les tableaux et formules ne sont pas coupés : 400 est une cible souple, pas une limite stricte ni un optimum démontré.

## Vérifications et limites

```bash
python -m unittest discover -s tests -v
```

Les tests hors ligne exercent les identifiants, la reprise d'ingestion, les frontières des enfants, RRF, la provenance et les branches du vrai graphe avec un modèle de test. Ils n'établissent pas la qualité actuarielle d'une réponse. Les résultats réels du guide sont vérifiés séparément par rapport à l'article ; ils ne sont pas distribués comme réponses universelles.

La vérification locale sur le guide retrouve les mêmes 128 parents, 195 enfants, trois parents RRF, images et références que l'article, en réutilisant ses résultats de parsing et d'encodage. Une génération réelle a aussi montré une imprécision sur la MSEP, commentée dans l'article : des références valides ne suffisent pas à garantir une interprétation correcte.

Les figures accompagnent les parents retenus, mais ne sont pas indexées par leurs pixels. RRF favorise l'accord entre moteurs et ne garantit pas que les trois parents suffisent. Les liens indiquent la section ou l'objet et les pages ; ils ne prouvent pas que chaque affirmation est exacte. Pour un PDF local, la citation ouvre sa copie locale, pas une URL publique.

Cette première version est prévue pour un utilisateur local, des PDF numériques et une ingestion à la fois. Elle n'inclut ni serveur, ni base vectorielle distante, ni reranker. Aucun document source ni clé privée n'est publié dans ce dépôt.

## Références

- [Docling et le chunking](https://docling-project.github.io/docling/concepts/chunking/)
- [Docling, exemples RAG et provenance](https://docling-project.github.io/docling/_generated/examples/visual_grounding/)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [BM25s](https://github.com/xhluca/bm25s)
- [RRF, Cormack et al.](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)

Le code est sous licence MIT. Les PDF utilisés restent soumis aux droits de leurs auteurs.
