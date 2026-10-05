# Docling Hybrid RAG

Le code à exécuter pour accompagner l'article **Construire un RAG documentaire pour l’assurance** de TRA Bi Néné Othniel. L'article explique les choix ; ce dépôt permet de les essayer sur un PDF avec sa propre clé API.

Les fonctions reprennent son pipeline : Docling conserve la structure, les parents regroupent les sections, les enfants servent à la recherche. BM25 et BGE-M3 recherchent tous deux les enfants, puis chaque enfant est remonté à son parent avant la fusion RRF. Le LLM choisit de répondre directement ou de rechercher, découpe la demande en sous-questions et fixe un budget de 3, 5 ou 7 parents. Chaque sous-question a sa propre recherche. Le modèle reçoit les parents retenus avec leurs images, répond avec des citations lisibles, puis un contrôle signale les paragraphes mal soutenus par leurs sources.

## Utiliser la bibliothèque dans un autre projet

Dans l'environnement virtuel de votre projet (Python 3.12 ou supérieur) :

```bash
python -m pip install "docling-hybrid-rag @ git+https://github.com/FranckTbn/docling-hybrid-rag.git"
```

```python
from docling_hybrid_rag import ingest_document, hybrid_retrieval, load_knowledge_base, build_context

for pdf in ["rapport-1.pdf", "rapport-2.pdf"]:          # autant de PDF que vous voulez, un par un
    ingest_document(pdf, "ma_base", title=pdf)

base = load_knowledge_base("ma_base")
hits = hybrid_retrieval("Quelle est la conclusion du rapport 2 ?", base)
context = build_context(hits["parent_ids"], base)         # textes, images et sources citables
```

La recherche, le contexte et les citations fonctionnent sans clé API : seul le workflow `create_workflow` appelle un LLM. Une même base accepte plusieurs PDF ; chaque passage garde le document dont il vient.

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
from docling_hybrid_rag import ingest_document

record = ingest_document("mon_document.pdf", "data/knowledge", title="Mon document")
```

Le premier argument accepte aussi une URL HTTP(S) renvoyant un PDF, notamment le guide de l'article :

```python
record = ingest_document(
    "https://www.institutdesactuaires.com/global/gene/link.php?doc_id=17739&fg=1",
    title="Guide de provisionnement des sinistres en assurance non-vie",
)
```

Une autre ingestion ajoute un document sans remplacer les précédents. Le SHA-256 du PDF distingue les versions. Réingérer exactement le même contenu reprend les étapes sauvegardées sans relancer le parsing ou l'encodeur. Un échec ne rend pas un document partiel cherchable. Si une option du parsing, le code d'une étape ou la version d'une bibliothèque change, seule cette étape et les suivantes sont recalculées.

La base `data/knowledge` contient un `catalog.json` qui relie chaque PDF à son dossier `documents/<empreinte>/`, nommé d'après les 16 premiers caractères du SHA-256 du PDF. Ce dossier conserve le PDF source, `document.json` et ses images dans `artifacts`, les parents/enfants dans `chunks.json` et les vecteurs dans `children-embeddings.npz`. Son `manifest.json` retient, pour le parsing, le chunking et les embeddings, les entrées utilisées et l'empreinte des fichiers produits. L'index BM25 de l'ensemble des parents est sauvegardé dans `indexes/bm25`. Les chemins internes sont relatifs pour déplacer cette base avec le projet.

## 2. Poser une question avec LangGraph

```python
from docling_hybrid_rag import create_workflow

rag = create_workflow("data/knowledge", env_path=".env")
# Le même identifiant permet de poursuivre cette conversation.
config = {"configurable": {"thread_id": "ma-conversation"}}
result = rag.invoke(
    {"question": "Comment la méthode de Mack mesure-t-elle l'incertitude des provisions ?"}, config
)
print(result["answer"])
```

Le LLM lit les trois derniers messages et choisit une branche :

- `direct` pour une conversation ou une demande sans preuve documentaire. Le modèle répond sans charger la base.
- `retrieve` pour une question documentaire. Le modèle choisit aussi le nombre de parents : 3 pour une demande ciblée, 5 pour une explication plus large, 7 pour une comparaison ou une synthèse. RRF détermine ensuite quels parents retourner.

La décision est structurée et validée : aucune autre branche ni aucun autre budget n'est accepté. `result["route_reason"]` expose un motif court et `result["context_k"]` indique le budget demandé, ou 0 pour une réponse directe. Il peut y avoir moins de parents si les candidats sont insuffisants. Ce choix adaptatif est une règle de départ à évaluer, pas un optimum démontré.

```python
suite = rag.invoke({"question": "Et quelles sont ses limites ?"}, config)
print(suite["answer"])
print(suite["route"], suite["context_k"], suite["route_reason"])
```

Le routeur utilise l'historique pour reformuler une relance en question autonome, visible dans `result["search_question"]`. Cette question sert à la recherche et à la réponse sourcée. Les anciennes réponses ne deviennent pas des preuves ; seules les sources retrouvées pour le tour courant peuvent être citées.

`InMemorySaver` conserve l'état de chaque conversation dans la RAM du processus Python. Le champ `messages` garde **trois messages au total**, questions et réponses confondues, et non trois échanges complets. Après une réponse, il peut donc commencer par un message assistant. À la question suivante, le routeur voit le dernier échange et la nouvelle question. Les prompts système et les images n'entrent pas dans cette fenêtre. Un sujet plus ancien peut être oublié ; il faut alors le repréciser.

Réutiliser le même graphe et le même `thread_id` pour continuer, ou changer d'identifiant pour une conversation séparée. La mémoire disparaît avec le processus ou un nouveau graphe. Les anciens checkpoints peuvent rester en RAM jusqu'à cette fermeture ; la limite de trois porte sur le champ `messages` courant et l'historique envoyé au modèle. Aucun serveur supplémentaire n'est nécessaire.

La clé API est désormais utilisée dès le routage, y compris pour une salutation. Un tour normal effectue un appel pour décider puis un autre pour répondre. La base et l'index BM25 restent chargés seulement à la première recherche, puis réutilisés. Recréer `rag` après une nouvelle ingestion pour actualiser les index et démarrer une nouvelle mémoire.

`result["context"]` permet de voir les parents transmis ; `result["sources"]` contient les références effectivement citées. Le notebook affiche le Markdown, les tableaux, les formules et les images du contexte.

### Sous-questions, reranking et contrôle

Pour une comparaison, le routeur renvoie jusqu'à trois sous-questions dans `result["sub_questions"]`. Chacune est recherchée séparément, en parallèle, avec sa part du budget : un thème très présent dans le document ne peut pas évincer les sources d'un autre. `result["themes"]` montre, pour chaque sous-question, les parents retenus et les reformulations éventuelles.

```python
from docling_hybrid_rag.settings import get_reranker

rag = create_workflow("data/knowledge", reranker=get_reranker(), relevance_threshold=0.2)
```

Le reranker `BAAI/bge-reranker-v2-m3` (environ 2,3 Go, téléchargé au premier usage) relit le meilleur enfant de chaque parent candidat face à sa sous-question. Avec `relevance_threshold`, une sous-question dont le meilleur passage reste sous ce score est reformulée une fois, puis déclarée sans passage retrouvé. Ce seuil est à calibrer sur vos propres questions ; la valeur ci-dessus n'est qu'un exemple.

Après la réponse, un appel LLM vérifie que chaque paragraphe est soutenu par les extraits qu'il cite. En cas de doute, une **alerte de vérification** et des pistes de relecture sont ajoutées au message. `result["support"]` contient le détail ; `verify=False` désactive ce contrôle.

### Expansion de requête par thésaurus

L'expansion reprend la méthode de l'article [Query Expansion](https://ornelle.quarto.pub/query-expansion/) de l'auteur : la requête originale est conservée, les libellés des concepts reconnus sont ajoutés avec un poids, et chaque terme pèse sur le score BM25 selon ce poids. Elle est facultative : sans thésaurus, la requête est inchangée.

```python
thesaurus = {"language": "fr", "concepts": [
    {"id": "ibnr", "prefLabel": "IBNR", "altLabel": ["sinistres survenus non déclarés"],
     "hiddenLabel": [], "broader": [], "narrower": [], "related": []},
]}
rag = create_workflow("data/knowledge", thesaurus=thesaurus)   # ou un chemin vers un fichier JSON
```

Par défaut, seuls `prefLabel` (poids 0,95) et `altLabel` (0,85) sont ajoutés ; `hiddenLabel` sert uniquement à reconnaître un concept. Les relations `broader`, `narrower` et `related` ne sont injectées que si `policy` les cite, car ce ne sont pas des synonymes. Le dense garde la question originale. La qualité des équivalences reste à valider par un expert du domaine.

## 3. Examiner la recherche indépendamment du modèle

```python
from docling_hybrid_rag import load_knowledge_base, hybrid_retrieval, build_context

base = load_knowledge_base("data/knowledge")
hits = hybrid_retrieval("Comment calculer l'incertitude avec Mack ?", base)
context = build_context(hits["parent_ids"], base)
```

BM25 est construit pendant l'ingestion sur les enfants de toute la base, avec les mots courants français ignorés et une racinisation française (« provisions » retrouve « provision »), puis sauvegardé. Le texte d'un enfant contient les titres de sa section. Les deux index sont rechargés sans réencodage ; seule une nouvelle question est encodée. Les vingt enfants candidats de chaque canal sont ramenés à des parents distincts avant RRF (constante 60) ; chaque parent garde son meilleur enfant dans chaque canal. L'appel direct à `hybrid_retrieval` garde par défaut trois parents, avec toutes leurs figures conservées. Dans le workflow, le LLM transmet son choix de 3, 5 ou 7 via `context_k`.

## Mesure sur dix PDF réels

La méthode est mesurée sur [Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) de Vectara : des articles arXiv au format PDF, des questions rédigées d'après une section précise, et la section attendue pour chacune (licence CC BY-NC 4.0, usage non commercial : les données ne sont pas dans ce dépôt). Dix articles de 8 à 24 pages, cent questions dont 45 portent sur du texte seul et 55 sur des tableaux ou des figures, sont ingérés dans une même base de 649 enfants.

```bash
python benchmarks/open_rag_bench.py download   # questions, sections attendues, PDF (pause entre deux requêtes arXiv)
python benchmarks/open_rag_bench.py ingest     # parse, découpe et indexe les dix PDF
python benchmarks/open_rag_bench.py evaluate   # BM25, dense, RRF, avec la portée des parents « section » puis « elements »
```

Un parent est jugé pertinent s'il partage assez de suites de quatre mots avec la section attendue : la mesure ne dépend pas du parsing. Les intervalles sont à 95 %, par rééchantillonnage des cent questions.

| Parents (3 premiers) | Section attendue retrouvée | Part de la section dans le contexte | Taille du contexte |
|---|---|---|---|
| Sections (192 parents) | 0,87 [0,80 à 0,93] | 0,67 [0,61 à 0,73] | 3 400 tokens |
| Éléments (522 parents) | 0,86 [0,79 à 0,93] | 0,42 [0,36 à 0,48] | 1 150 tokens |

Avec RRF, le bon document est toujours parmi les trois premiers parents et la bonne section l'est dans 87 % des cas (95 % parmi les cinq premiers). BM25 et le dense seuls obtiennent des résultats voisins, et leur fusion RRF est au même niveau ou un peu au-dessus : avec cent questions, les intervalles se recouvrent et aucun moteur ne se détache. Les parents « section » ne retrouvent pas plus souvent la section, mais ils mettent beaucoup plus de son contenu sous les yeux du modèle, surtout quand la question porte sur un tableau ou une figure voisins du passage (part couverte 0,56 contre 0,22 pour les questions texte, tableau et figure). Ce contexte coûte environ trois fois plus de tokens.

Limites : dix documents sans documents voisins en distracteurs rendent la recherche de document facile ; les questions sont écrites par un LLM d'après chaque section, donc plus proches du texte qu'une vraie demande ; la mesure ne juge ni la réponse du modèle ni les citations. Lancer `RAG_BENCHMARK_DIR=data/benchmarks/open-rag-bench-arxiv python -m unittest tests.test_open_rag_bench` vérifie que ces planchers tiennent après une modification.

## Correspondance avec l'article

| Partie de l'article | Code du dépôt |
|---|---|
| Configurer, parser, sauvegarder et recharger | `src/docling_hybrid_rag/parsing.py` |
| Construire les parents et les enfants | `src/docling_hybrid_rag/chunking.py` |
| Mesurer la recherche sur des questions annotées (références Docling) | `src/docling_hybrid_rag/evaluation.py` |
| Mesurer la recherche sur un benchmark public (passages de texte) | `src/docling_hybrid_rag/passage_evaluation.py`, `benchmarks/open_rag_bench.py` |
| Expansion de requête par thésaurus | `src/docling_hybrid_rag/expansion.py`, `src/docling_hybrid_rag/indexing.py` |
| Contrôle de soutien et alertes | `src/docling_hybrid_rag/verification.py` |
| Enrichir la base et reprendre les calculs | `src/docling_hybrid_rag/ingestion.py`, `src/docling_hybrid_rag/store.py` |
| BM25, dense, meilleurs enfants et RRF | `src/docling_hybrid_rag/retrieval.py` |
| Parents et images, registre des sources | `src/docling_hybrid_rag/context.py` |
| Prompt, réponse structurée, citations | `src/docling_hybrid_rag/answer.py` |
| Preuve nécessaire, branches et `invoke` | `src/docling_hybrid_rag/workflow.py` |

Les adaptations concernent les arguments des fonctions, les identifiants distincts entre PDF, les chemins portables et la persistance. Le parsing, les enfants autour de 400 tokens et les 20 candidats par canal restent ceux de l'article. Le workflow ajoute un routeur LLM et une mémoire courte ; son budget de 3, 5 ou 7 parents prolonge la démonstration de l'article à trois parents fixes. Un enfant ne dépasse pas 400 tokens : un tableau trop long est donc coupé entre ses lignes en plusieurs enfants, mais son parent le contient en entier. Ce budget n'est pas un optimum démontré.

## Vérifications et limites

```bash
python -m unittest discover -s tests -v
```

Les tests hors ligne exercent les identifiants, la reprise d'ingestion, les frontières des enfants, RRF, la provenance et les branches du vrai graphe avec un modèle de test. Ils n'établissent pas la qualité actuarielle d'une réponse. Les résultats réels du guide sont vérifiés séparément par rapport à l'article ; ils ne sont pas distribués comme réponses universelles.

Sur le guide, `tests/test_article_parity.py` (avec `RAG_ARTICLE_DIR`) refait le chunking et le compare aux chunks enregistrés de l'article. Le chunking garantit, et les tests vérifient, qu'aucun enfant n'est vide, que les enfants découpent exactement le texte de leur parent et qu'aucun élément du document n'appartient à deux parents. Des références valides ne suffisent pas à garantir une interprétation correcte : le contrôle de soutien vérifie qu'un extrait cité figure bien dans la source, pas que la conclusion est juste.

Les figures accompagnent les parents retenus, mais ne sont pas indexées par leurs pixels. RRF favorise l'accord entre moteurs et ne garantit pas que les parents choisis suffisent. Le routeur peut se tromper de branche ou de budget ; tester ses décisions sur ses propres questions. Les liens indiquent la section ou l'objet et les pages ; ils ne prouvent pas que chaque affirmation est exacte. Pour un PDF local, la citation ouvre sa copie locale, pas une URL publique.

Cette première version est prévue pour un utilisateur local, des PDF numériques et une ingestion à la fois. Elle n'inclut ni serveur ni base vectorielle distante. Le reranker est facultatif et se télécharge à part. Aucun document source ni clé privée n'est publié dans ce dépôt.

## Références

- [Docling et le chunking](https://docling-project.github.io/docling/concepts/chunking/)
- [Docling, exemples RAG et provenance](https://docling-project.github.io/docling/_generated/examples/visual_grounding/)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [BM25s](https://github.com/xhluca/bm25s)
- [RRF, Cormack et al.](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
- [LangGraph, routage par LLM](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [LangGraph, mémoire et identifiants de conversation](https://docs.langchain.com/oss/python/langgraph/persistence)

Le code est sous licence MIT. Les PDF utilisés restent soumis aux droits de leurs auteurs.
