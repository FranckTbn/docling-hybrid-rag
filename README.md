# Docling Hybrid RAG

Le code à exécuter pour accompagner l'article **Construire un RAG documentaire pour l’assurance** de TRA Bi Néné Othniel. L'article explique les choix ; ce dépôt permet de les essayer sur un PDF avec sa propre clé API.

Les fonctions reprennent son pipeline : Docling conserve la structure, les parents regroupent les sections, les enfants servent à la recherche dense. BM25 recherche les parents et BGE-M3 les enfants. RRF combine les classements au niveau des parents. Le LLM choisit de répondre directement ou de rechercher 3, 5 ou 7 parents. Il reçoit les parents retenus avec leurs images et répond avec des citations lisibles.

## Tester dans Google Colab

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FranckTbn/docling-hybrid-rag/blob/codex/colab-reader/demo_colab.ipynb)

**Version d'essai, validation complète sur Colab à confirmer.** Le bouton ouvre la branche d'essai préparée par l'auteur ; le lecteur n'a aucune branche à créer. Les étapes ci-dessous ne constituent pas encore une preuve d'ingestion ou de réponse réussie dans Colab.

Ouvrir le notebook avec le bouton, puis **enregistrer une copie dans son Drive personnel** pour pouvoir modifier les questions. Tout se déroule dans le navigateur, sans installation locale ni manipulation de branches ou de forks GitHub.

Avant d'exécuter les cellules, choisir **Exécution > Modifier le type d'exécution > GPU**, puis **T4 si disponible**. L'accès à un GPU dépend des ressources et des limites du compte Colab. Si aucun GPU n'est attribué, le contrôle du notebook doit le signaler avant l'ingestion.

Exécuter ensuite les quatre cellules principales dans l'ordre :

| Cellule | Action du lecteur | Rôle |
|---|---|---|
| 1. Installer | Cliquer sur Exécuter et attendre la fin | Préparer automatiquement l'environnement Python isolé nécessaire à l'exécution du RAG dans la machine Colab. Aucun redémarrage manuel du noyau n'est prévu. |
| 2. Configurer la clé | Dans **Secrets**, ajouter `OPENAI_API_KEY`, autoriser son accès au notebook, puis exécuter la cellule | Lire la clé sans l'afficher et effectuer un appel de contrôle avec le modèle choisi dans le formulaire. |
| 3. Ingérer le PDF | Exécuter la cellule avec le document public prérempli | Télécharger et ingérer les 30 pages de l'appendice actuariel de Retraite Québec. Attendre la confirmation de fin avant de poser une question. |
| 4. Poser une question | Modifier le champ du formulaire, puis exécuter | Interroger le document et afficher la réponse avec ses sources. |

Le modèle par défaut est `gpt-5.6-luna`. Le formulaire permet d'en choisir un autre accessible sur son compte, compatible avec **l'API Responses, le raisonnement, les images et la sortie JSON stricte**. Le test de connexion permet de repérer un problème d'accès avant l'ingestion ; il ne valide pas à lui seul la qualité des réponses documentaires.

La cellule 4 peut être rejouée avec une nouvelle question. Conserver la même conversation pour une relance comme « Et quelle condition limite ce transfert ? ». Cocher **Nouvelle conversation** pour un contrôle indépendant. La cellule 5, facultative, permet d'examiner le contexte transmis et les classements de recherche.

Le document d'exemple est [Calcul des rentes de Pierre et de Marie, appendice technique de la consultation de 2009](https://www.retraitequebec.gouv.qc.ca/sites/default/files/SiteCollectionDocuments/RetraiteQuebec/fr/publications/nos-programmes/regime-de-rentes/consultation-publique/cp_etude_impact_part2.pdf). Il illustre des calculs historiques, pas les règles actuelles du régime de retraite. Pour essayer un autre PDF numérique public, adapter sa source, son titre et la question dans les formulaires. L'OCR reste désactivé.

**Les appels OpenAI sont payants sur le compte du lecteur.** Ils comprennent le test de connexion, le routage et les réponses. Pour une question documentaire, le texte et les images des parents retenus sont transmis à OpenAI. La clé doit rester dans Secrets, jamais dans une cellule, une sortie ou un fichier partagé. Le parsing Docling utilise CUDA dans ce parcours ; BGE-M3 reste sur CPU, comme dans le code de l'article. Les modèles et les paramètres du pipeline original sont conservés, sans appliquer les petits lots de l'essai local.

Colab exécute le notebook sur une **machine privée et temporaire**. La copie dans Drive sauvegarde le notebook, pas automatiquement le PDF, les index ou les résultats présents dans la machine. Utiliser l'export facultatif du notebook pour récupérer les artefacts avant la suppression du runtime. La mémoire de conversation disparaît à l'arrêt du processus. Une nouvelle machine nécessite de refaire l'installation et l'ingestion. [Fonctionnement des machines Colab](https://research.google.com/colaboratory/faq.html)

## Utiliser le notebook en local

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

### 1. Enrichir la base

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

### 2. Poser une question avec LangGraph

```python
from lib import create_workflow

rag = create_workflow(data_dir="data", env_path=".env")
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

### 3. Examiner la recherche indépendamment du modèle

```python
from lib import load_knowledge_base, hybrid_retrieval, build_context

base = load_knowledge_base("data")
hits = hybrid_retrieval("Comment calculer l'incertitude avec Mack ?", base)
context = build_context(hits["parent_ids"], base)
```

BM25 est construit pendant l'ingestion sur l'ensemble des parents, avec les stopwords français pour le corpus et les questions, puis sauvegardé. Les deux index sont rechargés sans réencodage ; seule une nouvelle question est encodée. Les vingt candidats de chaque canal sont ramenés à des parents distincts avant RRF (constante 60). Un parent dense garde son meilleur enfant. L'appel direct à `hybrid_retrieval` garde par défaut trois parents, avec toutes leurs figures conservées. Dans le workflow, le LLM transmet son choix de 3, 5 ou 7 via `context_k`.

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

Les adaptations concernent les arguments des fonctions, les identifiants distincts entre PDF, les chemins portables et la persistance. Le parsing, les sections entières comme parents, les enfants autour de 400 tokens et les 20 candidats par canal restent ceux de l'article. Le workflow ajoute un routeur LLM et une mémoire courte ; son budget de 3, 5 ou 7 parents prolonge la démonstration de l'article à trois parents fixes. Les tableaux et formules ne sont pas coupés : 400 est une cible souple, pas une limite stricte ni un optimum démontré.

## Vérifications et limites

```bash
python -m unittest discover -s tests -v
```

Les tests hors ligne exercent les identifiants, la reprise d'ingestion, les frontières des enfants, RRF, la provenance et les branches du vrai graphe avec un modèle de test. Ils n'établissent pas la qualité actuarielle d'une réponse. Les résultats réels du guide sont vérifiés séparément par rapport à l'article ; ils ne sont pas distribués comme réponses universelles.

La vérification locale sur le guide retrouve les mêmes 128 parents, 195 enfants, trois parents RRF, images et références que l'article, en réutilisant ses résultats de parsing et d'encodage. Cette preuve concerne le guide de l'article, pas le nouveau PDF du parcours Colab. Une génération réelle a aussi montré une imprécision sur la MSEP, commentée dans l'article : des références valides ne suffisent pas à garantir une interprétation correcte.

Les figures accompagnent les parents retenus, mais ne sont pas indexées par leurs pixels. RRF favorise l'accord entre moteurs et ne garantit pas que les parents choisis suffisent. Le routeur peut se tromper de branche ou de budget ; tester ses décisions sur ses propres questions. Les liens indiquent la section ou l'objet et les pages ; ils ne prouvent pas que chaque affirmation est exacte. Pour un PDF local, la citation désigne sa copie locale ; son ouverture doit être vérifiée dans l'interface utilisée.

Cette première version est prévue pour un lecteur à la fois, des PDF numériques et une ingestion à la fois. Le parcours Colab prépare un processus Python isolé, sans serveur ni tunnel ; le notebook local reste disponible. Le projet n'inclut ni base vectorielle distante ni reranker. Aucun document source ni clé privée n'est publié dans ce dépôt.

## Références

- [Docling et le chunking](https://docling-project.github.io/docling/concepts/chunking/)
- [Docling, exemples RAG et provenance](https://docling-project.github.io/docling/_generated/examples/visual_grounding/)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [BM25s](https://github.com/xhluca/bm25s)
- [RRF, Cormack et al.](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
- [LangGraph, routage par LLM](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [LangGraph, mémoire et identifiants de conversation](https://docs.langchain.com/oss/python/langgraph/persistence)

Le code est sous licence MIT. Les PDF utilisés restent soumis aux droits de leurs auteurs.
