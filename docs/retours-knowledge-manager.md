# Retours d'usage : intégration dans Knowledge Manager

Ce fichier consigne les problèmes rencontrés en utilisant `docling-hybrid-rag` comme
dépendance d'une application, Knowledge Manager (FastAPI, une base de connaissance par
projet, ingestion dans un processus séparé, workflow appelé comme outil d'un agent).
Il sert à améliorer le package. Chaque point dit ce qui a été observé, pourquoi cela
gêne un usage en application et une piste. Rien ici n'a été modifié dans le package.

Versions concernées : 0.2.0 (copie de travail du Bureau, 2026-10-04) puis 0.3.0
installée depuis GitHub (`main`, commit `6f3fbb1`, 2026-10-05).

## Suite donnée dans la version 0.4.0

| Point | Réponse | Test |
|---|---|---|
| 1. Index BM25 refusé | Quand seules les versions de `bm25s` ou de `PyStemmer` diffèrent, `load_knowledge_base` reconstruit l'index (avertissement), sans toucher aux enfants ni aux vecteurs. Quand les enfants ont changé, le message renvoie vers `reindex(knowledge_dir)`, désormais public. | `test_pipeline` : `test_bm25_index_built_elsewhere_is_rebuilt_instead_of_refused`, `test_bm25_index_of_other_children_is_refused_with_the_reindex_hint` |
| 2. Mémoire d'un serveur | `create_workflow(memory=False)` : aucun `thread_id`, aucun point de reprise. `checkpointer=` accepte celui de votre choix. | `test_workflow` : `test_a_graph_without_memory_needs_no_thread_id_and_keeps_nothing`, `test_a_custom_checkpointer_replaces_the_ram_memory` |
| 3. Jetons | `result["usage"]` : une entrée par appel (`route_question`, `search_theme`, `direct`, `answer`, `verify`) avec modèle, jetons d'entrée, de sortie et total. `total_usage(result["usage"])` les additionne. | `test_workflow` : `test_each_model_call_reports_its_tokens_and_the_list_is_reset_every_turn`, `test_retrieval_turn_counts_the_router_and_the_support_check` |
| 4. Liens HTML | `create_workflow(link=plain_citation)` rend des citations en texte seul. `result["sources"]` est documenté comme l'interface stable (voir le README). | `test_workflow` : `test_plain_citations_leave_no_html_and_sources_stay_available` |
| 5. Passages d'un document | `document_passages(base, document)` renvoie les parents dans l'ordre, avec section, pages et texte, sans recharger le document. Les parents et les enfants portent leurs `pages`. | `test_pipeline` : `test_document_passages_give_the_parents_in_reading_order_with_their_pages` |
| 6. Modèles gardés en mémoire | `release_models()` vide l'encodeur et le reranker (environ 2 Go chacun). Le README documente `encoder=` et `reranker=` pour partager une seule instance. | `test_pipeline` : `test_release_models_forces_the_next_call_to_load_them_again` |

Le texte qui suit reste celui des retours d'origine.

## 1. Un index BM25 copié d'un autre environnement est refusé avec un message trompeur

Observé. La base de l'article (`Agentic AI Engineer/data/knowledge`, construite avec
`bm25s` 0.3.10) a été copiée dans Knowledge Manager, qui avait `bm25s` 0.3.12.
`load_knowledge_base` lève : « L'index BM25 ne correspond plus aux enfants ; réindexer
la base. » Les enfants n'avaient pas changé : seules les versions de `bm25s` différaient
(le contrat compare `versions`).

Gêne. Le message oriente vers les enfants, pas vers les versions. Et il n'existe pas de
fonction publique de réindexation : Knowledge Manager importe
`docling_hybrid_rag.indexing.save_lexical_index`, un module interne, puis réécrit
`catalog["bm25"]` à la main.

Pistes. Un message distinct quand seules les versions diffèrent. Une fonction publique
`reindex(knowledge_dir)`, ou une reconstruction automatique de BM25 (calcul peu coûteux,
entièrement dérivé des enfants) quand seules les versions ont changé.

## 2. La mémoire du workflow s'accumule dans un serveur

Observé. `create_workflow` compile le graphe avec `InMemorySaver` et exige un
`thread_id`. Une application qui pose une question par appel crée un identifiant neuf
à chaque fois ; les points de reprise restent en RAM jusqu'à l'arrêt du processus,
comme le README l'indique.

Gêne. Dans un serveur qui tourne des jours, la mémoire grandit sans limite pour un
usage qui n'a pas besoin d'historique.

Piste. Un paramètre `checkpointer` (dont `None` pour un usage sans mémoire), ou une
option `memory=False`.

## 3. Les jetons consommés par le workflow ne sont pas mesurables

Observé. Le routeur, la réponse et le contrôle de soutien appellent le modèle, mais le
résultat de `invoke` n'expose aucun usage.

Gêne. Knowledge Manager mesure les jetons de chaque appel LLM (règle du projet). Les
appels du workflow sont aujourd'hui les seuls qu'il ne peut pas compter.

Piste. Un champ `usage` dans le résultat : une entrée par appel (nœud, modèle, jetons
d'entrée et de sortie), lue dans `usage_metadata`.

## 4. La réponse contient des liens HTML vers des fichiers locaux

Observé. `result["answer"]` contient des balises `<a href="file:///…/source.pdf#page=…">`
pour un PDF local.

Gêne. Dans une application web, ces liens ne s'ouvrent pas et ne doivent pas être
exposés. Knowledge Manager retire le HTML et reconstruit ses citations depuis
`result["sources"]`, qui ne sont pas documentées comme un contrat stable.

Pistes. Documenter `sources` comme l'interface stable (clés, `pages`, `name`,
`document_title`, `text`). Proposer une réponse sans HTML, ou un paramètre qui
construit les liens (par exemple une fonction `source_url(document_id, page)`).

## 5. Lire tous les parents d'un PDF avec leurs pages demande `build_context`

Observé. Pour donner un document entier à un agent (avec la page de chaque parent),
Knowledge Manager filtre `knowledge_base.parents` par `metadata["document_id"]` puis
appelle `build_context` sur tous ces parents, ce qui recharge le document Docling et
résout chaque référence.

Gêne. C'est plus lourd que nécessaire, et l'application dépend de clés internes
(`parents`, `metadata["document_id"]`, `document_title`, `heading_path`, `parent_id`)
qui ne font pas partie de l'API publique.

Piste. Une fonction publique `document_passages(knowledge_dir, document_id)` qui
renvoie les parents dans l'ordre, avec texte, section et pages. Le commit local
`22e2502` (« chunk pages ») semble aller dans ce sens ; il n'est pas encore sur `main`.

## 6. Le modèle d'embeddings reste chargé pour toute la vie du processus

Observé. Le premier appel de recherche charge BGE-M3 : environ 190 s sur un Intel
i3-N305 à 8 Go de RAM, et environ 2 Go de mémoire qui ne sont jamais rendus
(`get_encoder` est mis en cache).

Gêne. Un serveur qui ne cherche que rarement dans des PDF garde ces 2 Go en
permanence ; sur cette machine, cela pousse le système à la pagination.

Pistes. Une fonction pour libérer l'encodeur, ou documenter l'option `encoder=` de
`create_workflow` comme moyen de partager et contrôler une seule instance.
