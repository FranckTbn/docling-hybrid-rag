# Architecture

`ingest_document` télécharge ou lit un PDF, vérifie son identité, appelle le pipeline Docling de l'article, construit ses parents/enfants et enregistre les vecteurs. Le manifeste n'ajoute que les documents terminés. Un échec laisse les étapes réussies réutilisables au prochain appel.

L'ingestion construit aussi BM25 sur l'ensemble des parents et sauvegarde un index identifié par leur contenu. `load_knowledge_base` rassemble les parents, enfants et vecteurs du manifeste et recharge cet index BM25. `hybrid_retrieval` classe les parents lexicaux et les enfants denses, déduplique les parents denses puis applique RRF. `build_context` recharge uniquement les documents sélectionnés et prépare leur texte, leurs images et le registre des sources.

`create_workflow` expose le graphe compilé avec `InMemorySaver`. `remember` ajoute la question et remet à zéro les résultats du tour précédent. Le réducteur de `messages` conserve trois messages utilisateur/assistant au total. `thread_id` isole les conversations ; la mémoire et les anciens checkpoints restent dans le processus Python.

`route_question` appelle le LLM avec cet historique. `RouteDecision` borne la branche à `direct` ou `retrieve` et le budget documentaire à 3, 5 ou 7 parents. Il fournit aussi une question autonome pour les relances. `direct` appelle le LLM sans ouvrir la base. `retrieve` utilise la question reformulée et le budget, puis passe par `context` et `answer`. Chaque branche ajoute seulement sa réponse finale à `messages`.

`answer.py` conserve le prompt documentaire et la validation de l'article : le contexte courant fait preuve, pas les anciennes réponses. Les noms et les liens de citation sont construits par le code. La mémoire ne modifie ni le parsing ni l'indexation. Les dépendants de l'interface `invoke` sont le README, `demo.ipynb`, `tests/test_pipeline.py` et `tests/test_workflow.py` : tous doivent fournir un `thread_id`.

Les modules de présentation du notebook n'interviennent pas dans les scores. Les changements du parsing et du chunking invalident la signature d'ingestion. Les changements de contexte ou de prompt concernent la génération, pas les embeddings. Les sources ont une identité `(document_id, self_ref)` ; un renvoi Docling seul ne distingue pas deux PDF.
