# Architecture

`ingest_document` télécharge ou lit un PDF, vérifie son identité, appelle le pipeline Docling de l'article, construit ses parents/enfants et enregistre les vecteurs. Le manifeste n'ajoute que les documents terminés. Un échec laisse les étapes réussies réutilisables au prochain appel.

L'ingestion construit aussi BM25 sur l'ensemble des parents et sauvegarde un index identifié par leur contenu. `load_knowledge_base` rassemble les parents, enfants et vecteurs du manifeste et recharge cet index BM25. `hybrid_retrieval` classe les parents lexicaux et les enfants denses, déduplique les parents denses puis applique RRF. `build_context` recharge uniquement les documents sélectionnés et prépare leur texte, leurs images et le registre des sources.

`create_workflow` expose le graphe compilé. Les salutations vont directement à `END`. Les autres questions passent par `retrieve`, `context`, `answer`, puis `END`. `answer.py` utilise le prompt et la validation de l'article. Les noms et les liens de citation sont construits par le code.

Les modules de présentation du notebook n'interviennent pas dans les scores. Les changements du parsing et du chunking invalident la signature d'ingestion. Les changements de contexte ou de prompt concernent la génération, pas les embeddings. Les sources ont une identité `(document_id, self_ref)` ; un renvoi Docling seul ne distingue pas deux PDF.
