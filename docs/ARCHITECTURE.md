# Architecture

## Entrées du lecteur

`demo_colab.ipynb` est l'entrée publique pour un débutant : ouverture depuis GitHub, copie dans Drive, sélection d'un GPU, installation, clé dans Secrets, ingestion, puis questions dans un formulaire. `demo.ipynb` conserve le parcours Python local. Les notebooks sont distribués sans sorties exécutées. Le parcours Colab est préparé ; sa validation complète reste à confirmer.

`colab_support.py` est le client du notebook Colab. Il utilise la bibliothèque standard pour préparer une venv isolée dans `/content`, installer le dépôt sans les extras Jupyter et communiquer avec `lib/colab_worker.py` par messages JSON sur les entrées et sorties du processus. Le noyau Colab garde ses propres bibliothèques et gère les affichages. Il n'importe pas les dépendances lourdes du RAG depuis la venv. Aucun serveur ou tunnel n'est nécessaire.

`lib/colab_worker.py` s'exécute avec le Python de cette venv. Il appelle les fonctions métier existantes et conserve le graphe compilé, les modèles chargés et l'identifiant de conversation entre les questions. Une nouvelle conversation change cet identifiant. Une ingestion actualise la base ; le graphe doit alors être recréé pour recharger les index. La fermeture du processus efface sa mémoire de conversation.

Le noyau lit `OPENAI_API_KEY` via Colab Secrets et la transmet au processus par son environnement, sans écrire de clé dans un fichier ou une sortie. Le modèle est choisi dans le formulaire et doit prendre en charge Responses, le raisonnement, la vision et le JSON strict. Le contrôle de connexion précède l'ingestion. Les journaux doivent rester distincts des messages JSON et ne jamais contenir la clé.

Docling utilise CUDA dans ce parcours après vérification de sa disponibilité ; BGE-M3 conserve `device="cpu"`. Le client et le worker ne modifient ni les modèles, ni le parsing, ni les budgets de recherche de l'article. Les petits lots expérimentés sur le PC local ne sont pas appliqués au parcours Colab. Un GPU choisi dans le menu ne constitue pas à lui seul une preuve d'utilisation par les modèles.

Les PDF, images, chunks et index restent dans la machine temporaire. Enregistrer le notebook dans Drive ne sauvegarde pas automatiquement ces artefacts. L'export facultatif permet de les récupérer ; un nouveau runtime nécessite une nouvelle installation et une ingestion ou une reprise depuis des artefacts compatibles.

## Pipeline partagé

`ingest_document` télécharge ou lit un PDF, vérifie son identité, appelle le pipeline Docling de l'article, construit ses parents/enfants et enregistre les vecteurs. Le manifeste n'ajoute que les documents terminés. Un échec laisse les étapes réussies réutilisables au prochain appel.

L'ingestion construit aussi BM25 sur l'ensemble des parents et sauvegarde un index identifié par leur contenu. `load_knowledge_base` rassemble les parents, enfants et vecteurs du manifeste et recharge cet index BM25. `hybrid_retrieval` classe les parents lexicaux et les enfants denses, déduplique les parents denses puis applique RRF. `build_context` recharge uniquement les documents sélectionnés et prépare leur texte, leurs images et le registre des sources.

`create_workflow` expose le graphe compilé avec `InMemorySaver`. `remember` ajoute la question et remet à zéro les résultats du tour précédent. Le réducteur de `messages` conserve trois messages utilisateur/assistant au total. `thread_id` isole les conversations ; la mémoire et les anciens checkpoints restent dans le processus Python.

`route_question` appelle le LLM avec cet historique. `RouteDecision` borne la branche à `direct` ou `retrieve` et le budget documentaire à 3, 5 ou 7 parents. Il fournit aussi une question autonome pour les relances. `direct` appelle le LLM sans ouvrir la base. `retrieve` utilise la question reformulée et le budget, puis passe par `context` et `answer`. Chaque branche ajoute seulement sa réponse finale à `messages`.

`answer.py` conserve le prompt documentaire et la validation de l'article : le contexte courant fait preuve, pas les anciennes réponses. Les noms et les liens de citation sont construits par le code. La mémoire ne modifie ni le parsing ni l'indexation. Les appelants de l'interface `invoke`, dont `demo.ipynb`, `lib/colab_worker.py`, `tests/test_pipeline.py` et `tests/test_workflow.py`, doivent fournir un `thread_id`. Le README, `demo_colab.ipynb` et `colab_support.py` dépendent aussi de ce contrat et du protocole du worker pour expliquer ou restituer la conversation.

Les modules de présentation du notebook n'interviennent pas dans les scores. Les changements du parsing et du chunking invalident la signature d'ingestion. Les changements de contexte ou de prompt concernent la génération, pas les embeddings. Les sources ont une identité `(document_id, self_ref)` ; un renvoi Docling seul ne distingue pas deux PDF.

## Vérification du parcours Colab

Les tests avec doubles doivent contrôler l'installation isolée, le protocole, les erreurs et la conservation de l'état sans charger de modèle. Ils ne remplacent pas un essai dans une copie personnelle du notebook : installation complète, Secrets, CUDA réellement utilisé par Docling, ingestion des 30 pages, question, relance, nouvelle conversation, inspection du contexte et export. Les résultats de cet essai doivent être distingués des preuves historiques du guide de l'article.
