# TP03 — Réponses aux questions

## 1. Démarrage

**Si un collègue reçoit seulement `mlflow.db`, que lui manque-t-il pour charger le pipeline ?** Les fichiers du modèle, qui sont dans `mlartifacts/` (la base ne contient que l'emplacement des artefacts, pas leur contenu). Il lui faut aussi un serveur lancé avec les mêmes options, le code du projet et un environnement compatible (`uv.lock`) pour pouvoir désérialiser le pickle.

**Expérience vs run ?** Une expérience (`telco-churn-tp03`) est un dossier logique qui regroupe des runs comparables. Un run est une exécution précise du code d'entraînement, avec ses paramètres, métriques, tags et artefacts. Ici : une expérience, un run par candidat.

**La page web exécute-t-elle le code d'entraînement ?** Non. C'est le client (`uv run python -m telco_churn.tracking`) qui entraîne et envoie les enregistrements au serveur. Le serveur stocke et l'interface affiche.

## 2. Enregistrement

**Différence entre métrique et artefact ?** Une métrique est une valeur numérique nommée (`val_average_precision = 0.662`), stockée dans la base et comparable entre runs dans l'interface. Un artefact est un fichier (matrice de confusion, `provenance.json`, modèle) stocké dans `mlartifacts/`.

**Qu'est-ce qui distingue deux allocations train/validation issues du même CSV ?** Pas le hash du CSV, qui est identique dans les deux cas. Ce qui les distingue : le fichier `split_assignment.csv` (client → split) et son SHA-256 (tag `split_assignment_sha256`), la seed et les proportions en paramètres, et le *digest* des datasets MLflow `telco_train` / `telco_valid` (calculé sur le contenu, `customerID` compris).

**Une seed garantit-elle les mêmes résultats sur toutes les machines ?** Non. Elle fixe le hasard du code, mais les résultats peuvent changer avec d'autres versions de bibliothèques, un autre OS ou CPU, ou du calcul en parallèle. Il faut aussi les versions (`uv.lock`) et l'environnement.

**Peut-on retrouver les données à partir du hash ?** Non. Un hash permet de *vérifier* qu'on a les bons octets, pas de les reconstruire. Le CSV doit être obtenu ailleurs (URL épinglée dans `references/data_source.md`).

**Que devient la traçabilité si on enregistre un commit avec des modifications non commitées ?** Le commit ne décrit plus exactement le code exécuté. C'est pour ça que le code a été commité avant les runs finaux et que l'on enregistre `git_dirty` et la liste des fichiers modifiés (ici seulement deux fichiers de documentation, pas de code).

## 3. Comparaison

**Pourquoi des populations de validation différentes rendent la comparaison difficile ?** L'écart de score mélangerait l'effet du modèle et celui des clients tirés (certains échantillons sont plus « faciles »). Avec la même allocation, la seule différence entre runs est le candidat.

**Pourquoi ne pas sélectionner plusieurs fois sur le test ?** À force de choisir ce qui marche sur le test, on s'ajuste à lui : il devient un deuxième jeu de validation et ne donne plus une estimation honnête des performances sur de nouveaux clients. Ici le test n'est pas évalué dans TP03.

## 4. Modèle et registre

**Que décrit le schéma d'entrée enregistré ?** Les six colonnes attendues par le pipeline après le parsing TP02, avec leur type : `tenure`, `MonthlyCharges`, `TotalCharges` en `double` et `Contract`, `InternetService`, `PaymentMethod` en `string`. Il décrit le format d'entrée, pas la validité métier des valeurs. Remarque : `mlflow.sklearn.load_model` renvoie l'objet sklearn brut, donc le schéma n'est pas appliqué automatiquement (il l'est via `mlflow.pyfunc`). C'est pourquoi on garde `prepare_features` comme contrôle explicite à l'entrée.

**Run ID vs URI du modèle ?** Le run ID identifie *l'exécution* (paramètres, métriques, données). L'URI `models:/m-…` identifie *l'objet modèle enregistré* produit par ce run. Le registre ajoute un nom et une version (`telco-churn` v1) et l'alias `candidate` indique quelle version joue ce rôle aujourd'hui.

**Pourquoi un chargement dans un nouveau processus est-il une meilleure preuve ?** Dans le notebook d'entraînement, l'objet est encore en mémoire, avec les variables, imports et états du training. Un nouveau processus prouve que tout ce qu'il faut est dans le stockage MLflow et le code du package, sans `fit` et sans données d'entraînement.

**Si l'alias pointe plus tard vers une autre version, l'objet déjà chargé change-t-il ?** Non. `load_model` télécharge une version précise au moment de l'appel. L'objet Python en mémoire reste l'ancienne version. Il faut recharger `models:/telco-churn@candidate` (redémarrer le service ou relancer le chargement) pour récupérer la nouvelle version. C'est pour ça qu'on note aussi le numéro de version à côté de l'alias.

## 5. Persistance

Voir `reports/tp03_summary.md`, section 6 : après redémarrage, les runs, la version 1 et l'alias sont toujours là, et l'inférence donne les mêmes scores.
