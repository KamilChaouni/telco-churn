# TP01 — Réponses aux questions

## 1. Projet et environnement

**pyproject.toml vs uv.lock vs .venv/**
- `pyproject.toml` : ce que le projet *demande* (métadonnées, `requires-python`, dépendances avec des bornes comme `pandas>=3.1.0`, point d'entrée `telco-inspect`). Écrit à la main, versionné.
- `uv.lock` : le résultat de la *résolution* : version exacte de chaque paquet, y compris transitifs (numpy, six, tzdata…), avec hashes et pour toutes les plateformes. Généré par uv, versionné, c'est lui qui rend l'install reproductible.
- `.venv/` : l'installation concrète sur *ma* machine (binaires, chemins absolus). Jetable, jamais versionné ni copié : on le recrée avec `uv sync --locked`.

**Pourquoi séparer notebooks et code réutilisable ?** Un notebook dépend de l'ordre d'exécution des cellules et d'un état caché, se diffe mal dans Git et ne s'importe pas. Le code du package s'importe, se teste (`pytest`), se lance en une commande et sera réutilisé par l'entraînement et la prédiction en TP02. Le notebook sert à explorer, pas à être le seul chemin exécutable.

**Où va le modèle entraîné, où va le code qui le construit ?** Le pipeline ajusté (sortie, ex. `joblib`) → `models/` (ignoré par Git, tracé par hash/métadonnées). Le code qui le construit → le package (`src/telco_churn/pipeline.py`, `training.py`).

**Cinq choix d'emplacement** : voir la section « Project layout » du README.

## 2. Identifier les données

**Deux fichiers de même nom peuvent-ils contenir des données différentes ?** Oui, et c'est arrivé ici : `datasets.csv` fourni en cours et `Telco-Customer-Churn.csv` du dépôt IBM ont le même contenu métier mais pas les mêmes octets (BOM, CRLF, lignes entre guillemets), donc des hashes différents, et pandas lit le premier comme une seule colonne. Le nom ne prouve rien ; le hash identifie les octets.

**Un hash identique prouve-t-il que les données métier sont correctes ?** Non. Il prouve seulement qu'on a exactement les mêmes octets que la référence. Si la source contenait des erreurs (valeurs fausses, `TotalCharges` vides), elles sont toujours là. Le hash garantit l'*identité*, pas la *qualité* ni la véracité.

**Pourquoi la cible et l'ID ne doivent pas devenir des entrées du modèle ?**
- `Churn` est ce qu'on veut prédire : l'utiliser en entrée = fuite de cible, score parfait en test et inutilisable en production où on ne la connaît pas.
- `customerID` est un identifiant unique et arbitraire : il n'a aucun sens causal, le modèle pourrait mémoriser des clients (sur-apprentissage) et ne généraliserait pas aux nouveaux clients. Il sert à tracer les lignes, pas à prédire.

## 3. Inspection

**Pourquoi une colonne de montant peut être lue comme texte ?** Parce qu'il suffit d'une valeur non numérique pour que pandas garde toute la colonne en texte. Ici `TotalCharges` contient 11 valeurs `" "` (un espace) ; ce sont 11 clients avec `tenure = 0`, pas encore facturés. `isna()` en trouve 0 : seul le test « vide ou espaces » (`astype("string").str.strip().eq("")`) les détecte. Autres causes possibles : séparateur décimal `,`, symbole monétaire, séparateur de milliers.

**Pourquoi sauvegarder un rapport plutôt qu'imprimer ?** Une sortie terminal disparaît. Un JSON est conservé, comparable automatiquement entre deux exécutions (c'est ce qu'on a fait pour la recréation), lisible par d'autres programmes, et associé au hash de la source et aux versions de l'environnement : c'est une preuve, pas un souvenir.

**Prédire toujours la classe majoritaire est-il une baseline utile en TP02 ?** Oui comme *plancher*, non comme objectif. Toujours prédire « No » donne 73,5 % d'accuracy mais 0 % de rappel sur les churners, qui sont précisément ceux qu'on veut retenir. Elle montre qu'une accuracy de 75 % n'est pas impressionnante ; il faudra regarder rappel, précision, F1, ROC-AUC / PR-AUC sur la classe `Yes`.

## 4. Recréation

`uv sync --locked` installe exactement le lock et échoue si `pyproject.toml` et `uv.lock` ne correspondent plus, au lieu de résoudre silencieusement de nouvelles versions. Résultat : mêmes versions, même hash, mêmes statistiques ; seuls l'horodatage et le chemin de l'exécutable diffèrent. Le lock ne fige pas l'OS, le matériel, les bibliothèques système ni les futurs détails d'entraînement (graine, threads BLAS…).

## 5. Git

**Qu'identifie un commit ?** Un instantané complet de tous les fichiers suivis, plus ses parents, son auteur, sa date et son message, le tout identifié par un hash SHA. Il permet de retrouver exactement le code d'une exécution.

**Qu'est-ce qui reste non enregistré si le working tree a des modifications ?** Tout ce qui n'est pas commité : modifications non indexées, fichiers indexés non commités, fichiers non suivis. Le hash du commit ne décrit alors pas le code réellement exécuté. Restent aussi hors Git par choix : `.venv`, les données brutes, les rapports générés, les modèles.

**Que mettre dans une bonne description de PR ?** Le pourquoi et le quoi du changement, comment il a été vérifié (commandes et résultats), ce qui n'est pas couvert, et une question précise pour le relecteur.

## Question finale

**Ce qu'un autre ingénieur peut reproduire maintenant :** le même environnement (versions exactes via `uv.lock`, Python 3.12), les mêmes octets d'entrée (vérifiés par SHA-256) et le même rapport d'inspection, à partir d'un commit donné.

**Ce qu'il manque pour reproduire un modèle entraîné :** le commit du code d'entraînement, la configuration et les hyperparamètres (`configs/`), les graines aléatoires et la définition exacte du split train/test, les versions des bibliothèques d'entraînement (scikit-learn…), le hash des données effectivement utilisées, l'artefact ajusté lui-même avec son hash, et les métriques obtenues, idéalement enregistrés ensemble comme métadonnées dans `models/`.

## Extensions optionnelles

- `git restore README.md` **supprime** la modification du working tree (irréversible si non commitée) ; `git restore --staged README.md` **désindexe** seulement, la modification reste dans le fichier.
- Hash d'une copie modifiée : même un seul octet changé (ou juste des fins de ligne CRLF) donne un hash totalement différent, comme on l'a vu avec `datasets.csv`.
