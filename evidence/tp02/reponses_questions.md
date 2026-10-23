# TP02 — Réponses aux questions

## 1. Premier modèle

**Que donne un système qui prédit toujours « pas de churn » ?** Sur la validation : 73,5 % d'accuracy, mais 0 churner détecté (rappel 0, précision 0, matrice : TN 1035, FP 0, FN 374, TP 0). C'est le `DummyClassifier(strategy="most_frequent")`.

**Une accuracy élevée peut-elle coexister avec zéro churner détecté ?** Oui, c'est exactement le cas de la baseline : la classe « No » représente 73,5 % des clients, donc ne jamais prédire « Yes » donne déjà 73,5 %.

**Sur quelles données ajuster la baseline ?** Sur le train, comme les autres modèles. Elle apprend la classe majoritaire du train.

**Quelles données peuvent influencer le choix du modèle ?** Train (apprentissage) et validation (comparaison). Le test est réservé : utilisé une seule fois, après la décision, pour le modèle choisi.

## 2. Pipeline

**fit / transform / predict ?** `fit` apprend des paramètres sur des données (médianes, moyennes/écarts-types, catégories, coefficients du modèle). `transform` applique ces paramètres appris sans les modifier. `predict` (ou `predict_proba`) fait passer les données dans toutes les transformations puis dans le classifieur.

**Que se passe-t-il si on ajuste un scaler sur la validation ?** Le scaler utilise des informations de la validation (fuite de données) et la validation n'est plus transformée comme le seraient de vraies nouvelles données. L'évaluation devient trop optimiste et non représentative de la production.

**Un Pipeline empêche-t-il automatiquement la fuite si on lui donne les mauvaises données ?** Non. Il garantit que tout est ajusté *sur ce qu'on lui donne* à `fit`. Si on lui donne train + validation, ou si le split est fait après un prétraitement appris sur tout le dataset, la fuite est là quand même.

**Qui ordonne quoi ?** `Pipeline` ordonne les étapes en séquence (préprocesseur puis classifieur ; imputer puis scaler à l'intérieur d'une branche). `ColumnTransformer` aiguille les colonnes : les 3 numériques vers la branche `num`, les 3 catégorielles vers `cat`, et ignore le reste (`remainder="drop"`).

**Pourquoi `errors="raise"` plutôt que `coerce` sans contrôle ?** `coerce` transforme silencieusement n'importe quel texte invalide (« 12,50 », « N/A ») en NaN, qui serait ensuite imputé sans que personne ne le sache. On convertit donc d'abord les blancs en NaN (cas connu et documenté), puis on vérifie qu'aucune autre valeur n'échoue : sinon erreur.

## 3. Comparaison

**Changer le split pour chaque modèle rendrait-il la comparaison plus lisible ?** Non : la différence de score mélangerait l'effet du modèle et l'effet du hasard du split. Même split, mêmes features, mêmes métriques pour tous.

**Le modèle avec la meilleure accuracy répond-il forcément au besoin ?** Non. L'équipe veut trouver les clients qui partent. La régression logistique a la meilleure accuracy (0,804) mais l'arbre a le meilleur rappel (0,602), et la forêt la meilleure average precision (0,662). Le bon choix dépend du métrique lié au besoin.

**Si deux candidats sont proches, qu'est-ce qui justifie le plus simple ?** Plus facile à expliquer, plus rapide, plus léger, moins de risque de sur-apprentissage, et une différence de 0,005 sur 1 409 clients peut être du bruit. D'où la règle fixée à l'avance : à 0,01 près, on garde le plus simple. Ici l'écart forêt/logistique est 0,014, donc la forêt est retenue — mais c'est serré, et la logistique serait un choix défendable.

**Une probabilité de 0,8 est-elle un risque de 80 % calibré ?** Pas automatiquement. Une forêt aléatoire produit des moyennes de votes d'arbres, souvent mal calibrées. Il faudrait vérifier avec une courbe de calibration ou un score de Brier, et éventuellement calibrer (`CalibratedClassifierCV`).

**Pourquoi l'average precision ?** La classe positive est minoritaire (26,5 %). L'AP mesure la qualité du classement des clients à risque sans dépendre d'un seuil, et la baseline vaut la proportion de positifs (0,265) : un bon repère.

## 4. Artefact

**Pourquoi sauvegarder seulement le classifieur ne suffit pas ?** Le classifieur attend des colonnes déjà imputées, mises à l'échelle et encodées en one-hot dans un ordre précis. Sans le préprocesseur ajusté, il faudrait refaire ces transformations à la main, avec des risques d'erreur ou de fuite.

**Quelles informations apprises vivent dans un imputer ou un encodeur ?** Imputer : les médianes du train (`statistics_`, ex. `TotalCharges` = 1373,625). Scaler : moyennes et écarts-types. Encodeur : la liste des catégories vues au train et l'ordre des colonnes produites.

**Deux fichiers `model.joblib` peuvent-ils contenir des modèles différents ?** Oui. Le nom ne dit rien : seuls le hash du fichier et les métadonnées (données, seed, paramètres, commit, versions) identifient le modèle.

**Pipeline sklearn vs workflow DAG (Kubeflow) ?** Un Pipeline sklearn est un seul objet Python qui enchaîne des transformations et un modèle, en mémoire, et se sauvegarde en un fichier. Un DAG Kubeflow orchestre des étapes séparées (ingestion, entraînement, évaluation, déploiement) qui tournent dans des conteneurs, peuvent être planifiées, relancées et échanger des artefacts. Le pipeline sklearn peut être une seule étape d'un DAG.

## 5. Vérifications (`uv run pytest -q`)

| Vérification | Test |
|---|---|
| IDs disjoints entre train/valid/test, split reproductible | `test_splits_disjoint_complete_and_reproducible` |
| Médianes de l'imputer = médianes du train (≠ full data : 1373,6 vs 1397,5 pour `TotalCharges`) | `test_imputer_median_learned_from_train_only` |
| `predict` ne modifie pas les statistiques | `test_predict_does_not_refit` |
| NaN / blanc numérique imputé, catégorie inconnue encodée en zéros | `test_missing_numeric_and_unknown_category_follow_policy` |
| Texte numérique invalide refusé | `test_malformed_numeric_text_is_rejected` |
| Colonne manquante → erreur visible | `test_missing_required_feature_fails_visibly` |
| Rechargement dans un nouveau processus → mêmes scores | `test_reload_in_fresh_process_gives_same_scores` + `scripts/check_reload.py` |
