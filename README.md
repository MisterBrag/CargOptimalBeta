## **Table des matières**

- **Introduction**
- **Technologies utilisées**
- **Installation et configuration**
- **Utilisation de l'application**
- **Structure du projet**
- **Guide du fichier CSV**
- **Paramétrage**
- **Architecture**
- **Contribuer au projet**
- **Licence et crédits**

---

## **Introduction**

CargOptimal est un projet d'optimisation des dimensions de boîtes pour le stockage et l'expédition de produits. Il utilise l'apprentissage automatique pour prédire les dimensions optimales des boîtes, visant à minimiser les coûts d'expédition et d'emballage, tout en garantissant une protection adéquate des produits.

---

## **Technologies utilisées**

- **Backend**: Python 3.9, Flask 2.0
- **Frontend**: HTML5, CSS3, JavaScript ES6
- **Machine Learning**: scikit-learn 0.24
- **Database**: SQLite 3 (ou autre selon le cas)

---

## **Installation et configuration**

1. Cloner le dépôt Git :
    
    ```
    git clone https://github.com/****/CargOptimal.git
    
    ```
    
2. Naviguer dans le dossier du projet :
    
    ```
    cd CargOptimal
    
    ```
    
3. Installer les dépendances avec pip (il est recommandé d'utiliser un environnement virtuel) :
    
    ```
    pip install -r requirements.txt
    
    ```
    
4. Configurer la base de données : Créez une nouvelle base de données SQLite avec le nom **`cargoptimal.db`** dans le dossier **`database`**.
5. Lancer l'application :
    
    ```
    python app.py
    
    ```
    

---

## **Utilisation de l'application**

Une fois l'application lancée, accédez à **`http://localhost:5000`** sur votre navigateur. Vous verrez une interface utilisateur intuitive guidant à travers le processus de collecte des données, analyse des résultats, entraînement du modèle, et prédiction des dimensions de boîtes optimisées.

---

## **Structure du projet**

```
CargOptimal/
│
├── app.py
├── static/
│   ├── styles.css
├── templates/
│   ├── index.html
│   ├── calcul_caracteristiques.html
│   ├── analyse_resultats.html
│   ├── optimisation.html
│   ├── prediction.html
│   └── model.html
├── model/
│   ├── model.pkl
├── data/
│   ├── Fashion.csv
└── README.md

```

---

## **Guide du fichier CSV**

Le fichier CSV utilisé pour la collecte des données doit respecter la structure suivante :

- colonne 1 : **`Nom du produit`**
- colonne 2 : **`Longueur de la boîte actuelle`**
- colonne 3 : **`Largeur de la boîte actuelle`**
- colonne 4 : **`Hauteur de la boîte actuelle`**

- Comment prendre les dimensions des produits ?
    
    Avant de commencer, assurez-vous d'avoir les outils nécessaires à portée de main. Vous aurez besoin d'un ruban à mesurer ou d'une règle flexible, d'un crayon et d'une feuille de papier pour noter les dimensions. Par exemple, si vous expédiez des livres, préparez ces outils pour mesurer la taille de chaque livre.
    
    **Mesurer un objet carré ou rectangulaire et entrer les dimensions dans le tableau:**
    
    Pour un objet carré ou rectangulaire, comme une boîte à chaussures, les mesures sont assez simples. Il y a trois dimensions à prendre en compte : la largeur, la hauteur et la profondeur. Assurez-vous de mesurer le côté le plus large pour la largeur, le côté le plus haut pour la hauteur et du devant vers l'arrière pour la profondeur. Ensuite, entrez ces mesures dans le tableau sous la forme "largeur x hauteur x profondeur". Par exemple, si votre boîte à chaussures mesure 30cm de large, 20cm de haut et 12cm de profondeur, vous entrerez "30 x 20 x 12" dans le tableau.
    
    **Mesurer un objet rond ou cylindrique (comme une bouteille) et entrer les dimensions dans le tableau:**
    
    Pour un objet rond, comme un ballon de football, mesurez le diamètre à son point le plus large. Pour une bouteille de vin ou un autre objet cylindrique, mesurez également la hauteur du bas vers le haut. Si l'objet est conique (comme un vase qui s'élargit vers le haut), mesurez aussi le diamètre à son point le plus large. Pour entrer ces mesures dans le tableau, vous pouvez considérer le diamètre comme la largeur et la profondeur. Par exemple, si le diamètre est de 10cm et la hauteur est de 30cm, vous entrerez "10 x 30 x 10" dans le tableau.
    
    **Mesurer un objet irrégulier et entrer les dimensions dans le tableau:**
    
    Pour un objet irrégulier, comme une statue, imaginez-le dans une boîte et mesurez cette boîte imaginaire. Il s'agit de la largeur, de la hauteur et de la profondeur maximales de l'objet à tous les points. Notez ces mesures et entrez-les dans le tableau sous la forme "largeur x hauteur x profondeur".
    
    **Mesurer des produits liquides ou semi-liquides et entrer les dimensions dans le tableau:**
    
    Pour les produits liquides ou semi-liquides contenus dans un emballage solide (par exemple, une bouteille d'huile d'olive), mesurez les dimensions physiques de l'emballage, comme vous le feriez pour un objet solide.
    
    Si le produit est contenu dans un emballage flexible ou déformable (par exemple, un sachet de sauce), mesurez les dimensions de l'emballage lorsqu'il est plein. Entrez ces informations dans le tableau sous la forme "largeur x hauteur x profondeur", en considérant le diamètre comme la largeur et la profondeur si le produit est de forme cylindrique.
    

## **Architecture**

L'architecture de l'application suit le modèle MVC (Modèle-Vue-Contrôleur), permettant une séparation des préoccupations et facilitant la maintenance et l'évolution du code.

- **Modèle**: Gère les données, y compris les opérations de base de données et les calculs liés à l'apprentissage automatique.
- **Vue**: Gère la présentation et la mise en forme des informations pour l'utilisateur.
- **Contrôleur**: Gère la logique de l'application, y compris la coordination entre le Modèle et la Vue.

---

## **Contribuer au projet**

Nous accueillons volontiers toute contribution au projet. Que vous souhaitiez corriger un bug, améliorer la documentation, ajouter une nouvelle fonctionnalité ou simplement poser une question, n'hésitez pas à ouvrir une issue ou une pull request.
