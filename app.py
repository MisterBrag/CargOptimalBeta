from flask import Flask, render_template, request, redirect, send_from_directory, jsonify, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import uuid
import os
from waitress import serve

app = Flask(__name__)
app.secret_key = 'secret-key'  # Remplacez ceci par une vraie clé secrète

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    pass

@login_manager.user_loader
def user_loader(username):
    if username != 'admin':
        return

    user = User()
    user.id = username
    return user

numeric_columns = ['volume_total', 'somme_largeur', 'somme_hauteur', 'somme_profondeur', 'max_largeur', 'max_hauteur', 'max_profondeur', 'boite_largeur','boite_hauteur','boite_profondeur']

def aggregate_product_features(df):
    product_dims = []
    for _, row in df.iterrows():
        width = row['somme_largeur']
        height = row['somme_hauteur']
        depth = row['somme_profondeur']
        product_dims.append((width, height, depth))
    print("Product dimensions (product_dims) :")
    print(product_dims)
    return product_dims


# Charger les données à partir du fichier CSV
def load_data(file):
    data = pd.read_csv(file.stream, sep=';')
    print("Données brutes après lecture du fichier CSV:")
    print(data)

    # Convertissez les colonnes numériques en nombres à virgule flottante
    data[numeric_columns] = data[numeric_columns].astype(float)
    print("Données après conversion en nombres à virgule flottante:")
    print(data)

    return data

# Extraire les caractéristiques cumulées des produits
def get_product_cumul_features(data):
    return data.iloc[:, :7].values.tolist()

# Extraire les dimensions des boîtes en carton
def get_box_dims(data):
    return data.iloc[:, 7:].values.tolist()

# Entraîner le modèle
def train_model(data):
    # Sélectionnez les colonnes de caractéristiques et la cible
    features = data.iloc[:, :7]  # Select all columns till 6th column (from volume_total to max_profondeur)
    target = data.iloc[:, 7:]  # Select all columns from 7th till end (from boite_largeur to boite_profondeur)

    # Entraînez le modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)

    # Retournez le modèle entraîné
    return model

# Enregistrer le modèle dans un fichier pickle
def save_optimal_box_dims(optimal_box_dims, filename):
    with open(f'{output_folder}/{filename}', 'wb') as f:
        pickle.dump(optimal_box_dims, f)

# Charger le model à partir du fichier pickle
def load_optimal_box_dims():
    with open('optimal_box_dims.pkl', 'rb') as f:
        optimal_box_dims = pickle.load(f)
    return optimal_box_dims


    # Génère le nom de fichier unique avec un UUID
def generate_unique_filename(prefix=""):
    unique_id = uuid.uuid4()
    return f"{prefix}_{unique_id}.pickle"


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'elonteub': 
            user = User()
            user.id = username
            login_user(user)
            return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
def home():
    return redirect('/login')


@app.route('/dashboard')
def index():
    return render_template('index.html')


@app.route('/calcul_caracteristiques', methods=['GET', 'POST'])
def calcul_caracteristiques():
    if request.method == 'POST':
        dimensions = request.get_json().get('dimensions')  # Utiliser request.get_json() à la place de request.form.get()

    return render_template('calcul_caracteristiques.html')

@app.route('/optimisation')
def optimisation():
    return render_template('optimisation.html')

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


# Supposons que les fichiers soient enregistrés dans un dossier appelé "output_files"
output_folder = "output_files"

# Créer le répertoire s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

@app.route('/train_sa', methods=['POST'])
def train_sa():
    # Charger les données d'entraînement
    file = request.files['file']
    data = load_data(file)


     # Entraîner le modèle
    model = train_model(data)


    # Générer des noms de fichiers uniques pour chaque entraînement de modèle
    model_filename = generate_unique_filename("trained_model")

    # Enregistrer le modèle dans un fichier pickle
    with open(f'{output_folder}/{model_filename}', 'wb') as f:
        pickle.dump(model, f)

    return f"""
    Modèle entraîné avec succès ! <br/>
    Les fichiers peuvent être téléchargés depuis : <br/>
    <a href="/download/{model_filename}">Cliquez ici pour télécharger le modèle</a> <br/>
    """

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(output_folder, filename, as_attachment=True)

@app.route('/predict_sa', methods=['POST'])
def predict_sa():
     # Récupérer les données du formulaire
    form_data = request.form.to_dict()

    # Calculer le nombre de produits
    num_products = int(form_data["num_products"])

    # Initialiser les valeurs pour le volume total, la somme et le max de chaque dimension
    volume_total = 0
    sum_dims = [0, 0, 0]
    max_dims = [0, 0, 0]

    # Parcourir chaque produit et calculer les valeurs
    for i in range(1, num_products + 1):
        width = float(form_data[f"product_width_{i}"])
        height = float(form_data[f"product_height_{i}"])
        depth = float(form_data[f"product_depth_{i}"])

        volume_total += width * height * depth
        sum_dims = [sum_dims[j] + val for j, val in enumerate([width, height, depth])]
        max_dims = [max(max_dims[j], val) for j, val in enumerate([width, height, depth])]

    # Construire le vecteur de caractéristiques
    x = [volume_total] + sum_dims + max_dims

    # Redimensionner en tant que tableau 2D, comme requis par sklearn
    X = np.array(x).reshape(1, -1)

    # Charger le modèle
    file_box_dims = request.files['file_box_dims']
    optimal_box_dims = pickle.load(file_box_dims.stream)

    # Faire la prédiction
    y_pred = optimal_box_dims.predict(X)

    # Construction de la réponse
    result = {
        'box_dimensions': {
            'length': y_pred[0][0],
            'width': y_pred[0][1],
            'height': y_pred[0][2]
        }
    }



    # Ajouter ici les informations sur le coût des boîtes et la quantité de matière utilisée
    custom_box_cost = 0.50  # Par exemple
    standard_box_cost = 1.00  # Par exemple
    custom_box_material = 1.00  # Par exemple, en m²
    standard_box_material = 1.50  # Par exemple, en m²
    trees_saved_per_m2 = 0.004  # Par exemple, nombre d'arbres sauvés par mètre carré de matériel économisé


    # Calculer les économies réalisées et le nombre d'arbres sauvés pour 100 colis envoyés
    savings = (standard_box_cost - custom_box_cost) * num_products * 100
    trees_saved = (standard_box_material - custom_box_material) * num_products * trees_saved_per_m2 * 100

    # Prévisions basées sur les données de l'année précédente
    previous_year_shipments = 250  # Ce nombre devrait être récupéré d'une base de données ou d'une autre source de données
    forecasted_shipments = 450  # Ce nombre devrait être basé sur un modèle de prévision ou une autre méthode d'estimation
    forecasted_savings = (standard_box_cost - custom_box_cost) * forecasted_shipments
    forecasted_trees_saved = (standard_box_material - custom_box_material) * forecasted_shipments * trees_saved_per_m2

    result['forecast'] = {
        'savings': savings,
        'trees_saved': trees_saved,
        'forecasted_savings': forecasted_savings,
        'forecasted_trees_saved': forecasted_trees_saved
    }

    result['message'] = f'''
    📦 Dimensions optimales de la boîte pour {num_products} produit(s) : {y_pred[0][0]:.1f} x {y_pred[0][1]:.1f} x {y_pred[0][2]:.1f}
    💰 Pour 100 colis envoyés, vous économisez : {savings} €
    🌳 Nombre d'arbres sauvés : {trees_saved} <br/>
    🚀 Basé sur les 250 colis que vous avez envoyés l'année dernière, et une prévision de 450 colis cette année, vous pourriez économiser jusqu'à {forecasted_savings} € et sauver jusqu'à {forecasted_trees_saved} arbres.
    Merci de contribuer à un avenir plus durable ! 🌍💚
    '''

    return jsonify(result), 200

@app.route('/analyse_resultats')
def analyse_resultats():
    return render_template('analyse_resultats.html')

@app.route('/model')
def modelcommunity():
    return render_template('model.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)

    #app.run(debug=True)
