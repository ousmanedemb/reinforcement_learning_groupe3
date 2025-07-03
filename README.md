# Système de Tarification Dynamique par Reinforcement Learning

## 📌 Description
Ce projet implémente une solution de tarification intelligente pour les services de transport utilisant l'apprentissage par renforcement. Le système apprend automatiquement à ajuster les prix en fonction de la demande, de l'offre et d'autres facteurs contextuels pour maximiser les revenus.

## 🛠 Technologies
- **Langage** : Python 3.10+
- **Bibliothèques principales** :
  - Stable Baselines3 (PPO)
  - Gymnasium
  - Scikit-learn
  - Pandas/Numpy
  - Gradio (interface)

## 🔧 Installation
1. Cloner le dépôt
2. Créer un environnement virtuel
   - python -m venv env
   - source env/bin/activate # Linux/Mac
   - env\Scripts\activate    # Windows
3. Installer les dépendances :
   - pip install -r requirements.txt
4. Entraînement du modèle
   - python reinforcement_learning.py
## Lancer le dashboard interactif
Le script démarre automatiquement une interface Gradio après l'entraînement accessible sur : http://localhost:7860

## 🖥️ Interface Gradio
📌 Dashboard principal
### Visualisation des courbes de reward, prix, ou quantité.

### Aperçu du journal de simulation.

## 🔍 Tester un prix
### Choisissez une ligne (step) du dataset.
Entrez un multiplicateur de prix (ex : 1.3x).
Obtenez instantanément :
le prix calculé,
la quantité estimée,
le revenu simulé.

![image](https://github.com/user-attachments/assets/f39c3ae8-d78a-4b08-bd38-bfb700734a13)

![image](https://github.com/user-attachments/assets/f6a8f28d-98ec-4910-b322-0a57ffaee42e)

![image](https://github.com/user-attachments/assets/e8fecac1-bc79-45ca-911f-4e459b2b10ce)


