# SNAKE·RL

Un agent **Deep Q-Network (DQN)** développé avec PyTorch qui apprend à jouer au Snake par apprentissage par renforcement, sans stratégie de déplacement prédéfinie.

**Démo en ligne :** [SNAKE·RL](https://lab-neural.netlify.app/)

Observe l'agent jouer, ses perceptions et ses Q-valeurs en direct, ou affronte-le dans un duel sur la même séquence de pommes.

## Comment ça marche

### L'environnement
- Grille de 20 × 20.
- 3 actions : tout droit, tourner à droite ou à gauche. Le demi-tour est impossible.
- Fin de partie en cas de collision ou de famine.

### L'état : 17 valeurs binaires

| Information | Valeurs |
|---|---:|
| Dangers à 1 case (devant, droite, gauche) | 3 |
| Dangers à 2 cases (devant, droite, gauche) | 3 |
| Direction du serpent | 4 |
| Position relative de la pomme | 4 |
| Pièges potentiels | 3 |
| **Total** | **17** |

Les pièges sont détectés par une recherche bornée à 120 cases dans l'espace accessible.

### Récompenses

| Événement | Récompense |
|---|---:|
| Pomme mangée | +10 |
| Collision ou famine | −10 |
| Rapprochement de la pomme | +1 |
| Éloignement de la pomme | −1 |

### Réseau et entraînement

Architecture : `Linear(17, 256) → ReLU → Linear(256, 3)`

| Paramètre | Valeur |
|---|---|
| Optimiseur | Adam |
| Taux d'apprentissage | 0,001 |
| Perte | MSE |
| Facteur d'actualisation γ | 0,99 |
| Mémoire de rejeu | 100 000 transitions |
| Mémoire longue | Jusqu'à 1 000 transitions en fin de partie |
| Réseau cible | Synchronisé toutes les 100 parties |
| Exploration | Décroît jusqu'à 0 après 300 parties |

`train_step()` met à jour le réseau principal à chaque étape. `train_long_memory()` utilise le Double DQN en fin de partie : le réseau principal sélectionne l'action et le réseau cible évalue sa valeur.

À l'inférence, l'agent choisit l'action ayant la Q-valeur la plus élevée, sans exploration aléatoire.

## Structure du dépôt

.
├── train_snake.py    # Entraînement
├── mainsnake.py      # réseau de neurone
├── snake.py          # observer l'agent jouer
├── snake_pygame.py   # interface et logique du jeu 
├── model.pth         # Poids entraînés
└── README.md

## Installation

pip install torch numpy pygame

## Utilisation

python train_snake.py
python evalue.py
python mainsnake.py

### Chemin du modèle

Le fichier `model.pth` est utilisé pour sauvegarder et charger le modèle entraîné. Son chemin doit être configuré manuellement dans les scripts concernés.

1. Ouvre `train_snake.py`, `evalue.py` et `mainsnake.py`.
2. Recherche les instructions `torch.save()` et `torch.load()`.
3. Renseigne le chemin vers ton fichier `model.pth` dans chaque script, en utilisant le même emplacement pour la sauvegarde et le chargement.

Exemple :

MODEL_PATH = r"C:\Users\toi\snake-rl\model.pth"

Adapte ce chemin à l'emplacement réel de ton fichier.

## Auteur

[**gabriel-tgd**](https://github.com/gabriel-tgd)