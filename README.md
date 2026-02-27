# Flappy Bird NEAT

Jeu Flappy Bird intégrant une intelligence artificielle basée sur NEAT (NeuroEvolution of Augmenting Topologies), permettant l’évolution progressive de réseaux de neurones artificiels sans architecture prédéfinie. Chaque oiseau est piloté par un réseau neuronal dont la topologie et les poids évoluent par sélection naturelle, mutation et reproduction à partir d’une population initiale.

À chaque frame, le réseau reçoit exactement trois entrées issues de l’état courant du jeu : la position verticale de l’oiseau (`bird.pos_y`), la distance absolue entre cette position et le bord supérieur du trou du tuyau cible (`abs(bird.pos_y - focus_pipe.height)`), ainsi que la distance absolue entre cette position et le bord inférieur du trou (`abs(bird.pos_y - focus_pipe.bottom)`). Ces valeurs sont utilisées brutes (en pixels), sans normalisation préalable. La sortie du réseau détermine l’action de saut. La fonction de fitness récompense principalement la survie et le franchissement des tuyaux, tout en pénalisant les collisions.

Le processus évolutif exploite la spéciation propre à NEAT afin de préserver les innovations structurelles et de limiter la convergence prématurée. Les meilleures architectures sont sélectionnées génération après génération, ce qui permet une amélioration progressive des performances.

Le système inclut une persistance complète de l’état d’entraînement : sauvegarde du meilleur génome, de la population entière avec ses espèces, du numéro de génération et de l’état courant du jeu (score, tuyaux actifs, positions). La restauration automatique au démarrage permet de reprendre l’entraînement sans réinitialiser la dynamique évolutive.

Projet initialement basé sur le tutoriel de Tech With Tim, puis refactorisé avec une séparation plus stricte entre logique de jeu, moteur d’évolution et couche de persistance.

Note : J’ai créé le projet initialement en suivant le tutoriel YouTube *Python Flappy Bird AI Tutorial (with NEAT)* de *Tech With Tim* ([https://www.youtube.com/watch?v=MMxFDaIOHsE](https://www.youtube.com/watch?v=MMxFDaIOHsE)). Par la suite, j’ai ajouté un système de sauvegarde automatique de l’état des oiseaux et du jeu à la fermeture.

Note: I initially created the project by following the YouTube tutorial *Python Flappy Bird AI Tutorial (with NEAT)* by *Tech With Tim* ([https://www.youtube.com/watch?v=MMxFDaIOHsE](https://www.youtube.com/watch?v=MMxFDaIOHsE)). After that, I added an automatic saving system for the birds’ state and the overall game state when the program closes.

## Technologies

| Technologie | Version |
|-------------|---------|
| Python | 3.11+ |
| pygame | 2.6.1 |
| neat-python | 1.1.0 |

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
make run
```

Ou directement :
```bash
python3 "Flappy Bird/Flappy_bird.py"
```

## Fonctionnalités

- Entraînement NEAT avec population de 20 oiseaux par génération
- Sauvegarde automatique à la fermeture :
  - Meilleur génome
  - Population entière et espèces
  - Numéro de génération
  - État visuel du jeu (tuyaux, score)
- Restauration automatique au démarrage
- Affichage en temps réel : score, génération, nombre d'oiseaux vivants
