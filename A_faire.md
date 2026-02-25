# À faire

## 1. Compréhension rapide
- `Flappy_bird.py` charge les sprites (oiseau, tuyaux, sol, fond) depuis le dossier `imgs` et configure la fenêtre Pygame en 500×800.
- La classe `Bird` gère la position verticale, l’animation des ailes et les collisions via des masques `pygame.mask`.
- `Pipe` instancie les obstacles avec un espace (`GAP`) fixe et marque quand un oiseau les a dépassés pour incrémenter le score.
- `Base` anime le sol sur deux sprites qui défilent, créant l’illusion d’un mouvement infini.
- `draw_window` compose le rendu (fond, tuyaux, oiseau.xN, score, génération) et est appelé à chaque frame.
- La fonction `main(genomes, config)` prépare les réseaux NEAT, gère la boucle de jeu, calcule le fitness et supprime les oiseaux qui percutent un obstacle ou sortent de l’écran.

## 2. Bugs à corriger
- Dans `main`, `output = nets[x].activate(...)` est calculé dans la boucle sur les oiseaux mais le test `if (output[0] > 0.5): bird.jump()` est placé **en dehors** de la boucle. Seul le dernier oiseau évalué saute (et `bird` n’est même plus défini si la liste est vide). Ce bloc doit rester dans la boucle pour que chaque oiseau puisse réagir à son propre réseau.
- La fonction `run` calcule `config_path = os.path.join(local_dir, "config-feeforward.txt")` mais appelle `run(adresse_fichier_de_configuration_neat)` en bas du fichier. Lancer le script depuis un autre répertoire casse la résolution (le chemin relatif n’existe plus). Utiliser `config_path` évite les erreurs `FileNotFoundError`.

## 3. DRY en priorité
- Le même schéma « lister tous les oiseaux, lister tous les tuyaux » est dupliqué pour le mouvement, la collision, la suppression et le calcul de fitness. Créer une fonction `update_birds(birds, pipes, nets, ge)` centraliserait ces quatre boucles.
- `draw_window` affiche trois fois un `STAT_FONT.render` quasi identique. Mettre un helper `blit_stat(label, value, y)` réduirait ces répétitions.

## 4. Sécurité
- Aucun des chargements d’images (`pygame.image.load`) ni de fichier de configuration NEAT n’est encapsulé. En cas de ressource manquante, le script crashe avec une trace pleine. Entourer ces appels d’un `try/except` et afficher un message compréhensible éviterait d’exposer l’utilisateur final à une pile Python brute.
- Les chemins d’accès (`adresse_images`, `adresse_fichier_de_configuration_neat`) sont codés en relatif et ne sont jamais normalisés. Ajouter `os.path.join(os.path.dirname(__file__), ...)` éviterait que quelqu’un exécute le script depuis un dossier non prévu et charge des fichiers arbitraires.
