"""
	Flappy Bird avec NEAT (NeuroEvolution of Augmenting Topologies).
	Jeu d apprentissage automatique ou des oiseaux apprennent a eviter les tuyaux.

	Usage :
		python Flappy_bird.py
"""
from pathlib import Path
import random
import pickle
import atexit
import time

import neat
import pygame

# ==============================================================================
# Initialisation Pygame
# ==============================================================================

pygame.font.init()

# ==============================================================================
# Constantes
# ==============================================================================

WIN_WIDTH = 500
WIN_HEIGHT = 800
GENERATION = 0

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "assets" / "imgs"
CONFIG_FILE = BASE_DIR / "NEAT_configuration_file.txt"
SAVE_FILE = BASE_DIR / "data" / "best_genome.pkl"
CHECKPOINT_FILE = BASE_DIR / "data" / "checkpoint.pkl"

BEST_GENOME = None
BEST_FITNESS = -float("inf")
CURRENT_POPULATION = None
GAME_STATE = None
FAST_DEATH_COUNT = 0
LAST_GEN_TIME = 0

# ==============================================================================
# Fonctions utilitaires
# ==============================================================================

"""
	Charge une image depuis le dossier imgs et la met a l echelle x2.

	@param filename Nom du fichier image a charger.
	@return Surface pygame mise a l echelle.
	@raises SystemExit Si l image ne peut pas etre chargee.
"""
def load_scaled_image(filename):
	path = IMAGES_DIR / filename
	try:
		image = pygame.image.load(path)
	except pygame.error as load_error:
		raise SystemExit(f"Impossible de charger l image requise : {path}\n{load_error}") from load_error
	return pygame.transform.scale2x(image)


"""
	Affiche une chaine avec retour a la ligne.

	@param message Chaine a afficher.
"""
def afficher(message):
	print(message)


"""
	Affiche une chaine sans retour a la ligne.

	@param message Chaine a afficher.
"""
def afficher_sans_newline(message):
	print(message, end='')

# ------------------------------------------------------------------------------
# Chargement des ressources
# ------------------------------------------------------------------------------

BIRD_IMGS = [
	load_scaled_image("bird1.png"),
	load_scaled_image("bird2.png"),
	load_scaled_image("bird3.png")
]
PIPE_IMG = load_scaled_image("pipe.png")
BASE_IMG = load_scaled_image("base.png")
BG_IMG = load_scaled_image("bg.png")
STAT_FONT = pygame.font.SysFont("comicsans", 50)

# ==============================================================================
# Classe Bird
# ==============================================================================

class Bird:
	"""
		Represente un oiseau controle par le reseau de neurones.
	"""
	IMGS = BIRD_IMGS
	MAX_ROTATION = 25
	ROT_VEL = 20
	ANIMATION_TIME = 5

	def __init__(self, pos_x, pos_y):
		self.pos_x = pos_x
		self.pos_y = pos_y
		self.tilt = 0
		self.tick_count = 0
		self.velocity = 0
		self.height = self.pos_y
		self.img_count = 0
		self.img = self.IMGS[0]

	"""
		Fait sauter l oiseau en appliquant une velocite negative.
	"""
	def jump(self):
		self.velocity = -10.5
		self.tick_count = 0
		self.height = self.pos_y

	"""
		Deplace l oiseau selon la physique du jeu (gravite + velocite).
	"""
	def move(self):
		self.tick_count += 1
		displacement = self.velocity * self.tick_count + 1.5 * self.tick_count ** 2
		displacement = self._clamp_displacement(displacement)
		self.pos_y = self.pos_y + displacement
		self._update_tilt(displacement)

	"""
		Limite le deplacement a une valeur maximale.

		@param displacement Deplacement calcule.
		@return Deplacement limite.
	"""
	def _clamp_displacement(self, displacement):
		if displacement >= 16:
			displacement = displacement / abs(displacement) * 16
		if displacement < 0:
			displacement -= 2
		return displacement

	"""
		Met a jour l inclinaison de l oiseau selon le deplacement.

		@param displacement Deplacement actuel.
	"""
	def _update_tilt(self, displacement):
		if displacement < 0 or self.pos_y < self.height + 50:
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION
		elif self.tilt > -90:
			self.tilt -= self.ROT_VEL

	"""
		Dessine l oiseau sur la fenetre avec animation et rotation.

		@param window Surface pygame de destination.
	"""
	def draw(self, window):
		self.img_count += 1
		self._select_animation_frame()
		self._handle_dive_animation()
		self._render_rotated(window)

	"""
		Selectionne l image d animation selon le compteur.
	"""
	def _select_animation_frame(self):
		anim_time = self.ANIMATION_TIME
		if self.img_count < anim_time:
			self.img = self.IMGS[0]
		elif self.img_count < anim_time * 2:
			self.img = self.IMGS[1]
		elif self.img_count < anim_time * 3:
			self.img = self.IMGS[2]
		elif self.img_count < anim_time * 4:
			self.img = self.IMGS[1]
		elif self.img_count < anim_time * 4 + 1:
			self.img = self.IMGS[0]
			self.img_count = 0

	"""
		Gere l animation de plongee quand l oiseau descend fortement.
	"""
	def _handle_dive_animation(self):
		if self.tilt <= -80:
			self.img = self.IMGS[1]
			self.img_count = self.ANIMATION_TIME * 2

	"""
		Effectue le rendu de l oiseau avec rotation.

		@param window Surface pygame de destination.
	"""
	def _render_rotated(self, window):
		rotated_image = pygame.transform.rotate(self.img, self.tilt)
		original_rect = self.img.get_rect(topleft=(self.pos_x, self.pos_y))
		new_rect = rotated_image.get_rect(center=original_rect.center)
		window.blit(rotated_image, new_rect.topleft)

	"""
		Retourne le masque de collision de l oiseau.

		@return Masque pygame pour detection de collision.
	"""
	def get_mask(self):
		return pygame.mask.from_surface(self.img)

# ==============================================================================
# Classe Pipe
# ==============================================================================

class Pipe:
	"""
		Represente un tuyau (obstacle) dans le jeu.
	"""
	GAP = 200
	VEL = 5

	def __init__(self, pos_x):
		self.pos_x = pos_x
		self.height = 0
		self.gap = 100
		self.top = 0
		self.bottom = 0
		self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
		self.PIPE_BOTTOM = PIPE_IMG
		self.passed = False
		self.set_height()

	"""
		Definit aleatoirement la hauteur du tuyau et calcule les positions.
	"""
	def set_height(self):
		self.height = random.randrange(50, 450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	"""
		Deplace le tuyau vers la gauche.
	"""
	def move(self):
		self.pos_x -= self.VEL

	"""
		Dessine les deux parties du tuyau sur la fenetre.

		@param window Surface pygame de destination.
	"""
	def draw(self, window):
		window.blit(self.PIPE_TOP, (self.pos_x, self.top))
		window.blit(self.PIPE_BOTTOM, (self.pos_x, self.bottom))

	"""
		Verifie la collision entre l oiseau et le tuyau.

		@param bird Oiseau a verifier.
		@return True si collision detectee.
	"""
	def collide(self, bird):
		bird_mask = bird.get_mask()
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
		top_offset = (self.pos_x - bird.pos_x, self.top - round(bird.pos_y))
		bottom_offset = (self.pos_x - bird.pos_x, self.bottom - round(bird.pos_y))
		bottom_point = bird_mask.overlap(bottom_mask, bottom_offset)
		top_point = bird_mask.overlap(top_mask, top_offset)
		return bool(top_point or bottom_point)

# ==============================================================================
# Classe Base
# ==============================================================================

class Base:
	"""
		Represente le sol defilant du jeu.
	"""
	VEL = 5
	WIDTH = BASE_IMG.get_width()
	IMG = BASE_IMG

	def __init__(self, pos_y):
		self.pos_y = pos_y
		self.pos_x1 = 0
		self.pos_x2 = self.WIDTH

	"""
		Deplace le sol et gere le bouclage infini.
	"""
	def move(self):
		self.pos_x1 -= self.VEL
		self.pos_x2 -= self.VEL
		self._wrap_around()

	"""
		Repositionne les segments de sol pour creer un defilement infini.
	"""
	def _wrap_around(self):
		if self.pos_x1 + self.WIDTH < 0:
			self.pos_x1 = self.pos_x2 + self.WIDTH
		if self.pos_x2 + self.WIDTH < 0:
			self.pos_x2 = self.pos_x1 + self.WIDTH

	"""
		Dessine les deux segments de sol sur la fenetre.

		@param window Surface pygame de destination.
	"""
	def draw(self, window):
		window.blit(self.IMG, (self.pos_x1, self.pos_y))
		window.blit(self.IMG, (self.pos_x2, self.pos_y))

# ==============================================================================
# Fonctions de rendu
# ==============================================================================

"""
	Dessine tous les elements du jeu sur la fenetre.

	@param window Surface pygame de destination.
	@param birds Liste des oiseaux a dessiner.
	@param pipes Liste des tuyaux a dessiner.
	@param base Sol du jeu.
	@param score Score actuel.
	@param generation Numero de generation actuel.
"""
def draw_window(window, birds, pipes, base, score, generation):
	window.blit(BG_IMG, (0, 0))
	_draw_pipes(window, pipes)
	_draw_stats(window, score, generation, len(birds))
	base.draw(window)
	_draw_birds(window, birds)
	pygame.display.update()


"""
	Dessine tous les tuyaux.

	@param window Surface pygame de destination.
	@param pipes Liste des tuyaux.
"""
def _draw_pipes(window, pipes):
	for pipe in pipes:
		pipe.draw(window)


"""
	Dessine les statistiques (score, generation, nombre d oiseaux).

	@param window Surface pygame de destination.
	@param score Score actuel.
	@param generation Numero de generation.
	@param bird_count Nombre d oiseaux vivants.
"""
def _draw_stats(window, score, generation, bird_count):
	score_text = STAT_FONT.render(f"Score: {score}", 1, (255, 255, 255))
	window.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))
	gen_text = STAT_FONT.render(f"Gen: {generation}", 1, (255, 255, 255))
	window.blit(gen_text, (WIN_WIDTH - 35 - gen_text.get_width(), 45))
	birds_text = STAT_FONT.render(f"Nomber of birds: {bird_count}", 1, (255, 255, 255))
	window.blit(birds_text, (WIN_WIDTH - 100 - birds_text.get_width(), 105))


"""
	Dessine tous les oiseaux.

	@param window Surface pygame de destination.
	@param birds Liste des oiseaux.
"""
def _draw_birds(window, birds):
	for bird in birds:
		bird.draw(window)

# ==============================================================================
# Fonctions de gestion des entites
# ==============================================================================

"""
	Supprime un oiseau et ses donnees associees des listes.

	@param index Index de l oiseau a supprimer.
	@param birds Liste des oiseaux.
	@param nets Liste des reseaux de neurones.
	@param genomes Liste des genomes.
"""
def remove_bird(index, birds, nets, genomes):
	birds.pop(index)
	nets.pop(index)
	genomes.pop(index)


"""
	Met a jour tous les oiseaux (mouvement et decision de saut).

	@param birds Liste des oiseaux.
	@param nets Liste des reseaux de neurones.
	@param genomes Liste des genomes.
	@param pipes Liste des tuyaux.
	@param pipe_index Index du tuyau de focus.
"""
def update_birds(birds, nets, genomes, pipes, pipe_index):
	if not birds:
		return
	focus_pipe = pipes[pipe_index]
	for idx, bird in enumerate(birds):
		bird.move()
		genomes[idx].fitness += 0.1
		output = nets[idx].activate((
			bird.pos_y,
			abs(bird.pos_y - focus_pipe.height),
			abs(bird.pos_y - focus_pipe.bottom)
		))
		if output[0] > 0.5:
			bird.jump()

# ------------------------------------------------------------------------------
# Gestion des tuyaux
# ------------------------------------------------------------------------------

"""
	Gere les tuyaux : collisions, passage et deplacement.

	@param pipes Liste des tuyaux.
	@param birds Liste des oiseaux.
	@param nets Liste des reseaux de neurones.
	@param genomes Liste des genomes.
	@return Tuple (ajouter_tuyau, tuyaux_a_supprimer).
"""
def handle_pipes(pipes, birds, nets, genomes):
	add_pipe = False
	removed = []
	for pipe in pipes:
		add_pipe = _check_pipe_collisions(pipe, birds, nets, genomes, add_pipe)
		if pipe.pos_x + pipe.PIPE_TOP.get_width() < 0:
			removed.append(pipe)
		pipe.move()
	return add_pipe, removed


"""
	Verifie les collisions et passages pour un tuyau donne.

	@param pipe Tuyau a verifier.
	@param birds Liste des oiseaux.
	@param nets Liste des reseaux de neurones.
	@param genomes Liste des genomes.
	@param add_pipe Etat actuel du flag d ajout.
	@return Nouveau etat du flag d ajout.
"""
def _check_pipe_collisions(pipe, birds, nets, genomes, add_pipe):
	for idx in range(len(birds) - 1, -1, -1):
		bird = birds[idx]
		if pipe.collide(bird):
			genomes[idx].fitness -= 1
			remove_bird(idx, birds, nets, genomes)
			continue
		if (not pipe.passed) and pipe.pos_x < bird.pos_x:
			pipe.passed = True
			add_pipe = True
	return add_pipe


"""
	Supprime les oiseaux sortis des limites de l ecran.

	@param birds Liste des oiseaux.
	@param nets Liste des reseaux de neurones.
	@param genomes Liste des genomes.
"""
def purge_out_of_bounds(birds, nets, genomes):
	for idx in range(len(birds) - 1, -1, -1):
		bird = birds[idx]
		if bird.pos_y + bird.img.get_height() >= 730 or bird.pos_y < 0:
			remove_bird(idx, birds, nets, genomes)

# ==============================================================================
# Fonctions principales NEAT
# ==============================================================================

"""
	Fonction de fitness appelee par NEAT pour chaque generation.

	@param genomes_list Liste des genomes a evaluer.
	@param config Configuration NEAT.
"""
def evaluate_genomes(genomes_list, config):
	global GENERATION, GAME_STATE
	_check_fast_death_reset()
	GENERATION += 1
	base, pipes, window, clock = _initialize_game()
	safe_x = _get_safe_bird_x(pipes)
	nets, genomes, birds = _initialize_generation(genomes_list, config, safe_x)
	_run_game_loop(birds, nets, genomes, pipes, base, window, clock)
	_reload_game_state_from_checkpoint()


"""
	Initialise les oiseaux et reseaux pour une generation.

	@param genomes_list Liste des genomes.
	@param config Configuration NEAT.
	@return Tuple (nets, genomes, birds).
"""
def _initialize_generation(genomes_list, config, bird_x=230):
	nets = []
	genomes = []
	birds = []
	for _, genome in genomes_list:
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		birds.append(Bird(bird_x, 350))
		genome.fitness = 0
		genomes.append(genome)
	return nets, genomes, birds


"""
	Initialise les elements du jeu.

	@return Tuple (base, pipes, window, clock).
"""
def _initialize_game():
	global GAME_STATE
	window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	clock = pygame.time.Clock()
	if GAME_STATE is not None:
		base, pipes = _restore_game_elements()
	else:
		base = Base(730)
		pipes = [Pipe(700)]
	return base, pipes, window, clock


"""
	Restaure les elements du jeu depuis letat sauvegarde.

	@return Tuple (base, pipes).
"""
def _restore_game_elements():
	global GAME_STATE
	base = Base(730)
	base.pos_x1 = GAME_STATE.get("base_x1", 0)
	base.pos_x2 = GAME_STATE.get("base_x2", base.WIDTH)
	pipes = _restore_pipes_with_health_check()
	saved_score = GAME_STATE.get("score", 0)
	print(f"Etat jeu restaure : {len(pipes)} tuyaux, score={saved_score}")
	return base, pipes


"""
	Restaure les tuyaux avec health check.
	Repositionne les tuyaux pour que les oiseaux ne meurent pas immediatement.

	@return Liste des tuyaux repositionnes.
"""
def _restore_pipes_with_health_check():
	global GAME_STATE
	pipes_data = GAME_STATE.get("pipes", [])
	if not pipes_data:
		return [Pipe(700)]
	pipes = []
	for pipe_data in pipes_data:
		pipe = Pipe(pipe_data["pos_x"])
		pipe.height = pipe_data["height"]
		pipe.passed = pipe_data["passed"]
		pipe.top = pipe.height - pipe.PIPE_TOP.get_height()
		pipe.bottom = pipe.height + pipe.GAP
		pipes.append(pipe)
	return pipes


"""
	Calcule la position X safe pour les oiseaux entre 2 tuyaux.

	@param pipes Liste des tuyaux.
	@return Position X safe pour les oiseaux.
"""
def _get_safe_bird_x(pipes):
	default_x = 230
	if not pipes:
		print(f"DEBUG: pas de pipes, bird_x={default_x}")
		return default_x
	pipes_sorted = sorted(pipes, key=lambda p: p.pos_x)
	print(f"DEBUG: pipes positions = {[p.pos_x for p in pipes_sorted]}")
	for idx in range(len(pipes_sorted) - 1):
		current_pipe = pipes_sorted[idx]
		next_pipe = pipes_sorted[idx + 1]
		gap_start = current_pipe.pos_x + current_pipe.PIPE_TOP.get_width() + 20
		gap_end = next_pipe.pos_x - 20
		if gap_end > gap_start:
			result = (gap_start + gap_end) // 2
			result = min(result, WIN_WIDTH - 100)
			print(f"DEBUG: gap trouve entre {gap_start} et {gap_end}, bird_x={result}")
			return result
	first_pipe = pipes_sorted[0]
	if first_pipe.pos_x > 100:
		result = first_pipe.pos_x // 2
		print(f"DEBUG: avant premier tuyau, bird_x={result}")
		return result
	print(f"DEBUG: fallback, bird_x={default_x}")
	return default_x

# ------------------------------------------------------------------------------
# Detection des morts rapides (reset automatique)
# ------------------------------------------------------------------------------

"""
	Detecte si 3 generations meurent en moins de 3 secondes apres restauration.
	Si oui, supprime les sauvegardes et reset le jeu.
"""
def _check_fast_death_reset():
	global FAST_DEATH_COUNT, LAST_GEN_TIME, GAME_STATE, GENERATION
	current_time = time.time()
	if GAME_STATE is None:
		FAST_DEATH_COUNT = 0
		LAST_GEN_TIME = current_time
		return
	time_since_last = current_time - LAST_GEN_TIME
	if time_since_last < 1.0:
		FAST_DEATH_COUNT += 1
		print(f"Mort rapide detectee ({FAST_DEATH_COUNT}/3)")
	else:
		FAST_DEATH_COUNT = 0
	LAST_GEN_TIME = current_time
	if FAST_DEATH_COUNT >= 3:
		print("3 morts rapides consecutives - Reset des sauvegardes")
		_delete_save_files()
		GAME_STATE = None
		GENERATION = 0
		FAST_DEATH_COUNT = 0


"""
	Supprime les fichiers de sauvegarde dans data/.
"""
def _delete_save_files():
	if SAVE_FILE.exists():
		SAVE_FILE.unlink()
		print(f"Supprime : {SAVE_FILE}")
	if CHECKPOINT_FILE.exists():
		CHECKPOINT_FILE.unlink()
		print(f"Supprime : {CHECKPOINT_FILE}")

# ------------------------------------------------------------------------------
# Gestion de letat du jeu
# ------------------------------------------------------------------------------

"""
	Met a jour letat global du jeu pour sauvegarde.

	@param birds Liste des oiseaux.
	@param pipes Liste des tuyaux.
	@param base Sol du jeu.
	@param score Score actuel.
"""
def _update_game_state(birds, pipes, base, score):
	global GAME_STATE
	if score <= 0:
		return
	birds_data = [{"pos_x": b.pos_x, "pos_y": b.pos_y, "velocity": b.velocity, "tilt": b.tilt} for b in birds]
	pipes_data = [{"pos_x": p.pos_x, "height": p.height, "passed": p.passed} for p in pipes]
	GAME_STATE = {
		"birds": birds_data,
		"pipes": pipes_data,
		"base_x1": base.pos_x1,
		"base_x2": base.pos_x2,
		"score": score
	}


"""
	Recupere le score depuis letat sauvegarde.

	@return Score sauvegarde ou 0.
"""
def _restore_game_state_score():
	global GAME_STATE
	if GAME_STATE is None:
		return 0
	return GAME_STATE.get("score", 0)


"""
	Recharge GAME_STATE depuis le fichier checkpoint.
	Appelee apres chaque generation pour reprendre a la derniere sauvegarde valide.
"""
def _reload_game_state_from_checkpoint():
	global GAME_STATE
	if not CHECKPOINT_FILE.exists():
		return
	try:
		with open(CHECKPOINT_FILE, "rb") as fichier:
			checkpoint_data = pickle.load(fichier)
		GAME_STATE = checkpoint_data.get("game_state", None)
	except (IOError, pickle.UnpicklingError):
		pass


# ------------------------------------------------------------------------------
# Boucle de jeu
# ------------------------------------------------------------------------------

"""
	Execute la boucle principale du jeu.

	@param birds Liste des oiseaux.
	@param nets Liste des reseaux de neurones.
	@param genomes Liste des genomes.
	@param pipes Liste des tuyaux.
	@param base Sol du jeu.
	@param window Fenetre pygame.
	@param clock Horloge pygame.
"""
def _run_game_loop(birds, nets, genomes, pipes, base, window, clock):
	global GAME_STATE
	score = _restore_game_state_score()
	running = True
	while running:
		clock.tick(30)
		if _handle_quit_event():
			break
		pipe_index = _get_focus_pipe_index(birds, pipes)
		if pipe_index == -1:
			break
		score, running = _process_frame(birds, nets, genomes, pipes, base, window, score, pipe_index)
		_update_game_state(birds, pipes, base, score)


"""
	Traite une frame du jeu.

	@param birds Liste des oiseaux.
	@param nets Liste des reseaux de neurones.
	@param genomes Liste des genomes.
	@param pipes Liste des tuyaux.
	@param base Sol du jeu.
	@param window Fenetre pygame.
	@param score Score actuel.
	@param pipe_index Index du tuyau de focus.
	@return Tuple (score, continuer).
"""
def _process_frame(birds, nets, genomes, pipes, base, window, score, pipe_index):
	update_birds(birds, nets, genomes, pipes, pipe_index)
	add_pipe, removed = handle_pipes(pipes, birds, nets, genomes)
	score = _process_pipe_events(add_pipe, removed, pipes, genomes, score)
	purge_out_of_bounds(birds, nets, genomes)
	base.move()
	draw_window(window, birds, pipes, base, score, GENERATION)
	return score, len(birds) > 0


"""
	Gere l evenement de fermeture de fenetre.

	@return True si l utilisateur veut quitter.
"""
def _handle_quit_event():
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			raise SystemExit(0)
	return False


"""
	Determine l index du tuyau sur lequel les oiseaux doivent se concentrer.

	@param birds Liste des oiseaux.
	@param pipes Liste des tuyaux.
	@return Index du tuyau ou -1 si plus d oiseaux.
"""
def _get_focus_pipe_index(birds, pipes):
	if not birds:
		return -1
	pipe_index = 0
	if len(pipes) > 1 and birds[0].pos_x > pipes[0].pos_x + pipes[0].PIPE_TOP.get_width():
		pipe_index = 1
	return pipe_index


"""
	Traite les evenements lies aux tuyaux (ajout, suppression, score).

	@param add_pipe Flag indiquant s il faut ajouter un tuyau.
	@param removed Liste des tuyaux a supprimer.
	@param pipes Liste des tuyaux.
	@param genomes Liste des genomes.
	@param score Score actuel.
	@return Score mis a jour.
"""
def _process_pipe_events(add_pipe, removed, pipes, genomes, score):
	if add_pipe:
		score += 1
		for genome in genomes:
			genome.fitness += 5
			update_best_genome(genome, genome.fitness)
		pipes.append(Pipe(600))
	for pipe_to_remove in removed:
		pipes.remove(pipe_to_remove)
	return score

# ==============================================================================
# Sauvegarde et chargement du meilleur genome
# ==============================================================================

"""
	Sauvegarde le meilleur genome dans un fichier pickle.
	Appelee automatiquement a la fermeture du programme.
"""
def save_best_genome():
	global BEST_GENOME, GENERATION, CURRENT_POPULATION
	if BEST_GENOME is None:
		return
	try:
		save_data = {
			"genome": BEST_GENOME,
			"generation": GENERATION
		}
		if CURRENT_POPULATION is not None:
			save_data["population"] = CURRENT_POPULATION.population
			save_data["species"] = CURRENT_POPULATION.species
		with open(SAVE_FILE, "wb") as fichier:
			pickle.dump(save_data, fichier)
		print(f"Sauvegarde : genome + population + generation {GENERATION}")
	except IOError as erreur_io:
		print(f"save_best_genome : {erreur_io}")


"""
	Charge le meilleur genome depuis un fichier pickle.

	@return Genome charge ou None si fichier inexistant.
"""
def load_best_genome():
	global GENERATION
	if not SAVE_FILE.exists():
		return None
	try:
		with open(SAVE_FILE, "rb") as fichier:
			save_data = pickle.load(fichier)
		if isinstance(save_data, dict):
			genome = save_data.get("genome")
			GENERATION = save_data.get("generation", 0)
		else:
			genome = save_data
		print(f"Charge : genome + generation {GENERATION} depuis {SAVE_FILE}")
		return genome
	except (IOError, pickle.UnpicklingError) as erreur:
		print(f"load_best_genome : {erreur}")
		return None


"""
	Met a jour le meilleur genome si le fitness actuel est superieur.

	@param genome Genome a evaluer.
	@param fitness Fitness du genome.
"""
def update_best_genome(genome, fitness):
	global BEST_GENOME, BEST_FITNESS
	if fitness > BEST_FITNESS:
		BEST_FITNESS = fitness
		BEST_GENOME = genome


# Enregistrement de la sauvegarde automatique a la fermeture
atexit.register(save_best_genome)


"""
	Sauvegarde le checkpoint complet (population, especes, generation).
	Appelee automatiquement a la fermeture du programme.
"""
def save_checkpoint():
	global CURRENT_POPULATION, GENERATION, GAME_STATE
	if CURRENT_POPULATION is None:
		return
	try:
		checkpoint_data = {
			"population": CURRENT_POPULATION.population,
			"species": CURRENT_POPULATION.species,
			"generation": GENERATION,
			"game_state": GAME_STATE
		}
		with open(CHECKPOINT_FILE, "wb") as fichier:
			pickle.dump(checkpoint_data, fichier)
		print(f"Checkpoint sauvegarde : generation {GENERATION}")
	except IOError as erreur_io:
		print(f"save_checkpoint : {erreur_io}")


"""
	Charge le checkpoint complet depuis un fichier pickle.

	@return Dictionnaire checkpoint ou None si fichier inexistant.
"""
def load_checkpoint():
	global GENERATION, GAME_STATE
	if not CHECKPOINT_FILE.exists():
		return None
	try:
		with open(CHECKPOINT_FILE, "rb") as fichier:
			checkpoint_data = pickle.load(fichier)
		GENERATION = checkpoint_data.get("generation", 0)
		GAME_STATE = checkpoint_data.get("game_state", None)
		print(f"Checkpoint charge : generation {GENERATION}")
		return checkpoint_data
	except (IOError, pickle.UnpicklingError) as erreur:
		print(f"load_checkpoint : {erreur}")
		return None


# Enregistrement sauvegarde checkpoint a la fermeture
atexit.register(save_checkpoint)

# ==============================================================================
# Configuration et lancement
# ==============================================================================

"""
	Charge la configuration NEAT depuis un fichier.

	@param config_path Chemin optionnel vers le fichier de configuration.
	@return Configuration NEAT.
	@raises SystemExit Si le fichier n existe pas.
"""
def load_config(config_path=None):
	target = Path(config_path) if config_path else CONFIG_FILE
	if not target.exists():
		raise SystemExit(f"Impossible de trouver la configuration NEAT : {target}")
	return neat.config.Config(
		neat.DefaultGenome,
		neat.DefaultReproduction,
		neat.DefaultSpeciesSet,
		neat.DefaultStagnation,
		str(target),
	)


"""
	Lance l entrainement NEAT avec les reporters.

	@param config_path Chemin optionnel vers le fichier de configuration.
"""
def run_training(config_path=None):
	global CURRENT_POPULATION
	config = load_config(config_path)
	population = _create_or_restore_population(config)
	CURRENT_POPULATION = population
	population.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	population.add_reporter(stats)
	population.run(evaluate_genomes, 3000)


"""
	Cree une nouvelle population ou restaure depuis un checkpoint.

	@param config Configuration NEAT.
	@return Population NEAT.
"""
def _create_or_restore_population(config):
	checkpoint = load_checkpoint()
	if checkpoint is None:
		return neat.Population(config)
	population = neat.Population(config)
	population.population = checkpoint["population"]
	population.species = checkpoint["species"]
	population.generation = checkpoint.get("generation", 0)
	print(f"Population restauree : {len(population.population)} genomes")
	return population


"""
	Injecte le genome sauvegarde dans la population initiale.

	@param population Population NEAT.
	@param config Configuration NEAT.
"""
def _inject_saved_genome(population, config):
	saved_genome = load_best_genome()
	if saved_genome is None:
		return
	global BEST_GENOME, BEST_FITNESS
	BEST_GENOME = saved_genome
	BEST_FITNESS = saved_genome.fitness if hasattr(saved_genome, "fitness") else 0
	if population.population:
		first_key = next(iter(population.population))
		saved_genome.key = first_key
		population.population[first_key] = saved_genome
		print(f"Genome injecte avec fitness={BEST_FITNESS}")

# ==============================================================================
# Main
# ==============================================================================

"""
	Point d entree du programme.
"""
def main():
	run_training()

# ==============================================================================
# Lancement du programme
# ==============================================================================

if __name__ == "__main__":
	main()
