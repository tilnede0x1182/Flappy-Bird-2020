from pathlib import Path
import random

import neat
import pygame

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "imgs"
CONFIG_FILE = BASE_DIR / "NEAT_configuration_file.txt"


def _load_scaled_image(filename):
    """Charge une image projet depuis le dossier imgs et la met à l'échelle."""
    path = IMAGES_DIR / filename
    try:
        image = pygame.image.load(path)
    except pygame.error as exc:  # pragma: no cover
        raise SystemExit(f"Impossible de charger l'image requise : {path}\n{exc}") from exc
    return pygame.transform.scale2x(image)


BIRD_IMGS = [_load_scaled_image("bird1.png"), _load_scaled_image("bird2.png"), _load_scaled_image("bird3.png")]
PIPE_IMG = _load_scaled_image("pipe.png")
BASE_IMG = _load_scaled_image("base.png")
BG_IMG = _load_scaled_image("bg.png")

STAT_FONT = pygame.font.SysFont("comicsans", 50)


def aff(var_str):
    print(var_str)


def affnn(var_str):
    print(var_str, end='')


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2
        if d >= 16:
            d = d / abs(d) * 16
        if d < 0:
            d -= 2
        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        elif self.tilt > -90:
            self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)
        return bool(t_point or b_point)


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, gen):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render(f"Score: {score}", 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render(f"Gen: {gen}", 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 35 - text.get_width(), 45))

    text = STAT_FONT.render(f"Nomber of birds: {len(birds)}", 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 100 - text.get_width(), 105))

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def remove_bird(index, birds, nets, ge):
    birds.pop(index)
    nets.pop(index)
    ge.pop(index)


def update_birds(birds, nets, ge, pipes, pipe_ind):
    if not birds:
        return
    focus_pipe = pipes[pipe_ind]
    for idx, bird in enumerate(birds):
        bird.move()
        ge[idx].fitness += 0.1
        output = nets[idx].activate((bird.y, abs(bird.y - focus_pipe.height), abs(bird.y - focus_pipe.bottom)))
        if output[0] > 0.5:
            bird.jump()


def handle_pipes(pipes, birds, nets, ge):
    add_pipe = False
    removed = []
    for pipe in pipes:
        for idx in range(len(birds) - 1, -1, -1):
            bird = birds[idx]
            if pipe.collide(bird):
                ge[idx].fitness -= 1
                remove_bird(idx, birds, nets, ge)
                continue
            if (not pipe.passed) and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
        if pipe.x + pipe.PIPE_TOP.get_width() < 0:
            removed.append(pipe)
        pipe.move()
    return add_pipe, removed


def purge_out_of_bounds(birds, nets, ge):
    for idx in range(len(birds) - 1, -1, -1):
        bird = birds[idx]
        if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
            remove_bird(idx, birds, nets, ge)


def main(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                raise SystemExit(0)

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            break

        update_birds(birds, nets, ge, pipes, pipe_ind)
        add_pipe, rem = handle_pipes(pipes, birds, nets, ge)

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        purge_out_of_bounds(birds, nets, ge)

        base.move()
        draw_window(win, birds, pipes, base, score, GEN)

        if not birds:
            break


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


def run(config_path=None):
    config = load_config(config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(main, 3000)


if __name__ == "__main__":
    run()
