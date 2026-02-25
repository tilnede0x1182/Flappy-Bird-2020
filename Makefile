run:
	python3 "Flappy Bird\Flappy_bird.py"

compile_run:
	rm -f "Flappy Bird/data/best_genome.pkl" "Flappy Bird/data/checkpoint.pkl"
	python3 "Flappy Bird\Flappy_bird.py"
