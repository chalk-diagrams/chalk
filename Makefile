.PHONY: help check autoformat notebook html clean

flake: 
	flake8 --show-source diagrams/*

black: 
	black --line-length 79 diagrams/*

type:
	mypy --strict --ignore-missing-imports diagrams/*.py

intro:
	python examples/intro.py

squares: 
	python examples/squares.py

hanoi: 
	python examples/hanoi.py

escher_square: 
	python examples/escher_square_limit.py

lattice: 
	python examples/lattice.py

images: squares hanoi intro escher_square

