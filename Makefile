.PHONY: help check autoformat notebook html clean

flake: 
	flake8 --show-source chalk/*.py

black: 
	black --line-length 79 chalk/*.py

type:
	mypy --strict --ignore-missing-imports chalk/*.py

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

lenet: 
	python examples/lenet.py

logo: 
	python examples/logo.py

hilbert:
	python examples/hilbert.py

koch:
	python examples/koch.py

tensor:
	python examples/tensor.py

latex:
	python examples/latex.py

hex_variation:
	python examples/hex_variation.py

images: squares hanoi intro escher_square lenet logo hilbert koch tensor latex hex_variation
