flake: 
	flake8 --show-source diagrams/*

black: 
	black --line-length 79 diagrams/*

type:
	mypy --strict diagrams/*.py
