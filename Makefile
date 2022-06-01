
# maintenance
.PHONY: flake black test type interrogate \
		clean cleanall style check

# installation
.PHONY: install installextras pipinstalltest

# generate examples
.PHONY: intro squares hanoi escher_square lattice lenet logo \
		hilbert koch tensor latex hex_variation images

####------------------------------------------------------------####

PACKAGE_NAME := chalk
TESTPYPI_DOWNLOAD_URL := "https://test.pypi.org/simple/"
PYPIPINSTALL := "python -m pip install -U --index-url"
PIPINSTALL_PYPITEST := "$(PYPIPINSTALL) $(TESTPYPI_DOWNLOAD_URL)"
PKG_INFO := "import pkginfo; dev = pkginfo.Develop('.'); print((dev.$${FIELD}))"

# This is where you store the eggfile
# and other generated archives
ARCHIVES_DIR := ".archives"

# Folder path for tests
TESTS_DIR := "tests"

# Interrogate will flag the test as FAILED if
# % success threshold is under the following value
INTERROGATE_FAIL_UNDER := 0  # ideally this should be 100

# Specify paths of requirements.txt and dev_requirements.txt
REQ_FILE := "requirements.txt"
DEV_REQ_FILE := "dev_requirements.txt"

####------------------------------------------------------------####

### Code maintenance

## Run flake8

flake:
	@ echo "✨ Applying formatter: flake8 ... ⏳"
	flake8 --show-source chalk/*.py setup.py \
		# tests \

## Run black

black:
	@ echo "✨ Applying formatter: black ... ⏳"
	black --target-version py38 --line-length 79 $(PACKAGE_NAME)/*.py setup.py \
		# tests \

## Run pytest

test:
	@ echo "✨ Run tests: pytest ... ⏳"
	@if [ -d "$(TESTS_DIR)" ]; then pytest $(TESTS_DIR); else echo "\n\t🔥 No tests configured yet. Skipping tests.\n"; fi

## Run mypy

type:
	@ echo "✨ Applying type checker: mypy ... ⏳"
	mypy --strict --ignore-missing-imports $(PACKAGE_NAME)/*.py \
		# tests \

## Run interrogate

interrogate:
	@ echo "✨ Applying doctest checker: interrogate ... ⏳"
	$(eval INTERROGATE_CONFIG := -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under $(INTERROGATE_FAIL_UNDER))
	$(eval INTERROGATE_COMMAND := interrogate $(INTERROGATE_CONFIG))
	# Check tests
	@if [ -d "$(TESTS_DIR)" ]; then $(INTERROGATE_COMMAND) $(TESTS_DIR); else echo "\n\t🔥 No tests configured yet. Skipping tests.\n"; fi
	# Check package
	@$(INTERROGATE_COMMAND) $(PACKAGE_NAME)

## Cleanup
#
# Instruction:
#
# make clean    : if only cleaning artifacts created after running,
#                 code, tests, etc.
# make cleanall : if cleaning all artifacts (including the ones
#                 generated after creating dist and wheels).
#
# Note: archives created (dist and wheels) are stored in
#       $(ARCHIVES_DIR). This is defined at the top of this Makefile.
#--------------------------------------------------------------------

clean:
	@ echo "🪣 Cleaning repository ... ⏳"
	rm -rf \
		.ipynb_checkpoints **/.ipynb_checkpoints \
		.pytest_cache **/.pytest_cache \
		**/__pycache__ **/**/__pycache__

cleanall: clean
	@ echo "🪣 Cleaning dist/archive files ... ⏳"
	rm -rf build/* dist/* $(PACKAGE_NAME).egg-info/* $(ARCHIVES_DIR)/*

## Style Checks and Unit Tests

style: clean black flake interrogate clean

check: style type test clean

####------------------------------------------------------------####

### Code Installation

## Install for development (from local repository)
#
# Instruction: Contributors will need to run...
#
# - "make installextras": if installing for the first time or want to
#                         update to the latest dev-requirements or
#                         other extra dependencies.
# - "make install"      : if only installing the local source (after
#                         making some changes) to the source code.
#--------------------------------------------------------------------

install: clean
	@echo "📀 Installing $(PACKAGE_NAME) from local source ... ⏳"
	if [ -f $(REQ_FILE) ]; then python -m pip install -r $(REQ_FILE); fi
	python -m pip install -Ue "."

installextras: install
	@echo "📀 Installing $(PACKAGE_NAME) extra-dependencies from PyPI ... ⏳"
	if [ -f $(DEV_REQ_FILE) ]; then python -m pip install -r $(DEV_REQ_FILE); fi

## Install from test.pypi.org
#
# Instruction:
#
# 🔥 This is useful if you want to test the latest released package
#    from the TestPyPI, before you push the release to PyPI.
#--------------------------------------------------------------------

pipinstalltest:
	@echo "💿 Installing $(PACKAGE_NAME) from TestPyPI ($(TESTPYPI_DOWNLOAD_URL)) ... ⏳"
	@if [ $(VERSION) ]; then $(PIPINSTALL_PYPITEST) $(PACKAGE_NAME)==$(VERSION); else $(PIPINSTALL_PYPITEST) $(PACKAGE_NAME); fi;


####------------------------------------------------------------####

### Generate Output for Examples

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
	@echo "🎁 Generate all examples ... ⏳"
