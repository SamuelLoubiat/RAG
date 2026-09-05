UV = uv
PYTHON = $(UV) run python
MAIN_SCRIPT = main.py

.PHONY: install run debug clean lint

install:
	$(UV) sync

run: install
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb src/__main__.py

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint: install
	$(UV) run flake8 . --exclude=.venv,data
	$(UV) run mypy src main.py --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs