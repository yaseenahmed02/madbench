.PHONY: setup test smoke figures

setup:
	bash scripts/setup/setup_all.sh

test:
	.venv/bin/python -m pytest tests -q

smoke:
	bash scripts/smoke.sh

figures:
	uv run --no-project --with pandas==3.0.3 --with matplotlib==3.10.9 python figures/make_track_a_figs.py
	uv run --no-project --with pandas==3.0.3 --with matplotlib==3.10.9 python figures/make_paper_figs.py
	uv run --no-project --with pandas==3.0.3 --with matplotlib==3.10.9 python figures/make_detail_figs.py
	uv run --no-project --with pandas==3.0.3 --with matplotlib==3.10.9 python figures/make_result_size_figs.py
