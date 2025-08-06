##

```bash
uv run pytest -q
uv run python da-gp/scripts/bench.py --obs_grid 100 --backends sklearn
uv run python da-gp/scripts/plot_perf.py bench.csv
uv run python da-gp/scripts/plot_posterior.py --n_obs 2000
latexmk -pdf main.tex          # or your usual build target
```