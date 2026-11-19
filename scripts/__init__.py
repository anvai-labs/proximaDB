# Makes `scripts` a regular package so `from scripts import …` resolves to this
# repo even when another project installs a top-level `scripts` package
# (a namespace package loses to any regular package on sys.path).
