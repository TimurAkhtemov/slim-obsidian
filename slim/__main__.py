"""`python -m slim` — the same entry point as the `slim` console script, so `devserver` can
respawn the server without depending on the console script being on PATH."""

from .cli import main

main()
