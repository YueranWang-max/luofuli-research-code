# Luofuli research code

Frozen study design: paper/CONFIRMATORY_PROTOCOL_V3.md and paper/EXTERNAL_VALIDATION_PROTOCOL_MINICPM5.md. Training entry points are in src/train. Evaluation and statistical analysis are in paper/. Use the archived protocol for model and environment versions. GPU scripts may require adapting local model and data paths.

To regenerate the eight manuscript figures, install R packages jsonlite and svglite, change to paper/figures and run Rscript render_redesign.R. The script reads the bundled figure inputs. To run Python tests with project dependencies installed, run python -m pytest from the repository root. Model weights, training checkpoints and full evaluation outputs are supplied separately or obtained from their respective sources.
