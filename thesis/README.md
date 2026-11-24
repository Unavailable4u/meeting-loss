# Thesis source: "Which Ideas Actually Save Compute?"

LaTeX source of the thesis on the `meeting-loss` project (meeting loss, selective backward pass, pre-registered compute-matched protocol).

## Build
    latexmk -pdf main.tex          # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
Works on Overleaf (upload the zip, main file `main.tex`, compiler pdfLaTeX). Needs only standard TeX Live packages
(tikz, tcolorbox, algorithm2e, natbib, booktabs, cleveref, ...). `main.pdf` is a pre-built copy.

## Edit first
* `metadata.tex`: name, degree, university, supervisor, date (title page).
* `frontmatter/declaration.tex` and `frontmatter/acknowledgements.tex`: red TODO boxes. Disclose AI assistance as your university requires.
* `references.bib`: header says which entries were checked against a web source and which were written from memory. Check the rest before submitting.

## Layout
    main.tex, preamble.tex, metadata.tex, references.bib
    frontmatter/   title page, declaration, abstract, acknowledgements, notation
    chapters/      ch01 introduction ... ch10 conclusion
    appendices/    A reproducibility, B hyper-parameters, C hypothesis register, D exploratory tables, E post-hoc decisions
    figures/       PDF figures (matplotlib) and TikZ diagrams (tikz_*.tex)
    tables/        LaTeX table bodies generated from the raw logs
    scripts/       Python that regenerates every figure and table from the repository's raw logs

## Regenerate figures and tables (optional)
    git clone https://github.com/Unavailable4u/meeting-loss.git
    export MEETING_LOSS_REPO=$PWD/meeting-loss
    pip install numpy pandas pyyaml matplotlib
    cd scripts && python make_tables.py && python fig_study1.py && python fig_study23.py
All numbers in the thesis come from these scripts (analysis commit b7cda46; locks prereg-v1 3378d36, prereg-v2 cc2830c, prereg-v3 a49e9e3).

## Known open items (also stated in Appendix A)
* The repository's own HYPOTHESES.md / FINDINGS.md did not yet record the H8-H10 verdicts; the thesis recomputes them from the raw logs (all three supported).
* The repository unit tests were not run while writing the thesis (no PyTorch in the writing environment). Run `pytest tests/`.
* Wall-clock time to target was not measured; all savings are counted FLOPs (one MLP on MNIST).
