# Technical report: "Which Ideas Actually Save Compute?"

LaTeX source of an independent research report on the `meeting-loss` project (meeting loss, selective backward pass, pre-registered compute-matched protocol).

Authors: S. M. Shuaib Islam Sayad and Shaikh Md Abu Ahad (Department of Electronics and Telecommunication, RUET). Not part of a degree programme.

## Build
    cd report
    latexmk -pdf main.tex          # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
Works on Overleaf (upload the `report/` folder, main file `main.tex`, compiler pdfLaTeX). Needs only standard TeX Live packages
(tikz, tcolorbox, algorithm2e, natbib, booktabs, cleveref, ...). `main.pdf` is a pre-built copy: rebuild and commit it after any change.

## Edit first
* `metadata.tex`: authors, affiliation, date (title page).
* `frontmatter/declaration.tex`: the statement on the use of AI assistance and the licence note (CC BY 4.0 for the report, MIT or Apache-2.0 for the code; see `../LICENSE.md`). Check that the wording matches what you did.
* `references.bib`: the header says which entries were checked against a web source and when. Re-check before submitting to a venue.

## Layout
    main.tex, preamble.tex, metadata.tex, references.bib
    frontmatter/   title page, statement of authorship, abstract, notation
    chapters/      ch01 introduction ... ch10 conclusion
    appendices/    A reproducibility, B hyper-parameters, C hypothesis register, D exploratory tables, E post-hoc decisions
    figures/       PDF figures (matplotlib) and TikZ diagrams (tikz_*.tex)
    tables/        LaTeX table bodies generated from the raw logs
    scripts/       Python that regenerates every figure and table from the repository's raw logs

## Regenerate figures and tables (optional)
From the repository root:

    pip install numpy pandas pyyaml matplotlib
    cd report/scripts && python make_tables.py && python fig_study1.py && python fig_study23.py

The scripts find the repository root themselves; set `MEETING_LOSS_REPO` to override. All numbers in the report come from these scripts (analysis commit b7cda46; locks prereg-v1 3378d36, prereg-v2 cc2830c, prereg-v3 a49e9e3). Regenerated figure PDFs differ byte-wise from the committed ones only in their embedded timestamps; the tables regenerate identically.

## Known open items (also stated in Appendix A)
* The repository unit tests were not run while the report was written (no PyTorch in the writing environment). Run `pytest tests/` and update Appendix A with the result.
* Wall-clock time to target was not measured; all savings are counted FLOPs (one MLP on MNIST).
* The statement on AI assistance in `frontmatter/declaration.tex` is still to be written.
