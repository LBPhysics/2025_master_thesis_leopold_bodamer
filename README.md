# Master Thesis – LaTeX Manuscript

This repository now focuses exclusively on the written thesis: LaTeX sources, bibliography, and curated figures. The numerical codes and Python tooling previously bundled here have moved to dedicated project under: 
https://github.com/LBPhysics/2025_master_thesis_python_leopold_bodamer.git

## Repository layout
```
Master_thesis/
├── figures/        # hand-drawn and exported graphics referenced in the text
├── latex/          # thesis sources (chapters, style file, bibliography)
├── .vscode/        # editor helpers for LaTeX workflows
├── .gitignore      # LaTeX- and figure-specific ignores
└── README.md       # this document
```

## Building the thesis
The LaTeX sources are set up for `latexmk`. From the repository root run:

```bash
latexmk -pdf -shell-escape -cd latex/main.tex
```

The command writes all auxiliary files alongside the sources and produces `latex/main.pdf`. Clean up with `latexmk -c -cd latex/main.tex` if you want to remove the generated intermediates.

## Figures
- `figures/` contains editable SVG drafts.

## Version control tips
- Commit generated PDFs sparingly—`latex/main.pdf` is ignored so that the history stays light.
- Keep auxiliary logs out of source control; `latexmk -c` is your friend before pushing.
- Use branches for major chapter rewrites to keep reviews focused.