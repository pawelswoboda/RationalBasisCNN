#!/bin/bash
# Compiles rational_basis_iclr2027.pdf via _build_wrapper.tex (see the comments there).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="/opt/software/pc2/EB-SW/software/texlive/20230313-GCC-12.3.0/bin/x86_64-linux:$PATH"
JOB=rational_basis_iclr2027
pdflatex -interaction=nonstopmode -jobname="$JOB" _build_wrapper.tex >/dev/null 2>&1 || true
bibtex "$JOB" >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode -jobname="$JOB" _build_wrapper.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode -jobname="$JOB" _build_wrapper.tex >/dev/null 2>&1 || true
echo "TeX errors: $(grep -c '^!' $JOB.log || true)"
grep -o 'Output written on [^)]*)' "$JOB.log" || echo "NO PDF PRODUCED"
