#!/usr/bin/env bash
# Copy pipeline figures and LaTeX tables into paper/, then zip paper/ for Overleaf upload.
# Usage (from the project root): bash scripts/sync_paper.sh
set -euo pipefail

cp output/figures/*.png paper/figures/
cp output/tables/tab_*.tex paper/tables/

rm -f paper.zip
(cd paper && zip -qr ../paper.zip . -x "*.aux" "*.log" "*.out" "*.bbl" "*.blg" "*.fls" "*.fdb_latexmk" "*.synctex.gz" "main.pdf")
echo "Wrote paper.zip: upload it in Overleaf via New Project > Upload Project."
