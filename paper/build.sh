#!/bin/sh
set -eu
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex supporting_information.tex
