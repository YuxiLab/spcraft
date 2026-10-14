.PHONY: all

all:
	python3 generate_figures.py
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build sparse_lu_superlu.tex > build/compile-pass-1.txt
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build sparse_lu_superlu.tex > build/compile-pass-2.txt
	cp build/sparse_lu_superlu.pdf sparse_lu_superlu.pdf
