PANDOC ?= pandoc
LATEXMK ?= latexmk
DOC ?= foundations
export PATH := /Library/TeX/texbin:/opt/homebrew/bin:$(PATH)

SOLVER_DOCS := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(SOLVER_DOCS)/../..)
OBJ_DIR := $(REPO_ROOT)/docs/obj
PDF_DIR := $(REPO_ROOT)/docs/pdf
SOURCE := $(SOLVER_DOCS)/$(DOC).md
TEX := $(OBJ_DIR)/$(DOC).tex
BUILT_PDF := $(OBJ_DIR)/$(DOC).pdf
PDF := $(PDF_DIR)/$(DOC).pdf
HEADER := $(SOLVER_DOCS)/pdf-header.tex
FILTER := $(SOLVER_DOCS)/pdf-layout.lua
FIGURES := $(wildcard $(SOLVER_DOCS)/figures/*.tex)

ifeq ($(DOC),README)
NOTE_OPTIONS := --toc --toc-depth=2
endif

.PHONY: pdf tex publish video
pdf: $(PDF)
tex: $(TEX)
publish: $(PDF)

video:
	node "$(REPO_ROOT)/scripts/render_lu_video.mjs"

$(TEX): $(SOURCE) $(HEADER) $(FILTER) $(FIGURES) $(SOLVER_DOCS)/Makefile
	mkdir -p "$(OBJ_DIR)"
	$(PANDOC) --from=markdown+tex_math_dollars --to=latex --standalone \
		--shift-heading-level-by=-1 \
		--metadata subtitle="SpCraft solver learning path" $(NOTE_OPTIONS) \
		--variable fontsize=11pt --variable papersize=letter \
		--variable colorlinks=true --variable linkcolor=accent --variable urlcolor=accent \
		--variable geometry="margin=0.85in,headheight=15pt,headsep=18pt,footskip=25pt" \
		--include-in-header="$(HEADER)" --lua-filter="$(FILTER)" \
		--output="$@.tmp" "$<"
	mv "$@.tmp" "$@"

$(BUILT_PDF): $(TEX)
	$(LATEXMK) -pdf -synctex=1 -halt-on-error -interaction=nonstopmode \
		-file-line-error -outdir="$(OBJ_DIR)" "$<"

$(PDF): $(BUILT_PDF)
	mkdir -p "$(PDF_DIR)"
	cp "$<" "$@"
