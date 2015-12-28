default: main.pdf

texfiles := $(wildcard *.tex)

main.pdf: $(texfiles) all.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

run:
	go main.pdf

clean:
	rm -f main.aux main.bbl main.blg main.log main.pdf
