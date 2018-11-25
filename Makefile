clean:
	rm -rf src/__pycache__
	rm -rf data/

download:
	git clone git@github.com:henriquepgomide/caRtola.git
	mv caRtola/data/ data/
	rm -rf caRtola

run:
	python3 src/main.py
