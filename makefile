clean:
	rm -rf build dist
	rm -rf *.__pycache__
	rm -rf **/__pycache__
	rm -rf *.egg-info
	rm -rf ./**/*.egg-info
	rm -f sklearn_deap/**/*.c
	rm -f sklearn_deap/**/*.so

compile:
	python setup.py build_ext --inplace

build:
	$(MAKE) clean
	$(MAKE) compile

.PHONY: build