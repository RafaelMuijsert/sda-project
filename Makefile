.PHONY: reproduce clean

reproduce:
	python src/family_history.py
	python src/lifestyle.py
	python src/contingency_heatmap.py
	# python src/chi_square.py
	# python src/ordinal_logistic_regression.py
	python src/binary_logistic_regression.py

clean:
	rm -f results/*.png

all: reproduce
	
