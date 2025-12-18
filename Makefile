.PHONY: reproduce clean

reproduce:
	python src/family_history.py
	python src/lifestyle.py
	python src/contingency_heatmap.py
	# python src/chi_square.py
	python src/logistic_regression.py

clean:
	rm -f figures/*.png

all: reproduce
	
