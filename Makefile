.PHONY: reproduce clean

reproduce:
	python src/data_preparation/prepare_data.py
	python src/exploratory_analysis/family_history.py
	python src/exploratory_analysis/lifestyle.py
	python src/exploratory_analysis/contingency_heatmap.py
	python src/modeling/chi_square.py
	python src/modeling/binary_logistic_regression.py
	python src/modeling/compare_hereditary_lifestyle.py
	python src/modeling/interaction_model.py

clean:
	rm -f results/*.png

all: reproduce
	
