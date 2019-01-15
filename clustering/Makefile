run:
	python k_means.py -i ../datasets/train_set.csv
	python pcolorit.py -i clustering_KMeans.csv
test:
	python k_means.py -i ../datasets/train_set.csv --test
	python pcolorit.py -i clustering_KMeans.csv
pep8:
	autopep8 -i *.py
clean:
	rm -rf clustering_KMeans.csv
	rm -rf clustering_KMeans.png