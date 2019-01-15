run:
	python classifiers.py -train ../datasets/train_set.csv -test ../datasets/test_set.csv
test:
	python classifiers.py -train ../datasets/train_set.csv -test ../datasets/test_set.csv --test
pep8:
	autopep8 -i *.py
clean:
	rm -rf roc_plots/*.png
	rm -rf EvaluationMetric_10fold.csv
	rm -rf testSet_categories.csv