Text classification and clustering using scikit-learn.

Each folder contains a Makefile with some options:

  * option **run** runs the python script
  * option **pep8** formats the python scripts to be PEP8 compliant (more [here](https://www.python.org/dev/peps/pep-0008/))
  * option **clean** removes any script generated files (results, images, etc.)
  * some folders contain a **test** option for testing purposes, this indicates algorithms to take as input less data (1000 documents) to run faster

IMPORTANT NOTES:

   * requirements.txt file contains all needed packages to run these python scripts (e.g. "pip install -r requirements.txt")
   * in the classification folder, classifiers.py file needs a python command to be executed before running ( "import nltk;nltk.download('punkt')" - see lines 212, 213)
   * Makefiles assume that a folder named "datasets" exists here, with data files "test_set.csv" and "train_set.csv"
   * for Ubuntu/Linux users packages python-dev and python-tk are required
