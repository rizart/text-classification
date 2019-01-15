# Author: Rizart Dona
# File: pcolorit.py

from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv

# parse arguments
parser = argparse.ArgumentParser(
    description="Produce image from K-Means csv file")
parser.add_argument("-i",
                    required=True,
                    dest="input",
                    help="Input K-Means csv file")
arg = parser.parse_args()

# setup x, y, z
x = []
y = []
z = []

# csv filename
filename = arg.input

# populate x, y, z
with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    categories = reader.fieldnames[1:]
    x = categories
    for row in reader:
        zt = []
        for c in categories:
            zt.append(float(row[c]))
        z.append(zt)
        y.append('Cluster' + row['ClusterId'])

# create pandas DataFrame from x, y, z
df = DataFrame(z, index=y, columns=x)

# create heatmap plot
plt.pcolor(df, cmap='YlOrRd')
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.colorbar()
plt.title(filename)

# save plot as .png image
plt.savefig('clustering_KMeans.png')
