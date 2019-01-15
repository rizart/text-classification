# Author: Rizart Dona
# File: k_means.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import argparse
import pandas
import time
import csv
import sys

# parse arguments
parser = argparse.ArgumentParser(
    description="Run K-Means clustering algorithm")
parser.add_argument("-i",
                    required=True,
                    dest="input",
                    help="Input csv training dataset")
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(feature=False)
arg = parser.parse_args()

# read data
data = pandas.read_csv(arg.input, sep="\t")

# test case
if arg.test:
    n_values = 1000
    data = data[0:n_values]

# get unique categories
unique_categories = list(set(data.Category))

# vectorize document content
vectorizer = TfidfVectorizer(stop_words='english')
sys.stdout.write("vectorizing documents .. ")
sys.stdout.flush()
stime = time.time()
X = vectorizer.fit_transform(data['Content'])
etime = time.time()
seconds = int(etime - stime)
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
sys.stdout.write("(%02d:%02d:%02d)\n" % (h, m, s))

# setup k-means
k_clusters = len(unique_categories)
km = KMeans(n_clusters=k_clusters, init='k-means++',
            max_iter=100, n_jobs=-1, n_init=1)

# run k-means
sys.stdout.write("running K-Means .. ")
sys.stdout.flush()
stime = time.time()
km.fit(X)
etime = time.time()
seconds = int(etime - stime)
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
sys.stdout.write("(%d:%02d:%02d)\n" % (h, m, s))


# get cluster id for each row
cluster_ids = list(km.labels_)

# get category for each row
categlist = []
for index, row in data.iterrows():
    categlist.append(row['Category'])

# zip them
cc = zip(cluster_ids, categlist)

# setup output map
mmap = {}
for i in xrange(k_clusters):
    mmap[i] = {}
    mmap[i]["total"] = 0
    for categ in unique_categories:
        mmap[i][categ] = 0

# calculate
for clusterId, category in cc:
    mmap[clusterId][category] += 1
    mmap[clusterId]['total'] += 1

# write to csv file
output_csv_file = 'clustering_KMeans.csv'
with open(output_csv_file, 'w') as csvfile:
    fieldnames = ['ClusterId'] + unique_categories
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    row = {}

    for clusterId in mmap:
        total = mmap[clusterId]["total"]
        row['ClusterId'] = clusterId
        for categ in unique_categories:
            cat = mmap[clusterId][categ]
            perc = float(cat) / float(total)
            row[categ] = '%.2f' % perc
        writer.writerow(row)
