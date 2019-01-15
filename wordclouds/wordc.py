# Author: Rizart Dona
# File: wordc.py

import wordcloud
import argparse
import pandas
import sys

# parse arguments
parser = argparse.ArgumentParser(description="Generate WordClouds")
parser.add_argument("-i",
                    required=True,
                    dest="input",
                    help="Input csv training dataset")
arg = parser.parse_args()

# read Data
df = pandas.read_csv(arg.input, sep="\t")

# initialize map
# cmap holds for every category
# a concatenated string from all contents
cmap = {}

sys.stdout.write("setting up documents .. ")
sys.stdout.flush()
# for every category assign content (include title) to map
for index, row in df.iterrows():

    category = row['Category']
    content = row['Content']
    title = row['Title']

    if cmap.has_key(category):
        cmap[category] += " " + content + " " + title
    else:
        cmap[category] = content + " " + title
sys.stdout.write("done\n")
sys.stdout.flush()


# assign stopwords
stopwords = wordcloud.STOPWORDS.copy()

# add a couple of extra stopwords (use only lowercase)
stopwords.add("said")
stopwords.add("people")


# for every category
for category, content in cmap.iteritems():
    sys.stdout.write("producing wordcloud for %s .. " % category)
    sys.stdout.flush()
    wordCloud = wordcloud.WordCloud(stopwords=stopwords)
    wordCloud.width = 1920
    wordCloud.height = 1080

    # generate a wordcloud
    tc = wordCloud.generate_from_text(content)

    # save to image
    image = tc.to_image()
    image.save('wordclouds/%s.png' % (category))
    sys.stdout.write("done\n")
    sys.stdout.flush()
