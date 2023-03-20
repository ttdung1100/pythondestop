class KMeans:
"""performs k-means clustering"""
def __init__(self, k):
self.k = k # number of clusters
self.means = None # means of clusters
def classify(self, input):
"""return the index of the cluster closest to the input"""
    return min(range(self.k),
key=lambda i: squared_distance(input, self.means[i]))

def train(self, inputs):
# choose k random points as the initial means
self.means = random.sample(inputs, self.k)
assignments = None

while True:
# Find new assignments
new_assignments = map(self.classify, inputs)
# If no assignments have changed, we're done.
if assignments == new_assignments:
    return
# Otherwise keep the new assignments,
assignments = new_assignments
# And compute new means based on the new assignments
for i in range(self.k):
# find all the points assigned to cluster i
i_points = [p for p, a in zip(inputs, assignments) if a == i]
# make sure i_points is not empty so don't divide by 0
if i_points:
self.means[i] = vector_mean(i_points)

random.seed(0) # so you get the same results as me
clusterer = KMeans(3)
clusterer.train(inputs)
print clusterer.means

random.seed(0)
clusterer = KMeans(2)
clusterer.train(inputs)
print clusterer.means


def squared_clustering_errors(inputs, k):
"""finds the total squared error from k-means clustering the inputs"""
clusterer = KMeans(k)
clusterer.train(inputs)
means = clusterer.means
assignments = map(clusterer.classify, inputs)
    return sum(squared_distance(input, means[cluster])
for input, cluster in zip(inputs, assignments))
# now plot from 1 up to len(inputs) clusters
ks = range(1, len(inputs) + 1)

errors = [squared_clustering_errors(inputs, k) for k in ks]
plt.plot(ks, errors)
plt.xticks(ks)
plt.xlabel("k")
plt.ylabel("total squared error")
plt.title("Total Error vs. # of Clusters")
plt.show()

path_to_png_file = r"C:\images\image.png" # wherever your image is
import matplotlib.image as mpimg
img = mpimg.imread(path_to_png_file)

top_row = img[0]
top_left_pixel = top_row[0]
red, green, blue = top_left_pixel

pixels = [pixel for row in img for pixel in row]


clusterer = KMeans(5)
clusterer.train(pixels) # this might take a while



def recolor(pixel):
cluster = clusterer.classify(pixel) # index of the closest cluster
    return clusterer.means[cluster] # mean of the closest cluster
new_img = [[recolor(pixel) for pixel in row] # recolor this row of pixels
for row in img] # for each row in the image

plt.imshow(new_img)
plt.axis('off')
plt.show()

leaf1 = ([10, 20],) # to make a 1-tuple you need the trailing comma
leaf2 = ([30, -15],) # otherwise Python treats the parentheses as parentheses


merged = (1, [leaf1, leaf2])


def is_leaf(cluster):
"""a cluster is a leaf if it has length 1"""
    return len(cluster) == 1
def get_children(cluster):
    """returns the two children of this cluster if it's a merged cluster;
raises an exception if this is a leaf cluster"""
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]


def get_values(cluster):
"""returns the value in this cluster (if it's a leaf cluster)
or all the values in the leaf clusters below it (if it's not)"""
    if is_leaf(cluster):
        return cluster # is already a 1-tuple containing value
    else:
        return [value
for child in get_children(cluster)
for value in get_values(child)]


def cluster_distance(cluster1, cluster2, distance_agg=min):
"""compute all the pairwise distances between cluster1 and cluster2
and apply _distance_agg_ to the resulting list"""
    return distance_agg([distance(input1, input2)
for input1 in get_values(cluster1)
for input2 in get_values(cluster2)])

def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0] # merge_order is first element of 2-tuple


def bottom_up_cluster(inputs, distance_agg=min):
# start with every input a leaf cluster / 1-tuple
clusters = [(input,) for input in inputs]
# as long as we have more than one cluster left...
while len(clusters) > 1:
# find the two closest clusters
c1, c2 = min([(cluster1, cluster2)

for i, cluster1 in enumerate(clusters)
for cluster2 in clusters[:i]],
key=lambda (x, y): cluster_distance(x, y, distance_agg))
# remove them from the list of clusters
clusters = [c for c in clusters if c != c1 and c != c2]
# merge them, using merge_order = # of clusters left
merged_cluster = (len(clusters), [c1, c2])
# and add their merge
clusters.append(merged_cluster)
# when there's only one cluster left, return it
    return clusters[0]


base_cluster = bottom_up_cluster(inputs)


def generate_clusters(base_cluster, num_clusters):
# start with a list with just the base cluster
    clusters = [base_cluster]
# as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
# choose the last-merged of our clusters
    next_cluster = min(clusters, key=get_merge_order)
# remove it from the list
    clusters = [c for c in clusters if c != next_cluster]
# and add its children to the list (i.e., unmerge it)
    clusters.extend(get_children(next_cluster))
# once we have enough clusters...
        return clusters


three_clusters = [get_values(cluster)
for cluster in generate_clusters(base_cluster, 3)]


for i, cluster, marker, color in zip([1, 2, 3],
three_clusters,
['D','o','*'],
['r','g','b']):
xs, ys = zip(*cluster) # magic unzipping trick
plt.scatter(xs, ys, color=color, marker=marker)
# put a number at the mean of the cluster
x, y = vector_mean(cluster)
plt.plot(x, y, marker='$' + str(i) + '$', color='black')
plt.title("User Locations -- 3 Bottom-Up Clusters, Min")
plt.xlabel("blocks east of city center")
plt.ylabel("blocks north of city center")
plt.show()


#CHAPTER 20
#Natural Language Processing



data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
("data science", 60, 70), ("analytics", 90, 3),
("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
("actionable insights", 40, 30), ("think out of the box", 45, 10),
("self-starter", 30, 50), ("customer focus", 65, 15),
("thought leadership", 35, 35)]


def text_size(total):
"""equals 8 if total is 0, 28 if total is 200"""
    return 8 + total / 200 * 20
for word, job_popularity, resume_popularity in data:
plt.text(job_popularity, resume_popularity, word,
ha='center', va='center',
size=text_size(job_popularity + resume_popularity))
plt.xlabel("Popularity on Job Postings")
plt.ylabel("Popularity on Resumes")
plt.axis([0, 100, 0, 100])
plt.xticks([])
plt.yticks([])
plt.show()


def fix_unicode(text):
    return text.replace(u"\u2019", "'")


from bs4 import BeautifulSoup
import requests
url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')
content = soup.find("div", "entry-content") # find entry-content div
regex = r"[\w']+|[\.]" # matches a word or a period
document = []
for paragraph in content("p"):
words = re.findall(regex, fix_unicode(paragraph.text))
document.extend(words)

bigrams = zip(document, document[1:])
transitions = defaultdict(list)
for prev, current in bigrams:
transitions[prev].append(current)


def generate_using_bigrams():
    current = "." # this means the next word will start a sentence
    result = []
    while True:
next_word_candidates = transitions[current] # bigrams (current, _)
current = random.choice(next_word_candidates) # choose one at random
result.append(current) # append it to results
if current == ".": return " ".join(result) # if "." we're done


trigrams = zip(document, document[1:], document[2:])
trigram_transitions = defaultdict(list)
starts = []


for prev, current, next in trigrams:
    if prev == ".": # if the previous "word" was a period
        starts.append(current) # then this is a start word
        trigram_transitions[(prev, current)].append(next)


def generate_using_trigrams():
    current = random.choice(starts) # choose a random starting word
    prev = "." # and precede it with a '.'
    result = [current]
while True:
next_word_candidates = trigram_transitions[(prev, current)]
next_word = random.choice(next_word_candidates)
prev, current = current, next_word
result.append(current)
if current == ".":
    return " ".join(result)


grammar = {
"_S" : ["_NP _VP"],
"_NP" : ["_N",
"_A _NP _P _A _N"],
"_VP" : ["_V",
"_V _NP"],
"_N" : ["data science", "Python", "regression"],
"_A" : ["big", "linear", "logistic"],
"_P" : ["about", "near"],
"_V" : ["learns", "trains", "tests", "is"]
}


def is_terminal(token):
    return token[0] != "_"


def expand(grammar, tokens):
for i, token in enumerate(tokens):
# skip over terminals
if is_terminal(token): continue
# if we get here, we found a non-terminal token
# so we need to choose a replacement at random
replacement = random.choice(grammar[token])
if is_terminal(replacement):
tokens[i] = replacement
else:
tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
# now call expand on the new list of tokens
    return expand(grammar, tokens)
# if we get here we had all terminals and are done
    return tokens

def generate_sentence(grammar):
    return expand(grammar, ["_S"])


random.random(
)

inverse_normal_cdf(random.random())


def roll_a_die():
    return random.choice([1,2,3,4,5,6])

def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2



def random_y_given_x(x):
"""equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()



def random_x_given_y(y):
    if y <= 7:
# if the total is 7 or less, the first die is equally likely to be
# 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
# if the total is 7 or more, the first die is equally likely to be
# (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)



def gibbs_sample(num_iters=100):
    x, y = 1, 2 # doesn't really matter
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y


def compare_distributions(num_samples=1000):
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
        return counts


documents[3][4]
documents[3][4]
document_topics[3][4]



def sample_from(weights):
"""returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random() # uniform between 0 and total
        for i, w in enumerate(weights):
            rnd -= w # return the smallest i such that
            if rnd <= 0: return i # weights[0] + ... + weights[i] >= rnd


documents = [
["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
["R", "Python", "statistics", "regression", "probability"],
["machine learning", "regression", "decision trees", "libsvm"],
["Python", "R", "Java", "C++", "Haskell", "programming languages"],
["statistics", "probability", "mathematics", "theory"],
["machine learning", "scikit-learn", "Mahout", "neural networks"],
["neural networks", "deep learning", "Big Data", "artificial intelligence"],
["Hadoop", "Java", "MapReduce", "Big Data"],
["statistics", "R", "statsmodels"],
["C++", "deep learning", "artificial intelligence", "probability"],
["pandas", "R", "Python"],
["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
["libsvm", "regression", "support vector machines"]
]


# a list of Counters, one for each document
document_topic_counts = [Counter() for _ in documents]


# a list of Counters, one for each topic
topic_word_counts = [Counter() for _ in range(K)]


# a list of numbers, one for each topic
topic_counts = [0 for _ in range(K)]


# a list of numbers, one for each document
document_lengths = map(len, documents)


distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)


D = len(documents)
document_topic_counts[3][1]
topic_word_counts[2]["nlp"]



def p_topic_given_document(topic, d, alpha=0.1):
"""the fraction of words in document _d_
that are assigned to _topic_ (plus some smoothing)"""
    return ((document_topic_counts[d][topic] + alpha) /
    (document_lengths[d] + K * alpha))
def p_word_given_topic(word, topic, beta=0.1):
"""the fraction of words assigned to _topic_
that equal _word_ (plus some smoothing)"""
    return ((topic_word_counts[topic][word] + beta) /
    (topic_counts[topic] + W * beta))




def topic_weight(d, word, k):
"""given a document and a word in that document,
return the weight for the kth topic"""

    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
    for k in range(K)])


random.seed(0)
document_topics = [[random.randrange(K) for word in document]
for document in documents]
for d in range(D):
for word, topic in zip(documents[d], document_topics[d]):
document_topic_counts[d][topic] += 1
topic_word_counts[topic][word] += 1
topic_counts[topic] += 1


for iter in range(1000):
for d in range(D):
for i, (word, topic) in enumerate(zip(documents[d],
document_topics[d])):
# remove this word / topic from the counts
# so that it doesn't influence the weights
document_topic_counts[d][topic] -= 1
topic_word_counts[topic][word] -= 1
topic_counts[topic] -= 1
document_lengths[d] -= 1
# choose a new topic based on the weights
new_topic = choose_new_topic(d, word)
document_topics[d][i] = new_topic
# and now add it back to the counts
document_topic_counts[d][new_topic] += 1
topic_word_counts[new_topic][word] += 1
topic_counts[new_topic] += 1
document_lengths[d] += 1


for k, word_counts in enumerate(topic_word_counts):
for word, count in word_counts.most_common():
if count > 0: print k, word, count


topic_names = ["Big Data and programming languages",
"Python and statistics",
"databases",
"machine learning"]


for document, topic_counts in zip(documents, document_topic_counts):
    print document
    for topic, count in topic_counts.most_common():
        if count > 0:
            print topic_names[topic], count,
            print
