#CHAPTER 21
#Network Analysis


users = [
{ "id": 0, "name": "Hero" },
{ "id": 1, "name": "Dunn" },
{ "id": 2, "name": "Sue" },
{ "id": 3, "name": "Chi" },
{ "id": 4, "name": "Thor" },
{ "id": 5, "name": "Clive" },
{ "id": 6, "name": "Hicks" },
{ "id": 7, "name": "Devin" },
{ "id": 8, "name": "Kate" },
{ "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]



for user in users:
    user["friends"] = []
for i, j in friendships:
# this works because users[i] is the user whose id is i
    users[i]["friends"].append(users[j]) # add i as a friend of j
    users[j]["friends"].append(users[i]) # add j as a friend of i


from collections import deque
def shortest_paths_from(from_user):
# a dictionary from "user_id" to *all* shortest paths to that user
    shortest_paths_to = { from_user["id"] : [[]] }
# a queue of (previous user, next user) that we need to check.
# starts out with all pairs (from_user, friend_of_from_user)


frontier = deque((from_user, friend)
for friend in from_user["friends"])
# keep going until we empty the queue
while frontier:
    prev_user, user = frontier.popleft() # remove the user who's
    user_id = user["id"] # first in the queue
# because of the way we're adding to the queue,
# necessarily we already know some shortest paths to prev_user
    paths_to_prev_user = shortest_paths_to[prev_user["id"]]
    new_paths_to_user = [path + [user_id] for path in paths_to_prev_user]
# it's possible we already know a shortest path
    old_paths_to_user = shortest_paths_to.get(user_id, [])
# what's the shortest path to here that we've seen so far?
if old_paths_to_user:
    min_path_length = len(old_paths_to_user[0])
else:
    min_path_length = float('inf')
# only keep paths that aren't too long and are actually new
    new_paths_to_user = [path
    for path in new_paths_to_user
        if len(path) <= min_path_length
            and path not in old_paths_to_user]
            shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user
# add never-seen neighbors to the frontier
            frontier.extend((user, friend)
    for friend in user["friends"]
        if friend["id"] not in shortest_paths_to)
            return shortest_paths_to

for user in users:
    user["shortest_paths"] = shortest_paths_from(user)

for user in users:
user["betweenness_centrality"] = 0.0
for source in users:
source_id = source["id"]
for target_id, paths in source["shortest_paths"].iteritems():

    if source_id < target_id: # don't double count
num_paths = len(paths) # how many shortest paths?
contrib = 1 / num_paths # contribution to centrality
for path in paths:
    for id in path:
        if id not in [source_id, target_id]:
            users[id]["betweenness_centrality"] += contrib


def farness(user):
"""the sum of the lengths of the shortest paths to each other user"""
    return sum(len(paths[0])
    for paths in user["shortest_paths"].values())

for user in users:
    user["closeness_centrality"] = 1 / farness(user)


def matrix_product_entry(A, B, i, j):
    return dot(get_row(A, i), get_column(B, j))



def matrix_multiply(A, B):
    n1, k1 = shape(A)
    n2, k2 = shape(B)
    if k1 != n2:
        raise ArithmeticError("incompatible shapes!")
        return make_matrix(n1, k2, partial(matrix_product_entry, A, B))


v = [1, 2, 3]
v_as_matrix = [[1],
[2],
[3]]


def vector_as_matrix(v):
"""returns the vector v (represented as a list) as a n x 1 matrix"""
    return [[v_i] for v_i in v]

def vector_from_matrix(v_as_matrix):
"""returns the n x 1 matrix as a list of values"""
    return [row[0] for row in v_as_matrix]


def matrix_operate(A, v):
v_as_matrix = vector_as_matrix(v)
product = matrix_multiply(A, v_as_matrix)
    return vector_from_matrix(product)


def find_eigenvector(A, tolerance=0.00001):
    guess = [random.random() for __ in A]
    while True:
        result = matrix_operate(A, guess)
        length = magnitude(result)
        next_guess = scalar_multiply(1/length, result)
        if distance(guess, next_guess) < tolerance:
            return next_guess, length # eigenvector, eigenvalue

guess = next_guess
rotate = [[ 0, 1],
[-1, 0]]

flip = [[0, 1],
[1, 0]]

def entry_fn(i, j):
    return 1 if (i, j) in friendships or (j, i) in friendships else 0
n = len(users)
adjacency_matrix = make_matrix(n, n, entry_fn)


matrix_operate(adjacency_matrix, eigenvector_centralities)

dot(get_row(adjacency_matrix, i), eigenvector_centralities)

endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2),
(2, 1), (1, 3), (2, 3), (3, 4), (5, 4),
(5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]
    for user in users:
        user["endorses"] = [] # add one list to track outgoing endorsements
        user["endorsed_by"] = [] # and another to track endorsements
        for source_id, target_id in endorsements:
            users[source_id]["endorses"].append(users[target_id])
            users[target_id]["endorsed_by"].append(users[source_id])
            endorsements_by_id = [(user["id"], len(user["endorsed_by"]))
            for user in users]
            sorted(endorsements_by_id,
            key=lambda (user_id, num_endorsements): num_endorsements,
            reverse=True)



def page_rank(users, damping = 0.85, num_iters = 100):
# initially distribute PageRank evenly
    num_users = len(users)
    pr = { user["id"] : 1 / num_users for user in users }
# this is the small fraction of PageRank
# that each node gets each iteration
    base_pr = (1 - damping) / num_users

for __ in range(num_iters):
    next_pr = { user["id"] : base_pr for user in users }
for user in users:
# distribute PageRank to outgoing links
    links_pr = pr[user["id"]] * damping
for endorsee in user["endorses"]:
    next_pr[endorsee["id"]] += links_pr / len(user["endorses"])

pr = next_pr
    return pr


#CHAPTER 22
#Recommender Systems

users_interests = [
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


popular_interests = Counter(interest
for user_interests in users_interests
for interest in user_interests).most_common()

[('Python', 4),
('R', 4),
('Java', 3),
('regression', 3),
('statistics', 3),
('probability', 3),
# ...
]


def most_popular_new_interests(user_interests, max_results=5):
    suggestions = [(interest, frequency)
    for interest, frequency in popular_interests
    if interest not in user_interests]
    return suggestions[:max_results]

["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"]

most_popular_new_interests(users_interests[1], 5)
# [('Python', 4), ('R', 4), ('Java', 3), ('regression', 3), ('statistics', 3)]


[('Java', 3),
('HBase', 3),
('Big Data', 3),
('neural networks', 2),
('Hadoop', 2)]


def cosine_similarity(v, w):
    return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))

unique_interests = sorted(list({ interest
for user_interests in users_interests
for interest in user_interests }))


['Big Data',
'C++',
'Cassandra',
'HBase',
'Hadoop',
'Haskell',
# ...
]


def make_user_interest_vector(user_interests):
"""given a list of interests, produce a vector whose ith element is 1
if unique_interests[i] is in the list, 0 otherwise"""
    return [1 if interest in user_interests else 0
    for interest in unique_interests]

user_interest_matrix = map(make_user_interest_vector, users_interests)



user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
for interest_vector_j in user_interest_matrix]
for interest_vector_i in user_interest_matrix]

def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity) # find other
for other_user_id, similarity in # users with
enumerate(user_similarities[user_id]) # nonzero
if user_id != other_user_id and similarity > 0] # similarity
    return sorted(pairs, # sort them
key=lambda (_, similarity): similarity, # most similar
reverse=True) # first

[(9, 0.5669467095138409),
(1, 0.3380617018914066),
(8, 0.1889822365046136),
(13, 0.1690308509457033),
(5, 0.1543033499620919)]

def user_based_suggestions(user_id, include_current_interests=False):
# sum up the similarities
    suggestions = defaultdict(float)
for other_user_id, similarity in most_similar_users_to(user_id):
for interest in users_interests[other_user_id]:
    suggestions[interest] += similarity
# convert them to a sorted list
    suggestions = sorted(suggestions.items(),
    key=lambda (_, weight): weight,
    reverse=True)
# and (maybe) exclude already-interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
        for suggestion, weight in suggestions
        if suggestion not in users_interests[user_id]]



[('MapReduce', 0.5669467095138409),
('MongoDB', 0.50709255283711),
('Postgres', 0.50709255283711),
('NoSQL', 0.3380617018914066),
('neural networks', 0.1889822365046136),
('deep learning', 0.1889822365046136),
('artificial intelligence', 0.1889822365046136),
#...
]


#Item-Based Collaborative Filtering



interest_user_matrix = [[user_interest_vector[j]
for user_interest_vector in user_interest_matrix]
for j, _ in enumerate(unique_interests)]

[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
for user_vector_j in interest_user_matrix]
for user_vector_i in interest_user_matrix]


def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
    for other_interest_id, similarity in enumerate(similarities)
    if interest_id != other_interest_id and similarity > 0]
        return sorted(pairs,
        key=lambda (_, similarity): similarity,
        reverse=True)


[('Hadoop', 0.8164965809277261),
('Java', 0.6666666666666666),
('MapReduce', 0.5773502691896258),
('Spark', 0.5773502691896258),
('Storm', 0.5773502691896258),
('Cassandra', 0.4082482904638631),
('artificial intelligence', 0.4082482904638631),
('deep learning', 0.4082482904638631),
('neural networks', 0.4082482904638631),
('HBase', 0.3333333333333333)]

def item_based_suggestions(user_id, include_current_interests=False):
# add up the similar interests
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
    for interest, similarity in similar_interests:
        suggestions[interest] += similarity
# sort them by weight
        suggestions = sorted(suggestions.items(),
        key=lambda (_, similarity): similarity,
        reverse=True)
        if include_current_interests:
            return suggestions
        else:
            return [(suggestion, weight)
            for suggestion, weight in suggestions
            if suggestion not in users_interests[user_id]]


[('MapReduce', 1.861807319565799),
('Postgres', 1.3164965809277263),
('MongoDB', 1.3164965809277263),
('NoSQL', 1.2844570503761732),
('programming languages', 0.5773502691896258),
('MySQL', 0.5773502691896258),
('Haskell', 0.5773502691896258),
('databases', 0.5773502691896258),
('neural networks', 0.4082482904638631),
('deep learning', 0.4082482904638631),
('C++', 0.4082482904638631),
('artificial intelligence', 0.4082482904638631),
('Python', 0.2886751345948129),
('R', 0.2886751345948129)]



#CHAPTER 23
#Databases and SQL



users = [[0, "Hero", 0],
[1, "Dunn", 2],
[2, "Sue", 3],
[3, "Chi", 3]]


CREATE TABLE users (
user_id INT NOT NULL,
name VARCHAR(200),
num_friends INT);

INSERT INTO users (user_id, name, num_friends) VALUES (0, 'Hero', 0);



class Table:
def __init__(self, columns):
self.columns = columns
self.rows = []
def __repr__(self):
"""pretty representation of the table: columns then rows"""
    return str(self.columns) + "\n" + "\n".join(map(str, self.rows))
def insert(self, row_values):
    if len(row_values) != len(self.columns):
    raise TypeError("wrong number of elements")
    row_dict = dict(zip(self.columns, row_values))

self.rows.append(row_dict)


users = Table(["user_id", "name", "num_friends"])
users.insert([0, "Hero", 0])
users.insert([1, "Dunn", 2])
users.insert([2, "Sue", 3])
users.insert([3, "Chi", 3])
users.insert([4, "Thor", 3])
users.insert([5, "Clive", 2])
users.insert([6, "Hicks", 3])
users.insert([7, "Devin", 2])
users.insert([8, "Kate", 2])
users.insert([9, "Klein", 3])
users.insert([10, "Jen", 1])


UPDATE users
SET num_friends = 3
WHERE user_id = 1;


def update(self, updates, predicate):
    for row in self.rows:
        if predicate(row):
            for column, new_value in updates.iteritems():
                row[column] = new_value
                users.update({'num_friends' : 3}, # set num_friends = 3
                lambda row: row['user_id'] == 1) # in rows where user_id == 1


DELETE FROM users;
DELETE FROM users WHERE user_id = 1;
def delete(self, predicate=lambda row: True):
"""delete all rows matching predicate
or all rows if no predicate supplied"""
self.rows = [row for row in self.rows if not(predicate(row))]
users.delete(lambda row: row["user_id"] == 1) # deletes rows with user_id == 1
users.delete() # deletes every row

SELECT * FROM users; -- get the entire contents
SELECT * FROM users LIMIT 2; -- get the first two rows
SELECT user_id FROM users; -- only get specific columns
SELECT user_id FROM users WHERE name = 'Dunn'; -- only get specific rows


SELECT LENGTH(name) AS name_length FROM users;


def select(self, keep_columns=None, additional_columns=None):
    if keep_columns is None: # if no columns specified,
        keep_columns = self.columns # return all columns
        if additional_columns is None:
            additional_columns = {}
# new table for results
            result_table = Table(keep_columns + additional_columns.keys())
for row in self.rows:
    new_row = [row[column] for column in keep_columns]
for column_name, calculation in additional_columns.iteritems():
    new_row.append(calculation(row))
    result_table.insert(new_row)
    return result_table

def where(self, predicate=lambda row: True):
"""return only the rows that satisfy the supplied predicate"""
    where_table = Table(self.columns)
    where_table.rows = filter(predicate, self.rows)
    return where_table

def limit(self, num_rows):
"""return only the first num_rows rows"""
    limit_table = Table(self.columns)
    limit_table.rows = self.rows[:num_rows]
    return limit_table


# SELECT * FROM users;
users.select()
# SELECT * FROM users LIMIT 2;
users.limit(2)
# SELECT user_id FROM users;
users.select(keep_columns=["user_id"])

# SELECT user_id FROM users WHERE name = 'Dunn';
users.where(lambda row: row["name"] == "Dunn") \
.select(keep_columns=["user_id"])
# SELECT LENGTH(name) AS name_length FROM users;
def name_length(row): return len(row["name"])
    users.select(keep_columns=[],
    additional_columns = { "name_length" : name_length })


SELECT LENGTH(name) as name_length,
MIN(user_id) AS min_user_id,
COUNT(*) AS num_users
FROM users
GROUP BY LENGTH(name);


SELECT SUBSTR(name, 1, 1) AS first_letter,
AVG(num_friends) AS avg_num_friends
FROM users
GROUP BY SUBSTR(name, 1, 1)
HAVING AVG(num_friends) > 1;

SELECT SUM(user_id) as user_id_sum
FROM users
WHERE user_id > 1;

def group_by(self, group_by_columns, aggregates, having=None):
    grouped_rows = defaultdict(list)
# populate groups
for row in self.rows:
    key = tuple(row[column] for column in group_by_columns)
    grouped_rows[key].append(row)
# result table consists of group_by columns and aggregates
    result_table = Table(group_by_columns + aggregates.keys())
    for key, rows in grouped_rows.iteritems():
        if having is None or having(rows):
            new_row = list(key)
            for aggregate_name, aggregate_fn in aggregates.iteritems():
                new_row.append(aggregate_fn(rows))
                result_table.insert(new_row)
                return result_table


def min_user_id(rows): return min(row["user_id"] for row in rows)
stats_by_length = users \
    .select(additional_columns={"name_length" : name_length}) \
        .group_by(group_by_columns=["name_length"],
        aggregates={ "min_user_id" : min_user_id,
        "num_users" : len })



def first_letter_of_name(row):
    return row["name"][0] if row["name"] else ""

def average_num_friends(rows):
    return sum(row["num_friends"] for row in rows) / len(rows)
def enough_friends(rows):
    return average_num_friends(rows) > 1

avg_friends_by_letter = users \
.select(additional_columns={'first_letter' : first_letter_of_name}) \
.group_by(group_by_columns=['first_letter'],
aggregates={ "avg_num_friends" : average_num_friends },
having=enough_friends)


def sum_user_ids(rows): return sum(row["user_id"] for row in rows)
user_id_sum = users \
.where(lambda row: row["user_id"] > 1) \
.group_by(group_by_columns=[],
aggregates={ "user_id_sum" : sum_user_ids })


SELECT * FROM users
ORDER BY name
LIMIT 2;


def order_by(self, order):
    new_table = self.select() # make a copy
    new_table.rows.sort(key=order)
    return new_table


friendliest_letters = avg_friends_by_letter \
.order_by(lambda row: -row["avg_num_friends"]) \
.limit(4)


CREATE TABLE user_interests (
user_id INT NOT NULL,
interest VARCHAR(100) NOT NULL
);


user_interests = Table(["user_id", "interest"])
user_interests.insert([0, "SQL"])
user_interests.insert([0, "NoSQL"])
user_interests.insert([2, "SQL"])
user_interests.insert([2, "MySQL"])


SELECT users.name
FROM users
JOIN user_interests
ON users.user_id = user_interests.user_id
WHERE user_interests.interest = 'SQL'


SELECT users.id, COUNT(user_interests.interest) AS num_interests
FROM users
LEFT JOIN user_interests
ON users.user_id = user_interests.user_id




def join(self, other_table, left_join=False):
join_on_columns = [c for c in self.columns # columns in
if c in other_table.columns] # both tables
additional_columns = [c for c in other_table.columns # columns only
if c not in join_on_columns] # in right table
# all columns from left table + additional_columns from right table
join_table = Table(self.columns + additional_columns)
for row in self.rows:

def is_join(other_row):
    return all(other_row[c] == row[c] for c in join_on_columns)

other_rows = other_table.where(is_join).rows
# each other row that matches this one produces a result row

for other_row in other_rows:
    join_table.insert([row[c] for c in self.columns] +
    [other_row[c] for c in additional_columns])
# if no rows match and it's a left join, output with Nones
if left_join and not other_rows:
    join_table.insert([row[c] for c in self.columns] +
    [None for c in additional_columns])
    return join_table

sql_users = users \
.join(user_interests) \
.where(lambda row: row["interest"] == "SQL") \
.select(keep_columns=["name"])



def count_interests(rows):
"""counts how many rows have non-None interests"""
    return len([row for row in rows if row["interest"] is not None])

user_interest_counts = users \
    .join(user_interests, left_join=True) \
        .group_by(group_by_columns=["user_id"],
        aggregates={"num_interests" : count_interests })



SELECT MIN(user_id) AS min_user_id FROM
(SELECT user_id FROM user_interests WHERE interest = 'SQL') sql_interests;



likes_sql_user_ids = user_interests \
.where(lambda row: row["interest"] == "SQL") \
.select(keep_columns=['user_id'])
likes_sql_user_ids.group_by(group_by_columns=[],
aggregates={ "min_user_id" : min_user_id })


SELECT users.name
FROM users
JOIN user_interests
ON users.user_id = user_interests.user_id
WHERE user_interests.interest = 'SQL'


user_interests \
.where(lambda row: row["interest"] == "SQL") \
.join(users) \
.select(["name"])


user_interests \
.join(users) \
.where(lambda row: row["interest"] == "SQL") \
.select(["name"])


#CHAPTER 24
#MapReduce



def word_count_old(documents):
"""word count not using MapReduce"""
    return Counter(word
    for document in documents
    for word in tokenize(document))



def wc_mapper(document):
"""for each word in the document, emit (word,1)"""
for word in tokenize(document):
    yield (word, 1)


def wc_reducer(word, counts):
"""sum up the counts for a word"""
yield (word, sum(counts))


def word_count(documents):
"""count the words in the input documents using MapReduce"""
# place to store grouped values
    collector = defaultdict(list)
    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)
            return [output
            for word, counts in collector.iteritems()
            for output in wc_reducer(word, counts)]


{ "data" : [1, 1],
"science" : [1, 1],
"big" : [1],
"fiction" : [1] }


[("data", 2), ("science", 2), ("big", 1), ("fiction", 1)]


""" Why MapReduce?
As mentioned earlier, the primary benefit of MapReduce is that it allows us to distrib‐
ute computations by moving the processing to the data. Imagine we want to wordcount across billions of documents.
Our original (non-MapReduce) approach requires the machine doing the processing
to have access to every document. This means that the documents all need to either
live on that machine or else be transferred to it during processing. More important, it
means that the machine can only process one document at a time. """

""" Imagine now that our billions of documents are scattered across 100 machines. With
the right infrastructure (and glossing over some of the details), we can do the follow‐
ing:
• Have each machine run the mapper on its documents, producing lots of (key,
value) pairs.
• Distribute those (key, value) pairs to a number of “reducing” machines, making
sure that the pairs corresponding to any given key all end up on the same
machine.
• Have each reducing machine group the pairs by key and then run the reducer on
each set of values.
• Return each (key, output) pair. """




def map_reduce(inputs, mapper, reducer):
"""runs MapReduce on the inputs using mapper and reducer"""
collector = defaultdict(list)
for input in inputs:
    for key, value in mapper(input):
        collector[key].append(value)
        return [output
        for key, values in collector.iteritems()
        for output in reducer(key,values)]

word_counts = map_reduce(documents, wc_mapper, wc_reducer)

def reduce_values_using(aggregation_fn, key, values):
"""reduces a key-values pair by applying aggregation_fn to the values"""
    yield (key, aggregation_fn(values))
def values_reducer(aggregation_fn):
"""turns a function (values -> output) into a reducer
that maps (key, values) -> (key, output)"""
    return partial(reduce_values_using, aggregation_fn)


sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values: len(set(values)))



{"id": 1,
"username" : "joelgrus",
"text" : "Is anyone interested in a data science book?",
"created_at" : datetime.datetime(2013, 12, 21, 11, 47, 0),
"liked_by" : ["data_guy", "data_gal", "mike"] }


def data_science_day_mapper(status_update):
"""yields (day_of_week, 1) if status_update contains "data science" """
    if "data science" in status_update["text"].lower():
    day_of_week = status_update["created_at"].weekday()
    yield (day_of_week, 1)
    data_science_days = map_reduce(status_updates,
    data_science_day_mapper,
    sum_reducer)



def words_per_user_mapper(status_update):
    user = status_update["username"]
    for word in tokenize(status_update["text"]):
    yield (user, (word, 1))



def most_popular_word_reducer(user, words_and_counts):
"""given a sequence of (word, count) pairs,
return the word with the highest total count"""
word_counts = Counter()
for word, count in words_and_counts:
word_counts[word] += count
word, count = word_counts.most_common(1)[0]
yield (user, (word, count))
user_words = map_reduce(status_updates,
words_per_user_mapper,
most_popular_word_reducer)

def liker_mapper(status_update):
user = status_update["username"]
for liker in status_update["liked_by"]:
yield (user, liker)
distinct_likers_per_user = map_reduce(status_updates,
liker_mapper,
count_distinct_reducer)



def matrix_multiply_mapper(m, element):
"""m is the common dimension (columns of A, rows of B)
element is a tuple (matrix_name, i, j, value)"""
name, i, j, value = element
if name == "A":
# A_ij is the jth entry in the sum for each C_ik, k=1..m
for k in range(m):
# group with other entries for C_ik
yield((i, k), (j, value))
else:
# B_ij is the i-th entry in the sum for each C_kj
for k in range(m):
# group with other entries for C_kj
yield((k, j), (i, value))
def matrix_multiply_reducer(m, key, indexed_values):
results_by_index = defaultdict(list)
for index, value in indexed_values:
results_by_index[index].append(value)
# sum up all the products of the positions with two results
sum_product = sum(results[0] * results[1]
for results in results_by_index.values()
if len(results) == 2)
if sum_product != 0.0:
yield (key, sum_product)



entries = [("A", 0, 0, 3), ("A", 0, 1, 2),
("B", 0, 0, 4), ("B", 0, 1, -1), ("B", 1, 0, 10)]
mapper = partial(matrix_multiply_mapper, 3)


reducer = partial(matrix_multiply_reducer, 3)
map_reduce(entries, mapper, reducer) # [((0, 1), -3), ((0, 0), 32)]