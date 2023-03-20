

empty_dict = {} # Pythonic
empty_dict2 = dict() # less Pythonic
grades = { "Joel" : 80, "Tim" : 95 } # dictionary literal


joels_grade = grades["Joel"] # equals 80


try:
kates_grade = grades["Kate"]
except KeyError:
print "no grade for Kate!"

joel_has_grade = "Joel" in grades # True
kate_has_grade = "Kate" in grades # False

joels_grade = grades.get("Joel", 0) # equals 80
kates_grade = grades.get("Kate", 0) # equals 0
no_ones_grade = grades.get("No One") # default default is None

grades["Tim"] = 99 # replaces the old value
grades["Kate"] = 100 # adds a third entry
num_students = len(grades) # equals 


tweet = {
"user" : "joelgrus",
"text" : "Data Science is Awesome",
"retweet_count" : 100,
"hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}

tweet_keys = tweet.keys() # list of keys
tweet_values = tweet.values() # list of values
tweet_items = tweet.items() # list of (key, value) tuples
"user" in tweet_keys # True, but uses a slow list in
"user" in tweet # more Pythonic, uses faster dict in
"joelgrus" in tweet_values # True


word_counts = {}
for word in document:
if word in word_counts:
word_counts[word] += 1
else:
word_counts[word] = 1



word_counts = {}
for word in document:
try:
word_counts[word] += 1
except KeyError:
word_counts[word] = 1


word_counts = {}
for word in document:
previous_count = word_counts.get(word, 0)
word_counts[word] = previous_count + 1


from collections import defaultdict
word_counts = defaultdict(int) # int() produces 0
for word in document:
word_counts[word] += 1


dd_list = defaultdict(list) # list() produces an empty list
dd_list[2].append(1) # now dd_list contains {2: [1]}
dd_dict = defaultdict(dict) # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle" # { "Joel" : { "City" : Seattle"}}
dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1 # now dd_pair contains {2: [0,1]}



from collections import Counter
c = Counter([0, 1, 2, 0]) # c is (basically) { 0 : 2, 1 : 1, 2 : 1 }




word_counts = Counter(document)


# print the 10 most common words and their counts
for word, count in word_counts.most_common(10):
print word, count








