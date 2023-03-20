def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner



def majority_vote(labels):
"""assumes that labels are ordered from nearest to farthest"""
vote_counts = Counter(labels)
winner, winner_count = vote_counts.most_common(1)[0]
num_winners = len([count
for count in vote_counts.values()
if count == winner_count])
if num_winners == 1:
return winner # unique winner, so return it
else:
return majority_vote(labels[:-1]) # try again without the farthest


def knn_classify(k, labeled_points, new_point):
"""each labeled point should be a pair (point, label)"""
# order the labeled points from nearest to farthest
by_distance = sorted(labeled_points,
key=lambda (point, _): distance(point, new_point))
# find the labels for the k closest
k_nearest_labels = [label for _, label in by_distance[:k]]
# and let them vote
return majority_vote(k_nearest_labels)




# each entry is ([longitude, latitude], favorite_language)
cities = [([-122.3 , 47.53], "Python"), # Seattle
([ -96.85, 32.85], "Java"), # Austin
([ -89.33, 43.13], "R"), # Madison
# ... and so on
]


# key is language, value is pair (longitudes, latitudes)
plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }
# we want each language to have a different marker and color
markers = { "Java" : "o", "Python" : "s", "R" : "^" }
colors = { "Java" : "r", "Python" : "b", "R" : "g" }
for (longitude, latitude), language in cities:
plots[language][0].append(longitude)
plots[language][1].append(latitude)
# create a scatter series for each language
for language, (x, y) in plots.iteritems():
plt.scatter(x, y, color=colors[language], marker=markers[language],
label=language, zorder=10)
plot_state_borders(plt) # pretend we have a function that does this
plt.legend(loc=0) # let matplotlib choose the location
plt.axis([-130,-60,20,55]) # set the axes
plt.title("Favorite Programming Languages")
plt.show()



# try several different values for k
for k in [1, 3, 5, 7]:
num_correct = 0
for city in cities:
location, actual_language = city
other_cities = [other_city
for other_city in cities
if other_city != city]
predicted_language = knn_classify(k, other_cities, location)
if predicted_language == actual_language:
num_correct += 1
print k, "neighbor[s]:", num_correct, "correct out of", len(cities)



plots = { "Java" : ([], []), "Python" : ([], []), "R" : ([], []) }
k = 1 # or 3, or 5, or ...
for longitude in range(-130, -60):
for latitude in range(20, 55):
predicted_language = knn_classify(k, cities, [longitude, latitude])
plots[predicted_language][0].append(longitude)
plots[predicted_language][1].append(latitude)



def random_point(dim):
return [random.random() for _ in range(dim)]


def random_distances(dim, num_pairs):
return [distance(random_point(dim), random_point(dim))
for _ in range(num_pairs)]



dimensions = range(1, 101)
avg_distances = []
min_distances = []
random.seed(0)
for dim in dimensions:
distances = random_distances(dim, 10000) # 10,000 random pairs
avg_distances.append(mean(distances)) # track the average
min_distances.append(min(distances)) # track the minimum


min_avg_ratio = [min_dist / avg_dist
for min_dist, avg_dist in zip(min_distances, avg_distances)]





