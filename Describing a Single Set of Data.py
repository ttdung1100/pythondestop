num_friends = [100, 49, 41, 40, 25,
# ... and lots more
]

friend_counts = Counter(num_friends)
xs = range(101) # largest value is 100
ys = [friend_counts[x] for x in xs] # height is just # of friends
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()

num_points = len(num_friends) # 204


largest_value = max(num_friends) # 100
smallest_value = min(num_friends) # 1

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0] # 1
second_smallest_value = sorted_values[1] # 1
second_largest_value = sorted_values[-2] # 49


#Central Tendencies

# this isn't right if you don't from __future__ import division
def mean(x):
    return sum(x)/len(x)

mean(num_friends) # 7.333333

def median(v):
"""finds the 'middle-most' value of v"""
n = len(v)
sorted_v = sorted(v)
midpoint = n // 2
if n % 2 == 1:
# if odd, return the middle value
    return sorted_v[midpoint]
else:
# if even, return the average of the middle values
        lo = midpoint - 1
        hi = midpoint
            return (sorted_v[lo] + sorted_v[hi]) / 2

median(num_friends) # 6.0


def quantile(x, p):
    p_index = int(p * len(x))
    return sorted(x)[p_index]
"""returns the pth-percentile value in x"""

quantile(num_friends, 0.10) # 1
quantile(num_friends, 0.25) # 3
quantile(num_friends, 0.75) # 9
quantile(num_friends, 0.90) # 13


def mode(x):
"""returns a list, might be more than one mode"""
counts = Counter(x)
max_count = max(counts.values())
    return [x_i for x_i, count in counts.iteritems()
    if count == max_count]

mode(num_friends) # 1 and 6


#Dispersion
# "range" already means something in Python, so we'll use a different name
def data_range(x):
    return max(x) - min(x)
data_range(num_friends) # 99

def de_mean(x):
"""translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]
def variance(x):
"""assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)
variance(num_friends) # 81.54

def standard_deviation(x):
    return math.sqrt(variance(x))
standard_deviation(num_friends) # 9.03


def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)
interquartile_range(num_friends) # 6


#Correlation
def covariance(x, y):
n = len(x)
        return dot(de_mean(x), de_mean(y)) / (n - 1)

covariance(num_friends, daily_minutes) # 22.43


def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero

correlation(num_friends, daily_minutes) # 0.25




outlier = num_friends.index(100) # index of outlier
num_friends_good = [x
for i, x in enumerate(num_friends)
if i != outlier]
daily_minutes_good = [x
for i, x in enumerate(daily_minutes)
if i != outlier]
correlation(num_friends_good, daily_minutes_good) # 0.57




#Simpson’s Paradox

x = [-2, -1, 0, 1, 2]
y = [ 2, 1, 0, 1, 2]

x = [-2, 1, 0, 1, 2]
y = [99.98, 99.99, 100, 100.01, 100.02]

