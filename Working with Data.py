def bucketize(point, bucket_size):
    """floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)
def make_histogram(points, bucket_size):
    """buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)
def plot_histogram(points, bucket_size, title=""):
histogram = make_histogram(points, bucket_size)
plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
plt.title(title)
plt.show()


random.seed(0)
# uniform between -100 and 100
uniform = [200 * random.random() - 100 for _ in range(10000)]
# normal distribution with mean 0, standard deviation 57
normal = [57 * inverse_normal_cdf(random.random())
for _ in range(10000)]


plot_histogram(uniform, 10, "Uniform Histogram")


plot_histogram(uniform, 10, "Uniform Histogram")


def random_normal():
"""returns a random draw from a standard normal distribution"""    
    return inverse_normal_cdf(random.random())
xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]


plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different Joint Distributions")
plt.show()


print correlation(xs, ys1) # 0.9
print correlation(xs, ys2) # -0.9


def correlation_matrix(data):
"""returns the num_columns x num_columns matrix whose (i, j)th entry
is the correlation between columns i and j of data"""
_, num_columns = shape(data)
def matrix_entry(i, j):
    return correlation(get_column(data, i), get_column(data, j))
    return make_matrix(num_columns, num_columns, matrix_entry)


import matplotlib.pyplot as plt
_, num_columns = shape(data)
fig, ax = plt.subplots(num_columns, num_columns)
for i in range(num_columns):
    for j in range(num_columns):
# scatter column_j on the x-axis vs column_i on the y-axis
if i != j: ax[i][j].scatter(get_column(data, j), get_column(data, i))
# unless i == j, in which case show the series name
else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
xycoords='axes fraction',
ha="center", va="center")
# then hide axis labels except left and bottom charts
if i < num_columns - 1: ax[i][j].xaxis.set_visible(False)
if j > 0: ax[i][j].yaxis.set_visible(False)
# fix the bottom right and top left axis labels, which are wrong because
# their charts only have text in them
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()

closing_price = float(row[2])


def parse_row(input_row, parsers):
"""given a list of parsers (some of which may be None)
apply the appropriate one to each element of the input_row"""
     return [parser(value) if parser is not None else value
    for value, parser in zip(input_row, parsers)]
    
def parse_rows_with(reader, parsers):
"""wrap a reader to apply the parsers to each of its rows"""
for row in reader:
yield parse_row(row, parsers)



def try_or_none(f):
"""wraps f to return None if f raises an exception
assumes f takes only one input"""
    def f_or_none(x):
        try: return f(x)
        except: return None
    return f_or_none
        
def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value
    for value, parser in zip(input_row, parsers)]

import dateutil.parser
data = []
with open("comma_delimited_stock_prices.csv", "rb") as f:
    reader = csv.reader(f)
    for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
        data.append(line)


for row in data:
    if any(x is None for x in row):
    print row
    
    def try_parse_field(field_name, value, parser_dict):
"""try to parse value using the appropriate function from parser_dict"""
        parser = parser_dict.get(field_name) # None if no such entry
        if parser is not None:
            return try_or_none(parser)(value)
else:
    return value

def parse_dict(input_dict, parser_dict):
        return { field_name : try_parse_field(field_name, value, parser_dict)
for field_name, value in input_dict.iteritems() }



data = [
{'closing_price': 102.06,
'date': datetime.datetime(2014, 8, 29, 0, 0),
'symbol': 'AAPL'},
# ...
]


max_aapl_price = max(row["closing_price"]
for row in data
if row["symbol"] == "AAPL")


# group rows by symbol
by_symbol = defaultdict(list)
for row in data:
by_symbol[row["symbol"]].append(row)
# use a dict comprehension to find the max for each symbol
max_price_by_symbol = { symbol : max(row["closing_price"]
for row in grouped_rows)
for symbol, grouped_rows in by_symbol.iteritems() }



def picker(field_name):
"""returns a function that picks a field out of a dict"""    
    return lambda row: row[field_name]
def pluck(field_name, rows):
"""turn a list of dicts into the list of field_name values"""    
    return map(picker(field_name), rows)


def group_by(grouper, rows, value_transform=None):
# key is output of grouper, value is list of rows    
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)
        if value_transform is None:
            return grouped
        else:
            return { key : value_transform(rows)
            for key, rows in grouped.iteritems() }



max_price_by_symbol = group_by(picker("symbol"),
data,
lambda rows: max(pluck("closing_price", rows)))


def percent_price_change(yesterday, today):
    return today["closing_price"] / yesterday["closing_price"] - 1

def day_over_day_changes(grouped_rows):
# sort the rows by date    
    ordered = sorted(grouped_rows, key=picker("date"))
# zip with an offset to get pairs of consecutive days
    return [{ "symbol" : today["symbol"],
    "date" : today["date"],
    "change" : percent_price_change(yesterday, today) }
    for yesterday, today in zip(ordered, ordered[1:])]


# key is symbol, value is list of "change" dicts
changes_by_symbol = group_by(picker("symbol"), data, day_over_day_changes)
# collect all "change" dicts into one big list
all_changes = [change
for changes in changes_by_symbol.values()
for change in changes]


max(all_changes, key=picker("change"))
# {'change': 0.3283582089552237,
# 'date': datetime.datetime(1997, 8, 6, 0, 0),
# 'symbol': 'AAPL'}
# see, e.g. http://news.cnet.com/2100-1001-202143.html

min(all_changes, key=picker("change"))
# {'change': -0.5193370165745856,
# 'date': datetime.datetime(2000, 9, 29, 0, 0),
# 'symbol': 'AAPL'}
# see, e.g. http://money.cnn.com/2000/09/29/markets/techwrap/




# to combine percent changes, we add 1 to each, multiply them, and subtract 1
# for instance, if we combine +10% and -20%, the overall change is
# (1 + 10%) * (1 - 20%) - 1 = 1.1 * .8 - 1 = -12%
def combine_pct_changes(pct_change1, pct_change2):
    return (1 + pct_change1) * (1 + pct_change2) - 1
def overall_change(changes):
    return reduce(combine_pct_changes, pluck("change", changes))

overall_change_by_month = group_by(lambda row: row['date'].month,
all_changes,
overall_change)


a_to_b = distance([63, 150], [67, 160]) # 10.77
a_to_c = distance([63, 150], [70, 171]) # 22.14
b_to_c = distance([67, 160], [70, 171]) # 11.40


a_to_b = distance([160, 150], [170.2, 160]) # 14.28
a_to_c = distance([160, 150], [177.8, 171]) # 27.53
b_to_c = distance([170.2, 160], [177.8, 171]) # 13.37


def scale(data_matrix):
"""returns the means and standard deviations of each column"""    
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix,j))
    for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix,j))
    for j in range(num_cols)]
    return means, stdevs


def rescale(data_matrix):
"""rescales the input data so that each column
has mean 0 and standard deviation 1
leaves alone columns with no deviation"""    
    means, stdevs = scale(data_matrix)
def rescaled(i, j):
    if stdevs[j] > 0:
        return (data_matrix[i][j] - means[j]) / stdevs[j]
    else:
            return data_matrix[i][j]
            num_rows, num_cols = shape(data_matrix)
            return make_matrix(num_rows, num_cols, rescaled)

def de_mean_matrix(A):
"""returns the result of subtracting from every value in A the mean
value of its column. the resulting matrix has mean 0 in every column"""
    nr, nc = shape(A)
    column_means, _ = scale(A)
    return make_matrix(nr, nc, lambda i, j: A[i][j] - column_means[j])

def direction(w):
    mag = magnitude(w)
    return [w_i / mag for w_i in w]

def directional_variance_i(x_i, w):
"""the variance of the row x_i in the direction determined by w"""    
    return dot(x_i, direction(w)) ** 2
def directional_variance(X, w):
"""the variance of the data in the direction determined w"""    
    return sum(directional_variance_i(x_i, w)
    for x_i in X)



def directional_variance_gradient_i(x_i, w):
"""the contribution of row x_i to the gradient of
the direction-w variance"""    
    projection_length = dot(x_i, direction(w))
    return [2 * projection_length * x_ij for x_ij in x_i]
def directional_variance_gradient(X, w):
    return vector_sum(directional_variance_gradient_i(x_i,w)
    for x_i in X)


def first_principal_component(X):
    guess = [1 for _ in X[0]]
    unscaled_maximizer = maximize_batch(
        partial(directional_variance, X), # is now a function of w
        partial(directional_variance_gradient, X), # is now a function of w
        guess)
        return direction(unscaled_maximizer)


# here there is no "y", so we just pass in a vector of Nones
# and functions that ignore that input
def first_principal_component_sgd(X):
    guess = [1 for _ in X[0]]
    unscaled_maximizer = maximize_stochastic(
        lambda x, _, w: directional_variance_i(x, w),
        lambda x, _, w: directional_variance_gradient_i(x, w),
        X,
        [None for _ in X], # the fake "y"
        guess)
        return direction(unscaled_maximizer)


def project(v, w):
"""return the projection of v onto the direction w"""    
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)

def remove_projection_from_vector(v, w):
"""projects v onto w and subtracts the result from v"""    
    return vector_subtract(v, project(v, w))
def remove_projection(X, w):
"""for each row of X
projects the row onto w, and subtracts the result from the row"""    
    return [remove_projection_from_vector(x_i, w) for x_i in X]


def principal_component_analysis(X, num_components):
    components = []
    for _ in range(num_components):
        component = first_principal_component(X)
        components.append(component)
        X = remove_projection(X, component)
        return components

def transform_vector(v, components):
    return [dot(v, w) for w in components]
    
def transform(X, components):
    return [transform_vector(x_i, components) for x_i in X]