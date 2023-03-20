def exp(base, power):
    return base ** power


def two_to_the(power):
    return exp(2, power)


from functools import partial
two_to_the = partial(exp, 2) # is now a function of one variable
print two_to_the(3) # 8    

square_of = partial(exp, power=2)
print square_of(3) # 9


def double(x):
    return 2 * x
xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs] # [2, 4, 6, 8]
twice_xs = map(double, xs) # same as above
list_doubler = partial(map, double) # *function* that doubles a list
twice_xs = list_doubler(xs) # again [2, 4, 6, 8]


def multiply(x, y): return x * y
products = map(multiply, [1, 2], [4, 5]) # [1 * 4, 2 * 5] = [4, 10]


def is_even(x):
"""True if x is even, False if x is odd"""
return x % 2 == 0
x_evens = [x for x in xs if is_even(x)] # [2, 4]
x_evens = filter(is_even, xs) # same as above
list_evener = partial(filter, is_even) # *function* that filters a list
x_evens = list_evener(xs) # again [2, 4]


x_product = reduce(multiply, xs) # = 1 * 2 * 3 * 4 = 24
list_product = partial(reduce, multiply) # *function* that reduces a list
x_product = list_product(xs) # again = 24


# not Pythonic
for i in range(len(documents)):
document = documents[i]
do_something(i, document)
# also not Pythonic
i = 0
for document in documents:
do_something(i, document)
i += 1

for i, document in enumerate(documents):
do_something(i, document)


for i in range(len(documents)): do_something(i) # not Pythonic
for i, _ in enumerate(documents): do_something(i) # Pythonic



list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
zip(list1, list2) # is [('a', 1), ('b', 2), ('c', 3)]



pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)


zip(('a', 1), ('b', 2), ('c', 3))

def add(a, b): return a + b
add(1, 2) # returns 3
add([1, 2]) # TypeError!
add(*[1, 2]) # returns 3


def doubler(f):
def g(x):
return 2 * f(x)
return g

def f1(x):
return x + 1
g = doubler(f1)
print g(3) # 8 (== ( 3 + 1) * 2)
print g(-1) # 0 (== (-1 + 1) * 2)



def f2(x, y):
return x + y
g = doubler(f2)
print g(1, 2) # TypeError: g() takes exactly 1 argument (2 given)


def magic(*args, **kwargs):
print "unnamed args:", args
print "keyword args:", kwargs
magic(1, 2, key="word", key2="word2")
# prints
# unnamed args: (1, 2)
# keyword args: {'key2': 'word2', 'key': 'word'}

def other_way_magic(x, y, z):
return x + y + z
x_y_list = [1, 2]
z_dict = { "z" : 3 }
print other_way_magic(*x_y_list, **z_dict) # 6


def doubler_correct(f):
"""works no matter what kind of inputs f expects"""
def g(*args, **kwargs):
"""whatever arguments g is supplied, pass them through to f"""
return 2 * f(*args, **kwargs)
return g
g = doubler_correct(f2)
print g(1, 2) # 6