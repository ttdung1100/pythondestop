

s = set()
s.add(1) # s is now { 1 }
s.add(2) # s is now { 1, 2 }
s.add(2) # s is still { 1, 2 }
x = len(s) # equals 2
y = 2 in s # equals True
z = 3 in s # equals False


stopwords_list = ["a","an","at"] + hundreds_of_other_words + ["yet", "you"]
"zip" in stopwords_list # False, but have to check every element
stopwords_set = set(stopwords_list)
"zip" in stopwords_set # very fast to check


item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list) # 6
item_set = set(item_list) # {1, 2, 3}
num_distinct_items = len(item_set) # 3
distinct_item_list = list(item_set) # [1, 2, 3]


if 1 > 2:
message = "if only 1 were greater than two..."
elif 1 > 3:
message = "elif stands for 'else if'"
else:
message = "when all else fails use else (if you want to)"

parity = "even" if x % 2 == 0 else "odd"


x = 0
while x < 10:
print x, "is less than 10"
x += 1

for x in range(10):
print x, "is less than 10"


for x in range(10):
if x == 3:
    continue # go immediately to the next iteration
if x == 5:
    break # quit the loop entirely
print x

one_is_less_than_two = 1 < 2 # equals True
true_equals_false = True == False # equals False

x = None
print x == None # prints True, but is not Pythonic
print x is None # prints True, and is Pythonic


s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""


first_char = s and s[0]


safe_x = x or 0

all([True, 1, { 3 }]) # True
all([True, 1, {}]) # False, {} is falsy
any([True, 1, {}]) # True, True is truthy
all([]) # True, no falsy elements in the list
any([]) # False, no truthy elements in the list



