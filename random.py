import random


four_uniform_randoms = [random.random() for _ in range(4)]
# [0.8444218515250481, # random.random() produces numbers
# 0.7579544029403025, # uniformly between 0 and 1
# 0.420571580830845, # it's the random function we'll use
# 0.25891675029296335] # most often


random.seed(10) # set the seed to 10
print random.random() # 0.57140259469
random.seed(10) # reset the seed to 10
print random.random() # 0.57140259469 again

random.randrange(10) # choose randomly from range(10) = [0, 1, ..., 9]
random.randrange(3, 6) # choose randomly from range(3, 6) = [3, 4, 5]

up_to_ten = range(10)
random.shuffle(up_to_ten)
print up_to_ten
# [2, 5, 1, 9, 7, 3, 8, 6, 4, 0] (your results will probably be different)

my_best_friend = random.choice(["Alice", "Bob", "Charlie"]) # "Bob" for me

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6) # [16, 36, 10, 6, 25, 9]

four_with_replacement = [random.choice(range(10)) for _ in range(4)]
# [9, 4, 4, 2]


import re
print all([ # all of these are true, because
not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
re.search("a", "cat"), # * 'cat' has an 'a' in it
not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
3 == len(re.split("[ab]", "carbs")), # * split on a or b to ['c','r','s']
"R-D-" == re.sub("[0-9]", "-", "R2D2") # * replace digits with dashes
]) # prints True


print all([ # all of these are true, because
not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
re.search("a", "cat"), # * 'cat' has an 'a' in it
not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
3 == len(re.split("[ab]", "carbs")), # * split on a or b to ['c','r','s']
"R-D-" == re.sub("[0-9]", "-", "R2D2")
])



