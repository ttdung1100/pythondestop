x = [4,1,2,3]
y = sorted(x) # is [1,2,3,4], x is unchanged
x.sort() # now x is [1,2,3,4]


# sort the list by absolute value from largest to smallest
x = sorted([-4,1,-2,3], key=abs, reverse=True) # is [-4,3,-2,1]
# sort the words and counts from highest count to lowest
wc = sorted(word_counts.items(),
key=lambda (word, count): count,
reverse=True)


even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]

square_dict = { x : x * x for x in range(5) } # { 0:0, 1:1, 2:4, 3:9, 4:16 }
square_set = { x * x for x in [1, -1] } # { 1 }

zeroes = [0 for _ in even_numbers] # has the same length as even_numbers


pairs = [(x, y)
for x in range(10)
for y in range(10)] # 100 pairs (0,0) (0,1) ... (9,8), (9,9)

increasing_pairs = [(x, y) # only pairs with x < y,
for x in range(10) # range(lo, hi) equals
for y in range(x + 1, 10)] # [lo, lo + 1, ..., hi - 1]


