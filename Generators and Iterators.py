def lazy_range(n):
"""a lazy version of range"""
i = 0
while i < n:
yield i
i += 1

for i in lazy_range(10):
do_something_with(i)

def natural_numbers():
"""returns 1, 2, 3, ..."""
n = 1
while True:
yield n
n += 1

lazy_evens_below_20 = (i for i in lazy_range(20) if i % 2 == 0)

