#Tutorial_EDIT python



def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
        raise ValueError('invalid user response')
                    print(reminder)


i = 5
def f(arg=i):
print(arg)
i = 6
f()




def f(a, L=[]):
    L.append(a)
    return L
print(f(1))
print(f(2))
print(f(3))



def f(a, L=None):
    if L is None:
        L = []
        L.append(a)
        return L

def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
print("-- This parrot wouldn't", action, end=' ')
print("if you put", voltage, "volts through it.")
print("-- Lovely plumage, the", type)
print("-- It's", state, "!")


parrot(1000) # 1 positional argument
parrot(voltage=1000) # 1 keyword argument
parrot(voltage=1000000, action='VOOOOOM') # 2 keyword arguments
parrot(action='VOOOOOM', voltage=1000000) # 2 keyword arguments
parrot('a million', 'bereft of life', 'jump') # 3 positional arguments
parrot('a thousand', state='pushing up the daisies') # 1 positional, 1 keyword




parrot() # required argument missing
parrot(voltage=5.0, 'dead') # non-keyword argument after a keyword argument
parrot(110, voltage=220) # duplicate value for the same argument
parrot(actor='John Cleese') # unknown keyword argument



def cheeseshop(kind, *arguments, **keywords):
print("-- Do you have any", kind, "?")
print("-- I'm sorry, we're all out of", kind)
for arg in arguments:
print(arg)
print("-" * 40)
for kw in keywords:
print(kw, ":", keywords[kw])


cheeseshop("Limburger", "It's very runny, sir.",
"It's really very, VERY runny, sir.",
shopkeeper="Michael Palin",
client="John Cleese",
sketch="Cheese Shop Sketch")


def write_multiple_items(file, separator, *args):
    file.write(separator.join(args))




# Fibonacci numbers module
def fib(n): # write Fibonacci series up to n
a, b = 0, 1
while a < n:
print(a, end=' ')
a, b = b, a+b
print()
def fib2(n): # return Fibonacci series up to n
result = []
a, b = 0, 1
while a < n:
result.append(a)
a, b = b, a+b
    return result

if __name__ == "__main__":
    import sys
fib(int(sys.argv[1]))






import sys
try:
    f = open('myfile.txt')
    s = f.readline()
    i = int(s.strip())
    except OSError as err:
        print("OS error: {0}".format(err))
    except ValueError:
        print("Could not convert data to an integer.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
    raise






for arg in sys.argv[1:]:
    try:
f = open(arg, 'r')
except OSError:
print('cannot open', arg)
else:
print(arg, 'has', len(f.readlines()), 'lines')
f.close()



class Error(Exception):
"""Base class for exceptions in this module."""
pass
class InputError(Error):
"""Exception raised for errors in the input.
Attributes:
expression -- input expression in which the error occurred
message -- explanation of the error
"""
def __init__(self, expression, message):
self.expression = expression
self.message = message
class TransitionError(Error):
"""Raised when an operation attempts a state transition that's not
allowed.
Attributes:
previous -- state at beginning of transition
next -- attempted new state
message -- explanation of why the specific transition is not allowed
"""
def __init__(self, previous, next, message):
self.previous = previous
self.next = next
self.message = message





for line in open("myfile.txt"):
print(line, end="")





with open("myfile.txt") as f:
for line in f:
print(line, end="")







def scope_test():
def do_local():
spam = "local spam"
def do_nonlocal():
nonlocal spam
spam = "nonlocal spam"
def do_global():
global spam
spam = "global spam"
spam = "test spam"
do_local()
print("After local assignment:", spam)
do_nonlocal()
print("After nonlocal assignment:", spam)
do_global()
print("After global assignment:", spam)
scope_test()
print("In global scope:", spam)





After local assignment: test spam
After nonlocal assignment: nonlocal spam
After global assignment: nonlocal spam
In global scope: global spam



class MyClass:
"""A simple example class"""
i = 12345
def f(self):
    return 'hello world'







def __init__(self):
self.data = []




x.counter = 1
while x.counter < 10:
x.counter = x.counter * 2
print(x.counter)
del x.counter

x.f()

xf = x.f
while True:
print(xf())


class Dog:
    kind = 'canine' # class variable shared by all instances
    def __init__(self, name):
        self.name = name # instance variable unique to each instance





class Dog:
def __init__(self, name):
    self.name = name
    self.tricks = [] # creates a new empty list for each dog
def add_trick(self, trick):
    self.tricks.append(trick)







# Function defined outside the class
def f1(self, x, y):
    return min(x, x+y)
class C:
f = f1
def g(self):
    return 'hello world'
h = g





class Bag:
    def __init__(self):
        self.data = []
    def add(self, x):
        self.data.append(x)
    def addtwice(self, x):
        self.add(x) 
        self.add(x)





class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)
def update(self, iterable):
    for item in iterable:
        self.items_list.append(item)
        __update = update # private copy of original update() method
class MappingSubclass(Mapping):
    
    def update(self, keys, values):
# provides new signature for update()
# but does not break __init__()
    for item in zip(keys, values):
    self.items_list.append(item)






class Employee:
pass
john = Employee() # Create an empty employee record
# Fill the fields of the record
john.name = 'John Doe'
john.dept = 'computer lab'
john.salary = 1000






for element in [1, 2, 3]:
print(element)
for element in (1, 2, 3):
print(element)
for key in {'one':1, 'two':2}:
print(key)
for char in "123":
print(char)
for line in open("myfile.txt"):
print(line, end='')





class Reverse:
"""Iterator for looping over a sequence backwards."""
def __init__(self, data):
self.data = data
self.index = len(data)
def __iter__(self):
    return self
def __next__(self):
    if self.index == 0:
    raise StopIteration

self.index = self.index - 1
    return self.data[self.index]







def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]






def average(values):
"""Computes the arithmetic mean of a list of numbers.
>>> print(average([20, 30, 70]))
40.0
"""
    return sum(values) / len(values)

import doctest
doctest.testmod() # automatically validate the embedded tests





import unittest
class TestStatisticalFunctions(unittest.TestCase):
def test_average(self):
    self.assertEqual(average([20, 30, 70]), 40.0)
    self.assertEqual(round(average([1, 5, 7]), 1), 4.3)
with self.assertRaises(ZeroDivisionError):
average([])
with self.assertRaises(TypeError):
average(20, 30, 70)
unittest.main() # Calling from the command line invokes all tests






import struct
with open('myfile.zip', 'rb') as f:
    data = f.read()
    start = 0
for i in range(3): # show the first 3 file headers
    start += 14
    fields = struct.unpack('<IIIHH', data[start:start+16])
    crc32, comp_size, uncomp_size, filenamesize, extra_size = fields
    start += 16
    filename = data[start:start+filenamesize]
    start += filenamesize
    extra = data[start:start+extra_size]
print(filename, hex(crc32), comp_size, uncomp_size)
start += extra_size + comp_size # skip to the next header




import threading, zipfile
class AsyncZip(threading.Thread):
    def __init__(self, infile, outfile):
        threading.Thread.__init__(self)
        self.infile = infile
        self.outfile = outfile


def run(self):
    f = zipfile.ZipFile(self.outfile, 'w', zipfile.ZIP_DEFLATED)
    f.write(self.infile)
    f.close()
print('Finished background zip of:', self.infile)
background = AsyncZip('mydata.txt', 'myarchive.zip')
background.start()
print('The main program continues to run in foreground.')
background.join() # Wait for the background task to finish
print('Main program waited until background was done.')





import logging
logging.debug('Debugging information')
logging.info('Informational message')
logging.warning('Warning:config file %s not found', 'server.conf')
logging.error('Error occurred')
logging.critical('Critical error -- shutting down')



unsearched = deque([starting_node])
def breadth_first_search(unsearched):
    node = unsearched.popleft()

for m in gen_moves(node):
    if is_goal(m):
        return m
        unsearched.append(m)




import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
with open(filename) as fobj:
startup_file = fobj.read()
exec(startup_file)



complex(real=3, imag=5)
complex(**{'real': 3, 'imag': 5})


complex(3, 5)
complex(*(3, 5))



from typing import List, Tuple
def remove_gray_shades(
colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
pass



from typing import List, Tuple
Color = Tuple[int, int, int]

def remove_gray_shades(colors: List[Color]) -> List[Color]:
    pass



class C:
field: 'annotation'

count: int = 0





