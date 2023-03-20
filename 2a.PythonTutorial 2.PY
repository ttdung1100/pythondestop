#2a.PythonTutorial 2

#CHAPTER
#FOUR
#MORE CONTROL FLOW TOOLS


x = int(input("Please enter an integer: "))

if x < 0:
    x = 0
    print(’Negative changed to zero’)
    elif x == 0:
        print(’Zero’)
        elif x == 1:
            print(’Single’)
            else:
                print(’More’)


a = [’cat’, ’window’, ’defenestrate’]
for x in a:
    print(x, len(x))
cat 3

for x in a[:]: # make a slice copy of the entire list
    if len(x) > 6: a.insert(0, x)
print a

print(a)



range(5, 10)
5 through 9
range(0, 10, 3)
0, 3, 6, 9
range(-10, -100, -30)
-10, -40, -70


a = [’Mary’, ’had’, ’a’, ’little’, ’lamb’]
for i in range(len(a)):
    print(i, a[i])




for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, ’equals’, x, ’*’, n//x)
            break
        else:
# loop fell through without finding a factor
print(n, ’is a prime number’)


#4.7.1 Default Argument Values


def ask_ok(prompt, retries=4, complaint=’Yes or no, please!’):
while True:
    ok = input(prompt)
if ok in (’y’, ’ye’, ’yes’):
return True
if ok in (’n’, ’no’, ’nop’, ’nope’):
return False
retries = retries - 1
if retries < 0:
raise IOError(’refusenik user’)
print(complaint)

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

#4.7.2 Keyword Arguments

def parrot(voltage, state=’a stiff’, action=’voom’, type=’Norwegian Blue’):
    print("-- This parrot wouldn’t", action, end=’ ’)
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It’s", state, "!")

parrot(1000) # 1 positional argument
parrot(voltage=1000) # 1 keyword argument
parrot(voltage=1000000, action=’VOOOOOM’) # 2 keyword arguments
parrot(action=’VOOOOOM’, voltage=1000000) # 2 keyword arguments
parrot(’a million’, ’bereft of life’, ’jump’) # 3 positional arguments
parrot(’a thousand’, state=’pushing up the daisies’) # 1 positional, 1 keyword

parrot() # required argument missing
parrot(voltage=5.0, ’dead’) # non-keyword argument after a keyword argument
parrot(110, voltage=220) # duplicate value for the same argument
parrot(actor=’John Cleese’) # unknown keyword argument



def cheeseshop(kind, *arguments, **keywords):
print("-- Do you have any", kind, "?")
print("-- I’m sorry, we’re all out of", kind)
for arg in arguments:
print(arg)
print("-" * 40)
keys = sorted(keywords.keys())
for kw in keys:
print(kw, ":", keywords[kw])


cheeseshop("Limburger", "It’s very runny, sir.",
"It’s really very, VERY runny, sir.",
shopkeeper="Michael Palin",
client="John Cleese",
sketch="Cheese Shop Sketch")



def write_multiple_items(file, separator, *args):
file.write(separator.join(args))


def concat(*args, sep="/"):
    return sep.join(args)

concat("earth", "mars", "venus")
’earth/mars/venus’

concat("earth", "mars", "venus", sep=".")
’earth.mars.venus’


#4.7.4 Unpacking Argument Lists
list(range(3, 6)) # normal call with separate arguments
[3, 4, 5]
args = [3, 6]
list(range(*args)) # call with arguments unpacked from a list
[3, 4, 5]




def parrot(voltage, state=’a stiff’, action=’voom’):
    print("-- This parrot wouldn’t", action, end=’ ’)
    print("if you put", voltage, "volts through it.", end=’ ’)
    print("E’s", state, "!")
    d = {"voltage": "four million", "state": "bleedin’ demised", "action": "VOOM"}
    parrot(**d)



# This parrot wouldn’t VOOM if you put four million volts through it. E’s bleedin’ demised


def make_incrementor(n):
    return lambda x: x + n
    f = make_incrementor(42)
    f(0)
    f(1)


def my_function():
    pass
"""Do nothing, but document it.

No, really, it doesn’t do anything.
"""

print(my_function.__doc__)
Do nothing, but document it.
No, really, it doesn’t do anything.



# DATA STRUCTURE
a = [66.25, 333, 333, 1, 1234.5]
print(a.count(333), a.count(66.25), a.count(’x’))
a.insert(2, -1)



a.append(333)
a
[66.25, 333, -1, 333, 1, 1234.5, 333]
a.index(333)
1
a.remove(333)
a
[66.25, -1, 333, 1, 1234.5, 333]
a.reverse()
a
[333, 1234.5, 1, 333, -1, 66.25]
a.sort()
a
[-1, 1, 66.25, 333, 333, 1234.5]






stack = [3, 4, 5]
stack.append(6)
stack.append(7)
stack
[3, 4, 5, 6, 7]
stack.pop()
7
stack
[3, 4, 5, 6]
stack.pop()
6
stack.pop()
5
stack
[3, 4]

from collections import deque
queue = deque(["Eric", "John", "Michael"])
queue.append("Terry") # Terry arrives
queue.append("Graham") # Graham arrives
queue.popleft() # The first to arrive now leaves
’Eric’
queue.popleft() # The second to arrive now leaves
’John’

queue # Remaining queue in order of arrival
deque([’Michael’, ’Terry’, ’Graham’])




squares = []
for x in range(10):
squares.append(x**2)

squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
squares = [x**2 for x in range(10)]


[(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]

combs = []
>>> for x in [1,2,3]:
... for y in [3,1,4]:
... if x != y:
... combs.append((x, y))
...
>>> combs
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]


vec = [-4, -2, 0, 2, 4]
>>> # create a new list with the values doubled
>>> [x*2 for x in vec]
[-8, -4, 0, 4, 8]
>>> # filter the list to exclude negative numbers
>>> [x for x in vec if x >= 0]
[0, 2, 4]
>>> # apply a function to all the elements
>>> [abs(x) for x in vec]
[4, 2, 0, 2, 4]
>>> # call a method on each element
>>> freshfruit = [’ banana’, ’ loganberry ’, ’passion fruit ’]



[weapon.strip() for weapon in freshfruit]
[’banana’, ’loganberry’, ’passion fruit’]
>>> # create a list of 2-tuples like (number, square)
>>> [(x, x**2) for x in range(6)]
[(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]
>>> # the tuple must be parenthesized, otherwise an error is raised
>>> [x, x**2 for x in range(6)]
File "<stdin>", line 1, in ?
[x, x**2 for x in range(6)]
^
SyntaxError: invalid syntax
>>> # flatten a list using a listcomp with two ’for’
>>> vec = [[1,2,3], [4,5,6], [7,8,9]]
>>> [num for elem in vec for num in elem]
[1, 2, 3, 4, 5, 6, 7, 8, 9]
List comprehensions can contain complex expressions and nested functions:
>>> from math import pi
>>> [str(round(pi, i)) for i in range(1, 6)]
[’3.1’, ’3.14’, ’3.142’, ’3.1416’, ’3.14159’]



matrix = [
... [1, 2, 3, 4],
... [5, 6, 7, 8],
... [9, 10, 11, 12],
... ]



[[row[i] for row in matrix] for i in range(4)]
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]


transposed = []
>>> for i in range(4):
... transposed.append([row[i] for row in matrix])
...
>>> transposed
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]


transposed = []
>>> for i in range(4):
... # the following 3 lines implement the nested listcomp
... transposed_row = []
... for row in matrix:
... transposed_row.append(row[i])
... transposed.append(transposed_row)


transposed
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]


zip(*matrix)
[(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]


a = [-1, 1, 66.25, 333, 333, 1234.5]
>>> del a[0]
>>> a
[1, 66.25, 333, 333, 1234.5]
>>> del a[2:4]
>>> a
[1, 66.25, 1234.5]
>>> del a[:]
>>> a


>>> del a



>>> t = 12345, 54321, ’hello!’
>>> t[0]
12345
>>> t
(12345, 54321, ’hello!’)
>>> # Tuples may be nested:
... u = t, (1, 2, 3, 4, 5)
>>> u
((12345, 54321, ’hello!’), (1, 2, 3, 4, 5))
>>> # Tuples are immutable:
... t[0] = 88888


Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: ’tuple’ object does not support item assignment
>>> # but they can contain mutable objects:
... v = ([1, 2, 3], [3, 2, 1])
>>> v
([1, 2, 3], [3, 2, 1])




>>> empty = ()
>>> singleton = ’hello’, # <-- note trailing comma
>>> len(empty)
0
>>> len(singleton)
1
>>> singleton
(’hello’,)
>>> x, y, z = t


>>> basket = {’apple’, ’orange’, ’apple’, ’pear’, ’orange’, ’banana’}
>>> print(basket) # show that duplicates have been removed
{’orange’, ’banana’, ’pear’, ’apple’}
>>> ’orange’ in basket # fast membership testing
True


>>> ’crabgrass’ in basket
False
>>> # Demonstrate set operations on unique letters from two words
...
>>> a = set(’abracadabra’)
>>> b = set(’alacazam’)
>>> a # unique letters in a
{’a’, ’r’, ’b’, ’c’, ’d’}
>>> a - b # letters in a but not in b
{’r’, ’d’, ’b’}
>>> a | b # letters in either a or b
{’a’, ’c’, ’r’, ’d’, ’b’, ’m’, ’z’, ’l’}
>>> a & b # letters in both a and b
{’a’, ’c’}
>>> a ^ b # letters in a or b but not both
{’r’, ’d’, ’b’, ’m’, ’z’, ’l’}
Like for lists, there is a set comprehension syntax:
>>> a = {x for x in ’abracadabra’ if x not in ’abc’}
>>> a
{’r’, ’d’}



>>> tel = {’jack’: 4098, ’sape’: 4139}
>>> tel[’guido’] = 4127
>>> tel
{’sape’: 4139, ’guido’: 4127, ’jack’: 4098}
>>> tel[’jack’]


4098
>>> del tel[’sape’]
>>> tel[’irv’] = 4127
>>> tel
{’guido’: 4127, ’irv’: 4127, ’jack’: 4098}
>>> list(tel.keys())
[’irv’, ’guido’, ’jack’]
>>> sorted(tel.keys())
[’guido’, ’irv’, ’jack’]
>>> ’guido’ in tel
True
>>> ’jack’ not in tel
False



>>> dict([(’sape’, 4139), (’guido’, 4127), (’jack’, 4098)])
{’sape’: 4139, ’jack’: 4098, ’guido’: 4127}



>>> {x: x**2 for x in (2, 4, 6)}
{2: 4, 4: 16, 6: 36}



>>> dict(sape=4139, guido=4127, jack=4098)
{’sape’: 4139, ’jack’: 4098, ’guido’: 4127}




#5.6 Looping Techniques


>>> knights = {’gallahad’: ’the pure’, ’robin’: ’the brave’}
>>> for k, v in knights.items():
... print(k, v)
...
gallahad the pure
robin the brave

>>> for i, v in enumerate([’tic’, ’tac’, ’toe’]):
... print(i, v)
...
0 tic
1 tac
2 toe


>>> questions = [’name’, ’quest’, ’favorite color’]
>>> answers = [’lancelot’, ’the holy grail’, ’blue’]
>>> for q, a in zip(questions, answers):
... print(’What is your {0}? It is {1}.’.format(q, a))


What is your name? It is lancelot.
What is your quest? It is the holy grail.
What is your favorite color? It is blue.


for i in reversed(range(1, 10, 2)):
    print(i)

basket = [’apple’, ’orange’, ’apple’, ’pear’, ’orange’, ’banana’]
>>> for f in sorted(set(basket)):
... print(f)
...
apple
banana
orange
pear


>>> string1, string2, string3 = ’’, ’Trondheim’, ’Hammer Dance’
>>> non_null = string1 or string2 or string3
>>> non_null
’Trondheim’


(1, 2, 3) < (1, 2, 4)
[1, 2, 3] < [1, 2, 4]
’ABC’ < ’C’ < ’Pascal’ < ’Python’
(1, 2, 3, 4) < (1, 2, 4)
(1, 2) < (1, 2, -1)
(1, 2, 3) == (1.0, 2.0, 3.0)
(1, 2, (’aa’, ’ab’)) < (1, 2, (’abc’, ’a’), 4)


def fib(n): # write Fibonacci series up to n
a, b = 0, 1
while b < n:
print(b, end=’ ’)
a, b = b, a+b
print()
def fib2(n): # return Fibonacci series up to n
result = []
a, b = 0, 1
while b < n:
result.append(b)
a, b = b, a+b
return result


>>> import fibo

fibo.fib(1000)
1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987
>>> fibo.fib2(100)


[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
>>> fibo.__name__
’fibo’


>>> fib = fibo.fib
>>> fib(500)
1 1 2 3 5 8 13 21 34 55 89 144 233 377


from fibo import fib, fib2
>>> fib(500)
1 1 2 3 5 8 13 21 34 55 89 144 233 377


from fibo import *
>>> fib(500)
1 1 2 3 5 8 13 21 34 55 89 144 233 377


#6.1.1 Executing modules as scripts

python fibo.py <arguments>

if __name__ == "__main__":
import sys
fib(int(sys.argv[1]))


$ python fibo.py 50
1 1 2 3 5 8 13 21 34


>>> import fibo
>>>



>>> import sys
>>> sys.ps1
’>>> ’
>>> sys.ps2
’... ’
>>> sys.ps1 = ’C> ’
C> print(’Yuck!’)
Yuck!
C>



>>> import sys
>>> sys.path.append(’/ufs/guido/lib/python’)


>>> import fibo, sys
>>> dir(fibo)
[’__name__’, ’fib’, ’fib2’]




>>> dir(sys)
[’__displayhook__’, ’__doc__’, ’__excepthook__’, ’__name__’, ’__stderr__’,
’__stdin__’, ’__stdout__’, ’_getframe’, ’api_version’, ’argv’,
’builtin_module_names’, ’byteorder’, ’callstats’, ’copyright’,
’displayhook’, ’exc_info’, ’excepthook’,
’exec_prefix’, ’executable’, ’exit’, ’getdefaultencoding’, ’getdlopenflags’,
’getrecursionlimit’, ’getrefcount’, ’hexversion’, ’maxint’, ’maxunicode’,
’meta_path’, ’modules’, ’path’, ’path_hooks’, ’path_importer_cache’,
’platform’, ’prefix’, ’ps1’, ’ps2’, ’setcheckinterval’, ’setdlopenflags’,
’setprofile’, ’setrecursionlimit’, ’settrace’, ’stderr’, ’stdin’, ’stdout’,
’version’, ’version_info’, ’warnoptions’]




>>> a = [1, 2, 3, 4, 5]
>>> import fibo
>>> fib = fibo.fib
>>> dir()
[’__builtins__’, ’__doc__’, ’__file__’, ’__name__’, ’a’, ’fib’, ’fibo’, ’sys’]





import builtins
>>> dir(builtins)



[’ArithmeticError’, ’AssertionError’, ’AttributeError’, ’BaseException’, ’Buffer
Error’, ’BytesWarning’, ’DeprecationWarning’, ’EOFError’, ’Ellipsis’, ’Environme
ntError’, ’Exception’, ’False’, ’FloatingPointError’, ’FutureWarning’, ’Generato
rExit’, ’IOError’, ’ImportError’, ’ImportWarning’, ’IndentationError’, ’IndexErr
or’, ’KeyError’, ’KeyboardInterrupt’, ’LookupError’, ’MemoryError’, ’NameError’,
’None’, ’NotImplemented’, ’NotImplementedError’, ’OSError’, ’OverflowError’, ’P
endingDeprecationWarning’, ’ReferenceError’, ’RuntimeError’, ’RuntimeWarning’, ’
StopIteration’, ’SyntaxError’, ’SyntaxWarning’, ’SystemError’, ’SystemExit’, ’Ta
bError’, ’True’, ’TypeError’, ’UnboundLocalError’, ’UnicodeDecodeError’, ’Unicod
eEncodeError’, ’UnicodeError’, ’UnicodeTranslateError’, ’UnicodeWarning’, ’UserW
arning’, ’ValueError’, ’Warning’, ’ZeroDivisionError’, ’__build_class__’, ’__deb
ug__’, ’__doc__’, ’__import__’, ’__name__’, ’__package__’, ’abs’, ’all’, ’any’,
’ascii’, ’bin’, ’bool’, ’bytearray’, ’bytes’, ’chr’, ’classmethod’, ’compile’, ’
complex’, ’copyright’, ’credits’, ’delattr’, ’dict’, ’dir’, ’divmod’, ’enumerate
’, ’eval’, ’exec’, ’exit’, ’filter’, ’float’, ’format’, ’frozenset’, ’getattr’,
’globals’, ’hasattr’, ’hash’, ’help’, ’hex’, ’id’, ’input’, ’int’, ’isinstance’,
’issubclass’, ’iter’, ’len’, ’license’, ’list’, ’locals’, ’map’, ’max’, ’memory
view’, ’min’, ’next’, ’object’, ’oct’, ’open’, ’ord’, ’pow’, ’print’, ’property’
, ’quit’, ’range’, ’repr’, ’reversed’, ’round’, ’set’, ’setattr’, ’slice’, ’sort
ed’, ’staticmethod’, ’str’, ’sum’, ’super’, ’tuple’, ’type’, ’vars’, ’zip’]


#CHAPTER
#SEVEN
#INPUT AND OUTPUT

>>> s = ’Hello, world.’
>>> str(s)
’Hello, world.’
>>> repr(s)
"’Hello, world.’"
>>> str(1/7)
’0.14285714285714285’
>>> x = 10 * 3.25
>>> y = 200 * 200
>>> s = ’The value of x is ’ + repr(x) + ’, and y is ’ + repr(y) + ’...’
>>> print(s)
The value of x is 32.5, and y is 40000...
>>> # The repr() of a string adds string quotes and backslashes:





... hello = ’hello, world\n’
>>> hellos = repr(hello)
>>> print(hellos)
’hello, world\n’
>>> # The argument to repr() may be any Python object:
... repr((x, y, (’spam’, ’eggs’)))
"(32.5, 40000, (’spam’, ’eggs’))"
Here are two ways to write a table of squares and cubes:
>>> for x in range(1, 11):
... print(repr(x).rjust(2), repr(x*x).rjust(3), end=’ ’)
... # Note use of ’end’ on previous line
... print(repr(x*x*x).rjust(4))
...
1 1 1
2 4 8
3 9 27
4 16 64
5 25 125
6 36 216
7 49 343
8 64 512
9 81 729
10 100 1000
>>> for x in range(1, 11):
... print(’{0:2d} {1:3d} {2:4d}’.format(x, x*x, x*x*x))
...
1 1 1
2 4 8
3 9 27
4 16 64
5 25 125
6 36 216
7 49 343
8 64 512
9 81 729
10 100 1000



>>> ’12’.zfill(5)
’00012’
>>> ’-3.14’.zfill(7)
’-003.14’   
>>> ’3.14159265359’.zfill(5)
’3.14159265359’





>>> print(’We are the {} who say "{}!"’.format(’knights’, ’Ni’))
We are the knights who say "Ni!"

>>> print(’{0} and {1}’.format(’spam’, ’eggs’))
spam and eggs
>>> print(’{1} and {0}’.format(’spam’, ’eggs’))
eggs and spam



>>> print(’This {food} is {adjective}.’.format(
... food=’spam’, adjective=’absolutely horrible’))
This spam is absolutely horrible.


>>> print(’The story of {0}, {1}, and {other}.’.format(’Bill’, ’Manfred’,
other=’Georg’))
The story of Bill, Manfred, and Georg.



>>> import math
>>> print(’The value of PI is approximately {}.’.format(math.pi))
The value of PI is approximately 3.14159265359.
>>> print(’The value of PI is approximately {!r}.’.format(math.pi))
The value of PI is approximately 3.141592653589793.


>>> import math
>>> print(’The value of PI is approximately {0:.3f}.’.format(math.pi))
The value of PI is approximately 3.142.


>>> table = {’Sjoerd’: 4127, ’Jack’: 4098, ’Dcab’: 7678}
>>> for name, phone in table.items():
... print(’{0:10} ==> {1:10d}’.format(name, phone))
...
Jack ==> 4098
Dcab ==> 7678
Sjoerd ==> 4127


>>> table = {’Sjoerd’: 4127, ’Jack’: 4098, ’Dcab’: 8637678}
>>> print(’Jack: {0[Jack]:d}; Sjoerd: {0[Sjoerd]:d}; ’
’Dcab: {0[Dcab]:d}’.format(table))
Jack: 4098; Sjoerd: 4127; Dcab: 8637678



>>> table = {’Sjoerd’: 4127, ’Jack’: 4098, ’Dcab’: 8637678}
>>> print(’Jack: {Jack:d}; Sjoerd: {Sjoerd:d}; Dcab: {Dcab:d}’.format(**table))
Jack: 4098; Sjoerd: 4127; Dcab: 8637678


>>> import math
>>> print(’The value of PI is approximately %5.3f.’ % math.pi)
The value of PI is approximately 3.142


>>> f = open(’/tmp/workfile’, ’w’)




>>> f.read()
’This is the entire file.\n’
>>> f.read()
’’


>>> f.readline()
’This is the first line of the file.\n’
>>> f.readline()
’Second line of the file\n’
>>> f.readline()
’’


>>> f.readlines()
[’This is the first line of the file.\n’, ’Second line of the file\n’]





>>> for line in f:
... print(line, end=’’)
...
This is the first line of the file.
Second line of the file


>>> f.write(’This is a test\n’)
15





value = (’the answer’, 42)
>>> s = str(value)
>>> f.write(s)
18




f = open(’/tmp/workfile’, ’rb+’)
>>> f.write(b’0123456789abcdef’)
16
>>> f.seek(5) # Go to the 6th byte in the file
5
>>> f.read(1)
b’5’
>>> f.seek(-3, 2) # Go to the 3rd byte before the end
13
>>> f.read(1)
b’d’





f.close()
>>> f.read()
Traceback (most recent call last):
File "<stdin>", line 1, in ?
ValueError: I/O operation on closed file



with open(’/tmp/workfile’, ’r’) as f:
... read_data = f.read()
>>> f.closed
True



#7.2.2 The pickle Module

pickle.dump(x, f)

x = pickle.load(f)


#ERRORS AND EXCEPTIONS


>>> while True print(’Hello world’)
File "<stdin>", line 1, in ?
while True print(’Hello world’)


>>> 10 * (1/0)
Traceback (most recent call last):
File "<stdin>", line 1, in ?
ZeroDivisionError: int division or modulo by zero
>>> 4 + spam*3
Traceback (most recent call last):
File "<stdin>", line 1, in ?
NameError: name ’spam’ is not defined
>>> ’2’ + 2
Traceback (most recent call last):
File "<stdin>", line 1, in ?
TypeError: Can’t convert ’int’ object to str implicitly


>>> while True:
... try:
... x = int(input("Please enter a number: "))
... break
... except ValueError:
... print("Oops! That was no valid number. Try again...")
...



... except (RuntimeError, TypeError, NameError):
... pass



import sys
try:
f = open(’myfile.txt’)



s = f.readline()
i = int(s.strip())
except IOError as err:
print("I/O error: {0}".format(err))
except ValueError:
print("Could not convert data to an integer.")
except:
print("Unexpected error:", sys.exc_info()[0])
raise



for arg in sys.argv[1:]:
try:
f = open(arg, ’r’)
except IOError:
print(’cannot open’, arg)
else:
print(arg, ’has’, len(f.readlines()), ’lines’)
f.close()



>>> try:
... raise Exception(’spam’, ’eggs’)
... except Exception as inst:
... print(type(inst)) # the exception instance
... print(inst.args) # arguments stored in .args
... print(inst) # __str__ allows args to be printed directly,
... # but may be overridden in exception subclasses
... x, y = inst.args # unpack args
... print(’x =’, x)
... print(’y =’, y)
...
<class ’Exception’>
(’spam’, ’eggs’)
(’spam’, ’eggs’)
x = spam
y = eggs


>>> def this_fails():
... x = 1/0
...





>>> try:
... this_fails()
... except ZeroDivisionError as err:
... print(’Handling run-time error:’, err)
...


Handling
Handling run-time error: int division or modulo by zero


>>> raise NameError(’HiThere’)
Traceback (most recent call last):
File "<stdin>", line 1, in ?
NameError: HiThere




>>> try:
... raise NameError(’HiThere’)
... except NameError:
... print(’An exception flew by!’)
... raise
...
An exception flew by!
Traceback (most recent call last):
File "<stdin>", line 2, in ?
NameError: HiThere




>>> class MyError(Exception):
... def __init__(self, value):
... self.value = value
... def __str__(self):
... return repr(self.value)
...
>>> try:
... raise MyError(2*2)
... except MyError as e:
... print(’My exception occurred, value:’, e.value)
...
My exception occurred, value: 4
>>> raise MyError(’oops!’)
Traceback (most recent call last):




File "<stdin>", line 1, in ?
__main__.MyError: ’oops!’

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
"""Raised when an operation attempts a state transition that’s not
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





>>> try:
... raise KeyboardInterrupt
... finally:

... print(’Goodbye, world!’)
...
Goodbye, world!
KeyboardInterrupt


>>> def divide(x, y):
... try:
... result = x / y
... except ZeroDivisionError:
... print("division by zero!")
... else:
... print("result is", result)
... finally:
... print("executing finally clause")
...
>>> divide(2, 1)
result is 2.0
executing finally clause
>>> divide(2, 0)
division by zero!
executing finally clause
>>> divide("2", "1")
executing finally clause
Traceback (most recent call last):
File "<stdin>", line 1, in ?
File "<stdin>", line 3, in divide
TypeError: unsupported operand type(s) for /: ’str’ and ’str’




for line in open("myfile.txt"):
print(line)




with open("myfile.txt") as f:
for line in f:
print(line)

#CHAPTER
#NINE
#CLASSES


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



#9.3.1 Class Definition Syntax
class MyClass:
"""A simple example class"""
i = 12345
def f(self):
return ’hello world’



x = MyClass()



def __init__(self):
self.data = []


x = MyClass()


>>> class Complex:
... def __init__(self, realpart, imagpart):
... self.r = realpart
... self.i = imagpart
...
>>> x = Complex(3.0, -4.5)
>>> x.r, x.i
(3.0, -4.5)




x.counter = 1
while x.counter < 10:
x.counter = x.counter * 2
print(x.counter)
del x.counter






x.f()



xf = x.f
while True:
print(xf())


# Function defined outside the class
def f1(self, x, y):
return min(x, x+y)
class C:
f = f1
def g(self):
return ’hello world’
h = g


class Bag:
def __init__(self):
self.data = []
def add(self, x):
self.data.append(x)
def addtwice(self, x):
self.add(x)
self.add(x)


class DerivedClassName(modname.BaseClassName):

#9.6 Private Variables

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
john.name = ’John Doe’
john.dept = ’computer lab’
john.salary = 1000




raise Class
raise Instance



raise Class()




class B(Exception):
pass
class C(B):
pass
class D(C):
pass
for c in [B, C, D]:
try:
raise c()
except D:
print("D")
except C:
print("C")
except B:
print("B")





#9.9 Iterators

for element in [1, 2, 3]:
print(element)
for element in (1, 2, 3):
print(element)
for key in {’one’:1, ’two’:2}:
print(key)
for char in "123":
print(char)
for line in open("myfile.txt"):
print(line)




>>> s = ’abc’
>>> it = iter(s)
>>> it
<iterator object at 0x00A1DB50>
>>> next(it)
’a’
>>> next(it)
’b’
>>> next(it)
’c’
>>> next(it)
Traceback (most recent call last):
File "<stdin>", line 1, in ?
next(it)
StopIteration


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
>>> rev = Reverse(’spam’)
>>> iter(rev)
<__main__.Reverse object at 0x00A1DB50>
>>> for char in rev:
... print(char)
...
m a p s 



#9.10 Generators


def reverse(data):
for index in range(len(data)-1, -1, -1):
yield data[index]
>>> for char in reverse(’golf’):
... print(char)
...
f l o g





>>> sum(i*i for i in range(10)) # sum of squares
285
>>> xvec = [10, 20, 30]
>>> yvec = [7, 5, 3]
>>> sum(x*y for x,y in zip(xvec, yvec)) # dot product
260
>>> from math import pi, sin
>>> sine_table = {x: sin(x*pi/180) for x in range(0, 91)}
>>> unique_words = set(word for line in page for word in line.split())
>>> valedictorian = max((student.gpa, student.name) for student in graduates)
>>> data = ’golf’
>>> list(data[i] for i in range(len(data)-1, -1, -1))
[’f’, ’l’, ’o’, ’g’]


#CHAPTER
#TEN
#BRIEF TOUR OF THE STANDARD LIBRARY


#10.1 Operating System Interface
>>> import os
>>> os.getcwd() # Return the current working directory
’C:\\Python31’
>>> os.chdir(’/server/accesslogs’) # Change current working directory
>>> os.system(’mkdir today’) # Run the command mkdir in the system shell
0



>>> import os
>>> dir(os)
<returns a list of all module functions>
>>> help(os)
<returns an extensive manual page created from the module’s docstrings>




>>> import shutil
>>> shutil.copyfile(’data.db’, ’archive.db’)
>>> shutil.move(’/build/executables’, ’installdir’)



>>> import glob
>>> glob.glob(’*.py’)
[’primes.py’, ’random.py’, ’quote.py’]






>>> import sys
>>> print(sys.argv)
[’demo.py’, ’one’, ’two’, ’three’]


#10.4 Error Output Redirection and Program Termination


>>> sys.stderr.write(’Warning, log file not found starting a new one\n’)
Warning, log file not found starting a new one
The most direct way to terminate a script is to use sys.exit().


#10.5 String Pattern Matching


>>> import re
>>> re.findall(r’\bf[a-z]*’, ’which foot or hand fell fastest’)
[’foot’, ’fell’, ’fastest’]
>>> re.sub(r’(\b[a-z]+) \1’, r’\1’, ’cat in the the hat’)
’cat in the hat’


>>> ’tea for too’.replace(’too’, ’two’)
’tea for two’

#10.6 Mathematics

>>> import math
>>> math.cos(math.pi / 4)
0.70710678118654757
>>> math.log(1024, 2)
10.0




>>> import random
>>> random.choice([’apple’, ’pear’, ’banana’])
’apple’
>>> random.sample(range(100), 10) # sampling without replacement
[30, 83, 16, 4, 8, 81, 41, 50, 18, 33]
>>> random.random() # random float
0.17970987693706186
>>> random.randrange(6) # random integer chosen from range(6)
4




#10.7 Internet Access

>>> from urllib.request import urlopen
>>> for line in urlopen(’http://tycho.usno.navy.mil/cgi-bin/timer.pl’):
... line = line.decode(’utf-8’) # Decoding the binary data to text.
... if ’EST’ in line or ’EDT’ in line: # look for Eastern Time
... print(line)
<BR>Nov. 25, 09:43:32 PM EST
>>> import smtplib
>>> server = smtplib.SMTP(’localhost’)
>>> server.sendmail(’soothsayer@example.org’, ’jcaesar@example.org’,
... """To: jcaesar@example.org
... From: soothsayer@example.org
...
... Beware the Ides of March.
... """)
>>> server.quit()
(Note that the second example needs a mailserver running on localhost.)



#10.8 Dates and Times

>>> # dates are easily constructed and formatted
>>> from datetime import date
>>> now = date.today()
>>> now
datetime.date(2003, 12, 2)
>>> now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B.")
’12-02-03. 02 Dec 2003 is a Tuesday on the 02 day of December.’
>>> # dates support calendar arithmetic
>>> birthday = date(1964, 7, 31)
>>> age = now - birthday
>>> age.days
14368



#10.9 Data Compression

>>> import zlib
>>> s = b’witch which has which witches wrist watch’
>>> len(s)
41
>>> t = zlib.compress(s)
>>> len(t)
37
>>> zlib.decompress(t)
b’witch which has which witches wrist watch’
>>> zlib.crc32(s)
226805979


#10.10 Performance Measurement

>>> from timeit import Timer
>>> Timer(’t=a; a=b; b=t’, ’a=1; b=2’).timeit()
0.57535828626024577
>>> Timer(’a,b = b,a’, ’a=1; b=2’).timeit()
0.54962537085770791



#10.11 Quality Control
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
self.assertRaises(ZeroDivisionError, average, [])
self.assertRaises(TypeError, average, 20, 30, 70)
unittest.main() # Calling from the command line invokes all tests



#CHAPTER
#ELEVEN
#BRIEF TOUR OF THE STANDARD
#LIBRARY – PART II


>>> import reprlib
>>> reprlib.repr(set(’supercalifragilisticexpialidocious’))
"set([’a’, ’c’, ’d’, ’e’, ’f’, ’g’, ...])"



>>> import pprint
>>> t = [[[[’black’, ’cyan’], ’white’, [’green’, ’red’]], [[’magenta’,
... ’yellow’], ’blue’]]]
...
>>> pprint.pprint(t, width=30)
[[[[’black’, ’cyan’],
’white’,
[’green’, ’red’]],
[[’magenta’, ’yellow’],
’blue’]]]


>>> import textwrap
>>> doc = """The wrap() method is just like fill() except that it returns
... a list of strings instead of one big string with newlines to separate
... the wrapped lines."""
...
>>> print(textwrap.fill(doc, width=40))
The wrap() method is just like fill()
except that it returns a list of strings
instead of one big string with newlines
to separate the wrapped lines.


>>> import locale
>>> locale.setlocale(locale.LC_ALL, ’English_United States.1252’)
’English_United States.1252’
>>> conv = locale.localeconv() # get a mapping of conventions
>>> x = 1234567.8
>>> locale.format("%d", x, grouping=True)
’1,234,567’
>>> locale.format_string("%s%.*f", (conv[’currency_symbol’],
... conv[’frac_digits’], x), grouping=True)
’$1,234,567.80’




>>> from string import Template
>>> t = Template(’${village}folk send $$10 to $cause.’)
>>> t.substitute(village=’Nottingham’, cause=’the ditch fund’)
’Nottinghamfolk send $10 to the ditch fund.’



>>> t = Template(’Return the $item to $owner.’)
>>> d = dict(item=’unladen swallow’)
>>> t.substitute(d)
Traceback (most recent call last):
. . .
KeyError: ’owner’
>>> t.safe_substitute(d)
’Return the unladen swallow to $owner.’



>>> import time, os.path
>>> photofiles = [’img_1074.jpg’, ’img_1076.jpg’, ’img_1077.jpg’]
>>> class BatchRename(Template):
... delimiter = ’%’
>>> fmt = input(’Enter rename style (%d-date %n-seqnum %f-format): ’)
Enter rename style (%d-date %n-seqnum %f-format): Ashley_%n%f
>>> t = BatchRename(fmt)
>>> date = time.strftime(’%d%b%y’)


>>> for i, filename in enumerate(photofiles):
... base, ext = os.path.splitext(filename)
... newname = t.substitute(d=date, n=i, f=ext)
... print(’{0} --> {1}’.format(filename, newname))
img_1074.jpg --> Ashley_0.jpg
img_1076.jpg --> Ashley_1.jpg
img_1077.jpg --> Ashley_2.jpg






import struct
data = open(’myfile.zip’, ’rb’).read()
start = 0
for i in range(3): # show the first 3 file headers
start += 14
fields = struct.unpack(’<IIIHH’, data[start:start+16])
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
    f = zipfile.ZipFile(self.outfile, ’w’, zipfile.ZIP_DEFLATED)
f.write(self.infile)
f.close()
print(’Finished background zip of:’, self.infile)
background = AsyncZip(’mydata.txt’, ’myarchive.zip’)
background.start()
print(’The main program continues to run in foreground.’)
background.join() # Wait for the background task to finish
print(’Main program waited until background was done.’)


#11.5 Logging

import logging
logging.debug(’Debugging information’)
logging.info(’Informational message’)
logging.warning(’Warning:config file %s not found’, ’server.conf’)
logging.error(’Error occurred’)
logging.critical(’Critical error -- shutting down’)




>>> import weakref, gc
>>> class A:
... def __init__(self, value):
... self.value = value
... def __repr__(self):
... return str(self.value)
...
>>> a = A(10) # create a reference
>>> d = weakref.WeakValueDictionary()
>>> d[’primary’] = a # does not create a reference
>>> d[’primary’] # fetch the object if it is still alive
10
>>> del a # remove the one reference
>>> gc.collect() # run garbage collection right away
0
>>> d[’primary’] # entry was automatically removed
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
d[’primary’] # entry was automatically removed
File "C:/python31/lib/weakref.py", line 46, in __getitem__
o = self.data[key]()
KeyError: ’primary’




>>> from array import array
>>> a = array(’H’, [4000, 10, 700, 22222])
>>> sum(a)
26932
>>> a[1:3]
array(’H’, [10, 700])




>>> from collections import deque
>>> d = deque(["task1", "task2", "task3"])
>>> d.append("task4")
>>> print("Handling", d.popleft())
Handling task1
unsearched = deque([starting_node])
def breadth_first_search(unsearched):
node = unsearched.popleft()




for m in gen_moves(node):
if is_goal(m):
return m
unsearched.append(m)



>>> import bisect
>>> scores = [(100, ’perl’), (200, ’tcl’), (400, ’lua’), (500, ’python’)]
>>> bisect.insort(scores, (300, ’ruby’))
>>> scores
[(100, ’perl’), (200, ’tcl’), (300, ’ruby’), (400, ’lua’), (500, ’python’)]





>>> from heapq import heapify, heappop, heappush
>>> data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
>>> heapify(data) # rearrange the list into heap order
>>> heappush(data, -5) # add a new entry
>>> [heappop(data) for i in range(3)] # fetch the three smallest entries
[-5, 0, 1]



>>> from decimal import *
>>> round(Decimal(’0.70’) * Decimal(’1.05’), 2)
Decimal(’0.74’)
>>> round(.70 * 1.05, 2)
0.73



>>> Decimal(’1.00’) % Decimal(’.10’)
Decimal(’0.00’)
>>> 1.00 % 0.10

0.09999999999999995
>>> sum([Decimal(’0.1’)]*10) == Decimal(’1.0’)
True
>>> sum([0.1]*10) == 1.0
False
The decimal module provides arithmetic with as much precision as needed:
>>> getcontext().prec = 36
>>> Decimal(1) / Decimal(7)
Decimal(’0.142857142857142857142857142857142857’)








