#Data Mining with Python



from enum import Enum
class Grade ( Enum ):
good = 1
bad = 2
ok = 3

def polynomial (a, b, c):
    return lambda x: a*x **2 + b*x + c
f = polynomial (3, -2, -2)
f(3)


import matplotlib . pyplot as plt
def plot_dirac ( location , *args , ** kwargs ):
    print ( args )
    print ( kwargs )
plt . plot ([ location , location ], [0, 1], *args , ** kwargs )
plot_dirac (2)
plt. hold ( True )
plot_dirac (3, linewidth =3)
plot_dirac (-2, ’r--’)
plt. axis ((-4 , 4, 0, 2))
plt. show ()




def polynomial (* args ):
    expons = range ( len( args ))[:: -1]
    return lambda x: sum ([ coef *x** expon for coef , expon in zip(args , expons )])
    f = polynomial (3, -2, -2)
    f(3) # Returned result is 19
    f = polynomial ( -2)
    f(3) # Returned result is -2




class WordsString (str ):
    def __call__ (self , index ):
        return self . split ()[ index ]

import os

from os import listdir

import nltk . tokenize


import numpy as np
import matplotlib . pyplot as plt
import networkx as nx
import pandas as pd
import statsmodels . api as sm
import statsmodels . formula . api as smf

import imager .io. jpg


from . import categorize
from .. io import jpg

from . import io



from pylab import *
t = linspace (0, 10, 1000)
plot (t, sin (2 * pi * 3 * t))
show ()



from numpy import linspace , pi , sin
from matplotlib . pyplot import plot , show
t = linspace (0, 10, 1000)
plot (t, sin (2 * pi * 3 * t))
show ()

import numpy as np
import matplotlib . pyplot as plt
t = np. linspace (0, 10, 1000)
plt. plot (t, np.sin (2 * np.pi * 3 * t))
plt. show ()


try :
import ConfigParser as configparser
except ImportError :
import configparser
try :
from cStringIO import StringIO
except ImportError :
try :
from StringIO import StringIO
except ImportError :
from io import StringIO
try:
import cPickle as pickle
except ImportError :
import pickle

try :
import cPickle as pickle
except ImportError :
import pickle


import json , numpy
json . dumps ( numpy . float32 (1.23))



from numpy import max , min
def mean_diff (a):
""" Compute the mean difference in a sequence .
Parameters
----------
a : array_like
"""
    return float (( max(a) - min(a)) / ( len(a) - 1))
def test_mean_diff ():
    assert mean_diff ([1. , 7., 3., 2., 5.]) == 1.5
    assert mean_diff ([7 , 3, 2, 1, 5]) == 1.5



def mean (x):
    return float (sum(x ))/ len(x)
    import numpy as np
def test_mean ():
    assert np. isnan ( mean ([]))
    assert mean ([4.2]) == 4.2
    assert mean ([1 , 4.3 , 4]) == 3.1

import numpy as np
def mean (x):
""" Compute mean of list of numbers .
Examples
>>> np. isnan ( mean ([]))
True
>>> mean ([4.2])
4.2
>>> mean ([1 , 4.3 , 4])
3.1
"""
try :
return float ( sum(x ))/ len(x)
except ZeroDivisionError :
return np. nan

setup .py # your distutils / setuptools Python package metadata
mypkg /
__init__ .py
appmodule .py
...
tests /
test_app .py
...



setup .py # your distutils / setuptools Python package metadata
mypkg /
__init__ .py
appmodule .py
...
test /
test_app .py
...


def function1 ():
result = []
for x in range (10):
for y in range (5):
if x * y > 10:
result . append ((x, y))
function1 . name = "For loop version "
def function2 ():
result = [(x, y) for x in range (10) for y in range (5) if x * y > 10]
function2 . name = " List comprehension version "
import timeit
for func in [ function1 , function2 ]:
print (" {:26} = {:5.2 f}". format ( func .name , timeit . timeit ( func )))





$ python timeit_example.py
For loop version = 10.14
List comprehension version = 8.67


$ pypy timeit_example.py
For loop version = 2.26
List comprehension version = 1.93



class WeightedText ( object ):
def __init__ (self , text , weight =1.0):
self . text = text
self . _weight = weight
@property
def weight ( self ):
return self . _weight


@weight . setter
def weight (self , value ):
if value < 0:
# Ensure weight is non - negative
value = 0.0
self . _weight = float ( value )
text = WeightedText (’Hello ’)
text . weight # = 1.0
text . weight = -10 # calls function with @weight . setter
text . weight # = 0.0


def main ():
# Actual script code goes here
if __name__ == ’__main__ ’:
main ()



/mymodule
__main__.py
__init__.py
mysubmodule.py



import sys
def main ( args ):
if len( args ) == 2: # The first value in args is the program name
print ( args [1])
else :
sys . exit (2)
if __name__ == ’__main__ ’:
main ( sys. argv )


from __future__ import print_function


def main ():
print (1 + "1")
if __name__ == ’__main__ ’:
main ()





import matplotlib . pyplot as plt
import random
from collections import deque
def random_walker ():
x = 0
while True :
yield x
if random . random () > 0.5:
x += 1
else :
x -= 1
def autoregressor (it ):
x = 0
while True :
x = 0.79 * x + 0.2 * it. next ()
yield x



class Animator ():
def __init__ (self , window_width =100):
self . random_walk = random_walker ()
self . autoregression = autoregressor ( self . random_walk )
self . data = deque ( maxlen = window_width )
self . fig = plt. figure ()
self .ax = self . fig. add_subplot (1, 1, 1)
self .line , = self .ax. plot ([] , [], linewidth =5, alpha =0.5)
def animate_step ( self ):
self . data . append ( self . autoregression . next ())
N = len( self . data )
self . line . set_data ( range (N), self . data )
self .ax. set_xlim (0, N)
abs_max = max (abs (min ( self . data )), abs (max ( self . data )))
abs_max = max ( abs_max , 1)
self .ax. set_ylim (- abs_max , abs_max )
def animate_infinitely ( self ):
while True :
self . animate_step ()
plt . pause (0.01)
animator = Animator (500)
animator . animate_infinitely ()


import matplotlib . pyplot as plt
from matplotlib import animation
import random
from collections import deque
def random_walker ():
x = 0
while True :
yield x
if random . random () > 0.5:
x += 1
else :
x -= 1
def autoregressor (it ):
x = 0
while True :
x = 0.70 * x + 0.2 * it. next ()
yield x



class Animator ():
def __init__ (self , window_width =100):
self . random_walk = random_walker ()
self . autoregression = autoregressor ( self . random_walk )
self . data = deque ( maxlen = window_width )
self . fig = plt. figure ()
self .ax = self . fig. add_subplot (1, 1, 1)
self .line , = self .ax. plot ([] , [], linewidth =5, alpha =0.5)
self .ax. set_xlim (0, window_width )
self .ax. set_ylim (-80, 80)
def init ( self ):
return self .line ,
def step (self , n):
self . data . append ( self . autoregression . next ())
self . line . set_data ( range ( len( self . data )), self . data )
return self .line ,
animator = Animator (500)
anim = animation . FuncAnimation ( animator .fig , animator .step ,
init_func = animator .init ,
interval =0, blit = True )
plt. show ()


from flask import Flask
import matplotlib . pyplot as plt



from StringIO import StringIO
app = Flask ( __name__ )
def plot_example ():
plt . plot ([1 , 2, 5, 2, 7, 2, 1, 3])
sio = StringIO ()
plt . savefig (sio )
return sio . getvalue (). encode (’base64 ’). strip ()
@app . route (’/’)
def index ():
return """ <html ><body >
<img src =" data : image / png; base64 ,{0}" >
</body ></ html > """ . format ( plot_example ())
if __name__ == ’__main__ ’:
app .run ()



import cherrypy
import vincent
# Vega Scaffold modified from https :// github .com/ trifacta / vega / wiki / Runtime
HTML = """
<html >
<head >
<script src =" http :// trifacta . github .io/ vega / lib/d3.v3. min.js"></ script >



<script src =" http :// trifacta . github .io/ vega / lib/d3. geo. projection . min.js"></ script >
<script src =" http :// trifacta . github .io/ vega / vega .js"></ script >
</head >
<body ><div id =" vis "></div ></ body >
<script type =" text / javascript ">
// parse a spec and create a visualization view
function parse ( spec ) {
vg. parse . spec (spec , function ( chart ) { chart ({ el :"# vis "}). update (); });
}
parse ("/ plot ");
</ script >
</html >
"""
class VegaExample :
@cherrypy . expose
def index ( self ):
return HTML
@cherrypy . expose
def plot ( self ):
bar = vincent .Bar ([2 , 4, 2, 6, 3])
return bar. to_json ()
cherrypy . quickstart ( VegaExample ())


import plotly
import numpy as np
plotly . tools . set_credentials_file ( username =’fnielsen ’,
api_key =’The API key goes here ’,
stream_ids =[ ’a stream id ’,
’another stream id ’])
x = np. linspace (0, 10)
y = np.sin(x)
graph_url = plotly . plotly . plot ([x, y])



import matplotlib . pyplot as plt , mpld3 , cherrypy
class Mpld3Example ():
@cherrypy . expose
def index ( self ):
fig = plt . figure ()
plt . plot ([1 , 2, 3, 2, 3, 1, 3])
return mpld3 . fig_to_html ( fig)
cherrypy . quickstart ( Mpld3Example ())



from bokeh . plotting import *
import numpy as np
output_file (’rand . html ’)
line (np. random . rand (10) , np. random . rand (10))
show ()
import webbrowser
webbrowser . open (os. path . join (os. getcwd (), ’rand . html ’))



import pandas as pd
A = pd. DataFrame ([[4 , 5, ’yes ’], [6.5 , 7, ’no ’], [8, 9, ’ok ’]],
index =[2 , 3, 6], columns =[ ’a’, ’b’, ’c’])


import pandas , numpy




import sympy
x = sympy . symbols (’x’)
f = sympy .sin (2* sympy .pi*x) * sympy .exp(-x **2)
df = f. diff ()
ddf = df. diff ()
p = sympy . plot (f, df , ddf , xlim =(-3, 3),
adaptive =False , nb_of_points =1000)
p [0]. line_color = (1, 0, 0)
p [1]. line_color = (0, 1, 0)
p [2]. line_color = (0, 0, 1)
p. show ()
ddf. evalf ( subs ={ ’x’: 0.75})


import requests
from lxml import etree
url = ’http :// www .w3.org /TR /2009/ REC -skos - reference -20090818/ ’
response = requests .get(url)
tree = etree . HTML ( response . content )


import re
import requests
url = ’http :// www .w3.org /TR /2009/ REC -skos - reference -20090818/ ’
response = requests .get(url)
editors = re. findall (’Editors :(.*?) </ dl >’, response .text ,
flags =re. UNICODE | re. DOTALL )[0]
editor_list = re. findall (’<a .*? >(.+?) </a>’, editors )

from bs4 import BeautifulSoup
import re
import requests
url = ’http :// www .w3.org /TR /2009/ REC -skos - reference -20090818/ ’
response = requests .get(url)
soup = BeautifulSoup ( response . content )
names = soup . find_all (’dt ’, text =re. compile (’Editors ?: ’))[0]. find_next (’dd ’). text




import matplotlib . pyplot as plt
import networkx as nx
import pandas as pd
import requests
from StringIO import StringIO
URL = (’http :// www . nature .com / ncomms /2014/140617/ ’
’ncomms5022 / extref / ncomms5022 -s2. zip ’)


data = pd. read_excel ( StringIO ( requests . get( URL ). content ),
’ Supplementary Data 1’, skiprows =3, header =4)
disease_graph = nx. DiGraph ()
disease_graph . add_edges_from ( data [[ ’Code ’, ’Code .1 ’]]. itertuples ( index = False ))
nx. draw (nx. ego_graph ( disease_graph , ’K35 ’))
plt. show ()



from lazy import lazy
from numpy import matrix
from scipy . linalg import svd
class Matrix ( matrix ):
@lazy
def U( self ):
U, s, Vh = svd (self , full_matrices = False )
return U
import numpy . random as npr
# Initialize with a large random matrix
X = Matrix (npr. random ((2000 , 300)))
X.U # Slow - first call computes the value and stores it
X.U # Fast - second call uses the cached value and is much faster



class Matrix ( matrix ):
def __init__ (self , *args , ** kwargs ):
matrix . __init__ (self , *args , ** kwargs )
self ._U , self ._s , self . _Vh = svd(self , full_matrices = False )
@property
def U( self ):
return self ._U


class Matrix ( matrix ):
def __init__ (self , *args , ** kwargs ):
matrix . __init__ (self , *args , ** kwargs )
self ._U , self ._s , self . _Vh = None , None , None
def svd ( self ):
self ._U , self ._s , self . _Vh = svd( self )
return self ._U , self ._s , self . _Vh
@property
def U( self ):
if self ._U is None :


self . svd ()
return self ._U
@property
def s( self ):
if self ._s is None :
self . svd ()
return self ._s
@property
def Vh( self ):
if self ._Vh is None :
self . svd ()
return self . _Vh


import logging
from logging import NullHandler
log = logging . getLogger ( __name__ )
# Avoid "No handlers " message if no logger
log. addHandler ( NullHandler ())
class Matrix ( object ):
""" Numerical matrix ."""
def __init__ (self , obj ):
""" Initialize matrix object ."""
log . debug (" Constructing matrix object ")
self . _matrix = obj
def __getitem__ (self , indices ):
""" Get element in matrix .
Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4]])
>>> m[0, 1]
2
"""
return self . _matrix [ indices [0]][ indices [1]]
def __setitem__ (self , indices , value ):
""" Set element in matrix .

Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4]])
>>> m[0, 1]
2
>>> m[0, 1] = 5
>>> m[0, 1]
5
"""
self . _matrix [ indices [0]][ indices [1]] = value
@property
def shape ( self ):
""" Return shape of matrix .
Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4], [5, 6]])
>>> m. shape
(3, 2)
"""
rows = len( self . _matrix )
if rows == 0:
rows = 1
columns = 0
else :
columns = len ( self . _matrix [0])
return (rows , columns )
def __abs__ ( self ):
""" Return the absolute value .
Examples
--------
>>> m = Matrix ([[1 , -1]])
>>> m_abs = abs(m)
>>> m_abs [0, 1]
1
"""
result = Matrix ([[ abs( element ) for element in row]
for row in self . _matrix ])
return result
def __add__ (self , other ):
""" Add number to matrix .
Parameters
----------
other : integer or Matrix
Returns

m : Matrix
Matrix of the same size as the original matrix
Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4]])
>>> m = m + 1
>>> m[0, 0]
2
>>> m = m + Matrix ([[5 , 6], [7, 8]])
>>> m[0, 0]
7
"""
if isinstance (other , int) or isinstance (other , float ):
result = [[ element + other for element in row]
for row in self . _matrix ]
elif isinstance (other , Matrix ):
result = [[ self [m, n] + other [m, n]
for n in range ( self . shape [1])]
for m in range ( self . shape [0])]
else :
raise TypeError
return Matrix ( result )
def __mul__ (self , other ):
""" Multiply number to matrix .
Parameters
----------
other : integer , float
Returns
-------
m : Matrix
Matrix with multiplication result
Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4]])
>>> m = m * 2
>>> m[0, 0]
2
"""
if isinstance (other , int) or isinstance (other , float ):
result = [[ element * other for element in row]
for row in self . _matrix ]
else :
raise TypeError
return Matrix ( result )
def __pow__ (self , other ):
""" Power of element with ‘other ‘ as the exponent


Parameters
----------
other : integer , float
Returns
-------
m : Matrix
Matrix with multiplication result
Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4]])
>>> m = m ** 3
>>> m[0, 1]
8
"""
if isinstance (other , int) or isinstance (other , float ):
result = [[ element ** other for element in row]
for row in self . _matrix ]
else :
raise TypeError
return Matrix ( result )
def __str__ ( self ):
""" Return string representation of matrix ."""
return str( self . _matrix )
def transpose ( self ):
""" Return transposed matrix .
Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4]])
>>> m = m. transpose ()
>>> m[0, 1]
3
"""
log . debug (" Transposing ")
# list necessary for Python 3 where zip is a generator
return Matrix ( list ( zip (* self . _matrix )))
@property
def T( self ):
""" Transposed of matrix .
Returns
-------
m : Matrix
Copy of matrix
Examples
--------
>>> m = Matrix ([[1 , 2], [3, 4]])
>>> m = m.T

>>> m[0, 1]
3
"""
log . debug (" Calling transpose ()")
return self . transpose ()



import pandas as pd
pima_tr = pd. read_csv (’pima .tr.csv ’, index_col =0)
pima_te = pd. read_csv (’pima .te.csv ’, index_col =0)

url = (’http :// ftp .ics .uci .edu /pub / machine - learning - databases /’
’pima - indians - diabetes /pima - indians - diabetes . data ’)
pima = pd. read_csv (url , names =[ ’npreg ’, ’glu ’, ’bp ’, ’skin ’,
’ins ’, ’bmi ’, ’ped ’, ’age ’, ’type ’])




import seaborn as sns
import matplotlib . pyplot as plt
sns. corrplot ( pima_tr )
plt. show ()

import statsmodels . api as sm
import statsmodels . formula . api as smf
model = smf.glm(’type ~ npreg + glu + bp + skin + bmi + ped + age ’,
data = pima_tr , family =sm. families . Binomial ()). fit ()
print ( model . summary ())


class NoClassifier ():
""" Classifier that predict all data as "No ". """
def predict (self , x):
return pd. Series (["No"] * x. shape [0])

def accuracy (truth , predicted ):
if len( truth ) != len( predicted ):
raise Exception (" Wrong sizes ...")
total = len( truth )
if total == 0:
return 0
hits = len( filter ( lambda (x, y): x == y, zip(truth , predicted )))
return float ( hits )/ total



from scipy . linalg import pinv
from numpy import asarray , hstack , mat , ones , where
class LinearClassifier ():
""" y = X*b and b = pinv (X) * y """
def __init__ ( self ):
self . _parameters = None
def from_labels (self , y):
return mat( where (y=="No", -1, 1)). T

def to_labels (self , y):
return pd. Series ( asarray ( where (y <0, "No", " Yes" )). flatten ())
def fit (self , x, y):
intercept = ones ((x. shape [0] , 1))
self . _parameters = pinv ( hstack (( mat(x), intercept ))) * self . from_labels (y)
def predict (self , x):
intercept = ones ((x. shape [0] , 1))
y_estimated = hstack (( mat(x), intercept )) * self . _parameters
return self . to_labels ( y_estimated )




from db import DemoDB
import networkx as nx
import numpy as np
import matplotlib . pyplot as plt

# Load Chinook database
db = DemoDB ()
# Construct graph
graph = nx. MultiDiGraph ()
for table in db. tables :
graph . add_node ( table .name , number_of_rows =len( table .all ()))
for key in table . foreign_keys :
graph . add_edge ( table .name , key. table )
# Position and size of nodes
pos = nx. layout . fruchterman_reingold_layout ( graph )
sizes = 100 + 50 * np. sqrt ([ attrs [’ number_of_rows ’]
for node , attrs in graph . nodes ( data = True )])
# Draw the components of the graph
nx. draw_networkx_edges (graph , pos=pos , node_color =’k’, alpha =0.1 , width =3)
nx. draw_networkx_nodes (graph , pos=pos , node_color =’k’, alpha =0.2 , linewidths =0. ,
node_size = sizes )
nx. draw_networkx_labels (graph , pos=pos , font_color =’k’, font_size =8)
plt. show ()





try :
import cPickle as pickle
except ImportError :
import pickle
from nltk . corpus import brown
from nltk . classify import apply_features , NaiveBayesClassifier
from nltk . tokenize import word_tokenize


unique_words = set ([ word . lower () for word in brown . words ()])
news_sentences = brown . sents ( categories =’news ’)
other_sentences = brown . sents ( categories =set( brown . categories ()) - set ([ ’news ’]))




def word_features ( sentence ):
features = { word : False for word in unique_words }
for word in sentence :
if word . isalpha ():
features [ word . lower ()] = True
return features
featuresets = apply_features ( word_features , (
[( sent , ’news ’) for sent in news_sentences ] +
[( sent , ’other ’) for sent in other_sentences ]))


classifier = NaiveBayesClassifier . train ( featuresets )
classifier = NaiveBayesClassifier . train ( featuresets )
classifier = NaiveBayesClassifier . train ( featuresets )



import ijson
from StringIO import StringIO
json_string = """ [
{" id ": 1, " content ": " hello "},
{" id ": 2, " content ": " world "}]
"""
sio = StringIO ( json_string )
objects = ijson . items (sio , ’item ’)
for obj in objects :
print (obj)



import collections
import gzip
import ijson
import os. path
filename = os. path . expanduser (’~/ data / wikidata /20140721. json .gz ’)
id_company = 783794 # https :// www. wikidata .org/ wiki / Q783794
def get_instance_of_ids ( subject ):
""" Return numeric ids for ’instance of ’ (P31 ) object for subject ."""
ids = []
if ’claims ’ in subject and ’P31 ’ in subject [’claims ’]:
for statement in subject [’claims ’][ ’P31 ’]:
try :
id = statement [’mainsnak ’][ ’datavalue ’][ ’value ’][ ’numeric -id ’]
ids. append (id)
except KeyError :
pass
return ids
objects = ijson . items ( gzip . open ( filename ), ’item ’)
labels = collections . defaultdict ( str)
for obj in objects :
for language in [’ro ’, ’de ’, ’en ’]:
if ’labels ’ in obj and language in obj[’labels ’]:
labels [ obj[’id ’]] = obj[’labels ’][ language ][ ’value ’]
break
ids = get_instance_of_ids (obj)
if id_company in ids:
print ( labels [obj[’id ’]])


