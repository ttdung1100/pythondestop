import numpy as np





print(np.__version__)
np.show_config()

Z = np.zeros(10)
print(Z)


python -c "import numpy; numpy.info(numpy.add)"




Z = np.zeros(10)
Z[4] = 1
print(Z)


Z = np.arange(10,50)
print(Z)


Z = np.arange(50)
Z = Z[::-1]


Z = np.arange(9).reshape(3,3)
print(Z)



nz = np.nonzero([1,2,0,0,4,0])
print(nz)

Z = np.eye(3)
print(Z)



Z = np.random.random((3,3,3))
print(Z)


Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)


Z = np.random.random(30)
m = Z.mean()
print(m)



Z = np.ones((10,10))
Z[1:-1,1:-1] = 0



0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1



Z = np.diag(1+np.arange(4),k=-1)
print(Z)





Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)


print(np.unravel_index(100,(6,7,8)))




Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)




Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)






color = np.dtype([("r", np.ubyte, 1),
("g", np.ubyte, 1),
("b", np.ubyte, 1),
("a", np.ubyte, 1)])




Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)





# Author: Evgeni Burovski
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1




# Author: Jake VanderPlas
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))



Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z





np.array(0) // np.array(0)
np.array(0) // np.array(0.)
np.array(0) / np.array(0)
np.array(0) / np.array(0.)



# Author: Charles R Harris
Z = np.random.uniform(-10,+10,10)
print (np.trunc(Z + np.copysign(0.5, Z)))


Z = np.random.uniform(0,10,10)
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))




Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)



def generate():
for x in xrange(10):
yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)



Z = np.linspace(0,1,12,endpoint=True)[1:-1]
print(Z)



Z = np.random.random(10)
Z.sort()
print(Z)




# Author: Evgeni Burovski
Z = np.arange(10)
np.add.reduce(Z)



A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)


Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1



Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)


Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)


Z = np.zeros((10,10), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
np.linspace(0,1,10))
print(Z)


# Author: Evgeni Burovski
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))


for dtype in [np.int8, np.int32, np.int64]:
print(np.iinfo(dtype).min)
print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
print(np.finfo(dtype).min)
print(np.finfo(dtype).max)
print(np.finfo(dtype).eps)




np.set_printoptions(threshold=np.nan)
Z = np.zeros((25,25))
print(Z)



Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])


Z = np.zeros(10, [ ('position', [ ('x', float, 1),
('y', float, 1)]),
('color', [ ('r', float, 1),
('g', float, 1),
('b', float, 1)])])
print(Z)


Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)
# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial
Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)


Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)


# File content:
# -------------
1,2,3,4,5
6,,,7,8
,,9,10,11
# -------------
Z = np.genfromtxt("missing.dat", delimiter=",")



Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
print(index, value)
for index in np.ndindex(Z.shape):
print(index, Z[index])





X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)


# Author: Divakar
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)



# Author: Warren Weckesser
X = np.random.rand(5, 10)
# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)
# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)


# Author: Steve Tjoa
Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])



# Author: Warren Weckesser
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())


Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)




class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
if obj is None:
    return
self.info = getattr(obj, 'name', "no name")
Z = NamedArray(np.arange(10), "range_10")
print (Z.name)



# Author: Brett Olsen
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)


# Author: Alan G Isaac
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)





# Author: Nadav Horesh
w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
n = len(np.unique(F))
print(np.unique(I))


A = np.random.randint(0,10,(3,4,3,4))
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)


# Author: Jaime Fernández del Río
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)





# Author: Mathieu Blondel
# Slow version
np.diag(np.dot(A, B))
# Fast version
np.sum(A * B.T, axis=1)
# Faster version
np.einsum("ij,ji->i", A, B)




# Author: Warren Weckesser
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)


A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])





# Author: Eelco Hoogendoorn
A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)




# Author: Nicolas P. Rougier
faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)


# 65. Given an array C that is a bincount, how to produce an array A such that 
# np.bincount(A) == C? 
# Author: Jaime Fernández del Río
C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)




# Author: Jaime Fernández del Río
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))


# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)



# Author: Nathaniel J. Smith
Z = np.random.randint(0,2,100)
np.logical_not(arr, out=arr)
Z = np.random.uniform(-1.0,1.0,100)
np.negative(arr, out=arr)



def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))
P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))




# Author: Italmassov Kuanysh
# based on distance function from previous question
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print np.array([distance(P0,P1,p_i) for p_i in p])







# Author: Nicolas Rougier
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill = 0
position = (1,1)
R = np.ones(shape, dtype=Z.dtype)*fill
P = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)
R_start = np.zeros((len(shape),)).astype(int)
R_stop = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop = (P+Rs//2)+Rs%2
R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()
r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)



# Author: Stefan van der Walt
Z = np.arange(1,15,dtype=uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)




 #Author: Stefan van der Walt
Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)






Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())



# Author: Chris Barker
Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)




# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices
class Symetric(np.ndarray):
    def __setitem__(self, (i,j), value):
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)
def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)
S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)



# Author: Stefan van der Walt
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)
# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.


# Author: Robert Kern
Z = np.ones(16,16)
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
np.arange(0, Z.shape[1], k), axis=1)





# Author: Nicolas Rougier
def iterate(Z):
    # Count neighbours
N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
Z[1:-1,0:-2] + Z[1:-1,2:] +
Z[2: ,0:-2] + Z[2: ,1:-1] + Z[2: ,2:])
# Apply rules
birth = (N==3) & (Z[1:-1,1:-1]==0)
survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
Z[...] = 0
Z[1:-1,1:-1][birth | survive] = 1
    return Z
Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)





Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
# Slow
print (Z[np.argsort(Z)[-n:]])
# Fast
print (Z[np.argpartition(-Z,n)[:n]])




# Author: Stefan Van der Walt
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)
    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T
    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]
    return ix
print (cartesian(([1, 2, 3], [4, 5], [6, 7])))



Z = np.array([("Hello", 2.5, 3),
("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
names='col1, col2, col3',
formats = 'S8, f8, i8')




#Author: Ryan G.
x = np.random.rand(5e7)
%timeit np.power(x,3)
1 loops, best of 3: 574 ms per loop
%timeit x*x*x
1 loops, best of 3: 429 ms per loop
%timeit np.einsum('i,i,i->i',x,x,x)
1 loops, best of 3: 244 ms per loop


# Author: Gabe Schwartz
A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))
C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows)




# Author: Robert Kern
Z = np.random.randint(0,5,(10,3))
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(Z)
print(U)



# Author: Warren Weckesser
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])
# Author: Daniel T. McDonald
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))



# Author: Jaime Fernández del Río
Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)






# Author: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/
np.einsum('i->', A) # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B) # np.inner(A, B)
np.einsum('i,j', A, B) # np.outer(A, B)





# Author: Bas Swinckels
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)
    dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
    r = np.zeros_like(x)
    r[1:] = np.cumsum(dr) # integrate path
    r_int = np.linspace(0, r.max(), 200) # regular spaced path
    x_int = np.interp(r_int, r, x) # integrate path
    y_int = np.interp(r_int, r, y)



# Author: Evgeni Burovski
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
[2.0, 0.0, 1.0, 1.0],
[1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])














