# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:03:59 2025

@author: Radha Sharma
"""


What is numpy?
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

At the core of the NumPy package, is the ndarray object. This encapsulates n-dimensional arrays of homogeneous data types

Numpy Arrays Vs Python Sequences
NumPy arrays have a fixed size at creation, unlike Python lists (which can grow dynamically). Changing the size of an ndarray will create a new array and delete the original.

The elements in a NumPy array are all required to be of the same data type, and thus will be the same size in memory.

NumPy arrays facilitate advanced mathematical and other types of operations on large numbers of data. Typically, such operations are executed more efficiently and with less code than is possible using Python’s built-in sequences.

A growing plethora of scientific and mathematical Python-based packages are using NumPy arrays; though these typically support Python-sequence input, they convert such input to NumPy arrays prior to processing, and they often output NumPy arrays.

Creating Numpy Arrays

# np.array
import numpy as np

a = np.array([1,2,3])
print(a)
     
[1 2 3]

# 2D and 3D
b = np.array([[1,2,3],[4,5,6]])
print(b)
     
[[1 2 3]
 [4 5 6]]

c = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(c)
     
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

# dtype
np.array([1,2,3],dtype=float)
     
array([1., 2., 3.])

# np.arange
np.arange(1,11,2)
     
array([1, 3, 5, 7, 9])

# with reshape
np.arange(16).reshape(2,2,2,2)
     
array([[[[ 0,  1],
         [ 2,  3]],

        [[ 4,  5],
         [ 6,  7]]],


       [[[ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15]]]])

# np.ones and np.zeros
np.ones((3,4))
     
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])

np.zeros((3,4))
     
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])

# np.random
np.random.random((3,4))
     
array([[0.85721156, 0.31248316, 0.08807828, 0.35230774],
       [0.96813914, 0.44681708, 0.56396358, 0.53020065],
       [0.03277116, 0.28543753, 0.09521082, 0.87967034]])

# np.linspace
np.linspace(-10,10,10,dtype=int)
     
array([-10,  -8,  -6,  -4,  -2,   1,   3,   5,   7,  10])

# np.identity
np.identity(3)
     
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
Array Attributes

a1 = np.arange(10,dtype=np.int32)
a2 = np.arange(12,dtype=float).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

a3
     
array([[[0, 1],
        [2, 3]],

       [[4, 5],
        [6, 7]]])

# ndim
a3.ndim
     
3

# shape
print(a3.shape)
a3
     
(2, 2, 2)
array([[[0, 1],
        [2, 3]],

       [[4, 5],
        [6, 7]]])

# size
print(a2.size)
a2
     
12
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.]])

# itemsize
a3.itemsize
     
8

# dtype
print(a1.dtype)
print(a2.dtype)
print(a3.dtype)


     
int32
float64
int64
Changing Datatype

# astype
a3.astype(np.int32)
     
array([[[0, 1],
        [2, 3]],

       [[4, 5],
        [6, 7]]], dtype=int32)
Array Operations

a1 = np.arange(12).reshape(3,4)
a2 = np.arange(12,24).reshape(3,4)

a2
     
array([[12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]])

# scalar operations

# arithmetic
a1 ** 2
     
array([[  0,   1,   4,   9],
       [ 16,  25,  36,  49],
       [ 64,  81, 100, 121]])

# relational
a2 == 15
     
array([[False, False, False,  True],
       [False, False, False, False],
       [False, False, False, False]])

# vector operations
# arithmetic
a1 ** a2
     
array([[                   0,                    1,                16384,
                    14348907],
       [          4294967296,         762939453125,      101559956668416,
           11398895185373143],
       [ 1152921504606846976, -1261475310744950487,  1864712049423024128,
         6839173302027254275]])
Array Functions

a1 = np.random.random((3,3))
a1 = np.round(a1*100)
a1
     
array([[43., 28., 71.],
       [27., 93., 36.],
       [31., 18.,  7.]])

# max/min/sum/prod
# 0 -> col and 1 -> row
np.prod(a1,axis=0)
     
array([35991., 46872., 17892.])

# mean/median/std/var
np.var(a1,axis=1)
     
array([317.55555556, 854.        ,  96.22222222])

# trigonomoetric functions
np.sin(a1)
     
array([[-0.83177474,  0.27090579,  0.95105465],
       [ 0.95637593, -0.94828214, -0.99177885],
       [-0.40403765, -0.75098725,  0.6569866 ]])

# dot product
a2 = np.arange(12).reshape(3,4)
a3 = np.arange(12,24).reshape(4,3)

np.dot(a2,a3)
     
array([[114, 120, 126],
       [378, 400, 422],
       [642, 680, 718]])

# log and exponents
np.exp(a1)
     
array([[4.72783947e+18, 1.44625706e+12, 6.83767123e+30],
       [5.32048241e+11, 2.45124554e+40, 4.31123155e+15],
       [2.90488497e+13, 6.56599691e+07, 1.09663316e+03]])

# round/floor/ceil

np.ceil(np.random.random((2,3))*100)
     
array([[48.,  4.,  6.],
       [ 3., 18., 82.]])
Indexing and Slicing

a1 = np.arange(10)
a2 = np.arange(12).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

a3
     
array([[[0, 1],
        [2, 3]],

       [[4, 5],
        [6, 7]]])

a1
     
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

a2
     
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

a2[1,0]
     
4

a3
     
array([[[0, 1],
        [2, 3]],

       [[4, 5],
        [6, 7]]])

a3[1,0,1]
     
5

a3[1,1,0]
     
6

a1
     
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

a1[2:5:2]
     
array([2, 4])

a2
     
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

a2[0:2,1::2]
     
array([[1, 3],
       [5, 7]])

a2[::2,1::2]
     
array([[ 1,  3],
       [ 9, 11]])

a2[1,::3]
     
array([4, 7])

a2[0,:]
     
array([0, 1, 2, 3])

a2[:,2]
     
array([ 2,  6, 10])

a2[1:,1:3]
     
array([[ 5,  6],
       [ 9, 10]])

a3 = np.arange(27).reshape(3,3,3)
a3
     
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])

a3[::2,0,::2]
     
array([[ 0,  2],
       [18, 20]])

a3[2,1:,1:]
     
array([[22, 23],
       [25, 26]])

a3[0,1,:]
     
array([3, 4, 5])


     


     


     


     


     
Iterating

a1

for i in a1:
  print(i)
     
0
1
2
3
4
5
6
7
8
9

a2
     
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

for i in a2:
  print(i)
     
[0 1 2 3]
[4 5 6 7]
[ 8  9 10 11]

a3
     
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])

for i in a3:
  print(i)
     
[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[ 9 10 11]
 [12 13 14]
 [15 16 17]]
[[18 19 20]
 [21 22 23]
 [24 25 26]]

for i in np.nditer(a3):
  print(i)
     
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
Reshaping

# reshape
     

# Transpose
np.transpose(a2)
a2.T
     
array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])

# ravel
a3.ravel()
     
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
Stacking

# horizontal stacking
a4 = np.arange(12).reshape(3,4)
a5 = np.arange(12,24).reshape(3,4)
a5
     
array([[12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]])

np.hstack((a4,a5))
     
array([[ 0,  1,  2,  3, 12, 13, 14, 15],
       [ 4,  5,  6,  7, 16, 17, 18, 19],
       [ 8,  9, 10, 11, 20, 21, 22, 23]])

# Vertical stacking
np.vstack((a4,a5))
     
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]])
Splitting

# horizontal splitting
a4
     
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

np.hsplit(a4,5)
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-227-59485ca7f23c> in <module>
----> 1 np.hsplit(a4,5)

<__array_function__ internals> in hsplit(*args, **kwargs)

/usr/local/lib/python3.8/dist-packages/numpy/lib/shape_base.py in hsplit(ary, indices_or_sections)
    938         raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    939     if ary.ndim > 1:
--> 940         return split(ary, indices_or_sections, 1)
    941     else:
    942         return split(ary, indices_or_sections, 0)

<__array_function__ internals> in split(*args, **kwargs)

/usr/local/lib/python3.8/dist-packages/numpy/lib/shape_base.py in split(ary, indices_or_sections, axis)
    870         N = ary.shape[axis]
    871         if N % sections:
--> 872             raise ValueError(
    873                 'array split does not result in an equal division') from None
    874     return array_split(ary, indices_or_sections, axis)

ValueError: array split does not result in an equal division

# vertical splitting
     

a5
     
array([[12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]])

np.vsplit(a5,2)
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-230-5b73f701499e> in <module>
----> 1 np.vsplit(a5,2)

<__array_function__ internals> in vsplit(*args, **kwargs)

/usr/local/lib/python3.8/dist-packages/numpy/lib/shape_base.py in vsplit(ary, indices_or_sections)
    989     if _nx.ndim(ary) < 2:
    990         raise ValueError('vsplit only works on arrays of 2 or more dimensions')
--> 991     return split(ary, indices_or_sections, 0)
    992 
    993 

<__array_function__ internals> in split(*args, **kwargs)

/usr/local/lib/python3.8/dist-packages/numpy/lib/shape_base.py in split(ary, indices_or_sections, axis)
    870         N = ary.shape[axis]
    871         if N % sections:
--> 872             raise ValueError(
    873                 'array split does not result in an equal division') from None
    874     return array_split(ary, indices_or_sections, axis)

ValueError: array split does not result in an equal division


     
##################################################################################




Numpy array vs Python lists

# speed
# list
a = [i for i in range(10000000)]
b = [i for i in range(10000000,20000000)]

c = []
import time

start = time.time()
for i in range(len(a)):
  c.append(a[i] + b[i])
print(time.time()-start)
     
3.2699835300445557

# numpy
import numpy as np
a = np.arange(10000000)
b = np.arange(10000000,20000000)

start = time.time()
c = a + b
print(time.time()-start)
     
0.06481003761291504

3.26/0.06
     
54.33333333333333

# memory
a = [i for i in range(10000000)]
import sys

sys.getsizeof(a)
     
81528048

a = np.arange(10000000,dtype=np.int8)
sys.getsizeof(a)
     
10000104

# convenience
     
Advanced Indexing

# Normal Indexing and slicing

a = np.arange(24).reshape(6,4)
a
     
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]])

a[1,2]
     
5

a[1:3,1:3]
     
array([[4, 5],
       [7, 8]])

# Fancy Indexing

a[:,[0,2,3]]
     
array([[ 0,  2,  3],
       [ 4,  6,  7],
       [ 8, 10, 11],
       [12, 14, 15],
       [16, 18, 19],
       [20, 22, 23]])


     

# Boolean Indexing
a = np.random.randint(1,100,24).reshape(6,4)
a
     
array([[76, 98, 99, 39],
       [91, 46, 88, 23],
       [45,  6, 83,  1],
       [37, 43, 78, 85],
       [54, 73, 61, 53],
       [40, 93, 85, 77]])

# find all numbers greater than 50
a[a > 50]
     
array([76, 98, 99, 91, 88, 83, 78, 85, 54, 73, 61, 53, 93, 85, 77])

# find out even numbers
a[a % 2 == 0]
     
array([76, 98, 46, 88,  6, 78, 54, 40])

# find all numbers greater than 50 and are even

a[(a > 50) & (a % 2 == 0)]
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-97-0e69559201d8> in <module>
      1 # find all numbers greater than 50 and are even
      2 
----> 3 a[(a > 50) and (a % 2 == 0)]

ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

# find all numbers not divisible by 7
a[~(a % 7 == 0)]
     
array([76, 99, 39, 46, 88, 23, 45,  6, 83,  1, 37, 43, 78, 85, 54, 73, 61,
       53, 40, 93, 85])
Broadcasting
The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations.

The smaller array is “broadcast” across the larger array so that they have compatible shapes.


# same shape
a = np.arange(6).reshape(2,3)
b = np.arange(6,12).reshape(2,3)

print(a)
print(b)

print(a+b)
     
[[0 1 2]
 [3 4 5]]
[[ 6  7  8]
 [ 9 10 11]]
[[ 6  8 10]
 [12 14 16]]

# diff shape
a = np.arange(6).reshape(2,3)
b = np.arange(3).reshape(1,3)

print(a)
print(b)

print(a+b)
     
[[0 1 2]
 [3 4 5]]
[[0 1 2]]
[[0 2 4]
 [3 5 7]]
Broadcasting Rules
1. Make the two arrays have the same number of dimensions.

If the numbers of dimensions of the two arrays are different, add new dimensions with size 1 to the head of the array with the smaller dimension.
2. Make each dimension of the two arrays the same size.

If the sizes of each dimension of the two arrays do not match, dimensions with size 1 are stretched to the size of the other array.
If there is a dimension whose size is not 1 in either of the two arrays, it cannot be broadcasted, and an error is raised.


# More examples

a = np.arange(12).reshape(4,3)
b = np.arange(3)

print(a)
print(b)

print(a+b)
     
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
[0 1 2]
[[ 0  2  4]
 [ 3  5  7]
 [ 6  8 10]
 [ 9 11 13]]

a = np.arange(12).reshape(3,4)
b = np.arange(3)

print(a)
print(b)

print(a+b)
     
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[0 1 2]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-104-fa6cbb589166> in <module>
      5 print(b)
      6 
----> 7 print(a+b)

ValueError: operands could not be broadcast together with shapes (3,4) (3,) 

a = np.arange(3).reshape(1,3)
b = np.arange(3).reshape(3,1)

print(a)
print(b)

print(a+b)
     
[[0 1 2]]
[[0]
 [1]
 [2]]
[[0 1 2]
 [1 2 3]
 [2 3 4]]

a = np.arange(3).reshape(1,3)
b = np.arange(4).reshape(4,1)

print(a)
print(b)

print(a + b)
     
[[0 1 2]]
[[0]
 [1]
 [2]
 [3]]
[[0 1 2]
 [1 2 3]
 [2 3 4]
 [3 4 5]]

a = np.array([1])
# shape -> (1,1)
b = np.arange(4).reshape(2,2)
# shape -> (2,2)

print(a)
print(b)

print(a+b)
     
[1]
[[0 1]
 [2 3]]
[[1 2]
 [3 4]]

a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(4,3)

print(a)
print(b)

print(a+b)
     
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-109-c590a65467e5> in <module>
      5 print(b)
      6 
----> 7 print(a+b)

ValueError: operands could not be broadcast together with shapes (3,4) (4,3) 

a = np.arange(16).reshape(4,4)
b = np.arange(4).reshape(2,2)

print(a)
print(b)

print(a+b)
     
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
[[0 1]
 [2 3]]
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-110-57df50a0058a> in <module>
      5 print(b)
      6 
----> 7 print(a+b)

ValueError: operands could not be broadcast together with shapes (4,4) (2,2) 
Working with mathematical formulas

a = np.arange(10)
np.sin(a)
     
array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
       -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])

# sigmoid
def sigmoid(array):
  return 1/(1 + np.exp(-(array)))


a = np.arange(100)

sigmoid(a)
     
array([0.5       , 0.73105858, 0.88079708, 0.95257413, 0.98201379,
       0.99330715, 0.99752738, 0.99908895, 0.99966465, 0.99987661,
       0.9999546 , 0.9999833 , 0.99999386, 0.99999774, 0.99999917,
       0.99999969, 0.99999989, 0.99999996, 0.99999998, 0.99999999,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ])

# mean squared error

actual = np.random.randint(1,50,25)
predicted = np.random.randint(1,50,25)
     

def mse(actual,predicted):
  return np.mean((actual - predicted)**2)

mse(actual,predicted)
     
500.12

# binary cross entropy
np.mean((actual - predicted)**2)
     
500.12

actual
     
array([ 5,  3,  9,  7,  3, 36, 49, 28, 20, 40,  2, 23, 29, 18, 30, 23,  7,
       40, 15, 11, 27, 44, 32, 28, 10])
Working with missing values

# Working with missing values -> np.nan
a = np.array([1,2,3,4,np.nan,6])
a
     
array([ 1.,  2.,  3.,  4., nan,  6.])

a[~np.isnan(a)]
     
array([1., 2., 3., 4., 6.])
Plotting Graphs

# plotting a 2D plot
# x = y
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
y = x

plt.plot(x,y)
     
[<matplotlib.lines.Line2D at 0x7f6f78e18f70>]


# y = x^2
x = np.linspace(-10,10,100)
y = x**2

plt.plot(x,y)
     
[<matplotlib.lines.Line2D at 0x7f6f87acf100>]


# y = sin(x)
x = np.linspace(-10,10,100)
y = np.sin(x)

plt.plot(x,y)
     
[<matplotlib.lines.Line2D at 0x7f6f5d1d0100>]


# y = xlog(x)
x = np.linspace(-10,10,100)
y = x * np.log(x)

plt.plot(x,y)
     
<ipython-input-137-4b3958c08378>:3: RuntimeWarning: invalid value encountered in log
  y = x * np.log(x)
[<matplotlib.lines.Line2D at 0x7f6f57ab62e0>]


# sigmoid
x = np.linspace(-10,10,100)
y = 1/(1+np.exp(-x))

plt.plot(x,y)
     
[<matplotlib.lines.Line2D at 0x7f6f5401e100>]

Meshgrids

# Meshgrids
     


     
####################################################################################




np.sort
Return a sorted copy of an array.

https://numpy.org/doc/stable/reference/generated/numpy.sort.html


# code
import numpy as np
a = np.random.randint(1,100,15)
a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

b = np.random.randint(1,100,24).reshape(6,4)
b
     
array([[12, 52, 42,  6],
       [29, 18, 47, 55],
       [61, 93, 83,  9],
       [38, 63, 44, 85],
       [ 8, 87, 31, 72],
       [40, 71,  2,  7]])

np.sort(a)[::-1]
     
array([94, 92, 78, 68, 53, 50, 38, 37, 30, 28, 21, 11,  9,  5,  2])

np.sort(b,axis=0)
     
array([[ 8, 18,  2,  6],
       [12, 52, 31,  7],
       [29, 63, 42,  9],
       [38, 71, 44, 55],
       [40, 87, 47, 72],
       [61, 93, 83, 85]])
np.append
The numpy.append() appends values along the mentioned axis at the end of the array

https://numpy.org/doc/stable/reference/generated/numpy.append.html


# code
np.append(a,200)
     
array([ 11,  53,  28,  50,  38,  37,  94,  92,   5,  30,  68,   9,  78,
         2,  21, 200])

b
     
array([[12, 52, 42,  6],
       [29, 18, 47, 55],
       [61, 93, 83,  9],
       [38, 63, 44, 85],
       [ 8, 87, 31, 72],
       [40, 71,  2,  7]])

np.append(b,np.random.random((b.shape[0],1)),axis=1)
     
array([[12.        , 52.        , 42.        ,  6.        ,  0.22006275],
       [29.        , 18.        , 47.        , 55.        ,  0.81740634],
       [61.        , 93.        , 83.        ,  9.        ,  0.89146072],
       [38.        , 63.        , 44.        , 85.        ,  0.84519124],
       [ 8.        , 87.        , 31.        , 72.        ,  0.24007274],
       [40.        , 71.        ,  2.        ,  7.        ,  0.48056374]])
np.concatenate
numpy.concatenate() function concatenate a sequence of arrays along an existing axis.

https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html


# code
c = np.arange(6).reshape(2,3)
d = np.arange(6,12).reshape(2,3)

print(c)
print(d)
     
[[0 1 2]
 [3 4 5]]
[[ 6  7  8]
 [ 9 10 11]]

np.concatenate((c,d),axis=0)
     
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])

np.concatenate((c,d),axis=1)
     
array([[ 0,  1,  2,  6,  7,  8],
       [ 3,  4,  5,  9, 10, 11]])
np.unique
With the help of np.unique() method, we can get the unique values from an array given as parameter in np.unique() method.

https://numpy.org/doc/stable/reference/generated/numpy.unique.html/


# code
e = np.array([1,1,2,2,3,3,4,4,5,5,6,6])
     

np.unique(e)
     
array([1, 2, 3, 4, 5, 6])
np.expand_dims
With the help of Numpy.expand_dims() method, we can get the expanded dimensions of an array

https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html


# code
a.shape
     
(15,)

np.expand_dims(a,axis=0).shape
     
(1, 15)

np.expand_dims(a,axis=1)
     
array([[11],
       [53],
       [28],
       [50],
       [38],
       [37],
       [94],
       [92],
       [ 5],
       [30],
       [68],
       [ 9],
       [78],
       [ 2],
       [21]])
np.where
The numpy.where() function returns the indices of elements in an input array where the given condition is satisfied.

https://numpy.org/doc/stable/reference/generated/numpy.where.html


a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

# find all indices with value greater than 50
np.where(a>50)
     
(array([ 1,  6,  7, 10, 12]),)

# replace all values > 50 with 0
np.where(a>50,0,a)
     
array([11,  0, 28, 50, 38, 37,  0,  0,  5, 30,  0,  9,  0,  2, 21])

np.where(a%2 == 0,0,a)
     
array([11, 53,  0,  0,  0, 37,  0,  0,  5,  0,  0,  9,  0,  0, 21])
np.argmax
The numpy.argmax() function returns indices of the max element of the array in a particular axis.

https://numpy.org/doc/stable/reference/generated/numpy.argmax.html


# code
a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

np.argmax(a)
     
6

b
     
array([[12, 52, 42,  6],
       [29, 18, 47, 55],
       [61, 93, 83,  9],
       [38, 63, 44, 85],
       [ 8, 87, 31, 72],
       [40, 71,  2,  7]])

np.argmax(b,axis=0)
     
array([2, 2, 2, 3])

np.argmax(b,axis=1)
     
array([1, 3, 1, 3, 1, 1])

# np.argmin
np.argmin(a)
     
13
np.cumsum
numpy.cumsum() function is used when we want to compute the cumulative sum of array elements over a given axis.

https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html


a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

np.cumsum(a)
     
array([ 11,  64,  92, 142, 180, 217, 311, 403, 408, 438, 506, 515, 593,
       595, 616])

b
     
array([[12, 52, 42,  6],
       [29, 18, 47, 55],
       [61, 93, 83,  9],
       [38, 63, 44, 85],
       [ 8, 87, 31, 72],
       [40, 71,  2,  7]])

np.cumsum(b,axis=1)
     
array([[ 12,  64, 106, 112],
       [ 29,  47,  94, 149],
       [ 61, 154, 237, 246],
       [ 38, 101, 145, 230],
       [  8,  95, 126, 198],
       [ 40, 111, 113, 120]])

np.cumsum(b)
     
array([  12,   64,  106,  112,  141,  159,  206,  261,  322,  415,  498,
        507,  545,  608,  652,  737,  745,  832,  863,  935,  975, 1046,
       1048, 1055])

# np.cumprod
np.cumprod(a)
     
array([                  11,                  583,                16324,
                     816200,             31015600,           1147577200,
               107872256800,        9924247625600,       49621238128000,
           1488637143840000,   101227325781120000,   911045932030080000,
       -2725393596491966464, -5450787192983932928, -3786066610405281792])

a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])
np.percentile
numpy.percentile()function used to compute the nth percentile of the given data (array elements) along the specified axis.

https://numpy.org/doc/stable/reference/generated/numpy.percentile.html


a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

np.percentile(a,50)
     
37.0

np.median(a)
     
37.0
np.histogram
Numpy has a built-in numpy.histogram() function which represents the frequency of data distribution in the graphical form.

https://numpy.org/doc/stable/reference/generated/numpy.histogram.html


# code
a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

np.histogram(a,bins=[0,50,100])
     
(array([9, 6]), array([  0,  50, 100]))
np.corrcoef
Return Pearson product-moment correlation coefficients.

https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html


salary = np.array([20000,40000,25000,35000,60000])
experience = np.array([1,3,2,4,2])

np.corrcoef(salary,experience)
     
array([[1.        , 0.25344572],
       [0.25344572, 1.        ]])
np.isin
With the help of numpy.isin() method, we can see that one array having values are checked in a different numpy array having different elements with different sizes.

https://numpy.org/doc/stable/reference/generated/numpy.isin.html


# code
a

     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

items = [10,20,30,40,50,60,70,80,90,100]

a[np.isin(a,items)]
     
array([50, 30])
np.flip
The numpy.flip() function reverses the order of array elements along the specified axis, preserving the shape of the array.

https://numpy.org/doc/stable/reference/generated/numpy.flip.html


# code
a
     
array([11, 53, 28, 50, 38, 37, 94, 92,  5, 30, 68,  9, 78,  2, 21])

np.flip(a)
     
array([21,  2, 78,  9, 68, 30,  5, 92, 94, 37, 38, 50, 28, 53, 11])

b
     
array([[12, 52, 42,  6],
       [29, 18, 47, 55],
       [61, 93, 83,  9],
       [38, 63, 44, 85],
       [ 8, 87, 31, 72],
       [40, 71,  2,  7]])

np.flip(b,axis=1)
     
array([[ 6, 42, 52, 12],
       [55, 47, 18, 29],
       [ 9, 83, 93, 61],
       [85, 44, 63, 38],
       [72, 31, 87,  8],
       [ 7,  2, 71, 40]])
np.put
The numpy.put() function replaces specific elements of an array with given values of p_array. Array indexed works on flattened array.

https://numpy.org/doc/stable/reference/generated/numpy.put.html


# code
a
     
array([110, 530,  28,  50,  38,  37,  94,  92,   5,  30,  68,   9,  78,
         2,  21])

np.put(a,[0,1],[110,530])
     
np.delete
The numpy.delete() function returns a new array with the deletion of sub-arrays along with the mentioned axis.

https://numpy.org/doc/stable/reference/generated/numpy.delete.html


# code
a
     
array([110, 530,  28,  50,  38,  37,  94,  92,   5,  30,  68,   9,  78,
         2,  21])

np.delete(a,[0,2,4])
     
array([530,  50,  37,  94,  92,   5,  30,  68,   9,  78,   2,  21])
Set functions
np.union1d
np.intersect1d
np.setdiff1d
np.setxor1d
np.in1d

m = np.array([1,2,3,4,5])
n = np.array([3,4,5,6,7])

np.union1d(m,n)
     
array([1, 2, 3, 4, 5, 6, 7])

np.intersect1d(m,n)
     
array([3, 4, 5])

np.setdiff1d(n,m)
     
array([6, 7])

np.setxor1d(m,n)
     
array([1, 2, 6, 7])

m[np.in1d(m,1)]
     
array([1])
np.clip
numpy.clip() function is used to Clip (limit) the values in an array.

https://numpy.org/doc/stable/reference/generated/numpy.clip.html


# code
a
     
array([110, 530,  28,  50,  38,  37,  94,  92,   5,  30,  68,   9,  78,
         2,  21])

np.clip(a,a_min=25,a_max=75)
     
array([75, 75, 28, 50, 38, 37, 75, 75, 25, 30, 68, 25, 75, 25, 25])

# 17. np.swapaxes
     

# 18. np.uniform
     

# 19. np.count_nonzero
     

# 21. np.tile
# https://www.kaggle.com/code/abhayparashar31/best-numpy-functions-for-data-science-50?scriptVersionId=98816580
     

# 22. np.repeat
# https://towardsdatascience.com/10-numpy-functions-you-should-know-1dc4863764c5
     


# 25. np.allclose and equals
     


     


     


     


     


     


     
###################################################################################



What is Pandas
Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.

https://pandas.pydata.org/about/index.html

Pandas Series
A Pandas Series is like a column in a table. It is a 1-D array holding data of any type.

Importing Pandas

import numpy as np
import pandas as pd
     
Series from lists

# string
country = ['India','Pakistan','USA','Nepal','Srilanka']

pd.Series(country)
     
0       India
1    Pakistan
2         USA
3       Nepal
4    Srilanka
dtype: object

# integers
runs = [13,24,56,78,100]

runs_ser = pd.Series(runs)
     

# custom index
marks = [67,57,89,100]
subjects = ['maths','english','science','hindi']

pd.Series(marks,index=subjects)
     
maths       67
english     57
science     89
hindi      100
dtype: int64

# setting a name
marks = pd.Series(marks,index=subjects,name='Nitish ke marks')
marks
     
maths       67
english     57
science     89
hindi      100
Name: Nitish ke marks, dtype: int64
Series from dict

marks = {
    'maths':67,
    'english':57,
    'science':89,
    'hindi':100
}

marks_series = pd.Series(marks,name='nitish ke marks')
marks_series
     
maths       67
english     57
science     89
hindi      100
Name: nitish ke marks, dtype: int64
Series Attributes

# size
marks_series.size
     
4

# dtype
marks_series.dtype
     
dtype('int64')

# name
marks_series.name
     
'nitish ke marks'

# is_unique
marks_series.is_unique

pd.Series([1,1,2,3,4,5]).is_unique
     
False

# index
marks_series.index
     
Index(['maths', 'english', 'science', 'hindi'], dtype='object')

runs_ser.index
     
RangeIndex(start=0, stop=5, step=1)

# values
marks_series.values
     
array([ 67,  57,  89, 100])
Series using read_csv

# with one col
subs = pd.read_csv('/content/subs.csv',squeeze=True)
subs
     
0       48
1       57
2       40
3       43
4       44
      ... 
360    231
361    226
362    155
363    144
364    172
Name: Subscribers gained, Length: 365, dtype: int64

# with 2 cols
vk = pd.read_csv('/content/kohli_ipl.csv',index_col='match_no',squeeze=True)
vk
     
match_no
1       1
2      23
3      13
4      12
5       1
       ..
211     0
212    20
213    73
214    25
215     7
Name: runs, Length: 215, dtype: int64

movies = pd.read_csv('/content/bollywood.csv',index_col='movie',squeeze=True)
movies
     
movie
Uri: The Surgical Strike                   Vicky Kaushal
Battalion 609                                Vicky Ahuja
The Accidental Prime Minister (film)         Anupam Kher
Why Cheat India                            Emraan Hashmi
Evening Shadows                         Mona Ambegaonkar
                                              ...       
Hum Tumhare Hain Sanam                    Shah Rukh Khan
Aankhen (2002 film)                     Amitabh Bachchan
Saathiya (film)                             Vivek Oberoi
Company (film)                                Ajay Devgn
Awara Paagal Deewana                        Akshay Kumar
Name: lead, Length: 1500, dtype: object
Series methods

# head and tail
subs.head()
     
0    48
1    57
2    40
3    43
4    44
Name: Subscribers gained, dtype: int64

vk.head(3)
     
match_no
1     1
2    23
3    13
Name: runs, dtype: int64

vk.tail(10)
     
match_no
206     0
207     0
208     9
209    58
210    30
211     0
212    20
213    73
214    25
215     7
Name: runs, dtype: int64

# sample
movies.sample(5)
     
movie
Arjun: The Warrior Prince    Yudhveer Bakoliya
Viceroy's House (film)         Hugh Bonneville
Joggers' Park (film)           Victor Banerjee
Tere Mere Phere                   Vinay Pathak
Mission Mangal                    Akshay Kumar
Name: lead, dtype: object

# value_counts -> movies
movies.value_counts()
     
Akshay Kumar        48
Amitabh Bachchan    45
Ajay Devgn          38
Salman Khan         31
Sanjay Dutt         26
                    ..
Diganth              1
Parveen Kaur         1
Seema Azmi           1
Akanksha Puri        1
Edwin Fernandes      1
Name: lead, Length: 566, dtype: int64

# sort_values -> inplace
vk.sort_values(ascending=False).head(1).values[0]
     
113

vk.sort_values(ascending=False)
     
match_no
128    113
126    109
123    108
164    100
120    100
      ... 
93       0
211      0
130      0
8        0
135      0
Name: runs, Length: 215, dtype: int64

# sort_index -> inplace -> movies
movies.sort_index(ascending=False,inplace=True)
     

movies
     
movie
Zor Lagaa Ke...Haiya!            Meghan Jadhav
Zokkomon                       Darsheel Safary
Zindagi Tere Naam           Mithun Chakraborty
Zindagi Na Milegi Dobara        Hrithik Roshan
Zindagi 50-50                      Veena Malik
                                   ...        
2 States (2014 film)              Arjun Kapoor
1971 (2007 film)                Manoj Bajpayee
1920: The Evil Returns             Vicky Ahuja
1920: London                     Sharman Joshi
1920 (film)                   Rajniesh Duggall
Name: lead, Length: 1500, dtype: object

vk.sort_values(inplace=True)
     

vk
     
match_no
87       0
211      0
207      0
206      0
91       0
      ... 
164    100
120    100
123    108
126    109
128    113
Name: runs, Length: 215, dtype: int64
Series Maths Methods

# count
vk.count()
     
215

# sum -> product
subs.sum()
     
49510

# mean -> median -> mode -> std -> var
subs.mean()
print(vk.median())
print(movies.mode())
print(subs.std())
print(vk.var())
     
24.0
0    Akshay Kumar
dtype: object
62.6750230372527
688.0024777222343

# min/max
subs.max()
     
396

# describe
subs.describe()
     
count    365.000000
mean     135.643836
std       62.675023
min       33.000000
25%       88.000000
50%      123.000000
75%      177.000000
max      396.000000
Name: Subscribers gained, dtype: float64
Series Indexing

# integer indexing
x = pd.Series([12,13,14,35,46,57,58,79,9])
x
     
0    12
1    13
2    14
3    35
4    46
5    57
6    58
7    79
8     9
dtype: int64

# negative indexing
x[-1]
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.8/dist-packages/pandas/core/indexes/range.py in get_loc(self, key, method, tolerance)
    384                 try:
--> 385                     return self._range.index(new_key)
    386                 except ValueError as err:

ValueError: -1 is not in range

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
<ipython-input-103-60055cb99cf0> in <module>
      1 # negative indexing
----> 2 x[-1]

/usr/local/lib/python3.8/dist-packages/pandas/core/series.py in __getitem__(self, key)
    940 
    941         elif key_is_scalar:
--> 942             return self._get_value(key)
    943 
    944         if is_hashable(key):

/usr/local/lib/python3.8/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1049 
   1050         # Similar to Index.get_value, but we do not fall back to positional
-> 1051         loc = self.index.get_loc(label)
   1052         return self.index._get_values_for_loc(self, loc, label)
   1053 

/usr/local/lib/python3.8/dist-packages/pandas/core/indexes/range.py in get_loc(self, key, method, tolerance)
    385                     return self._range.index(new_key)
    386                 except ValueError as err:
--> 387                     raise KeyError(key) from err
    388             raise KeyError(key)
    389         return super().get_loc(key, method=method, tolerance=tolerance)

KeyError: -1

movies
     
movie
Zor Lagaa Ke...Haiya!            Meghan Jadhav
Zokkomon                       Darsheel Safary
Zindagi Tere Naam           Mithun Chakraborty
Zindagi Na Milegi Dobara        Hrithik Roshan
Zindagi 50-50                      Veena Malik
                                   ...        
2 States (2014 film)              Arjun Kapoor
1971 (2007 film)                Manoj Bajpayee
1920: The Evil Returns             Vicky Ahuja
1920: London                     Sharman Joshi
1920 (film)                   Rajniesh Duggall
Name: lead, Length: 1500, dtype: object

vk[-1]
     
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.8/dist-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   3360             try:
-> 3361                 return self._engine.get_loc(casted_key)
   3362             except KeyError as err:

/usr/local/lib/python3.8/dist-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()

/usr/local/lib/python3.8/dist-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: -1

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
<ipython-input-107-8ecb27b68523> in <module>
----> 1 vk[-1]

/usr/local/lib/python3.8/dist-packages/pandas/core/series.py in __getitem__(self, key)
    940 
    941         elif key_is_scalar:
--> 942             return self._get_value(key)
    943 
    944         if is_hashable(key):

/usr/local/lib/python3.8/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1049 
   1050         # Similar to Index.get_value, but we do not fall back to positional
-> 1051         loc = self.index.get_loc(label)
   1052         return self.index._get_values_for_loc(self, loc, label)
   1053 

/usr/local/lib/python3.8/dist-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   3361                 return self._engine.get_loc(casted_key)
   3362             except KeyError as err:
-> 3363                 raise KeyError(key) from err
   3364 
   3365         if is_scalar(key) and isna(key) and not self.hasnans:

KeyError: -1

marks_series[-1]
     
100

# slicing
vk[5:16]
     
match_no
6      9
7     34
8      0
9     21
10     3
11    10
12    38
13     3
14    11
15    50
16     2
Name: runs, dtype: int64

# negative slicing
vk[-5:]
     
match_no
211     0
212    20
213    73
214    25
215     7
Name: runs, dtype: int64

movies[::2]
     
movie
Zor Lagaa Ke...Haiya!         Meghan Jadhav
Zindagi Tere Naam        Mithun Chakraborty
Zindagi 50-50                   Veena Malik
Zinda (film)                    Sanjay Dutt
Zid (2014 film)              Mannara Chopra
                                ...        
3 Storeys                       Aisha Ahmed
3 Deewarein                Naseeruddin Shah
22 Yards                        Barun Sobti
1971 (2007 film)             Manoj Bajpayee
1920: London                  Sharman Joshi
Name: lead, Length: 750, dtype: object

# fancy indexing
vk[[1,3,4,5]]
     
match_no
1     1
3    13
4    12
5     1
Name: runs, dtype: int64

# indexing with labels -> fancy indexing
movies['2 States (2014 film)']
     
'Arjun Kapoor'
Editing Series

# using indexing
marks_series[1] = 100
marks_series
     
maths       67
english    100
science     89
hindi      100
Name: nitish ke marks, dtype: int64

# what if an index does not exist
marks_series['evs'] = 100
     

marks_series
     
maths       67
english    100
science     89
hindi      100
sst         90
evs        100
Name: nitish ke marks, dtype: int64

# slicing
runs_ser[2:4] = [100,100]
runs_ser
     
0     13
1     24
2    100
3    100
4    100
dtype: int64

# fancy indexing
runs_ser[[0,3,4]] = [0,0,0]
runs_ser
     
0      0
1     24
2    100
3      0
4      0
dtype: int64

# using index label
movies['2 States (2014 film)'] = 'Alia Bhatt'
movies
     
movie
Zor Lagaa Ke...Haiya!            Meghan Jadhav
Zokkomon                       Darsheel Safary
Zindagi Tere Naam           Mithun Chakraborty
Zindagi Na Milegi Dobara        Hrithik Roshan
Zindagi 50-50                      Veena Malik
                                   ...        
2 States (2014 film)                Alia Bhatt
1971 (2007 film)                Manoj Bajpayee
1920: The Evil Returns             Vicky Ahuja
1920: London                     Sharman Joshi
1920 (film)                   Rajniesh Duggall
Name: lead, Length: 1500, dtype: object
Copy and Views


     
Series with Python Functionalities

# len/type/dir/sorted/max/min
print(len(subs))
print(type(subs))
print(dir(subs))
print(sorted(subs))
print(min(subs))
print(max(subs))
     
365
<class 'pandas.core.series.Series'>
['T', '_AXIS_LEN', '_AXIS_ORDERS', '_AXIS_REVERSED', '_AXIS_TO_AXIS_NUMBER', '_HANDLED_TYPES', '__abs__', '__add__', '__and__', '__annotations__', '__array__', '__array_priority__', '__array_ufunc__', '__array_wrap__', '__bool__', '__class__', '__contains__', '__copy__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__divmod__', '__doc__', '__eq__', '__finalize__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__imod__', '__imul__', '__init__', '__init_subclass__', '__int__', '__invert__', '__ior__', '__ipow__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__long__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__weakref__', '__xor__', '_accessors', '_accum_func', '_add_numeric_operations', '_agg_by_level', '_agg_examples_doc', '_agg_see_also_doc', '_align_frame', '_align_series', '_arith_method', '_as_manager', '_attrs', '_binop', '_can_hold_na', '_check_inplace_and_allows_duplicate_labels', '_check_inplace_setting', '_check_is_chained_assignment_possible', '_check_label_or_level_ambiguity', '_check_setitem_copy', '_clear_item_cache', '_clip_with_one_bound', '_clip_with_scalar', '_cmp_method', '_consolidate', '_consolidate_inplace', '_construct_axes_dict', '_construct_axes_from_arguments', '_construct_result', '_constructor', '_constructor_expanddim', '_convert', '_convert_dtypes', '_data', '_dir_additions', '_dir_deletions', '_drop_axis', '_drop_labels_or_levels', '_duplicated', '_find_valid_index', '_flags', '_from_mgr', '_get_axis', '_get_axis_name', '_get_axis_number', '_get_axis_resolvers', '_get_block_manager_axis', '_get_bool_data', '_get_cacher', '_get_cleaned_column_resolvers', '_get_index_resolvers', '_get_label_or_level_values', '_get_numeric_data', '_get_value', '_get_values', '_get_values_tuple', '_get_with', '_gotitem', '_hidden_attrs', '_index', '_indexed_same', '_info_axis', '_info_axis_name', '_info_axis_number', '_init_dict', '_init_mgr', '_inplace_method', '_internal_names', '_internal_names_set', '_is_cached', '_is_copy', '_is_label_or_level_reference', '_is_label_reference', '_is_level_reference', '_is_mixed_type', '_is_view', '_item_cache', '_ixs', '_logical_func', '_logical_method', '_map_values', '_maybe_update_cacher', '_memory_usage', '_metadata', '_mgr', '_min_count_stat_function', '_name', '_needs_reindex_multi', '_protect_consolidate', '_reduce', '_reindex_axes', '_reindex_indexer', '_reindex_multi', '_reindex_with_indexers', '_replace_single', '_repr_data_resource_', '_repr_latex_', '_reset_cache', '_reset_cacher', '_set_as_cached', '_set_axis', '_set_axis_name', '_set_axis_nocheck', '_set_is_copy', '_set_labels', '_set_name', '_set_value', '_set_values', '_set_with', '_set_with_engine', '_slice', '_stat_axis', '_stat_axis_name', '_stat_axis_number', '_stat_function', '_stat_function_ddof', '_take_with_is_copy', '_typ', '_update_inplace', '_validate_dtype', '_values', '_where', 'abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all', 'any', 'append', 'apply', 'argmax', 'argmin', 'argsort', 'array', 'asfreq', 'asof', 'astype', 'at', 'at_time', 'attrs', 'autocorr', 'axes', 'backfill', 'between', 'between_time', 'bfill', 'bool', 'clip', 'combine', 'combine_first', 'compare', 'convert_dtypes', 'copy', 'corr', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'div', 'divide', 'divmod', 'dot', 'drop', 'drop_duplicates', 'droplevel', 'dropna', 'dtype', 'dtypes', 'duplicated', 'empty', 'eq', 'equals', 'ewm', 'expanding', 'explode', 'factorize', 'ffill', 'fillna', 'filter', 'first', 'first_valid_index', 'flags', 'floordiv', 'ge', 'get', 'groupby', 'gt', 'hasnans', 'head', 'hist', 'iat', 'idxmax', 'idxmin', 'iloc', 'index', 'infer_objects', 'interpolate', 'is_monotonic', 'is_monotonic_decreasing', 'is_monotonic_increasing', 'is_unique', 'isin', 'isna', 'isnull', 'item', 'items', 'iteritems', 'keys', 'kurt', 'kurtosis', 'last', 'last_valid_index', 'le', 'loc', 'lt', 'mad', 'map', 'mask', 'max', 'mean', 'median', 'memory_usage', 'min', 'mod', 'mode', 'mul', 'multiply', 'name', 'nbytes', 'ndim', 'ne', 'nlargest', 'notna', 'notnull', 'nsmallest', 'nunique', 'pad', 'pct_change', 'pipe', 'plot', 'pop', 'pow', 'prod', 'product', 'quantile', 'radd', 'rank', 'ravel', 'rdiv', 'rdivmod', 'reindex', 'reindex_like', 'rename', 'rename_axis', 'reorder_levels', 'repeat', 'replace', 'resample', 'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling', 'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'searchsorted', 'sem', 'set_axis', 'set_flags', 'shape', 'shift', 'size', 'skew', 'slice_shift', 'sort_index', 'sort_values', 'squeeze', 'std', 'sub', 'subtract', 'sum', 'swapaxes', 'swaplevel', 'tail', 'take', 'to_clipboard', 'to_csv', 'to_dict', 'to_excel', 'to_frame', 'to_hdf', 'to_json', 'to_latex', 'to_list', 'to_markdown', 'to_numpy', 'to_period', 'to_pickle', 'to_sql', 'to_string', 'to_timestamp', 'to_xarray', 'transform', 'transpose', 'truediv', 'truncate', 'tz_convert', 'tz_localize', 'unique', 'unstack', 'update', 'value_counts', 'values', 'var', 'view', 'where', 'xs']
[33, 33, 35, 37, 39, 40, 40, 40, 40, 42, 42, 43, 44, 44, 44, 45, 46, 46, 48, 49, 49, 49, 49, 50, 50, 50, 51, 54, 56, 56, 56, 56, 57, 61, 62, 64, 65, 65, 66, 66, 66, 66, 67, 68, 70, 70, 70, 71, 71, 72, 72, 72, 72, 72, 73, 74, 74, 75, 76, 76, 76, 76, 77, 77, 78, 78, 78, 79, 79, 80, 80, 80, 81, 81, 82, 82, 83, 83, 83, 84, 84, 84, 85, 86, 86, 86, 87, 87, 87, 87, 88, 88, 88, 88, 88, 89, 89, 89, 90, 90, 90, 90, 91, 92, 92, 92, 93, 93, 93, 93, 95, 95, 96, 96, 96, 96, 97, 97, 98, 98, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 103, 103, 104, 104, 104, 105, 105, 105, 105, 105, 105, 105, 105, 105, 108, 108, 108, 108, 108, 108, 109, 109, 110, 110, 110, 111, 111, 112, 113, 113, 113, 114, 114, 114, 114, 115, 115, 115, 115, 117, 117, 117, 118, 118, 119, 119, 119, 119, 120, 122, 123, 123, 123, 123, 123, 124, 125, 126, 127, 128, 128, 129, 130, 131, 131, 132, 132, 134, 134, 134, 135, 135, 136, 136, 136, 137, 138, 138, 138, 139, 140, 144, 145, 146, 146, 146, 146, 147, 149, 150, 150, 150, 150, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 156, 156, 156, 156, 157, 157, 157, 157, 158, 158, 159, 159, 160, 160, 160, 160, 162, 164, 166, 167, 167, 168, 170, 170, 170, 170, 171, 172, 172, 173, 173, 173, 174, 174, 175, 175, 176, 176, 177, 178, 179, 179, 180, 180, 180, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 185, 186, 186, 186, 188, 189, 190, 190, 192, 192, 192, 196, 196, 196, 197, 197, 202, 202, 202, 203, 204, 206, 207, 209, 210, 210, 211, 212, 213, 214, 216, 219, 220, 221, 221, 222, 222, 224, 225, 225, 226, 227, 228, 229, 230, 231, 233, 236, 236, 237, 241, 243, 244, 245, 247, 249, 254, 254, 258, 259, 259, 261, 261, 265, 267, 268, 269, 276, 276, 290, 295, 301, 306, 312, 396]
33
396

# type conversion
list(marks_series)
     
[67, 100, 89, 100, 90, 100]

dict(marks_series)
     
{'maths': 67,
 'english': 100,
 'science': 89,
 'hindi': 100,
 'sst': 90,
 'evs': 100}

# membership operator

'2 States (2014 film)' in movies
     
True

'Alia Bhatt' in movies.values
     
True

movies
     
movie
Zor Lagaa Ke...Haiya!            Meghan Jadhav
Zokkomon                       Darsheel Safary
Zindagi Tere Naam           Mithun Chakraborty
Zindagi Na Milegi Dobara        Hrithik Roshan
Zindagi 50-50                      Veena Malik
                                   ...        
2 States (2014 film)                Alia Bhatt
1971 (2007 film)                Manoj Bajpayee
1920: The Evil Returns             Vicky Ahuja
1920: London                     Sharman Joshi
1920 (film)                   Rajniesh Duggall
Name: lead, Length: 1500, dtype: object

# looping
for i in movies.index:
  print(i)
     
Zor Lagaa Ke...Haiya!
Zokkomon
Zindagi Tere Naam
Zindagi Na Milegi Dobara
Zindagi 50-50
Zindaggi Rocks
Zinda (film)
Zila Ghaziabad
Zid (2014 film)
Zero (2018 film)
Zeher
Zed Plus
Zameer: The Fire Within
Zameen (2003 film)
Zamaanat
Yuvvraaj
Yuva
Yun Hota Toh Kya Hota
Youngistaan
Yeh Saali Aashiqui
Yeh Mera India
Yeh Lamhe Judaai Ke
Yeh Khula Aasmaan
Yeh Jawaani Hai Deewani
Yeh Hai India
Yeh Hai Bakrapur
Yeh Dooriyan
Yeh Dil
Yatra (2007 film)
Yamla Pagla Deewana: Phir Se
Yamla Pagla Deewana
Yakeen (2005 film)
Yadvi – The Dignified Princess
Yaaram (2019 film)
Ya Rab
Xcuse Me
Woodstock Villa
Woh Lamhe...
Why Cheat India
What's Your Raashee?
What the Fish
Well Done Abba
Welcome to Sajjanpur
Welcome Back (film)
Welcome 2 Karachi
Welcome (2007 film)
Wedding Pullav
Wedding Anniversary
Waris Shah: Ishq Daa Waaris
War Chhod Na Yaar
Waqt: The Race Against Time
Wanted (2009 film)
Wake Up Sid
Wake Up India
Wajah Tum Ho
Waiting (2015 film)
Waisa Bhi Hota Hai Part II
Wah Taj
Wafa: A Deadly Love Story
Waarrior Savitri
W (2014 film)
Vodka Diaries
Vivah
Vishwaroopam
Viruddh... Family Comes First
Vidyaarthi
Victory (2009 film)
Vicky Donor
Viceroy's House (film)
Via Darjeeling
Veerey Ki Wedding
Veerappan (2016 film)
Veer-Zaara
Veer (2010 film)
Valentine's Night
Vaastu Shastra (film)
Vaah! Life Ho Toh Aisi!
Vaada Raha
Vaada (film)
Uvaa
Utthaan
Utt Pataang
Uri: The Surgical Strike
United Six
Union Leader (film)
Ungli
Umrika
Umrao Jaan (2006 film)
Umar (film)
Ujda Chaman
Ugly (film)
Udta Punjab
Udaan (2010 film)
U R My Jaan
U Me Aur Hum
Turning 30
Tumsa Nahin Dekha: A Love Story
Tumhari Sulu
Tumbbad
Tum Milo Toh Sahi
Tum Mile
Tum Bin II
Tulsi (film)
Tujhe Meri Kasam
Tubelight (2017 Hindi film)
Trump Card (film)
Trapped (2016 Hindi film)
Traffic Signal (film)
Traffic (2016 film)
Total Siyapaa
Total Dhamaal
Toonpur Ka Super Hero
Tom Dick and Harry (2006 film)
Toilet: Ek Prem Katha
Toh Baat Pakki!
Titoo MBA
Titli (2014 film)
Tiger Zinda Hai
Thugs of Hindostan
Thodi Life Thoda Magic
Thoda Tum Badlo Thoda Hum
Thoda Pyaar Thoda Magic
Thoda Lutf Thoda Ishq
The Zoya Factor (film)
The Xposé
The Train (2007 film)
The Tashkent Files
The Stoneman Murders
The Sky Is Pink
The Silent Heroes
The Shaukeens
The Pink Mirror
The Namesake (film)
The Lunchbox
The Last Lear
The Killer (2006 film)
The Journey of Karma
The Japanese Wife
The Hero: Love Story of a Spy
The Ghazi Attack
The Final Exit
The Film Emotional Atyachar
The Film
The Dirty Picture
The Bypass
The Blueberry Hunt
The Blue Umbrella (2005 film)
The Accidental Prime Minister (film)
Thanks Maa
Thank You (2011 film)
Thackeray (film)
Tezz
Tevar
Teri Meri Kahaani (film)
Tere Naam
Tere Naal Love Ho Gaya
Tere Mere Phere
Tere Bin Laden: Dead or Alive
Tere Bin Laden
Tera Kya Hoga Johnny
Tell Me O Kkhuda
Tehzeeb (2003 film)
Teesri Aankh: The Hidden Camera
Tees Maar Khan (2010 film)
Teen Thay Bhai
Teen Patti (film)
Te3n
Taxi No. 9211
Tathastu
Tashan (film)
Tanu Weds Manu: Returns
Tanu Weds Manu: Returns
Tanu Weds Manu
Tango Charlie
Tamanchey
Talvar (film)
Talaash: The Hunt Begins...
Talaash: The Answer Lies Within
Take It Easy (2015 film)
Taj Mahal: An Eternal Love Story
Tahaan
Table No. 21
Taarzan: The Wonder Car
Taare Zameen Par
Ta Ra Rum Pum
Sweetiee Weds NRI
Swami (2007 film)
Swades
Super Nani
Super Model (film)
Super 30 (film)
Suno Sasurjee
Sunglass (film)
Sunday (2008 film)
Summer 2007
Sultan (2016 film)
Sulemani Keeda
Sukhmani: Hope for Life
Sui Dhaaga
Stumped (film)
Student of the Year 2
Student of the Year
Strings of Passion
Striker (2010 film)
Stree (2018 film)
Strangers (2007 Hindi film)
Staying Alive (2012 film)
Station (2014 film)
Stanley Ka Dabba
Ssukh
Sssshhh...
Speed (2007 film)
Special 26
Spark (2014 film)
Souten: The Other Woman
Sorry Daddy
Sorry Bhai!
Sooper Se Ooper
Sonu Ke Titu Ki Sweety
Sons of Ram
Soni (film)
Sonchiriya
Sonali Cable
Son of Sardaar
Socha Na Tha
Soch Lo
Sixteen (2013 Indian film)
Sirf (film)
Singham Returns
Singham
Singh Saab the Great
Singh Is Kinng
Singh Is Bliing
Simran (film)
Simmba
Silsiilay
Sikandar (2009 film)
Siddharth (2013 film)
Shukriya: Till Death Do Us Apart
Shuddh Desi Romance
Shubh Mangal Saavdhan
Showbiz (film)
Shortkut
Shortcut Safari
Shortcut Romeo
Shorgul
Shor in the City
Shootout at Lokhandwala
Sholay
Shivaay
Shiva (2006 film)
Shirin Farhad Ki Toh Nikal Padi
Ship of Theseus (film)
Shikhar (film)
Sheesha (2005 film)
Sheen (film)
Shart: The Challenge
Sharafat Gayi Tel Lene
Shanghai (2012 film)
Shamitabh
Shakalaka Boom Boom
Shaitan (film)
Shahid (film)
Shagird (2011 film)
Shabri
Shabnam Mausi
Shabd (film)
Shab (film)
Shaapit
Shaandaar
Shaadi Se Pehle
Shaadi No. 1
Shaadi Mein Zaroor Aana
Shaadi Ke Side Effects
Shaadi Karke Phas Gaya Yaar
Shaadi Ka Laddoo
Setters (film)
Sehar
Section 375
Secret Superstar
Second Hand Husband
Say Salaam India
Satyameva Jayate (2018 film)
Satyagraha (film)
Satya 2
Satta (film)
Satrangee Parachute
Satellite Shankar
Sarkar Raj
Sarkar 3
Sarkar (2005 film)
Sarhad Paar
Sarbjit (film)
Santa Banta Pvt Ltd
Sankat City
Sanju
Sandwich (2006 film)
Sanam Re
Samrat & Co.
Samay: When Time Strikes
Sallu Ki Shaadi
Salaam-e-Ishq: A Tribute to Love
Salaam Namaste
Sahi Dhandhe Galat Bande
Saheb Biwi Aur Gangster Returns
Saheb Biwi Aur Gangster 3
Saheb Biwi Aur Gangster
Sadiyaan
Sadda Adda
Sacred Evil – A True Story
Sachin: A Billion Dreams
Sabki Bajegi Band
Saaya (2003 film)
Saawariya
Saawan... The Love Season
Saathiya (film)
Saat Uchakkey
Saas Bahu Aur Sensex
Saare Jahaan Se Mehnga
Saansein
Saankal
Saand Ki Aankh
Saaho
Rustom (film)
Rush (2012 film)
Running Shaadi
Run (2004 film)
Rules: Pyaar Ka Superhit Formula
Rukh (film)
Rudraksh (film)
Roy (film)
Rough Book
Rokkk
Rok Sako To Rok Lo
Rog
Rocky Handsome
Rockstar (2011 film)
Rocket Singh: Salesman of the Year
Rock On!!
Rock On 2
Roar: Tigers of the Sundarbans
Roadside Romeo
Road to Sangam
Riyasat (film)
Risknamaa
Risk (2007 film)
Right Yaaa Wrong
Right Here Right Now (film)
Ribbon (film)
Revolver Rani
Revati (film)
Red: The Dark Side
Red Swastik
Red Alert: The War Within
Rebellious Flower
Rascals (2011 film)
Raqeeb
Rann (film)
Rangrezz
Rangoon (2017 Hindi film)
Rang Rasiya
Rang De Basanti
Ranchi Diaries
Ranbanka
Ramprasad Ki Tehrvi
Ramji Londonwaley
Ramayana: The Epic
Raman Raghav 2.0
Ramaiya Vastavaiya
Ramaa: The Saviour
Rakhtbeej
Rakht
Rajma Chawal
Rajjo
Raja Natwarlal
Raja Bhaiya (film)
Raincoat (film)
Raid (2018 film)
Rahasya
Ragini MMS 2
Ragini MMS
Raghu Romeo
Raees (film)
Race 3
Race 2
Race (2008 film)
Rab Ne Bana Di Jodi
Raazi
Raaz: The Mystery Continues
Raaz: Reboot
Raaz (2002 film)
Raavan
Raat Gayi Baat Gayi?
Raanjhanaa
Raag Desh (film)
Raabta (film)
Ra.One
Quick Gun Murugun
Queen (2014 film)
Qissa (film)
Qayamat: City Under Threat
Qarib Qarib Singlle
Qaidi Band
Pyare Mohan
Pyaar Mein Twist
Pyaar Ke Side Effects
Pyaar Ka Punchnama 2
Pyaar Ka Punchnama
Pyaar Impossible!
Purani Jeans
Prince (2010 film)
Prem Ratan Dhan Payo
Prem Kaa Game
Prateeksha
Prassthanam
Pranaam
Prague (2013 film)
Praan Jaye Par Shaan Na Jaye
Poster Boys
Popcorn Khao! Mast Ho Jao
Policegiri
Police Force: An Inside Story
Players (2012 film)
Plan (film)
Pizza (2014 film)
Pink (2016 film)
Pinjar (film)
Piku
Pihu
Photograph (film)
Phoonk 2
Phobia (2016 film)
Phir Milenge
Phir Kabhi
Phir Hera Pheri
Phillauri (film)
Phhir
Phata Poster Nikhla Hero
Phas Gaye Re Obama
Phantom (2015 film)
Phamous
Pehchaan: The Face of Truth
Peepli Live
Paying Guests
Patiala House (film)
Pati Patni Aur Woh (2019 film)
Patel Ki Punjabi Shaadi
Pataakha
Parwana (2003 film)
Partner (2007 film)
Parmanu: The Story of Pokhran
Parineeta (2005 film)
Parched
Paranthe Wali Gali
Pankh
Panchlait
Paltan (film)
Pal Pal Dil Ke Paas
Paisa Vasool
Paheli
Page 3 (film)
Pagalpanti (2019 film)
Padmashree Laloo Prasad Yadav
Padmaavat
Paathshaala
Paap
Paanch Ghantey Mien Paanch Crore
Paan Singh Tomar (film)
Paa (film)
PM Narendra Modi
PK (film)
P Se Pyaar F Se Faraar
P Se PM Tak
Oye Lucky! Lucky Oye!
Out of Control (2003 film)
One by Two (2014 film)
One Two Three
One Day: Justice Delivered
Once Upon ay Time in Mumbai Dobaara!
Once Upon a Time in Mumbaai
Omkara (2006 film)
Omerta (film)
Om-Dar-B-Dar
Om Shanti Om
Om (2003 film)
Ok Jaanu
Oh My God (2008 film)
October (2018 film)
OMG – Oh My God!
O Teri
Nothing but Life
Notebook (2019 film)
Not a Love Story (2011 film)
Noor (film)
No Smoking (2007 film)
No Problem (2010 film)
No One Killed Jessica
No Entry
Nishabd
Nirdosh
Nil Battey Sannata
Newton (film)
New York (2009 film)
Netaji Subhas Chandra Bose: The Forgotten Hero
Nehlle Pe Dehlla
Neerja
Neal 'n' Nikki
Nayee Padosan
Nawabzaade
Nautanki Saala!
Naughty @ 40
Nasha (film)
Naqaab
Nanu Ki Jaanu
Nanhe Jaisalmer
Namastey London
Namaste England
Naksha
Naina (2005 film)
Naam Shabana
Naach (2004 film)
Na Ghar Ke Na Ghaat Ke
NH10 (film)
NH-8 Road to Nidhivan
My Wife's Murder
My Name Is Khan
My Friend Pinto
My Brother…Nikhil
My Bollywood Bride
My Birthday Song
Muskaan
Musafir (2004 film)
Murder 3
Murder 2
Murder (2004 film)
Murari the Mad Gentleman
Munna Michael
Munna Bhai M.B.B.S.
Mummy Punjabi
Mumbhai Connection
Mumbai Se Aaya Mera Dost
Mumbai Salsa
Mumbai Meri Jaan
Mumbai Matinee
Mumbai Mast Kallander
Mumbai Delhi Mumbai
Mumbai Can Dance Saala
Mumbai 125 KM
Mulk (film)
Mukkabaaz
Mukhbiir
Mujhse Shaadi Karogi
Mujhse Fraaandship Karoge
Mughal-e-Azam
Mubarakan
Mr. X (2015 film)
Mr. Singh Mrs. Mehta
Mr. Bhatti on Chutti
Mr Prime Minister
Motu Patlu: King of Kings
Motichoor Chaknachoor
Morning Raga
Monsoon Shootout
Monica (film)
Money Hai Toh Honey Hai
Mom (film)
Mohenjo Daro (film)
Mohalla Assi
Moh Maya Money
Mittal v/s Mittal
Mitron
Mission Mangal
Mission Istaanbul
Missing (2018 film)
Miss Tanakpur Haazir Ho
Mirzya (film)
Mirch
Miley Naa Miley Hum
Milenge Milenge
Mickey Virus
Meri Pyaari Bindu
Meri Biwi Ka Jawaab Nahin
Mere Pyare Prime Minister
Mere Jeevan Saathi (2006 film)
Mere Genie Uncle
Mere Dost Picture Abhi Baki Hai
Mere Dad Ki Maruti
Mere Brother Ki Dulhan
Mere Baap Pehle Aap
Mercury (film)
Memories in March
Meinu Ek Ladki Chaahiye
Meeruthiya Gangsters
Meerabai Not Out
Meenaxi: A Tale of Three Cities
Maximum (film)
Mausam (2011 film)
Matrubhoomi
Matru Ki Bijlee Ka Mandola
Mastram
Mastizaade
Masti (2004 film)
Masaan
Mary Kom (film)
Married 2 America
Market (2003 film)
Marjaavaan
Marigold (2007 film)
Margarita with a Straw
Mardaani 2
Mardaani
Mard Ko Dard Nahi Hota
Maqbool
Mantra (2016 film)
Manto (2018 film)
Manorama Six Feet Under
Manmarziyaan
Manjunath (film)
Manjhi – The Mountain Man
Manikarnika: The Queen of Jhansi
Mangal Pandey: The Rising
Malik Ek
Malamaal Weekly
Maine Pyaar Kyun Kiya?
Maine Gandhi Ko Nahin Mara
Main Tera Hero
Main Prem Ki Diwani Hoon
Main Meri Patni Aur Woh
Main Madhuri Dixit Banna Chahti Hoon
Main Krishna Hoon
Main Hoon Part-Time Killer
Main Hoon Na
Main Aurr Mrs Khanna
Main Aur Mr. Riight
Main Aisa Hi Hoon
Mai (2013 film)
Magic Magic 3D
Madras Cafe
Madhoshi
Made in China (2019 film)
Madaari
Mad About Dance
Machine (2017 film)
Machhli Jal Ki Rani Hai
Maazii
Maatr
Maan Gaye Mughal-e-Azam
MSG: The Warrior Lion Heart
MSG: The Messenger
MSG-2 The Messenger
MP3: Mera Pehla Pehla Pyaar
M.S. Dhoni: The Untold Story
M Cream
Luv U Soniyo
Luv U Alia
Lucky: No Time for Love
Lucky Kabootar
Lucknow Central
Luckhnowi Ishq
Luck by Chance
Luck (2009 film)
Loveshhuda
Love per Square Foot
Love in Bombay
Love U...Mr. Kalakaar!
Love Story 2050
Love Sonia
Love Shagun
Love Sex Aur Dhokha
Love Ke Chakkar Mein
Love Games (film)
Love Breakups Zindagi
Love Aaj Kal
Lootera
London Dreams
Loins of Punjab Presents
Login (film)
Little Zizou
Listen... Amaya
Lipstick Under My Burkha
Life in a... Metro
Life Partner
Life Mein Kabhie Kabhiee
Life Ki Toh Lag Gayi
Life Is Beautiful (2014 film)
Life Express (2010 film)
Lekar Hum Deewana Dil
Lamhaa
Lakshya (film)
Lakshmi (2014 film)
Lakeer – Forbidden Lines
Laila Majnu (2018 film)
Lahore (film)
Lage Raho Munna Bhai
Lafangey Parindey
Ladies vs Ricky Bahl
Laal Rang
Laaga Chunari Mein Daag
LOC Kargil
Kyun! Ho Gaya Na...
Kyon Ki
Kyaa Super Kool Hain Hum
Kyaa Kool Hain Hum 3
Kyaa Kool Hai Hum
Kya Love Story Hai
Kya Dilli Kya Lahore
Kushti (film)
Kurbaan (2009 film)
Kuku Mathur Ki Jhand Ho Gayi
Kudiyon Ka Hai Zamana
Kuchh Meetha Ho Jaye
Kuchh Bheege Alfaaz
Kuch Naa Kaho
Kuch Kuch Locha Hai
Kucch To Hai
Kucch Luv Jaisaa
Krrish
Krishna Cottage
Krishna Aur Kans
Krazzy 4
Koyelaanchal
Koi... Mil Gaya
Koi Mere Dil Mein Hai
Koi Aap Sa
Knock Out (2010 film)
Kites (film)
Kisse Pyaar Karoon
Kisna: The Warrior Poet
Kismat Love Paisa Dilli
Kismat Konnection
Kismat (2004 film)
Kisaan
Kis Kisko Pyaar Karoon
Kis Kis Ki Kismat
Kill Dil
Kick (2014 film)
Ki & Ka
Khwahish
Khwaabb
Khushi (2003 Hindi film)
Khuda Kasam
Khoya Khoya Chand
Khosla Ka Ghosla
Khoobsurat (2014 film)
Khichdi: The Movie
Khel – No Ordinary Game
Khel Toh Ab Shuru Hoga
Khatta Meetha (2010 film)
Khap (film)
Khamoshiyan
Khamoshi (2019 film)
Khamoshh... Khauff Ki Raat
Khamosh Pani
Khakee
Khajoor Pe Atke
Kesari (film)
Keep Safe Distance (film)
Kaun Kitne Paani Mein
Kaun Hai Jo Sapno Mein Aaya
Katti Batti
Kash Aap Hamare Hote
Kasak (2005 film)
Karzzzz
Karwaan
Karthik Calling Karthik
Karma Aur Holi
Karle Pyaar Karle
Karar: The Deal
Karam (film)
Kapoor & Sons
Kaminey
Kalyug (2005 film)
Kalank
Kal Ho Naa Ho
Kaise Kahoon Ke... Pyaar Hai
Kai Po Che!
Kahin Hai Mera Pyar
Kahaani
Kagaar: Life on the Edge
Kadvi Hawa
Kabul Express
Kabir Singh
Kabhi Alvida Naa Kehna
Kaashi in Search of Ganga
Kaante
Kaanchi: The Unbreakable
Kaalo
Kaalakaandi
Kaal (2005 film)
Kaagaz Ke Fools
Kaabil
Just Married (2007 film)
Jurm (2005 film)
Junooniyat
Junglee (2019 film)
Julie 2
Julie (2004 film)
Jugni (2016 film)
Judwaa 2
Judgementall Hai Kya
Jolly LLB
Joker (2012 film)
Johnny Gaddaar
John Day (film)
Joggers' Park (film)
Jodi Breakers
Jodhaa Akbar
Jo Hum Chahein
Jo Bole So Nihaal (film)
Jism (2003 film)
Jimmy (2008 film)
Jigyaasa
Jigariyaa
Jia Aur Jia
Jhootha Kahin Ka
Jhootha Hi Sahi
Jhoom Barabar Jhoom
Jhankaar Beats
Jhalki
Jeena Isi Ka Naam Hai (film)
Jeena Hai Toh Thok Daal
Jazbaa
Jayantabhai Ki Luv Story
Jawani Diwani: A Youthful Joyride
Jattu Engineer
Jannat (film)
Janasheen
James (2005 film)
Jalpari: The Desert Mermaid
Jalebi (film)
Jal (film)
Jajantaram Mamantaram
Jail (2009 film)
Jai Veeru
Jai Jawaan Jai Kisaan (film)
Jai Ho (film)
Jai Gangaajal
Jai Chiranjeeva
Jahan Jaaeyega Hamen Paaeyega
Jagga Jasoos
Jackpot (2013 film)
Jack and Dil
Jabariya Jodi
Jab We Met
Jab Tak Hai Jaan
Jab Harry Met Sejal
Jaane Kyun De Yaaron
Jaane Kahan Se Aayi Hai
Jaane Hoga Kya
Jaan-E-Mann
Jaal: The Trap
JD (film)
It's a Wonderful Afterlife
Issaq
Island City (2015 film)
Isi Life Mein
Ishqiya
Ishqeria
Ishqedarriyaan
Ishq Vishk
Ishq Ke Parindey
Ishq Hai Tumse
Ishq Forever
Ishq Click
Ishkq in Paris
Ishaqzaade
Irudhi Suttru
Irada (2017 film)
Iqraar by Chance
Iqbal (film)
Inteqam: The Perfect Game
Inteha (2003 film)
Insan
Insaaf: The Justice
Inkaar (2013 film)
Indu Sarkar
Indian Babu
India's Most Wanted (film)
Impatient Vivek
I See You (2006 film)
I Proud to Be an Indian
I Love NY (2015 film)
I Love Desi
I Hate Luv Storys
I Am Kalam
I Am (2010 Indian film)
Hyderabad Blues 2
Hunterrr
Hungama (2003 film)
Humshakals
Humpty Sharma Ki Dulhania
Humne Jeena Seekh Liya
Humko Tumse Pyaar Hai
Humko Deewana Kar Gaye
Hume Tumse Pyaar Kitna
Hum Tumhare Hain Sanam
Hum Tum Shabana
Hum Tum Aur Ghost
Hum Tum
Hum Hai Raahi Car Ke
Hum Chaar
Hulchul (2004 film)
Housefull 4
Housefull 2
Housefull (2010 film)
Hotel Salvation
Hostel (2011 film)
Horror Story (film)
Hope Aur Hum
Honour Killing (film)
Honeymoon Travels Pvt. Ltd.
Home Delivery
Holiday: A Soldier Is Never Off Duty
Holiday (2006 film)
Hisss
Hindi Medium
Hind Ka Napak Ko Jawab: MSG Lion Heart 2
Himmatwala (2013 film)
Highway (2014 Hindi film)
High Jack (film)
Hichki
Heyy Babyy
Hey Bro
Heropanti
Heroine (2012 film)
Heroes (2008 film)
Hero (2015 Hindi film)
Help (film)
Hello Darling
Hello (2008 film)
Helicopter Eela
Heartless (2014 film)
Hazaaron Khwaishein Aisi
Hawayein
Hawas (2004 film)
Hawaizaada
Hawaa Hawaai
Hawa (film)
Hava Aney Dey
Haunted – 3D
Hatya (2004 film)
Hattrick (film)
Hate Story 4
Hate Story 2
Hate Story
Hastey Hastey
Haseena Parkar
Hasee Toh Phasee
Hari Puttar: A Comedy of Terrors
Haraamkhor
Happy Phirr Bhag Jayegi
Happy New Year (2014 film)
Happy Husbands (2011 film)
Happy Ending (film)
Happy Bhag Jayegi
Hanuman (2005 film)
Hamid (film)
Hamari Adhuri Kahani
Halla Bol
Halkaa
Half Girlfriend (film)
Haider (film)
Hai Apna Dil Toh Awara
Haasil
Haal-e-Dil
Guzaarish (film)
Guru (2007 film)
Gunday
Gumnaam – The Mystery
Gully Boy
Gulabi Gang (film)
Gulaal (film)
Gulaab Gang
Guest iin London
Guddu Rangeela
Guddu Ki Gun
Green Card Fever
Great Grand Masti
Grand Masti
Gour Hari Dastaan
Gori Tere Pyaar Mein
Good Newwz
Good Boy Bad Boy
Gone Kesh
Golmaal: Fun Unlimited
Golmaal Returns
Golmaal Again
Gollu Aur Pappu
Goliyon Ki Raasleela Ram-Leela
Gold (2018 film)
God Tussi Great Ho
Goal (2007 Hindi film)
Go Goa Gone
Go (2007 film)
Global Baba
Girlfriend (2004 film)
Gippi
Ghost (2019 film)
Ghost (2012 film)
Ghayal: Once Again
Ghanchakkar (film)
Ghajini (2008 film)
Genius (2018 Hindi film)
Gayab
Gauri: The Unborn
Gattu
Garv: Pride & Honour
Garam Masala (2005 film)
Gangster (2006 film)
Gangs of Wasseypur – Part 2
Gangs of Wasseypur
Gangoobai
Gangaajal
Gang of Ghosts
Gandhi My Father
Game (2011 film)
Gali Guleiyan
Gabbar Is Back
Gabbar Is Back
G Kutta Se
Fun2shh... Dudes in the 10th Century
Fun – Can Be Dangerous Sometimes
Fukrey Returns
Fukrey
Fugly (film)
Fuddu
FryDay
Fruit and Nut (film)
From Sydney with Love
Fredrick (film)
Freaky Ali
Fraud Saiyaan
Fox (film)
Force 2
Force (2011 film)
Footpath (2003 film)
Fool & Final
Flavors (film)
Flat 211
Fitoor
Firangi
Firaaq
Finding Fanny
Filmistaan
Fight Club – Members Only
Fida
Fever (2016 film)
Ferrari Ki Sawaari
Fatso!
Fashion (2008 film)
Fareb (2005 film)
Fanaa (2006 film)
Fan (film)
Familywala
Family of Thakurganj
Family (2006 film)
F.A.L.T.U
Evening Shadows
Escape from Taliban
Entertainment (2014 film)
English Vinglish
Enemmy
Elaan (2005 film)
Eklavya: The Royal Guard
Ekkees Toppon Ki Salaami
Ekk Deewana Tha
Ek: The Power of One
Ek Vivaah... Aisa Bhi
Ek Villain
Ek Thi Rani Aisi Bhi
Ek Thi Daayan
Ek Tha Tiger
Ek Second... Jo Zindagi Badal De?
Ek Se Bure Do
Ek Se Badhkar Ek (2004 film)
Ek Paheli Leela
Ek Main Aur Ekk Tu
Ek Khiladi Ek Haseena (film)
Ek Kahani Julie Ki
Ek Hasina Thi (film)
Ek Haseena Thi Ek Deewana Tha
Ek Din 24 Ghante
Ek Chalis Ki Last Local
Ek Aur Ek Gyarah
Ek Alag Mausam
Ek Ajnabee
Eight: The Power of Shani
Dus Kahaniyaan
Dus
Dunno Y... Na Jaane Kyon
Dum Maaro Dum (film)
Dum Laga Ke Haisha
Dum (2003 Hindi film)
Dulha Mil Gaya
Dude Where's the Party?
Drona (2008 film)
Drishyam (2015 film)
Dreams (2006 film)
Dream Girl (2019 film)
Double Dhamaal
Double Cross (2005 film)
Dosti: Friends Forever
Dostana (2008 film)
Dor (film)
Dongari Ka Raja
Don't Stop Dreaming
Don Muthu Swami
Don 2
Don (2006 Hindi film)
Dolly Ki Doli
Dobara
Dobaara: See Your Evil
Do Lafzon Ki Kahani (film)
Do Dooni Chaar
Dishoom
Dishkiyaoon
Dirty Politics (film)
Direct Ishq
Dilwale (2015 film)
Dilliwali Zaalim Girlfriend
Dil Toh Deewana Hai
Dil Toh Baccha Hai Ji
Dil Pardesi Ho Gayaa
Dil Ne Jise Apna Kahaa
Dil Maange More
Dil Kabaddi
Dil Ka Rishta
Dil Juunglee
Dil Jo Na Keh Saka
Dil Jo Bhi Kahey...
Dil Dosti Etc
Dil Diya Hai
Dil Dhadakne Do
Dil Bole Hadippa!
Dil Bechara Pyaar Ka Maara
Dil Bechara
Dhund (2003 film)
Dhoop
Dhoondte Reh Jaaoge
Dhoom 3
Dhoom 2
Dhoom
Dhol (film)
Dhokha
Dhobi Ghat (film)
Dharti Kahe Pukar Ke (2006 film)
Dharm (film)
Dharam Sankat Mein
Dhanak
Dhamaal
Dhadak
Devi (2016 film)
Devdas (2002 Hindi film)
Devaki (2005 film)
Dev (2004 film)
Detective Byomkesh Bakshy!
Desi Kattey
Desi Boyz
Deshdrohi
Department (film)
Delhi-6
Delhi Safari
Delhi Belly (film)
Dekh Tamasha Dekh
Dehraadun Diary
Deewane Huye Paagal
Deewaar (2004 film)
Dedh Ishqiya
Dear Zindagi
Dear Maya
Dear Friend Hitler
Dear Dad (film)
Deadline: Sirf 24 Ghante
De Taali
De De Pyaar De
De Dana Dan
Days of Tafree
Dasvidaniya
Dassehra
Darwaaza Bandh Rakho
Darr @ the Mall
Darna Zaroori Hai
Darna Mana Hai
Darling (2007 Indian film)
Dangerous Ishhq
Dangal (film)
Damadamm!
Daddy Cool (2009 Hindi film)
Daddy (2017 film)
Dabangg 3
Dabangg 2
Dabangg
Daawat-e-Ishq
Daas Dev
D-Day (2013 film)
D (film)
Crook (film)
Creature 3D
Crazy Cukkad Family
Court (film)
Corporate (2006 film)
Contract (2008 film)
Company (film)
Commando: A One Man Army
Commando 3 (film)
Coffee with D
Coffee Bloom
Cocktail (2012 film)
Click (2010 film)
Classic – Dance of Love
CityLights (2014 film)
City of Gold (2010 film)
Cigarette Ki Tarah
Chura Liyaa Hai Tumne
Chup Chup Ke
Chori Chori (2003 film)
Chor Chor Super Chor
Chocolate (2005 film)
Chittagong (film)
Chintu Ji
Chingaari
Chinar Daastaan-E-Ishq
Chillar Party
Children of War (2014 film)
Children of Heaven
Chicken Curry Law
Chhota Bheem and the Throne of Bali
Chhodon Naa Yaar
Chetna: The Excitement
Chennai Express
Chehraa
Chef (2017 film)
Cheeni Kum
Chatur Singh Two Star
Chashme Baddoor (2013 film)
Chase (2010 film)
Chargesheet (film)
Charas (2004 film)
Chandni Chowk to China
Chand Sa Roshan Chehra
Chand Ke Paar Chalo (film)
Chance Pe Dance
Chamku
Chameli (film)
Chalte Chalte (2003 film)
Chalo Dilli
Challo Driver
Chalk n Duster
Chal Pichchur Banate Hain
Chal Chala Chal
Chakravyuh (2012 film)
Chak De! India
Chaarfutiya Chhokare
Chaar Din Ki Chandni
Chaalis Chauraasi
Chaahat – Ek Nasha
Cash (2007 film)
Calendar Girls (2015 film)
Calcutta Mail
Calapor (film)
C Kkompany
Bypass Road (film)
Bunty Aur Babli
Bumper Draw
Bumm Bumm Bole
Bumboo
Bullett Raja
Bullet: Ek Dhamaka
Buddha in a Traffic Jam
Buddha Mar Gaya
Bubble Gum (film)
Brothers (2015 film)
Brij Mohan Amar Rahe
Breakaway (2011 film)
Break Ke Baad
Brahman Naman
Boss (2013 Hindi film)
Border (1997 film)
Boom (film)
Bombay to Goa (2007 film)
Bombay to Bangkok
Bombay Velvet
Bombay Talkies (film)
Bombairiya
Bollywood Diaries
Bol Bachchan
Bodyguard (2011 Hindi film)
Bobby Jasoos
Bluffmaster!
Blue (2009 film)
Bloody Isshq
Blood Money (2012 film)
Blood Brothers (2007 Indian film)
Blackmail (2005 film)
Black Friday (2007 film)
Black (2005 film)
Bittoo Boss
Bioscopewala
Bin Bulaye Baraati
Billu
Big Brother (2007 film)
Bhram
Bhopal: A Prayer for Rain
Bhoothnath Returns
Bhoothnath
Bhoot Unkle
Bhoot Returns
Bhoot (film)
Bhoomi (film)
Bhool Bhulaiyaa
Bhola in Bollywood
Bheja Fry 2
Bheja Fry (film)
Bhavesh Joshi Superhero
Bhanwarey
Bhaiaji Superhit
Bhagmati (2005 film)
Bhagam Bhag
Bhaag Milkha Bhaag
Bhaag Johnny
Bezubaan Ishq
Beyond the Clouds (2017 film)
Bewakoofiyaan
Bewafaa (2005 film)
Being Cyrus
Beiimaan Love
Behen Hogi Teri
Begum Jaan
Befikre
Bbuddah... Hoga Terra Baap
Bazaar E Husn
Batti Gul Meter Chalu
Battalion 609
Batla House
Basti (film)
Bas Ek Pal
Barsaat (2005 film)
Barkhaa
Barfi!
Bareilly Ki Barfi
Bardaasht
Barah Aana
Bank Chor
Banjo (2016 film)
Bangistan
Bang Bang!
Bandook
Band Baaja Baaraat
Banaras (2006 film)
Balwinder Singh Famous Ho Gaya
Bala (2019 film)
Bajrangi Bhaijaan
Bajirao Mastani
Bajatey Raho
Baghban (2003 film)
Badrinath Ki Dulhania
Badmashiyaan
Badlapur Boys
Badlapur (film)
Badla (2019 film)
Badhaai Ho
Bachna Ae Haseeno
Bachke Rehna Re Baba
Bachche Kachche Sachche
Baby (2015 Hindi film)
Babumoshai Bandookbaaz
Babuji Ek Ticket Bambai
Babloo Happy Hai
Baazaar
Baaz: A Bird in Danger
Baat Bann Gayi
Baar Baar Dekho
Baaghi 2
Baaghi (2016 film)
Baabul (2006 film)
Baabarr
B.A. Pass
Azhar (film)
Awarapan
Awara Paagal Deewana
Aval (2017 film)
Aurangzeb (film)
Aur Pappu Paas Ho Gaya
Ata Pata Laapata
Asambhav
Aryan: Unbreakable
Article 15 (film)
Armaan (2003 film)
Arjun: The Warrior Prince
Arjun Patiala
Apne
Apna Sapna Money Money
Apna Asmaan
Apartment (film)
Apaharan
Anwar (2007 film)
Anuradha (2014 film)
Anthony Kaun Hai?
Antardwand
Anna (2016 film)
Ankur Arora Murder Case
Ankhon Dekhi
Ankahee (2006 film)
Anjaane (2005 film)
Anjaana Anjaani
Angel (2011 film)
Andhadhun
Andaaz
Andaaz
Anamika (2008 film)
Anaarkali of Aarah
Amit Sahni Ki List
Amavas
Always Kabhi Kabhi
Aloo Chaat (film)
Alone (2015 Hindi film)
Allah Ke Banday
All the Best: Fun Begins
All Is Well (2015 film)
Aligarh (film)
Albert Pinto Ko Gussa Kyun Aata Hai?
Alag
Aladin (film)
Aksar 2
Aksar
Akaash Vani
Ajji
Ajab Prem Ki Ghazab Kahani
Ajab Gazabb Love
Aiyyaa
Aiyaary
Aitraaz
Aisa Yeh Jahaan
Aisa Kyon Hota Hai?
Airlift (film)
Ahista Ahista (2006 film)
Agnipankh
Agneepath (2012 film)
Aggar (film)
Agent Vinod (2012 film)
Aetbaar
Ae Dil Hai Mushkil
Adharm (2006 film)
Action Replayy
Action Jackson (2014 film)
Acid Factory
Accident on Hill Road
Ab Tumhare Hawale Watan Saathiyo
Ab Tak Chhappan 2
Aazaan
Aasma: The Sky Is the Limit
Aashiqui.in
Aashiqui 2
Aashiq Banaya Aapne
Aashayein
Aarakshan
Aapko Pehle Bhi Kahin Dekha Hai
Aap Ki Khatir (2006 film)
Aap Kaa Surroor
Aankhen (2002 film)
Aanch
Aan: Men at Work
Aalaap (film)
Aakrosh (2010 film)
Aakhari Decision
Aaja Nachle
Aaj Ka Andha Kanoon
Aagey Se Right
Aag (2007 film)
Aabra Ka Daabra
Aa Gaya Hero
Aa Dekhen Zara
ABCD 2
A Gentleman
A Flying Jatt
A Flat (film)
A Decent Arrangement
?: A Question Mark
99.9 FM (film)
99 (2009 film)
88 Antop Hill
7½ Phere
7 Khoon Maaf
7 Hours to Go
68 Pages
5 Weddings
404 (film)
3G (film)
36 China Town
31st October (film)
3 Storeys
3 Idiots
3 Deewarein
3 A.M. (2014 film)
22 Yards
2 States (2014 film)
1971 (2007 film)
1920: The Evil Returns
1920: London
1920 (film)

# Arithmetic Operators(Broadcasting)
100 + marks_series
     
maths      167
english    200
science    189
hindi      200
sst        190
evs        200
Name: nitish ke marks, dtype: int64

# Relational Operators

vk >= 50
     
match_no
1      False
2      False
3      False
4      False
5      False
       ...  
211    False
212    False
213     True
214    False
215    False
Name: runs, Length: 215, dtype: bool
Boolean Indexing on Series

# Find no of 50's and 100's scored by kohli
vk[vk >= 50].size
     
50

# find number of ducks
vk[vk == 0].size
     
9

# Count number of day when I had more than 200 subs a day
subs[subs > 200].size
     
59

# find actors who have done more than 20 movies
num_movies = movies.value_counts()
num_movies[num_movies > 20]
     
Akshay Kumar        48
Amitabh Bachchan    45
Ajay Devgn          38
Salman Khan         31
Sanjay Dutt         26
Shah Rukh Khan      22
Emraan Hashmi       21
Name: lead, dtype: int64
Plotting Graphs on Series

subs.plot()
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f54e0531a60>


movies.value_counts().head(20).plot(kind='pie')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f54e04f6850>

Some Important Series Methods

# astype
# between
# clip
# drop_duplicates
# isnull
# dropna
# fillna
# isin
# apply
# copy
     

import numpy as np
import pandas as pd
     

subs = pd.read_csv('/content/subs.csv',squeeze=True)
subs
     
0       48
1       57
2       40
3       43
4       44
      ... 
360    231
361    226
362    155
363    144
364    172
Name: Subscribers gained, Length: 365, dtype: int64

vk = pd.read_csv('/content/kohli_ipl.csv',index_col='match_no',squeeze=True)
vk
     
match_no
1       1
2      23
3      13
4      12
5       1
       ..
211     0
212    20
213    73
214    25
215     7
Name: runs, Length: 215, dtype: int64

movies = pd.read_csv('/content/bollywood.csv',index_col='movie',squeeze=True)
movies
     
movie
Uri: The Surgical Strike                   Vicky Kaushal
Battalion 609                                Vicky Ahuja
The Accidental Prime Minister (film)         Anupam Kher
Why Cheat India                            Emraan Hashmi
Evening Shadows                         Mona Ambegaonkar
                                              ...       
Hum Tumhare Hain Sanam                    Shah Rukh Khan
Aankhen (2002 film)                     Amitabh Bachchan
Saathiya (film)                             Vivek Oberoi
Company (film)                                Ajay Devgn
Awara Paagal Deewana                        Akshay Kumar
Name: lead, Length: 1500, dtype: object

# astype
import sys
sys.getsizeof(vk)
     
3456

sys.getsizeof(vk.astype('int16'))
     
2166

# between
vk[vk.between(51,99)].size
     
43


     

# clip
subs
     
0       48
1       57
2       40
3       43
4       44
      ... 
360    231
361    226
362    155
363    144
364    172
Name: Subscribers gained, Length: 365, dtype: int64

subs.clip(100,200)
     
0      100
1      100
2      100
3      100
4      100
      ... 
360    200
361    200
362    155
363    144
364    172
Name: Subscribers gained, Length: 365, dtype: int64

# drop_duplicates
temp = pd.Series([1,1,2,2,3,3,4,4])
temp
     
0    1
1    1
2    2
3    2
4    3
5    3
6    4
7    4
dtype: int64

temp.drop_duplicates(keep='last')
     
1    1
3    2
5    3
7    4
dtype: int64

temp.duplicated().sum()
     
4

vk.duplicated().sum()
     
137

movies.drop_duplicates()
     
movie
Uri: The Surgical Strike                   Vicky Kaushal
Battalion 609                                Vicky Ahuja
The Accidental Prime Minister (film)         Anupam Kher
Why Cheat India                            Emraan Hashmi
Evening Shadows                         Mona Ambegaonkar
                                              ...       
Sssshhh...                              Tanishaa Mukerji
Rules: Pyaar Ka Superhit Formula                  Tanuja
Right Here Right Now (film)                        Ankit
Talaash: The Hunt Begins...                Rakhee Gulzar
The Pink Mirror                          Edwin Fernandes
Name: lead, Length: 566, dtype: object

temp = pd.Series([1,2,3,np.nan,5,6,np.nan,8,np.nan,10])
temp
     
0     1.0
1     2.0
2     3.0
3     NaN
4     5.0
5     6.0
6     NaN
7     8.0
8     NaN
9    10.0
dtype: float64

temp.size
     
10

temp.count()
     
7

# isnull
temp.isnull().sum()
     
3


     

# dropna
temp.dropna()
     
0     1.0
1     2.0
2     3.0
4     5.0
5     6.0
7     8.0
9    10.0
dtype: float64


     

# fillna
temp.fillna(temp.mean())
     
0     1.0
1     2.0
2     3.0
3     5.0
4     5.0
5     6.0
6     5.0
7     8.0
8     5.0
9    10.0
dtype: float64


     

# isin
vk[(vk == 49) | (vk == 99)]
     
match_no
82    99
86    49
Name: runs, dtype: int64

vk[vk.isin([49,99])]
     
match_no
82    99
86    49
Name: runs, dtype: int64


     

# apply
movies
     
movie
Uri: The Surgical Strike                   Vicky Kaushal
Battalion 609                                Vicky Ahuja
The Accidental Prime Minister (film)         Anupam Kher
Why Cheat India                            Emraan Hashmi
Evening Shadows                         Mona Ambegaonkar
                                              ...       
Hum Tumhare Hain Sanam                    Shah Rukh Khan
Aankhen (2002 film)                     Amitabh Bachchan
Saathiya (film)                             Vivek Oberoi
Company (film)                                Ajay Devgn
Awara Paagal Deewana                        Akshay Kumar
Name: lead, Length: 1500, dtype: object

movies.apply(lambda x:x.split()[0].upper())
     
movie
Uri: The Surgical Strike                  VICKY
Battalion 609                             VICKY
The Accidental Prime Minister (film)     ANUPAM
Why Cheat India                          EMRAAN
Evening Shadows                            MONA
                                         ...   
Hum Tumhare Hain Sanam                     SHAH
Aankhen (2002 film)                     AMITABH
Saathiya (film)                           VIVEK
Company (film)                             AJAY
Awara Paagal Deewana                     AKSHAY
Name: lead, Length: 1500, dtype: object

subs
     
0       48
1       57
2       40
3       43
4       44
      ... 
360    231
361    226
362    155
363    144
364    172
Name: Subscribers gained, Length: 365, dtype: int64

subs.apply(lambda x:'good day' if x > subs.mean() else 'bad day')
     
0       bad day
1       bad day
2       bad day
3       bad day
4       bad day
         ...   
360    good day
361    good day
362    good day
363    good day
364    good day
Name: Subscribers gained, Length: 365, dtype: object

subs.mean()
     
135.64383561643837

# copy
     

vk
     
match_no
1       1
2      23
3      13
4      12
5       1
       ..
211     0
212    20
213    73
214    25
215     7
Name: runs, Length: 215, dtype: int64

new = vk.head()
     

new
     
match_no
1     1
2    23
3    13
4    12
5     1
Name: runs, dtype: int64

new[1] = 1
     

new = vk.head().copy()
     

new[1] = 100
     

new
     
match_no
1    100
2     23
3     13
4     12
5      1
Name: runs, dtype: int64

vk
     
match_no
1       1
2      23
3      13
4      12
5       1
       ..
211     0
212    20
213    73
214    25
215     7
Name: runs, Length: 215, dtype: int64


     
###################################################################################




import numpy as np
import pandas as pd
     
Creating DataFrame

# using lists
student_data = [
    [100,80,10],
    [90,70,7],
    [120,100,14],
    [80,50,2]
]

pd.DataFrame(student_data,columns=['iq','marks','package'])
     
iq	marks	package
0	100	80	10
1	90	70	7
2	120	100	14
3	80	50	2

# using dicts

student_dict = {
    'name':['nitish','ankit','rupesh','rishabh','amit','ankita'],
    'iq':[100,90,120,80,0,0],
    'marks':[80,70,100,50,0,0],
    'package':[10,7,14,2,0,0]
}

students = pd.DataFrame(student_dict)
students.set_index('name',inplace=True)
students
     
iq	marks	package
name			
nitish	100	80	10
ankit	90	70	7
rupesh	120	100	14
rishabh	80	50	2
amit	0	0	0
ankita	0	0	0

# using read_csv
movies = pd.read_csv('movies.csv')
movies
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
1	Battalion 609	tt9472208	NaN	https://en.wikipedia.org/wiki/Battalion_609	Battalion 609	Battalion 609	0	2019	131	War	4.1	73	The story revolves around a cricket match betw...	The story of Battalion 609 revolves around a c...	NaN	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	NaN	11 January 2019 (India)
2	The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Accidental_P...	The Accidental Prime Minister	The Accidental Prime Minister	0	2019	112	Biography|Drama	6.1	5549	Based on the memoir by Indian policy analyst S...	Explores Manmohan Singh's tenure as the Prime ...	NaN	Anupam Kher|Akshaye Khanna|Aahana Kumra|Atul S...	NaN	11 January 2019 (USA)
3	Why Cheat India	tt8108208	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Why_Cheat_India	Why Cheat India	Why Cheat India	0	2019	121	Crime|Drama	6.0	1891	The movie focuses on existing malpractices in ...	The movie focuses on existing malpractices in ...	NaN	Emraan Hashmi|Shreya Dhanwanthary|Snighdadeep ...	NaN	18 January 2019 (USA)
4	Evening Shadows	tt6028796	NaN	https://en.wikipedia.org/wiki/Evening_Shadows	Evening Shadows	Evening Shadows	0	2018	102	Drama	7.3	280	While gay rights and marriage equality has bee...	Under the 'Evening Shadows' truth often plays...	NaN	Mona Ambegaonkar|Ananth Narayan Mahadevan|Deva...	17 wins & 1 nomination	11 January 2019 (India)
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1624	Tera Mera Saath Rahen	tt0301250	https://upload.wikimedia.org/wikipedia/en/2/2b...	https://en.wikipedia.org/wiki/Tera_Mera_Saath_...	Tera Mera Saath Rahen	Tera Mera Saath Rahen	0	2001	148	Drama	4.9	278	Raj Dixit lives with his younger brother Rahu...	A man is torn between his handicapped brother ...	NaN	Ajay Devgn|Sonali Bendre|Namrata Shirodkar|Pre...	NaN	7 November 2001 (India)
1625	Yeh Zindagi Ka Safar	tt0298607	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yeh_Zindagi_Ka_S...	Yeh Zindagi Ka Safar	Yeh Zindagi Ka Safar	0	2001	146	Drama	3.0	133	Hindi pop-star Sarina Devan lives a wealthy ...	A singer finds out she was adopted when the ed...	NaN	Ameesha Patel|Jimmy Sheirgill|Nafisa Ali|Gulsh...	NaN	16 November 2001 (India)
1626	Sabse Bada Sukh	tt0069204	NaN	https://en.wikipedia.org/wiki/Sabse_Bada_Sukh	Sabse Bada Sukh	Sabse Bada Sukh	0	2018	\N	Comedy|Drama	6.1	13	Village born Lalloo re-locates to Bombay and ...	Village born Lalloo re-locates to Bombay and ...	NaN	Vijay Arora|Asrani|Rajni Bala|Kumud Damle|Utpa...	NaN	NaN
1627	Daaka	tt10833860	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Daaka	Daaka	Daaka	0	2019	136	Action	7.4	38	Shinda tries robbing a bank so he can be wealt...	Shinda tries robbing a bank so he can be wealt...	NaN	Gippy Grewal|Zareen Khan|	NaN	1 November 2019 (USA)
1628	Humsafar	tt2403201	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Humsafar	Humsafar	Humsafar	0	2011	35	Drama|Romance	9.0	2968	Sara and Ashar are childhood friends who share...	Ashar and Khirad are forced to get married due...	NaN	Fawad Khan|	NaN	TV Series (2011–2012)
1629 rows × 18 columns


ipl = pd.read_csv('ipl-matches.csv')
ipl
     
ID	City	Date	Season	MatchNumber	Team1	Team2	Venue	TossWinner	TossDecision	SuperOver	WinningTeam	WonBy	Margin	method	Player_of_Match	Team1Players	Team2Players	Umpire1	Umpire2
0	1312200	Ahmedabad	2022-05-29	2022	Final	Rajasthan Royals	Gujarat Titans	Narendra Modi Stadium, Ahmedabad	Rajasthan Royals	bat	N	Gujarat Titans	Wickets	7.0	NaN	HH Pandya	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	['WP Saha', 'Shubman Gill', 'MS Wade', 'HH Pan...	CB Gaffaney	Nitin Menon
1	1312199	Ahmedabad	2022-05-27	2022	Qualifier 2	Royal Challengers Bangalore	Rajasthan Royals	Narendra Modi Stadium, Ahmedabad	Rajasthan Royals	field	N	Rajasthan Royals	Wickets	7.0	NaN	JC Buttler	['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ ...	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	CB Gaffaney	Nitin Menon
2	1312198	Kolkata	2022-05-25	2022	Eliminator	Royal Challengers Bangalore	Lucknow Super Giants	Eden Gardens, Kolkata	Lucknow Super Giants	field	N	Royal Challengers Bangalore	Runs	14.0	NaN	RM Patidar	['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ ...	['Q de Kock', 'KL Rahul', 'M Vohra', 'DJ Hooda...	J Madanagopal	MA Gough
3	1312197	Kolkata	2022-05-24	2022	Qualifier 1	Rajasthan Royals	Gujarat Titans	Eden Gardens, Kolkata	Gujarat Titans	field	N	Gujarat Titans	Wickets	7.0	NaN	DA Miller	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	['WP Saha', 'Shubman Gill', 'MS Wade', 'HH Pan...	BNJ Oxenford	VK Sharma
4	1304116	Mumbai	2022-05-22	2022	70	Sunrisers Hyderabad	Punjab Kings	Wankhede Stadium, Mumbai	Sunrisers Hyderabad	bat	N	Punjab Kings	Wickets	5.0	NaN	Harpreet Brar	['PK Garg', 'Abhishek Sharma', 'RA Tripathi', ...	['JM Bairstow', 'S Dhawan', 'M Shahrukh Khan',...	AK Chaudhary	NA Patwardhan
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
945	335986	Kolkata	2008-04-20	2007/08	4	Kolkata Knight Riders	Deccan Chargers	Eden Gardens	Deccan Chargers	bat	N	Kolkata Knight Riders	Wickets	5.0	NaN	DJ Hussey	['WP Saha', 'BB McCullum', 'RT Ponting', 'SC G...	['AC Gilchrist', 'Y Venugopal Rao', 'VVS Laxma...	BF Bowden	K Hariharan
946	335985	Mumbai	2008-04-20	2007/08	5	Mumbai Indians	Royal Challengers Bangalore	Wankhede Stadium	Mumbai Indians	bat	N	Royal Challengers Bangalore	Wickets	5.0	NaN	MV Boucher	['L Ronchi', 'ST Jayasuriya', 'DJ Thornely', '...	['S Chanderpaul', 'R Dravid', 'LRPL Taylor', '...	SJ Davis	DJ Harper
947	335984	Delhi	2008-04-19	2007/08	3	Delhi Daredevils	Rajasthan Royals	Feroz Shah Kotla	Rajasthan Royals	bat	N	Delhi Daredevils	Wickets	9.0	NaN	MF Maharoof	['G Gambhir', 'V Sehwag', 'S Dhawan', 'MK Tiwa...	['T Kohli', 'YK Pathan', 'SR Watson', 'M Kaif'...	Aleem Dar	GA Pratapkumar
948	335983	Chandigarh	2008-04-19	2007/08	2	Kings XI Punjab	Chennai Super Kings	Punjab Cricket Association Stadium, Mohali	Chennai Super Kings	bat	N	Chennai Super Kings	Runs	33.0	NaN	MEK Hussey	['K Goel', 'JR Hopes', 'KC Sangakkara', 'Yuvra...	['PA Patel', 'ML Hayden', 'MEK Hussey', 'MS Dh...	MR Benson	SL Shastri
949	335982	Bangalore	2008-04-18	2007/08	1	Royal Challengers Bangalore	Kolkata Knight Riders	M Chinnaswamy Stadium	Royal Challengers Bangalore	field	N	Kolkata Knight Riders	Runs	140.0	NaN	BB McCullum	['R Dravid', 'W Jaffer', 'V Kohli', 'JH Kallis...	['SC Ganguly', 'BB McCullum', 'RT Ponting', 'D...	Asad Rauf	RE Koertzen
950 rows × 20 columns

DataFrame Attributes and Methods

# shape
movies.shape
ipl.shape
     
(950, 20)

# dtypes
movies.dtypes
ipl.dtypes
     
ID                   int64
City                object
Date                object
Season              object
MatchNumber         object
Team1               object
Team2               object
Venue               object
TossWinner          object
TossDecision        object
SuperOver           object
WinningTeam         object
WonBy               object
Margin             float64
method              object
Player_of_Match     object
Team1Players        object
Team2Players        object
Umpire1             object
Umpire2             object
dtype: object

# index
movies.index
ipl.index
     
RangeIndex(start=0, stop=950, step=1)

# columns
movies.columns
ipl.columns
student.columns
     
Index(['iq', 'marks', 'package'], dtype='object')

# values
student.values
ipl.values
     
array([[1312200, 'Ahmedabad', '2022-05-29', ...,
        "['WP Saha', 'Shubman Gill', 'MS Wade', 'HH Pandya', 'DA Miller', 'R Tewatia', 'Rashid Khan', 'R Sai Kishore', 'LH Ferguson', 'Yash Dayal', 'Mohammed Shami']",
        'CB Gaffaney', 'Nitin Menon'],
       [1312199, 'Ahmedabad', '2022-05-27', ...,
        "['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D Padikkal', 'SO Hetmyer', 'R Parag', 'R Ashwin', 'TA Boult', 'YS Chahal', 'M Prasidh Krishna', 'OC McCoy']",
        'CB Gaffaney', 'Nitin Menon'],
       [1312198, 'Kolkata', '2022-05-25', ...,
        "['Q de Kock', 'KL Rahul', 'M Vohra', 'DJ Hooda', 'MP Stoinis', 'E Lewis', 'KH Pandya', 'PVD Chameera', 'Mohsin Khan', 'Avesh Khan', 'Ravi Bishnoi']",
        'J Madanagopal', 'MA Gough'],
       ...,
       [335984, 'Delhi', '2008-04-19', ...,
        "['T Kohli', 'YK Pathan', 'SR Watson', 'M Kaif', 'DS Lehmann', 'RA Jadeja', 'M Rawat', 'D Salunkhe', 'SK Warne', 'SK Trivedi', 'MM Patel']",
        'Aleem Dar', 'GA Pratapkumar'],
       [335983, 'Chandigarh', '2008-04-19', ...,
        "['PA Patel', 'ML Hayden', 'MEK Hussey', 'MS Dhoni', 'SK Raina', 'JDP Oram', 'S Badrinath', 'Joginder Sharma', 'P Amarnath', 'MS Gony', 'M Muralitharan']",
        'MR Benson', 'SL Shastri'],
       [335982, 'Bangalore', '2008-04-18', ...,
        "['SC Ganguly', 'BB McCullum', 'RT Ponting', 'DJ Hussey', 'Mohammad Hafeez', 'LR Shukla', 'WP Saha', 'AB Agarkar', 'AB Dinda', 'M Kartik', 'I Sharma']",
        'Asad Rauf', 'RE Koertzen']], dtype=object)

# head and tail
movies.head(2)
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
1	Battalion 609	tt9472208	NaN	https://en.wikipedia.org/wiki/Battalion_609	Battalion 609	Battalion 609	0	2019	131	War	4.1	73	The story revolves around a cricket match betw...	The story of Battalion 609 revolves around a c...	NaN	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	NaN	11 January 2019 (India)

ipl.tail(2)
     
ID	City	Date	Season	MatchNumber	Team1	Team2	Venue	TossWinner	TossDecision	SuperOver	WinningTeam	WonBy	Margin	method	Player_of_Match	Team1Players	Team2Players	Umpire1	Umpire2
948	335983	Chandigarh	2008-04-19	2007/08	2	Kings XI Punjab	Chennai Super Kings	Punjab Cricket Association Stadium, Mohali	Chennai Super Kings	bat	N	Chennai Super Kings	Runs	33.0	NaN	MEK Hussey	['K Goel', 'JR Hopes', 'KC Sangakkara', 'Yuvra...	['PA Patel', 'ML Hayden', 'MEK Hussey', 'MS Dh...	MR Benson	SL Shastri
949	335982	Bangalore	2008-04-18	2007/08	1	Royal Challengers Bangalore	Kolkata Knight Riders	M Chinnaswamy Stadium	Royal Challengers Bangalore	field	N	Kolkata Knight Riders	Runs	140.0	NaN	BB McCullum	['R Dravid', 'W Jaffer', 'V Kohli', 'JH Kallis...	['SC Ganguly', 'BB McCullum', 'RT Ponting', 'D...	Asad Rauf	RE Koertzen

# sample
ipl.sample(5)
     
ID	City	Date	Season	MatchNumber	Team1	Team2	Venue	TossWinner	TossDecision	SuperOver	WinningTeam	WonBy	Margin	method	Player_of_Match	Team1Players	Team2Players	Umpire1	Umpire2
336	1082628	Mumbai	2017-05-01	2017	38	Mumbai Indians	Royal Challengers Bangalore	Wankhede Stadium	Royal Challengers Bangalore	bat	N	Mumbai Indians	Wickets	5.0	NaN	RG Sharma	['PA Patel', 'JC Buttler', 'N Rana', 'RG Sharm...	['V Kohli', 'Mandeep Singh', 'TM Head', 'AB de...	AK Chaudhary	CB Gaffaney
98	1254107	Sharjah	2021-09-25	2021	37	Punjab Kings	Sunrisers Hyderabad	Sharjah Cricket Stadium	Sunrisers Hyderabad	field	N	Punjab Kings	Runs	5.0	NaN	JO Holder	['KL Rahul', 'MA Agarwal', 'CH Gayle', 'AK Mar...	['DA Warner', 'WP Saha', 'KS Williamson', 'MK ...	RK Illingworth	YC Barde
890	392182	Cape Town	2009-04-18	2009	2	Royal Challengers Bangalore	Rajasthan Royals	Newlands	Royal Challengers Bangalore	bat	N	Royal Challengers Bangalore	Runs	75.0	NaN	R Dravid	['JD Ryder', 'RV Uthappa', 'LRPL Taylor', 'KP ...	['GC Smith', 'SA Asnodkar', 'NK Patel', 'T Hen...	BR Doctrove	RB Tiffin
157	1216533	Abu Dhabi	2020-10-19	2020/21	37	Chennai Super Kings	Rajasthan Royals	Sheikh Zayed Stadium	Chennai Super Kings	bat	N	Rajasthan Royals	Wickets	7.0	NaN	JC Buttler	['SM Curran', 'F du Plessis', 'SR Watson', 'AT...	['BA Stokes', 'RV Uthappa', 'SV Samson', 'SPD ...	CB Gaffaney	VK Sharma
386	980991	Chandigarh	2016-05-15	2016	46	Kings XI Punjab	Sunrisers Hyderabad	Punjab Cricket Association IS Bindra Stadium, ...	Kings XI Punjab	bat	N	Sunrisers Hyderabad	Wickets	7.0	NaN	HM Amla	['HM Amla', 'M Vijay', 'WP Saha', 'Gurkeerat S...	['DA Warner', 'S Dhawan', 'DJ Hooda', 'Yuvraj ...	KN Ananthapadmanabhan	M Erasmus

# info
movies.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1629 entries, 0 to 1628
Data columns (total 18 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   title_x           1629 non-null   object 
 1   imdb_id           1629 non-null   object 
 2   poster_path       1526 non-null   object 
 3   wiki_link         1629 non-null   object 
 4   title_y           1629 non-null   object 
 5   original_title    1629 non-null   object 
 6   is_adult          1629 non-null   int64  
 7   year_of_release   1629 non-null   int64  
 8   runtime           1629 non-null   object 
 9   genres            1629 non-null   object 
 10  imdb_rating       1629 non-null   float64
 11  imdb_votes        1629 non-null   int64  
 12  story             1609 non-null   object 
 13  summary           1629 non-null   object 
 14  tagline           557 non-null    object 
 15  actors            1624 non-null   object 
 16  wins_nominations  707 non-null    object 
 17  release_date      1522 non-null   object 
dtypes: float64(1), int64(3), object(14)
memory usage: 229.2+ KB

ipl.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 950 entries, 0 to 949
Data columns (total 20 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   ID               950 non-null    int64  
 1   City             899 non-null    object 
 2   Date             950 non-null    object 
 3   Season           950 non-null    object 
 4   MatchNumber      950 non-null    object 
 5   Team1            950 non-null    object 
 6   Team2            950 non-null    object 
 7   Venue            950 non-null    object 
 8   TossWinner       950 non-null    object 
 9   TossDecision     950 non-null    object 
 10  SuperOver        946 non-null    object 
 11  WinningTeam      946 non-null    object 
 12  WonBy            950 non-null    object 
 13  Margin           932 non-null    float64
 14  method           19 non-null     object 
 15  Player_of_Match  946 non-null    object 
 16  Team1Players     950 non-null    object 
 17  Team2Players     950 non-null    object 
 18  Umpire1          950 non-null    object 
 19  Umpire2          950 non-null    object 
dtypes: float64(1), int64(1), object(18)
memory usage: 148.6+ KB

# describe
movies.describe()
     
is_adult	year_of_release	imdb_rating	imdb_votes
count	1629.0	1629.000000	1629.000000	1629.000000
mean	0.0	2010.263966	5.557459	5384.263352
std	0.0	5.381542	1.567609	14552.103231
min	0.0	2001.000000	0.000000	0.000000
25%	0.0	2005.000000	4.400000	233.000000
50%	0.0	2011.000000	5.600000	1000.000000
75%	0.0	2015.000000	6.800000	4287.000000
max	0.0	2019.000000	9.400000	310481.000000

ipl.describe()
     
ID	Margin
count	9.500000e+02	932.000000
mean	8.304852e+05	17.056867
std	3.375678e+05	21.633109
min	3.359820e+05	1.000000
25%	5.012612e+05	6.000000
50%	8.297380e+05	8.000000
75%	1.175372e+06	19.000000
max	1.312200e+06	146.000000

# isnull
movies.isnull().sum()
     
title_x                0
imdb_id                0
poster_path          103
wiki_link              0
title_y                0
original_title         0
is_adult               0
year_of_release        0
runtime                0
genres                 0
imdb_rating            0
imdb_votes             0
story                 20
summary                0
tagline             1072
actors                 5
wins_nominations     922
release_date         107
dtype: int64

# duplicated
movies.duplicated().sum()
     
0

students.duplicated().sum()
     
1

# rename
students
     
iq	percent	lpa
0	100	80	10
1	90	70	7
2	120	100	14
3	80	50	2
4	0	0	0
5	0	0	0

students.rename(columns={'marks':'percent','package':'lpa'},inplace=True)
     
Math Methods

# sum -> axis argument
students.sum(axis=0)
     
iq         390
percent    300
lpa         33
dtype: int64

students.mean(axis=1)
     
0    63.333333
1    55.666667
2    78.000000
3    44.000000
4     0.000000
5     0.000000
dtype: float64

students.var()
     
iq         2710.0
percent    1760.0
lpa          33.5
dtype: float64


     
Selecting cols from a DataFrame

# single cols
movies['title_x']
     
0                   Uri: The Surgical Strike
1                              Battalion 609
2       The Accidental Prime Minister (film)
3                            Why Cheat India
4                            Evening Shadows
                        ...                 
1624                   Tera Mera Saath Rahen
1625                    Yeh Zindagi Ka Safar
1626                         Sabse Bada Sukh
1627                                   Daaka
1628                                Humsafar
Name: title_x, Length: 1629, dtype: object

ipl['Venue']
     
0                Narendra Modi Stadium, Ahmedabad
1                Narendra Modi Stadium, Ahmedabad
2                           Eden Gardens, Kolkata
3                           Eden Gardens, Kolkata
4                        Wankhede Stadium, Mumbai
                          ...                    
945                                  Eden Gardens
946                              Wankhede Stadium
947                              Feroz Shah Kotla
948    Punjab Cricket Association Stadium, Mohali
949                         M Chinnaswamy Stadium
Name: Venue, Length: 950, dtype: object

# multiple cols
movies[['year_of_release','actors','title_x']]
     
year_of_release	actors	title_x
0	2019	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	Uri: The Surgical Strike
1	2019	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	Battalion 609
2	2019	Anupam Kher|Akshaye Khanna|Aahana Kumra|Atul S...	The Accidental Prime Minister (film)
3	2019	Emraan Hashmi|Shreya Dhanwanthary|Snighdadeep ...	Why Cheat India
4	2018	Mona Ambegaonkar|Ananth Narayan Mahadevan|Deva...	Evening Shadows
...	...	...	...
1624	2001	Ajay Devgn|Sonali Bendre|Namrata Shirodkar|Pre...	Tera Mera Saath Rahen
1625	2001	Ameesha Patel|Jimmy Sheirgill|Nafisa Ali|Gulsh...	Yeh Zindagi Ka Safar
1626	2018	Vijay Arora|Asrani|Rajni Bala|Kumud Damle|Utpa...	Sabse Bada Sukh
1627	2019	Gippy Grewal|Zareen Khan|	Daaka
1628	2011	Fawad Khan|	Humsafar
1629 rows × 3 columns


ipl[['Team1','Team2','WinningTeam']]
     
Team1	Team2	WinningTeam
0	Rajasthan Royals	Gujarat Titans	Gujarat Titans
1	Royal Challengers Bangalore	Rajasthan Royals	Rajasthan Royals
2	Royal Challengers Bangalore	Lucknow Super Giants	Royal Challengers Bangalore
3	Rajasthan Royals	Gujarat Titans	Gujarat Titans
4	Sunrisers Hyderabad	Punjab Kings	Punjab Kings
...	...	...	...
945	Kolkata Knight Riders	Deccan Chargers	Kolkata Knight Riders
946	Mumbai Indians	Royal Challengers Bangalore	Royal Challengers Bangalore
947	Delhi Daredevils	Rajasthan Royals	Delhi Daredevils
948	Kings XI Punjab	Chennai Super Kings	Chennai Super Kings
949	Royal Challengers Bangalore	Kolkata Knight Riders	Kolkata Knight Riders
950 rows × 3 columns

Selecting rows from a DataFrame
iloc - searches using index positions
loc - searches using index labels

# single row
movies.iloc[5]
     
title_x                                                   Soni (film)
imdb_id                                                     tt6078866
poster_path         https://upload.wikimedia.org/wikipedia/en/thum...
wiki_link                   https://en.wikipedia.org/wiki/Soni_(film)
title_y                                                          Soni
original_title                                                   Soni
is_adult                                                            0
year_of_release                                                  2018
runtime                                                            97
genres                                                          Drama
imdb_rating                                                       7.2
imdb_votes                                                       1595
story               Soni  a young policewoman in Delhi  and her su...
summary             While fighting crimes against women in Delhi  ...
tagline                                                           NaN
actors              Geetika Vidya Ohlyan|Saloni Batra|Vikas Shukla...
wins_nominations                               3 wins & 5 nominations
release_date                                    18 January 2019 (USA)
Name: 5, dtype: object

# multiple row
movies.iloc[:5]
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
1	Battalion 609	tt9472208	NaN	https://en.wikipedia.org/wiki/Battalion_609	Battalion 609	Battalion 609	0	2019	131	War	4.1	73	The story revolves around a cricket match betw...	The story of Battalion 609 revolves around a c...	NaN	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	NaN	11 January 2019 (India)
2	The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Accidental_P...	The Accidental Prime Minister	The Accidental Prime Minister	0	2019	112	Biography|Drama	6.1	5549	Based on the memoir by Indian policy analyst S...	Explores Manmohan Singh's tenure as the Prime ...	NaN	Anupam Kher|Akshaye Khanna|Aahana Kumra|Atul S...	NaN	11 January 2019 (USA)
3	Why Cheat India	tt8108208	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Why_Cheat_India	Why Cheat India	Why Cheat India	0	2019	121	Crime|Drama	6.0	1891	The movie focuses on existing malpractices in ...	The movie focuses on existing malpractices in ...	NaN	Emraan Hashmi|Shreya Dhanwanthary|Snighdadeep ...	NaN	18 January 2019 (USA)
4	Evening Shadows	tt6028796	NaN	https://en.wikipedia.org/wiki/Evening_Shadows	Evening Shadows	Evening Shadows	0	2018	102	Drama	7.3	280	While gay rights and marriage equality has bee...	Under the 'Evening Shadows' truth often plays...	NaN	Mona Ambegaonkar|Ananth Narayan Mahadevan|Deva...	17 wins & 1 nomination	11 January 2019 (India)

# fancy indexing
movies.iloc[[0,4,5]]
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
4	Evening Shadows	tt6028796	NaN	https://en.wikipedia.org/wiki/Evening_Shadows	Evening Shadows	Evening Shadows	0	2018	102	Drama	7.3	280	While gay rights and marriage equality has bee...	Under the 'Evening Shadows' truth often plays...	NaN	Mona Ambegaonkar|Ananth Narayan Mahadevan|Deva...	17 wins & 1 nomination	11 January 2019 (India)
5	Soni (film)	tt6078866	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Soni_(film)	Soni	Soni	0	2018	97	Drama	7.2	1595	Soni a young policewoman in Delhi and her su...	While fighting crimes against women in Delhi ...	NaN	Geetika Vidya Ohlyan|Saloni Batra|Vikas Shukla...	3 wins & 5 nominations	18 January 2019 (USA)

# loc
students
     
iq	marks	package
name			
nitish	100	80	10
ankit	90	70	7
rupesh	120	100	14
rishabh	80	50	2
amit	0	0	0
ankita	0	0	0

students.loc['nitish']
     
iq         100
marks       80
package     10
Name: nitish, dtype: int64

students.loc['nitish':'rishabh':2]
     
iq	marks	package
name			
nitish	100	80	10
rupesh	120	100	14

students.loc[['nitish','ankita','rupesh']]
     
iq	marks	package
name			
nitish	100	80	10
ankita	0	0	0
rupesh	120	100	14

students.iloc[[0,3,4]]
     
iq	marks	package
name			
nitish	100	80	10
rishabh	80	50	2
amit	0	0	0
Selecting both rows and cols

movies.iloc[0:3,0:3]
     
title_x	imdb_id	poster_path
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...
1	Battalion 609	tt9472208	NaN
2	The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...

movies.loc[0:2,'title_x':'poster_path']
     
title_x	imdb_id	poster_path
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...
1	Battalion 609	tt9472208	NaN
2	The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...
Filtering a DataFrame

ipl.head(2)
     
ID	City	Date	Season	MatchNumber	Team1	Team2	Venue	TossWinner	TossDecision	SuperOver	WinningTeam	WonBy	Margin	method	Player_of_Match	Team1Players	Team2Players	Umpire1	Umpire2
0	1312200	Ahmedabad	2022-05-29	2022	Final	Rajasthan Royals	Gujarat Titans	Narendra Modi Stadium, Ahmedabad	Rajasthan Royals	bat	N	Gujarat Titans	Wickets	7.0	NaN	HH Pandya	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	['WP Saha', 'Shubman Gill', 'MS Wade', 'HH Pan...	CB Gaffaney	Nitin Menon
1	1312199	Ahmedabad	2022-05-27	2022	Qualifier 2	Royal Challengers Bangalore	Rajasthan Royals	Narendra Modi Stadium, Ahmedabad	Rajasthan Royals	field	N	Rajasthan Royals	Wickets	7.0	NaN	JC Buttler	['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ ...	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	CB Gaffaney	Nitin Menon

# find all the final winners
mask = ipl['MatchNumber'] == 'Final'
new_df = ipl[mask]
new_df[['Season','WinningTeam']]

ipl[ipl['MatchNumber'] == 'Final'][['Season','WinningTeam']]
     
Season	WinningTeam
0	2022	Gujarat Titans
74	2021	Chennai Super Kings
134	2020/21	Mumbai Indians
194	2019	Mumbai Indians
254	2018	Chennai Super Kings
314	2017	Mumbai Indians
373	2016	Sunrisers Hyderabad
433	2015	Mumbai Indians
492	2014	Kolkata Knight Riders
552	2013	Mumbai Indians
628	2012	Kolkata Knight Riders
702	2011	Chennai Super Kings
775	2009/10	Chennai Super Kings
835	2009	Deccan Chargers
892	2007/08	Rajasthan Royals

# how many super over finishes have occured
ipl[ipl['SuperOver'] == 'Y'].shape[0]
     
14

# how many matches has csk won in kolkata
ipl[(ipl['City'] == 'Kolkata') & (ipl['WinningTeam'] == 'Chennai Super Kings')].shape[0]
     
5

# toss winner is match winner in percentage
(ipl[ipl['TossWinner'] == ipl['WinningTeam']].shape[0]/ipl.shape[0])*100
     
51.473684210526315

# movies with rating higher than 8 and votes>10000
movies[(movies['imdb_rating'] > 8.5) & (movies['imdb_votes'] > 10000)].shape[0]
     
0

# Action movies with rating higher than 7.5
# mask1 = movies['genres'].str.split('|').apply(lambda x:'Action' in x)
mask1 = movies['genres'].str.contains('Action')
mask2 = movies['imdb_rating'] > 7.5

movies[mask1 & mask2]
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
41	Family of Thakurganj	tt8897986	https://upload.wikimedia.org/wikipedia/en/9/99...	https://en.wikipedia.org/wiki/Family_of_Thakur...	Family of Thakurganj	Family of Thakurganj	0	2019	127	Action|Drama	9.4	895	The film is based on small town of North India...	The film is based on small town of North India...	NaN	Jimmy Sheirgill|Mahie Gill|Nandish Singh|Prana...	NaN	19 July 2019 (India)
84	Mukkabaaz	tt7180544	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Mukkabaaz	The Brawler	Mukkabaaz	0	2017	154	Action|Drama|Sport	8.1	5434	A boxer (Shravan) belonging to upper cast tra...	A boxer struggles to make his mark in the boxi...	NaN	Viineet Kumar|Jimmy Sheirgill|Zoya Hussain|Rav...	3 wins & 6 nominations	12 January 2018 (USA)
106	Raazi	tt7098658	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Raazi	Raazi	Raazi	0	2018	138	Action|Drama|Thriller	7.8	20289	Hidayat Khan is the son of an Indian freedom f...	A Kashmiri woman agrees to marry a Pakistani a...	An incredible true story	Alia Bhatt|Vicky Kaushal|Rajit Kapoor|Shishir ...	21 wins & 26 nominations	11 May 2018 (USA)
110	Parmanu: The Story of Pokhran	tt6826438	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Parmanu:_The_Sto...	Parmanu: The Story of Pokhran	Parmanu: The Story of Pokhran	0	2018	129	Action|Drama|History	7.7	18292	Captain Ashwat Raina's efforts to turn India i...	Ashwat Raina and his teammates arrive in Pokhr...	1998| India: one secret operation| six Indians...	John Abraham|Boman Irani|Diana Penty|Anuja Sat...	NaN	25 May 2018 (USA)
112	Bhavesh Joshi Superhero	tt6129302	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Bhavesh_Joshi_Su...	Bhavesh Joshi Superhero	Bhavesh Joshi Superhero	0	2018	154	Action|Drama	7.6	4928	Bhavesh Joshi Superhero is an action film abou...	The origin story of Bhavesh Joshi an Indian s...	This year| justice will have a new name.	Harshvardhan Kapoor|Priyanshu Painyuli|Ashish ...	2 nominations	1 June 2018 (USA)
169	The Ghazi Attack	tt6299040	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Ghazi_Attack...	The Ghazi Attack	The Ghazi Attack	0	2017	116	Action|Thriller|War	7.6	10332	In 1971 amid rising tensions between India an...	A Pakistani submarine Ghazi plans to secretly...	The war you did not know about	Rana Daggubati|Kay Kay Menon|Atul Kulkarni|Om ...	1 win & 7 nominations	17 February 2017 (USA)
219	Raag Desh (film)	tt6080746	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Raagdesh	Raag Desh	Raag Desh	0	2017	135	Action|Drama|History	8.3	341	A period film based on the historic 1945 India...	A period film based on the historic 1945 India...	NaN	Kunal Kapoor|Amit Sadh|Mohit Marwah|Kenneth De...	NaN	28 July 2017 (India)
258	Irudhi Suttru	tt5310090	https://upload.wikimedia.org/wikipedia/en/f/fe...	https://en.wikipedia.org/wiki/Saala_Khadoos	Saala Khadoos	Saala Khadoos	0	2016	109	Action|Drama|Sport	7.6	10507	An under-fire boxing coach Prabhu is transfer...	The story of a former boxer who quits boxing f...	NaN	Madhavan|Ritika Singh|Mumtaz Sorcar|Nassar|Rad...	9 wins & 2 nominations	29 January 2016 (USA)
280	Laal Rang	tt5600714	NaN	https://en.wikipedia.org/wiki/Laal_Rang	Laal Rang	Laal Rang	0	2016	147	Action|Crime|Drama	8.0	3741	The friendship of two men is tested when thing...	The friendship of two men is tested when thing...	Every job good or bad| must be done with honesty.	Randeep Hooda|Akshay Oberoi|Rajniesh Duggall|P...	NaN	22 April 2016 (India)
297	Udta Punjab	tt4434004	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Udta_Punjab	Udta Punjab	Udta Punjab	0	2016	148	Action|Crime|Drama	7.8	23995	What on earth can a rock star a migrant labor...	A story that revolves around drug abuse in the...	NaN	Shahid Kapoor|Alia Bhatt|Kareena Kapoor|Diljit...	11 wins & 19 nominations	17 June 2016 (USA)
354	Dangal (film)	tt5074352	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Dangal_(film)	Dangal	Dangal	0	2016	161	Action|Biography|Drama	8.4	131338	Biopic of Mahavir Singh Phogat who taught wre...	Former wrestler Mahavir Singh Phogat and his t...	You think our girls are any lesser than boys?	Aamir Khan|Fatima Sana Shaikh|Sanya Malhotra|S...	23 wins & 4 nominations	21 December 2016 (USA)
362	Bajrangi Bhaijaan	tt3863552	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Bajrangi_Bhaijaan	Bajrangi Bhaijaan	Bajrangi Bhaijaan	0	2015	163	Action|Comedy|Drama	8.0	65877	A little mute girl from a Pakistani village ge...	An Indian man with a magnanimous heart takes a...	NaN	Salman Khan|Harshaali Malhotra|Nawazuddin Sidd...	25 wins & 13 nominations	17 July 2015 (USA)
365	Baby (2015 Hindi film)	tt3848892	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Baby_(2015_Hindi...	Baby	Baby	0	2015	159	Action|Thriller	8.0	49426	The country is perpetually under threat from t...	An elite counter-intelligence unit learns of a...	History Is Made By Those Who Give A Damn!	Akshay Kumar|Danny Denzongpa|Rana Daggubati|Ta...	1 win	23 January 2015 (India)
393	Detective Byomkesh Bakshy!	tt3447364	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Detective_Byomke...	Detective Byomkesh Bakshy!	Detective Byomkesh Bakshy!	0	2015	139	Action|Mystery|Thriller	7.6	14674	CALCUTTA 1943 A WAR - A MYSTERY - and A DETECT...	While investigating the disappearance of a che...	Expect The Unexpected	Sushant Singh Rajput|Anand Tiwari|Neeraj Kabi|...	NaN	3 April 2015 (USA)
449	Titli (2014 film)	tt3019620	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Titli_(2014_film)	Titli	Titli	0	2014	116	Action|Drama|Thriller	7.6	3677	In the badlands of Delhi's dystopic underbelly...	A Hindi feature film set in the lower depths o...	Daring| Desireable| Dangerous	Nawazuddin Siddiqui|Niharika Singh|Anil George...	4 wins & 5 nominations	20 June 2014 (USA)
536	Haider (film)	tt3390572	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Haider_(film)	Haider	Haider	0	2014	160	Action|Crime|Drama	8.1	46912	Vishal Bhardwaj's adaptation of William Shakes...	A young man returns to Kashmir after his fathe...	NaN	Tabu|Shahid Kapoor|Shraddha Kapoor|Kay Kay Men...	28 wins & 24 nominations	2 October 2014 (USA)
589	Vishwaroopam	tt2199711	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Vishwaroop_(Hind...	Vishwaroopam	Vishwaroopam	0	2013	148	Action|Thriller	8.2	38016	Vishwanathan a Kathak dance teacher in New Yo...	When a classical dancer's suspecting wife sets...	NaN	Kamal Haasan|Rahul Bose|Shekhar Kapur|Pooja Ku...	5 wins & 11 nominations	25 January 2013 (India)
625	Madras Cafe	tt2855648	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Madras_Cafe	Madras Cafe	Madras Cafe	0	2013	130	Action|Drama|Thriller	7.7	21393	An Indian Intelligence agent (portrayed by Joh...	An Indian intelligence agent journeys to a war...	NaN	John Abraham|Nargis Fakhri|Raashi Khanna|Praka...	10 wins & 10 nominations	23 August 2013 (India)
668	Paan Singh Tomar (film)	tt1620933	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Paan_Singh_Tomar...	Paan Singh Tomar	Paan Singh Tomar	0	2012	135	Action|Biography|Crime	8.2	29994	Paan Singh Tomar is a Hindi-language film bas...	The story of Paan Singh Tomar an Indian athle...	NaN	Irrfan Khan|	10 wins & 11 nominations	2 March 2012 (USA)
693	Gangs of Wasseypur	tt1954470	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Gangs_of_Wasseypur	Gangs of Wasseypur	Gangs of Wasseypur	0	2012	321	Action|Comedy|Crime	8.2	71636	Shahid Khan is exiled after impersonating the ...	A clash between Sultan and Shahid Khan leads t...	NaN	Manoj Bajpayee|Richa Chadha|Nawazuddin Siddiqu...	12 wins & 43 nominations	2 August 2012 (Singapore)
694	Gangs of Wasseypur – Part 2	tt1954470	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Gangs_of_Wasseyp...	Gangs of Wasseypur	Gangs of Wasseypur	0	2012	321	Action|Comedy|Crime	8.2	71636	Shahid Khan is exiled after impersonating the ...	A clash between Sultan and Shahid Khan leads t...	NaN	Manoj Bajpayee|Richa Chadha|Nawazuddin Siddiqu...	12 wins & 43 nominations	2 August 2012 (Singapore)
982	Jodhaa Akbar	tt0449994	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Jodhaa_Akbar	Jodhaa Akbar	Jodhaa Akbar	0	2008	213	Action|Drama|History	7.6	27541	Jodhaa Akbar is a sixteenth century love story...	A sixteenth century love story about a marriag...	NaN	Hrithik Roshan|Aishwarya Rai Bachchan|Sonu Soo...	32 wins & 21 nominations	15 February 2008 (USA)
1039	1971 (2007 film)	tt0983990	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/1971_(2007_film)	1971	1971	0	2007	160	Action|Drama|War	7.9	1121	Based on true facts the film revolves around ...	Based on true facts the film revolves around ...	Honor the heroes.......	Manoj Bajpayee|Ravi Kishan|Deepak Dobriyal|	1 win	9 March 2007 (India)
1058	Black Friday (2007 film)	tt0400234	https://upload.wikimedia.org/wikipedia/en/5/58...	https://en.wikipedia.org/wiki/Black_Friday_(20...	Black Friday	Black Friday	0	2004	143	Action|Crime|Drama	8.5	16761	A dramatic presentation of the bomb blasts tha...	Black Friday is a film about the investigation...	The story of the Bombay bomb blasts	Kay Kay Menon|Pavan Malhotra|Aditya Srivastava...	3 nominations	9 February 2007 (India)
1188	Omkara (2006 film)	tt0488414	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Omkara_(2006_film)	Omkara	Omkara	0	2006	155	Action|Crime|Drama	8.1	17594	Advocate Raghunath Mishra has arranged the mar...	A politically-minded enforcer's misguided trus...	NaN	Ajay Devgn|Saif Ali Khan|Vivek Oberoi|Kareena ...	19 wins & 20 nominations	28 July 2006 (USA)
1293	Sarkar (2005 film)	tt0432047	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Sarkar_(2005_film)	Sarkar	Sarkar	0	2005	124	Action|Crime|Drama	7.6	14694	Meet Subhash Nagre - a wealthy and influential...	The authority of a man who runs a parallel go...	'There are no Rights and Wrongs. Only Power' -...	Amitabh Bachchan|Abhishek Bachchan|Kay Kay Men...	2 wins & 10 nominations	1 July 2005 (India)
1294	Sehar	tt0477857	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Sehar	Sehar	Sehar	0	2005	125	Action|Crime|Drama	7.8	1861	At the tender age of 8 Ajay Kumar is traumatiz...	Ajay Kumar the newly appointed honest SSP of ...	NaN	Arshad Warsi|Pankaj Kapur|Mahima Chaudhry|Sush...	NaN	29 July 2005 (India)
1361	Lakshya (film)	tt0323013	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Lakshya_(film)	Lakshya	Lakshya	0	2004	186	Action|Drama|Romance	7.9	18777	Karan is a lazy good-for-nothing who lives on ...	An aimless jobless irresponsible grown man j...	It took him 24 years and 18000 feet to find hi...	Hrithik Roshan|Preity Zinta|Amitabh Bachchan|O...	4 wins & 10 nominations	18 June 2004 (USA)
1432	Gangaajal	tt0373856	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Gangaajal	Gangaajal	Gangaajal	0	2003	157	Action|Crime|Drama	7.8	14295	An SP Amit Kumar who is given charge of Tezpur...	An IPS officer motivates and leads a dysfuncti...	NaN	Ajay Devgn|Gracy Singh|Mohan Joshi|Yashpal Sha...	4 wins & 29 nominations	29 August 2003 (India)
1495	Company (film)	tt0296574	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Company_(film)	Company	Company	0	2002	155	Action|Crime|Drama	8.0	13474	Mallik is a henchman of Aslam Bhai a Mumbai u...	A small-time gangster named Chandu teams up wi...	A law & order enterprise	Ajay Devgn|Mohanlal|Manisha Koirala|Seema Bisw...	16 wins & 9 nominations	15 April 2002 (India)
1554	The Legend of Bhagat Singh	tt0319736	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Legend_of_Bh...	The Legend of Bhagat Singh	The Legend of Bhagat Singh	0	2002	155	Action|Biography|Drama	8.1	13455	Bhagat was born in British India during the ye...	The story of a young revolutionary who raised ...	NaN	Ajay Devgn|Sushant Singh|D. Santosh|Akhilendra...	11 wins & 5 nominations	7 June 2002 (India)
1607	Nayak (2001 Hindi film)	tt0291376	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Nayak_(2001_Hind...	Nayak: The Real Hero	Nayak: The Real Hero	0	2001	187	Action|Drama|Thriller	7.8	12522	Employed as a camera-man at a popular televisi...	A man accepts a challenge by the chief ministe...	Fight the power	Anil Kapoor|Rani Mukerji|Amrish Puri|Johnny Le...	2 nominations	7 September 2001 (India)

# write a function that can return the track record of 2 teams against each other
     
Adding new cols

# completely new
movies['Country'] = 'India'
movies.head()
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date	Country
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)	India
1	Battalion 609	tt9472208	NaN	https://en.wikipedia.org/wiki/Battalion_609	Battalion 609	Battalion 609	0	2019	131	War	4.1	73	The story revolves around a cricket match betw...	The story of Battalion 609 revolves around a c...	NaN	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	NaN	11 January 2019 (India)	India
2	The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Accidental_P...	The Accidental Prime Minister	The Accidental Prime Minister	0	2019	112	Biography|Drama	6.1	5549	Based on the memoir by Indian policy analyst S...	Explores Manmohan Singh's tenure as the Prime ...	NaN	Anupam Kher|Akshaye Khanna|Aahana Kumra|Atul S...	NaN	11 January 2019 (USA)	India
3	Why Cheat India	tt8108208	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Why_Cheat_India	Why Cheat India	Why Cheat India	0	2019	121	Crime|Drama	6.0	1891	The movie focuses on existing malpractices in ...	The movie focuses on existing malpractices in ...	NaN	Emraan Hashmi|Shreya Dhanwanthary|Snighdadeep ...	NaN	18 January 2019 (USA)	India
4	Evening Shadows	tt6028796	NaN	https://en.wikipedia.org/wiki/Evening_Shadows	Evening Shadows	Evening Shadows	0	2018	102	Drama	7.3	280	While gay rights and marriage equality has bee...	Under the 'Evening Shadows' truth often plays...	NaN	Mona Ambegaonkar|Ananth Narayan Mahadevan|Deva...	17 wins & 1 nomination	11 January 2019 (India)	India

# from existing ones
movies.dropna(inplace=True)
     

movies['lead actor'] = movies['actors'].str.split('|').apply(lambda x:x[0])
movies.head()
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date	Country	lead actor
11	Gully Boy	tt2395469	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Gully_Boy	Gully Boy	Gully Boy	0	2019	153	Drama|Music	8.2	22440	Gully Boy is a film about a 22-year-old boy "M...	A coming-of-age story based on the lives of st...	Apna Time Aayega!	Ranveer Singh|Alia Bhatt|Siddhant Chaturvedi|V...	6 wins & 3 nominations	14 February 2019 (USA)	India	Ranveer Singh
34	Yeh Hai India	tt5525846	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yeh_Hai_India	Yeh Hai India	Yeh Hai India	0	2017	128	Action|Adventure|Drama	5.7	169	Yeh Hai India follows the story of a 25 years...	Yeh Hai India follows the story of a 25 years...	A Film for Every Indian	Gavie Chahal|Mohan Agashe|Mohan Joshi|Lom Harsh|	2 wins & 1 nomination	24 May 2019 (India)	India	Gavie Chahal
37	Article 15 (film)	tt10324144	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Article_15_(film)	Article 15	Article 15	0	2019	130	Crime|Drama	8.3	13417	In the rural heartlands of India an upright p...	In the rural heartlands of India an upright p...	Farq Bahut Kar Liya| Ab Farq Laayenge.	Ayushmann Khurrana|Nassar|Manoj Pahwa|Kumud Mi...	1 win	28 June 2019 (USA)	India	Ayushmann Khurrana
87	Aiyaary	tt6774212	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Aiyaary	Aiyaary	Aiyaary	0	2018	157	Action|Thriller	5.2	3538	General Gurinder Singh comes with a proposal t...	After finding out about an illegal arms deal ...	The Ultimate Trickery	Sidharth Malhotra|Manoj Bajpayee|Rakul Preet S...	1 nomination	16 February 2018 (USA)	India	Sidharth Malhotra
96	Raid (2018 film)	tt7363076	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Raid_(2018_film)	Raid	Raid	0	2018	122	Action|Crime|Drama	7.4	13159	Set in the 80s in Uttar Pradesh India Raid i...	A fearless income tax officer raids the mansio...	Heroes don't always come in uniform	Ajay Devgn|Saurabh Shukla|Ileana D'Cruz|Amit S...	2 wins & 3 nominations	16 March 2018 (India)	India	Ajay Devgn

movies.info()
     
<class 'pandas.core.frame.DataFrame'>
Int64Index: 298 entries, 11 to 1623
Data columns (total 19 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   title_x           298 non-null    object 
 1   imdb_id           298 non-null    object 
 2   poster_path       298 non-null    object 
 3   wiki_link         298 non-null    object 
 4   title_y           298 non-null    object 
 5   original_title    298 non-null    object 
 6   is_adult          298 non-null    int64  
 7   year_of_release   298 non-null    int64  
 8   runtime           298 non-null    object 
 9   genres            298 non-null    object 
 10  imdb_rating       298 non-null    float64
 11  imdb_votes        298 non-null    int64  
 12  story             298 non-null    object 
 13  summary           298 non-null    object 
 14  tagline           298 non-null    object 
 15  actors            298 non-null    object 
 16  wins_nominations  298 non-null    object 
 17  release_date      298 non-null    object 
 18  Country           298 non-null    object 
dtypes: float64(1), int64(3), object(15)
memory usage: 46.6+ KB
Important DataFrame Functions

# astype
ipl.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 950 entries, 0 to 949
Data columns (total 20 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   ID               950 non-null    int64  
 1   City             899 non-null    object 
 2   Date             950 non-null    object 
 3   Season           950 non-null    object 
 4   MatchNumber      950 non-null    object 
 5   Team1            950 non-null    object 
 6   Team2            950 non-null    object 
 7   Venue            950 non-null    object 
 8   TossWinner       950 non-null    object 
 9   TossDecision     950 non-null    object 
 10  SuperOver        946 non-null    object 
 11  WinningTeam      946 non-null    object 
 12  WonBy            950 non-null    object 
 13  Margin           932 non-null    float64
 14  method           19 non-null     object 
 15  Player_of_Match  946 non-null    object 
 16  Team1Players     950 non-null    object 
 17  Team2Players     950 non-null    object 
 18  Umpire1          950 non-null    object 
 19  Umpire2          950 non-null    object 
dtypes: float64(1), int64(1), object(18)
memory usage: 148.6+ KB

ipl['ID'] = ipl['ID'].astype('int32')
     

ipl.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 950 entries, 0 to 949
Data columns (total 20 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   ID               950 non-null    int32  
 1   City             899 non-null    object 
 2   Date             950 non-null    object 
 3   Season           950 non-null    object 
 4   MatchNumber      950 non-null    object 
 5   Team1            950 non-null    object 
 6   Team2            950 non-null    object 
 7   Venue            950 non-null    object 
 8   TossWinner       950 non-null    object 
 9   TossDecision     950 non-null    object 
 10  SuperOver        946 non-null    object 
 11  WinningTeam      946 non-null    object 
 12  WonBy            950 non-null    object 
 13  Margin           932 non-null    float64
 14  method           19 non-null     object 
 15  Player_of_Match  946 non-null    object 
 16  Team1Players     950 non-null    object 
 17  Team2Players     950 non-null    object 
 18  Umpire1          950 non-null    object 
 19  Umpire2          950 non-null    object 
dtypes: float64(1), int32(1), object(18)
memory usage: 144.9+ KB

# ipl['Season'] = ipl['Season'].astype('category')
ipl['Team1'] = ipl['Team1'].astype('category')
ipl['Team2'] = ipl['Team2'].astype('category')
     

ipl.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 950 entries, 0 to 949
Data columns (total 20 columns):
 #   Column           Non-Null Count  Dtype   
---  ------           --------------  -----   
 0   ID               950 non-null    int32   
 1   City             899 non-null    object  
 2   Date             950 non-null    object  
 3   Season           950 non-null    category
 4   MatchNumber      950 non-null    object  
 5   Team1            950 non-null    category
 6   Team2            950 non-null    category
 7   Venue            950 non-null    object  
 8   TossWinner       950 non-null    object  
 9   TossDecision     950 non-null    object  
 10  SuperOver        946 non-null    object  
 11  WinningTeam      946 non-null    object  
 12  WonBy            950 non-null    object  
 13  Margin           932 non-null    float64 
 14  method           19 non-null     object  
 15  Player_of_Match  946 non-null    object  
 16  Team1Players     950 non-null    object  
 17  Team2Players     950 non-null    object  
 18  Umpire1          950 non-null    object  
 19  Umpire2          950 non-null    object  
dtypes: category(3), float64(1), int32(1), object(15)
memory usage: 127.4+ KB


     

# value_counts
     

# find which player has won most potm -> in finals and qualifiers
     

# Toss decision plot
     

# how many matches each team has played
     

# sort_values -> ascending -> na_position -> inplace -> multiple cols
     

#################################################################################



More Important Functions

# value_counts
# sort_values
# rank
# sort index
# set index
# rename index -> rename
# reset index
# unique & nunique
# isnull/notnull/hasnans
# dropna
# fillna
# drop_duplicates
# drop
# apply
# isin
# corr
# nlargest -> nsmallest
# insert
# copy
     

import numpy as np
import pandas as pd
     

a = pd.Series([1,1,1,2,2,3])
a.value_counts()
     
1    3
2    2
3    1
dtype: int64

# value_counts(series and dataframe)


marks = pd.DataFrame([
    [100,80,10],
    [90,70,7],
    [120,100,14],
    [80,70,14],
    [80,70,14]
],columns=['iq','marks','package'])

marks
     
iq	marks	package
0	100	80	10
1	90	70	7
2	120	100	14
3	80	70	14
4	80	70	14

marks.value_counts()
     
iq   marks  package
80   70     14         2
90   70     7          1
100  80     10         1
120  100    14         1
dtype: int64

ipl = pd.read_csv('ipl-matches.csv')
ipl[~ipl['MatchNumber'].str.isdigit()]['Player_of_Match'].value_counts()
     
KA Pollard           3
F du Plessis         3
SK Raina             3
A Kumble             2
MK Pandey            2
YK Pathan            2
M Vijay              2
JJ Bumrah            2
AB de Villiers       2
SR Watson            2
HH Pandya            1
Harbhajan Singh      1
A Nehra              1
V Sehwag             1
UT Yadav             1
MS Bisla             1
BJ Hodge             1
MEK Hussey           1
MS Dhoni             1
CH Gayle             1
MM Patel             1
DE Bollinger         1
AC Gilchrist         1
RG Sharma            1
DA Warner            1
MC Henriques         1
JC Buttler           1
RM Patidar           1
DA Miller            1
VR Iyer              1
SP Narine            1
RD Gaikwad           1
TA Boult             1
MP Stoinis           1
KS Williamson        1
RR Pant              1
SA Yadav             1
Rashid Khan          1
AD Russell           1
KH Pandya            1
KV Sharma            1
NM Coulter-Nile      1
Washington Sundar    1
BCJ Cutting          1
M Ntini              1
Name: Player_of_Match, dtype: int64

# find which player has won most potm -> in finals and qualifiers
     

# Toss decision plot
ipl['TossDecision'].value_counts().plot(kind='pie')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f034efd49d0>


# how many matches each team has played
(ipl['Team2'].value_counts() + ipl['Team1'].value_counts()).sort_values(ascending=False)
     
Mumbai Indians                 231
Royal Challengers Bangalore    226
Kolkata Knight Riders          223
Chennai Super Kings            208
Rajasthan Royals               192
Kings XI Punjab                190
Delhi Daredevils               161
Sunrisers Hyderabad            152
Deccan Chargers                 75
Delhi Capitals                  63
Pune Warriors                   46
Gujarat Lions                   30
Punjab Kings                    28
Gujarat Titans                  16
Rising Pune Supergiant          16
Lucknow Super Giants            15
Kochi Tuskers Kerala            14
Rising Pune Supergiants         14
dtype: int64

# sort_values(series and dataframe) -> ascending -> na_position -> inplace -> multiple cols
     

x = pd.Series([12,14,1,56,89])
x
     
0    12
1    14
2     1
3    56
4    89
dtype: int64

x.sort_values(ascending=False)
     
4    89
3    56
1    14
0    12
2     1
dtype: int64

movies = pd.read_csv('movies.csv')
movies.head(4)
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
1	Battalion 609	tt9472208	NaN	https://en.wikipedia.org/wiki/Battalion_609	Battalion 609	Battalion 609	0	2019	131	War	4.1	73	The story revolves around a cricket match betw...	The story of Battalion 609 revolves around a c...	NaN	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	NaN	11 January 2019 (India)
2	The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Accidental_P...	The Accidental Prime Minister	The Accidental Prime Minister	0	2019	112	Biography|Drama	6.1	5549	Based on the memoir by Indian policy analyst S...	Explores Manmohan Singh's tenure as the Prime ...	NaN	Anupam Kher|Akshaye Khanna|Aahana Kumra|Atul S...	NaN	11 January 2019 (USA)
3	Why Cheat India	tt8108208	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Why_Cheat_India	Why Cheat India	Why Cheat India	0	2019	121	Crime|Drama	6.0	1891	The movie focuses on existing malpractices in ...	The movie focuses on existing malpractices in ...	NaN	Emraan Hashmi|Shreya Dhanwanthary|Snighdadeep ...	NaN	18 January 2019 (USA)

movies.sort_values('title_x',ascending=False)
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
1623	Zubeidaa	tt0255713	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Zubeidaa	Zubeidaa	Zubeidaa	0	2001	153	Biography|Drama|History	6.2	1384	The film begins with Riyaz (Rajat Kapoor) Zub...	Zubeidaa an aspiring Muslim actress marries ...	The Story of a Princess	Karisma Kapoor|Rekha|Manoj Bajpayee|Rajit Kapo...	3 wins & 13 nominations	19 January 2001 (India)
939	Zor Lagaa Ke...Haiya!	tt1479857	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Zor_Lagaa_Ke...H...	Zor Lagaa Ke... Haiya!	Zor Lagaa Ke... Haiya!	0	2009	\N	Comedy|Drama|Family	6.4	46	A tree narrates the story of four Mumbai-based...	Children build a tree-house to spy on a beggar...	NaN	Meghan Jadhav|Mithun Chakraborty|Riya Sen|Seem...	NaN	NaN
756	Zokkomon	tt1605790	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Zokkomon	Zokkomon	Zokkomon	0	2011	109	Action|Adventure	4.0	274	After the passing of his parents in an acciden...	An orphan is abused and abandoned believed to...	Betrayal. Friendship. Bravery.	Darsheel Safary|Anupam Kher|Manjari Fadnnis|Ti...	NaN	22 April 2011 (India)
670	Zindagi Tere Naam	tt2164702	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Zindagi_Tere_Naam	Zindagi Tere Naam	Zindagi Tere Naam	0	2012	120	Romance	4.7	27	Mr. Singh an elderly gentleman relates to hi...	Mr. Singh an elderly gentleman relates to hi...	NaN	Mithun Chakraborty|Ranjeeta Kaur|Priyanka Meht...	1 win	16 March 2012 (India)
778	Zindagi Na Milegi Dobara	tt1562872	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Zindagi_Na_Mileg...	Zindagi Na Milegi Dobara	Zindagi Na Milegi Dobara	0	2011	155	Comedy|Drama	8.1	60826	Three friends decide to turn their fantasy vac...	Three friends decide to turn their fantasy vac...	NaN	Hrithik Roshan|Farhan Akhtar|Abhay Deol|Katrin...	30 wins & 22 nominations	15 July 2011 (India)
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
1039	1971 (2007 film)	tt0983990	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/1971_(2007_film)	1971	1971	0	2007	160	Action|Drama|War	7.9	1121	Based on true facts the film revolves around ...	Based on true facts the film revolves around ...	Honor the heroes.......	Manoj Bajpayee|Ravi Kishan|Deepak Dobriyal|	1 win	9 March 2007 (India)
723	1920: The Evil Returns	tt2222550	https://upload.wikimedia.org/wikipedia/en/e/e7...	https://en.wikipedia.org/wiki/1920:_The_Evil_R...	1920: Evil Returns	1920: Evil Returns	0	2012	124	Drama|Horror|Romance	4.8	1587	This story revolves around a famous poet who m...	This story revolves around a famous poet who m...	Possession is back	Vicky Ahuja|Tia Bajpai|Irma Jämhammar|Sharad K...	NaN	2 November 2012 (India)
287	1920: London	tt5638500	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/1920_London	1920 London	1920 London	0	2016	120	Horror|Mystery	4.1	1373	Shivangi (Meera Chopra) lives in London with h...	After her husband is possessed by an evil spir...	Fear strikes again	Sharman Joshi|Meera Chopra|Vishal Karwal|Suren...	NaN	6 May 2016 (USA)
1021	1920 (film)	tt1301698	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/1920_(film)	1920	1920	0	2008	138	Horror|Mystery|Romance	6.4	2588	A devotee of Bhagwan Shri Hanuman Arjun Singh...	After forsaking his family and religion a hus...	A Love Made in Heaven...A Revenge Born in Hell...	Rajniesh Duggall|Adah Sharma|Anjori Alagh|Raj ...	NaN	12 September 2008 (India)
1498	16 December (film)	tt0313844	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/16_December_(film)	16-Dec	16-Dec	0	2002	158	Action|Thriller	6.9	1091	16 December 1971 was the day when India won t...	Indian intelligence agents race against time t...	NaN	Danny Denzongpa|Gulshan Grover|Milind Soman|Di...	2 nominations	22 March 2002 (India)
1629 rows × 18 columns



     

students = pd.DataFrame(
    {
        'name':['nitish','ankit','rupesh',np.nan,'mrityunjay',np.nan,'rishabh',np.nan,'aditya',np.nan],
        'college':['bit','iit','vit',np.nan,np.nan,'vlsi','ssit',np.nan,np.nan,'git'],
        'branch':['eee','it','cse',np.nan,'me','ce','civ','cse','bio',np.nan],
        'cgpa':[6.66,8.25,6.41,np.nan,5.6,9.0,7.4,10,7.4,np.nan],
        'package':[4,5,6,np.nan,6,7,8,9,np.nan,np.nan]

    }
)

students
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
3	NaN	NaN	NaN	NaN	NaN
4	mrityunjay	NaN	me	5.60	6.0
5	NaN	vlsi	ce	9.00	7.0
6	rishabh	ssit	civ	7.40	8.0
7	NaN	NaN	cse	10.00	9.0
8	aditya	NaN	bio	7.40	NaN
9	NaN	git	NaN	NaN	NaN

students.sort_values('name',na_position='first',ascending=False,inplace=True)
     

students
     
name	college	branch	cgpa	package
3	NaN	NaN	NaN	NaN	NaN
5	NaN	vlsi	ce	9.00	7.0
7	NaN	NaN	cse	10.00	9.0
9	NaN	git	NaN	NaN	NaN
2	rupesh	vit	cse	6.41	6.0
6	rishabh	ssit	civ	7.40	8.0
0	nitish	bit	eee	6.66	4.0
4	mrityunjay	NaN	me	5.60	6.0
1	ankit	iit	it	8.25	5.0
8	aditya	NaN	bio	7.40	NaN

movies.sort_values(['year_of_release','title_x'],ascending=[True,False])
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
1623	Zubeidaa	tt0255713	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Zubeidaa	Zubeidaa	Zubeidaa	0	2001	153	Biography|Drama|History	6.2	1384	The film begins with Riyaz (Rajat Kapoor) Zub...	Zubeidaa an aspiring Muslim actress marries ...	The Story of a Princess	Karisma Kapoor|Rekha|Manoj Bajpayee|Rajit Kapo...	3 wins & 13 nominations	19 January 2001 (India)
1625	Yeh Zindagi Ka Safar	tt0298607	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yeh_Zindagi_Ka_S...	Yeh Zindagi Ka Safar	Yeh Zindagi Ka Safar	0	2001	146	Drama	3.0	133	Hindi pop-star Sarina Devan lives a wealthy ...	A singer finds out she was adopted when the ed...	NaN	Ameesha Patel|Jimmy Sheirgill|Nafisa Ali|Gulsh...	NaN	16 November 2001 (India)
1622	Yeh Teraa Ghar Yeh Meraa Ghar	tt0298606	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yeh_Teraa_Ghar_Y...	Yeh Teraa Ghar Yeh Meraa Ghar	Yeh Teraa Ghar Yeh Meraa Ghar	0	2001	175	Comedy|Drama	5.7	704	In debt; Dayashankar Pandey is forced to go to...	In debt; Dayashankar Pandey is forced to go to...	NaN	Sunil Shetty|Mahima Chaudhry|Paresh Rawal|Saur...	1 nomination	12 October 2001 (India)
1620	Yeh Raaste Hain Pyaar Ke	tt0292740	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yeh_Raaste_Hain_...	Yeh Raaste Hain Pyaar Ke	Yeh Raaste Hain Pyaar Ke	0	2001	149	Drama|Romance	4.0	607	Two con artistes and car thieves Vicky (Ajay ...	Two con artistes and car thieves Vicky (Ajay ...	Love is a journey... not a destination	Ajay Devgn|Madhuri Dixit|Preity Zinta|Vikram G...	NaN	10 August 2001 (India)
1573	Yaadein (2001 film)	tt0248617	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yaadein_(2001_film)	Yaadein...	Yaadein...	0	2001	171	Drama|Musical|Romance	4.4	3034	Raj Singh Puri is best friends with L.K. Malho...	Raj Singh Puri is best friends with L.K. Malho...	memories to cherish...	Jackie Shroff|Hrithik Roshan|Kareena Kapoor|Am...	1 nomination	27 June 2001 (India)
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
37	Article 15 (film)	tt10324144	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Article_15_(film)	Article 15	Article 15	0	2019	130	Crime|Drama	8.3	13417	In the rural heartlands of India an upright p...	In the rural heartlands of India an upright p...	Farq Bahut Kar Liya| Ab Farq Laayenge.	Ayushmann Khurrana|Nassar|Manoj Pahwa|Kumud Mi...	1 win	28 June 2019 (USA)
46	Arjun Patiala	tt7881524	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Arjun_Patiala	Arjun Patiala	Arjun Patiala	0	2019	107	Action|Comedy	4.1	676	Arjun Patiala(Diljit Dosanjh)has recently been...	This spoof comedy narrates the story of a cop ...	NaN	Diljit Dosanjh|Kriti Sanon|Varun Sharma|Ronit ...	NaN	26 July 2019 (USA)
10	Amavas	tt8396186	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Amavas	Amavas	Amavas	0	2019	134	Horror|Thriller	2.8	235	Far away from the bustle of the city a young ...	The lives of a couple turn into a nightmare a...	NaN	Ali Asgar|Vivan Bhatena|Nargis Fakhri|Sachiin ...	NaN	8 February 2019 (India)
26	Albert Pinto Ko Gussa Kyun Aata Hai?	tt4355838	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Albert_Pinto_Ko_...	Albert Pinto Ko Gussa Kyun Aata Hai?	Albert Pinto Ko Gussa Kyun Aata Hai?	0	2019	100	Drama	4.8	56	Albert leaves his house one morning without te...	Albert Pinto goes missing one day and his girl...	NaN	Manav Kaul|Nandita Das|	NaN	12 April 2019 (India)
21	22 Yards	tt9496212	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/22_Yards	22 Yards	22 Yards	0	2019	126	Sport	5.3	124	A dramatic portrayal of a victorious tale of a...	A dramatic portrayal of a victorious tale of a...	NaN	Barun Sobti|Rajit Kapur|Panchhi Bora|Kartikey ...	NaN	15 March 2019 (India)
1629 rows × 18 columns



     

# rank(series)
batsman = pd.read_csv('batsman_runs_ipl.csv')
batsman.head()
     
batter	batsman_run
0	A Ashish Reddy	280
1	A Badoni	161
2	A Chandila	4
3	A Chopra	53
4	A Choudhary	25

batsman['batting_rank'] = batsman['batsman_run'].rank(ascending=False)
batsman.sort_values('batting_rank')
     
batter	batsman_run	batting_rank
569	V Kohli	6634	1.0
462	S Dhawan	6244	2.0
130	DA Warner	5883	3.0
430	RG Sharma	5881	4.0
493	SK Raina	5536	5.0
...	...	...	...
512	SS Cottrell	0	594.0
466	S Kaushik	0	594.0
203	IC Pandey	0	594.0
467	S Ladda	0	594.0
468	S Lamichhane	0	594.0
605 rows × 3 columns


# sort_index(series and dataframe)
     

marks = {
    'maths':67,
    'english':57,
    'science':89,
    'hindi':100
}

marks_series = pd.Series(marks)
marks_series
     
maths       67
english     57
science     89
hindi      100
dtype: int64

marks_series.sort_index(ascending=False)
     
science     89
maths       67
hindi      100
english     57
dtype: int64

movies.sort_index(ascending=False)
     
title_x	imdb_id	poster_path	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
1628	Humsafar	tt2403201	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Humsafar	Humsafar	Humsafar	0	2011	35	Drama|Romance	9.0	2968	Sara and Ashar are childhood friends who share...	Ashar and Khirad are forced to get married due...	NaN	Fawad Khan|	NaN	TV Series (2011–2012)
1627	Daaka	tt10833860	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Daaka	Daaka	Daaka	0	2019	136	Action	7.4	38	Shinda tries robbing a bank so he can be wealt...	Shinda tries robbing a bank so he can be wealt...	NaN	Gippy Grewal|Zareen Khan|	NaN	1 November 2019 (USA)
1626	Sabse Bada Sukh	tt0069204	NaN	https://en.wikipedia.org/wiki/Sabse_Bada_Sukh	Sabse Bada Sukh	Sabse Bada Sukh	0	2018	\N	Comedy|Drama	6.1	13	Village born Lalloo re-locates to Bombay and ...	Village born Lalloo re-locates to Bombay and ...	NaN	Vijay Arora|Asrani|Rajni Bala|Kumud Damle|Utpa...	NaN	NaN
1625	Yeh Zindagi Ka Safar	tt0298607	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yeh_Zindagi_Ka_S...	Yeh Zindagi Ka Safar	Yeh Zindagi Ka Safar	0	2001	146	Drama	3.0	133	Hindi pop-star Sarina Devan lives a wealthy ...	A singer finds out she was adopted when the ed...	NaN	Ameesha Patel|Jimmy Sheirgill|Nafisa Ali|Gulsh...	NaN	16 November 2001 (India)
1624	Tera Mera Saath Rahen	tt0301250	https://upload.wikimedia.org/wikipedia/en/2/2b...	https://en.wikipedia.org/wiki/Tera_Mera_Saath_...	Tera Mera Saath Rahen	Tera Mera Saath Rahen	0	2001	148	Drama	4.9	278	Raj Dixit lives with his younger brother Rahu...	A man is torn between his handicapped brother ...	NaN	Ajay Devgn|Sonali Bendre|Namrata Shirodkar|Pre...	NaN	7 November 2001 (India)
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
4	Evening Shadows	tt6028796	NaN	https://en.wikipedia.org/wiki/Evening_Shadows	Evening Shadows	Evening Shadows	0	2018	102	Drama	7.3	280	While gay rights and marriage equality has bee...	Under the 'Evening Shadows' truth often plays...	NaN	Mona Ambegaonkar|Ananth Narayan Mahadevan|Deva...	17 wins & 1 nomination	11 January 2019 (India)
3	Why Cheat India	tt8108208	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Why_Cheat_India	Why Cheat India	Why Cheat India	0	2019	121	Crime|Drama	6.0	1891	The movie focuses on existing malpractices in ...	The movie focuses on existing malpractices in ...	NaN	Emraan Hashmi|Shreya Dhanwanthary|Snighdadeep ...	NaN	18 January 2019 (USA)
2	The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Accidental_P...	The Accidental Prime Minister	The Accidental Prime Minister	0	2019	112	Biography|Drama	6.1	5549	Based on the memoir by Indian policy analyst S...	Explores Manmohan Singh's tenure as the Prime ...	NaN	Anupam Kher|Akshaye Khanna|Aahana Kumra|Atul S...	NaN	11 January 2019 (USA)
1	Battalion 609	tt9472208	NaN	https://en.wikipedia.org/wiki/Battalion_609	Battalion 609	Battalion 609	0	2019	131	War	4.1	73	The story revolves around a cricket match betw...	The story of Battalion 609 revolves around a c...	NaN	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	NaN	11 January 2019 (India)
0	Uri: The Surgical Strike	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
1629 rows × 18 columns


# set_index(dataframe) -> inplace
batsman.set_index('batter',inplace=True)
     

batsman
     
batsman_run	batting_rank
batter		
A Ashish Reddy	280	166.5
A Badoni	161	226.0
A Chandila	4	535.0
A Chopra	53	329.0
A Choudhary	25	402.5
...	...	...
Yash Dayal	0	594.0
Yashpal Singh	47	343.0
Younis Khan	3	547.5
Yuvraj Singh	2754	27.0
Z Khan	117	256.0
605 rows × 2 columns


# reset_index(series + dataframe) -> drop parameter
batsman.reset_index(inplace=True)
     
batter	batsman_run	batting_rank
0	A Ashish Reddy	280	166.5
1	A Badoni	161	226.0
2	A Chandila	4	535.0
3	A Chopra	53	329.0
4	A Choudhary	25	402.5
...	...	...	...
600	Yash Dayal	0	594.0
601	Yashpal Singh	47	343.0
602	Younis Khan	3	547.5
603	Yuvraj Singh	2754	27.0
604	Z Khan	117	256.0
605 rows × 3 columns


batsman
     
batsman_run	batting_rank
batter		
A Ashish Reddy	280	166.5
A Badoni	161	226.0
A Chandila	4	535.0
A Chopra	53	329.0
A Choudhary	25	402.5
...	...	...
Yash Dayal	0	594.0
Yashpal Singh	47	343.0
Younis Khan	3	547.5
Yuvraj Singh	2754	27.0
Z Khan	117	256.0
605 rows × 2 columns


# how to replace existing index without loosing
batsman.reset_index().set_index('batting_rank')
     
batter	batsman_run
batting_rank		
166.5	A Ashish Reddy	280
226.0	A Badoni	161
535.0	A Chandila	4
329.0	A Chopra	53
402.5	A Choudhary	25
...	...	...
594.0	Yash Dayal	0
343.0	Yashpal Singh	47
547.5	Younis Khan	3
27.0	Yuvraj Singh	2754
256.0	Z Khan	117
605 rows × 2 columns


# series to dataframe using reset_index
marks_series.reset_index()
     
index	0
0	maths	67
1	english	57
2	science	89
3	hindi	100

# rename(dataframe) -> index
     

movies.set_index('title_x',inplace=True)
     

movies.rename(columns={'imdb_id':'imdb','poster_path':'link'},inplace=True)
     

movies.rename(index={'Uri: The Surgical Strike':'Uri','Battalion 609':'Battalion'})
     
imdb	link	wiki_link	title_y	original_title	is_adult	year_of_release	runtime	genres	imdb_rating	imdb_votes	story	summary	tagline	actors	wins_nominations	release_date
title_x																	
Uri	tt8291224	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Uri:_The_Surgica...	Uri: The Surgical Strike	Uri: The Surgical Strike	0	2019	138	Action|Drama|War	8.4	35112	Divided over five chapters the film chronicle...	Indian army special forces execute a covert op...	NaN	Vicky Kaushal|Paresh Rawal|Mohit Raina|Yami Ga...	4 wins	11 January 2019 (USA)
Battalion	tt9472208	NaN	https://en.wikipedia.org/wiki/Battalion_609	Battalion 609	Battalion 609	0	2019	131	War	4.1	73	The story revolves around a cricket match betw...	The story of Battalion 609 revolves around a c...	NaN	Vicky Ahuja|Shoaib Ibrahim|Shrikant Kamat|Elen...	NaN	11 January 2019 (India)
The Accidental Prime Minister (film)	tt6986710	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/The_Accidental_P...	The Accidental Prime Minister	The Accidental Prime Minister	0	2019	112	Biography|Drama	6.1	5549	Based on the memoir by Indian policy analyst S...	Explores Manmohan Singh's tenure as the Prime ...	NaN	Anupam Kher|Akshaye Khanna|Aahana Kumra|Atul S...	NaN	11 January 2019 (USA)
Why Cheat India	tt8108208	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Why_Cheat_India	Why Cheat India	Why Cheat India	0	2019	121	Crime|Drama	6.0	1891	The movie focuses on existing malpractices in ...	The movie focuses on existing malpractices in ...	NaN	Emraan Hashmi|Shreya Dhanwanthary|Snighdadeep ...	NaN	18 January 2019 (USA)
Evening Shadows	tt6028796	NaN	https://en.wikipedia.org/wiki/Evening_Shadows	Evening Shadows	Evening Shadows	0	2018	102	Drama	7.3	280	While gay rights and marriage equality has bee...	Under the 'Evening Shadows' truth often plays...	NaN	Mona Ambegaonkar|Ananth Narayan Mahadevan|Deva...	17 wins & 1 nomination	11 January 2019 (India)
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
Tera Mera Saath Rahen	tt0301250	https://upload.wikimedia.org/wikipedia/en/2/2b...	https://en.wikipedia.org/wiki/Tera_Mera_Saath_...	Tera Mera Saath Rahen	Tera Mera Saath Rahen	0	2001	148	Drama	4.9	278	Raj Dixit lives with his younger brother Rahu...	A man is torn between his handicapped brother ...	NaN	Ajay Devgn|Sonali Bendre|Namrata Shirodkar|Pre...	NaN	7 November 2001 (India)
Yeh Zindagi Ka Safar	tt0298607	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Yeh_Zindagi_Ka_S...	Yeh Zindagi Ka Safar	Yeh Zindagi Ka Safar	0	2001	146	Drama	3.0	133	Hindi pop-star Sarina Devan lives a wealthy ...	A singer finds out she was adopted when the ed...	NaN	Ameesha Patel|Jimmy Sheirgill|Nafisa Ali|Gulsh...	NaN	16 November 2001 (India)
Sabse Bada Sukh	tt0069204	NaN	https://en.wikipedia.org/wiki/Sabse_Bada_Sukh	Sabse Bada Sukh	Sabse Bada Sukh	0	2018	\N	Comedy|Drama	6.1	13	Village born Lalloo re-locates to Bombay and ...	Village born Lalloo re-locates to Bombay and ...	NaN	Vijay Arora|Asrani|Rajni Bala|Kumud Damle|Utpa...	NaN	NaN
Daaka	tt10833860	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Daaka	Daaka	Daaka	0	2019	136	Action	7.4	38	Shinda tries robbing a bank so he can be wealt...	Shinda tries robbing a bank so he can be wealt...	NaN	Gippy Grewal|Zareen Khan|	NaN	1 November 2019 (USA)
Humsafar	tt2403201	https://upload.wikimedia.org/wikipedia/en/thum...	https://en.wikipedia.org/wiki/Humsafar	Humsafar	Humsafar	0	2011	35	Drama|Romance	9.0	2968	Sara and Ashar are childhood friends who share...	Ashar and Khirad are forced to get married due...	NaN	Fawad Khan|	NaN	TV Series (2011–2012)
1629 rows × 17 columns


# unique(series)
temp = pd.Series([1,1,2,2,3,3,4,4,5,5,np.nan,np.nan])
print(temp)
     
0     1.0
1     1.0
2     2.0
3     2.0
4     3.0
5     3.0
6     4.0
7     4.0
8     5.0
9     5.0
10    NaN
11    NaN
dtype: float64

len(temp.unique())
     
6

temp.nunique()
     
5

len(ipl['Season'].unique())
     
15

# nunique(series + dataframe) -> does not count nan -> dropna parameter
ipl['Season'].nunique()
     
15

# isnull(series + dataframe)
students['name'][students['name'].isnull()]
     
3    NaN
5    NaN
7    NaN
9    NaN
Name: name, dtype: object

# notnull(series + dataframe)
students['name'][students['name'].notnull()]
     
0        nitish
1         ankit
2        rupesh
4    mrityunjay
6       rishabh
8        aditya
Name: name, dtype: object

# hasnans(series)
students['name'].hasnans
     
True

students
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
3	NaN	NaN	NaN	NaN	NaN
4	mrityunjay	NaN	me	5.60	6.0
5	NaN	vlsi	ce	9.00	7.0
6	rishabh	ssit	civ	7.40	8.0
7	NaN	NaN	cse	10.00	9.0
8	aditya	NaN	bio	7.40	NaN
9	NaN	git	NaN	NaN	NaN

students.isnull()
     
name	college	branch	cgpa	package
0	False	False	False	False	False
1	False	False	False	False	False
2	False	False	False	False	False
3	True	True	True	True	True
4	False	True	False	False	False
5	True	False	False	False	False
6	False	False	False	False	False
7	True	True	False	False	False
8	False	True	False	False	True
9	True	False	True	True	True

students.notnull()
     
name	college	branch	cgpa	package
0	True	True	True	True	True
1	True	True	True	True	True
2	True	True	True	True	True
3	False	False	False	False	False
4	True	False	True	True	True
5	False	True	True	True	True
6	True	True	True	True	True
7	False	False	True	True	True
8	True	False	True	True	False
9	False	True	False	False	False

# dropna(series + dataframe) -> how parameter -> works like or
students['name'].dropna()
     
0        nitish
1         ankit
2        rupesh
4    mrityunjay
6       rishabh
8        aditya
Name: name, dtype: object

students
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
3	NaN	NaN	NaN	NaN	NaN
4	mrityunjay	NaN	me	5.60	6.0
5	NaN	vlsi	ce	9.00	7.0
6	rishabh	ssit	civ	7.40	8.0
7	NaN	NaN	cse	10.00	9.0
8	aditya	NaN	bio	7.40	NaN
9	NaN	git	NaN	NaN	NaN

students.dropna(how='any')
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
6	rishabh	ssit	civ	7.40	8.0

students.dropna(how='all')
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
4	mrityunjay	NaN	me	5.60	6.0
5	NaN	vlsi	ce	9.00	7.0
6	rishabh	ssit	civ	7.40	8.0
7	NaN	NaN	cse	10.00	9.0
8	aditya	NaN	bio	7.40	NaN
9	NaN	git	NaN	NaN	NaN

students.dropna(subset=['name'])
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
4	mrityunjay	NaN	me	5.60	6.0
6	rishabh	ssit	civ	7.40	8.0
8	aditya	NaN	bio	7.40	NaN

students.dropna(subset=['name','college'])
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
6	rishabh	ssit	civ	7.40	8.0

students.dropna(inplace=True)
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
3	NaN	NaN	NaN	NaN	NaN
4	mrityunjay	NaN	me	5.60	6.0
5	NaN	vlsi	ce	9.00	7.0
6	rishabh	ssit	civ	7.40	8.0
7	NaN	NaN	cse	10.00	9.0
8	aditya	NaN	bio	7.40	NaN
9	NaN	git	NaN	NaN	NaN

# fillna(series + dataframe)
students['name'].fillna('unknown')
     
0        nitish
1         ankit
2        rupesh
3       unknown
4    mrityunjay
5       unknown
6       rishabh
7       unknown
8        aditya
9       unknown
Name: name, dtype: object

students
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
3	NaN	NaN	NaN	NaN	NaN
4	mrityunjay	NaN	me	5.60	6.0
5	NaN	vlsi	ce	9.00	7.0
6	rishabh	ssit	civ	7.40	8.0
7	NaN	NaN	cse	10.00	9.0
8	aditya	NaN	bio	7.40	NaN
9	NaN	git	NaN	NaN	NaN

students['package'].fillna(students['package'].mean())
     
0    4.000000
1    5.000000
2    6.000000
3    6.428571
4    6.000000
5    7.000000
6    8.000000
7    9.000000
8    6.428571
9    6.428571
Name: package, dtype: float64

students['name'].fillna(method='bfill')
     
0        nitish
1         ankit
2        rupesh
3    mrityunjay
4    mrityunjay
5       rishabh
6       rishabh
7        aditya
8        aditya
9           NaN
Name: name, dtype: object

# drop_duplicates(series + dataframe) -> works like and -> duplicated()
     

temp = pd.Series([1,1,1,2,3,3,4,4])
temp.drop_duplicates()
     
0    1
3    2
4    3
6    4
dtype: int64

marks.drop_duplicates(keep='last')
     
iq	marks	package
0	100	80	10
1	90	70	7
2	120	100	14
4	80	70	14

# find the last match played by virat kohli in Delhi
ipl['all_players'] = ipl['Team1Players'] + ipl['Team2Players']
ipl.head()
     
ID	City	Date	Season	MatchNumber	Team1	Team2	Venue	TossWinner	TossDecision	...	WinningTeam	WonBy	Margin	method	Player_of_Match	Team1Players	Team2Players	Umpire1	Umpire2	all_players
0	1312200	Ahmedabad	2022-05-29	2022	Final	Rajasthan Royals	Gujarat Titans	Narendra Modi Stadium, Ahmedabad	Rajasthan Royals	bat	...	Gujarat Titans	Wickets	7.0	NaN	HH Pandya	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	['WP Saha', 'Shubman Gill', 'MS Wade', 'HH Pan...	CB Gaffaney	Nitin Menon	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...
1	1312199	Ahmedabad	2022-05-27	2022	Qualifier 2	Royal Challengers Bangalore	Rajasthan Royals	Narendra Modi Stadium, Ahmedabad	Rajasthan Royals	field	...	Rajasthan Royals	Wickets	7.0	NaN	JC Buttler	['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ ...	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	CB Gaffaney	Nitin Menon	['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ ...
2	1312198	Kolkata	2022-05-25	2022	Eliminator	Royal Challengers Bangalore	Lucknow Super Giants	Eden Gardens, Kolkata	Lucknow Super Giants	field	...	Royal Challengers Bangalore	Runs	14.0	NaN	RM Patidar	['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ ...	['Q de Kock', 'KL Rahul', 'M Vohra', 'DJ Hooda...	J Madanagopal	MA Gough	['V Kohli', 'F du Plessis', 'RM Patidar', 'GJ ...
3	1312197	Kolkata	2022-05-24	2022	Qualifier 1	Rajasthan Royals	Gujarat Titans	Eden Gardens, Kolkata	Gujarat Titans	field	...	Gujarat Titans	Wickets	7.0	NaN	DA Miller	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...	['WP Saha', 'Shubman Gill', 'MS Wade', 'HH Pan...	BNJ Oxenford	VK Sharma	['YBK Jaiswal', 'JC Buttler', 'SV Samson', 'D ...
4	1304116	Mumbai	2022-05-22	2022	70	Sunrisers Hyderabad	Punjab Kings	Wankhede Stadium, Mumbai	Sunrisers Hyderabad	bat	...	Punjab Kings	Wickets	5.0	NaN	Harpreet Brar	['PK Garg', 'Abhishek Sharma', 'RA Tripathi', ...	['JM Bairstow', 'S Dhawan', 'M Shahrukh Khan',...	AK Chaudhary	NA Patwardhan	['PK Garg', 'Abhishek Sharma', 'RA Tripathi', ...
5 rows × 21 columns


def did_kohli_play(players_list):
  return 'V Kohli' in players_list
     

ipl['did_kohli_play'] = ipl['all_players'].apply(did_kohli_play)
ipl[(ipl['City'] == 'Delhi') & (ipl['did_kohli_play'] == True)].drop_duplicates(subset=['City','did_kohli_play'],keep='first')
     
ID	City	Date	Season	MatchNumber	Team1	Team2	Venue	TossWinner	TossDecision	...	WonBy	Margin	method	Player_of_Match	Team1Players	Team2Players	Umpire1	Umpire2	all_players	did_kohli_play
208	1178421	Delhi	2019-04-28	2019	46	Delhi Capitals	Royal Challengers Bangalore	Arun Jaitley Stadium	Delhi Capitals	bat	...	Runs	16.0	NaN	S Dhawan	['PP Shaw', 'S Dhawan', 'SS Iyer', 'RR Pant', ...	['PA Patel', 'V Kohli', 'AB de Villiers', 'S D...	BNJ Oxenford	KN Ananthapadmanabhan	['PP Shaw', 'S Dhawan', 'SS Iyer', 'RR Pant', ...	True
1 rows × 22 columns


students.drop_duplicates()
     

# drop(series + dataframe)
temp = pd.Series([10,2,3,16,45,78,10])
temp
     
0    10
1     2
2     3
3    16
4    45
5    78
6    10
dtype: int64

temp.drop(index=[0,6])
     
1     2
2     3
3    16
4    45
5    78
dtype: int64

students
     
name	college	branch	cgpa	package
0	nitish	bit	eee	6.66	4.0
1	ankit	iit	it	8.25	5.0
2	rupesh	vit	cse	6.41	6.0
3	NaN	NaN	NaN	NaN	NaN
4	mrityunjay	NaN	me	5.60	6.0
5	NaN	vlsi	ce	9.00	7.0
6	rishabh	ssit	civ	7.40	8.0
7	NaN	NaN	cse	10.00	9.0
8	aditya	NaN	bio	7.40	NaN
9	NaN	git	NaN	NaN	NaN

students.drop(columns=['branch','cgpa'],inplace=True)
     
name	college	package
0	nitish	bit	4.0
1	ankit	iit	5.0
2	rupesh	vit	6.0
3	NaN	NaN	NaN
4	mrityunjay	NaN	6.0
5	NaN	vlsi	7.0
6	rishabh	ssit	8.0
7	NaN	NaN	9.0
8	aditya	NaN	NaN
9	NaN	git	NaN

students.set_index('name').drop(index=['nitish','aditya'])
     
college	branch	cgpa	package
name				
ankit	iit	it	8.25	5.0
rupesh	vit	cse	6.41	6.0
NaN	NaN	NaN	NaN	NaN
mrityunjay	NaN	me	5.60	6.0
NaN	vlsi	ce	9.00	7.0
rishabh	ssit	civ	7.40	8.0
NaN	NaN	cse	10.00	9.0
NaN	git	NaN	NaN	NaN

# apply(series + dataframe)
temp = pd.Series([10,20,30,40,50])

temp
     
0    10
1    20
2    30
3    40
4    50
dtype: int64

def sigmoid(value):
  return 1/1+np.exp(-value)
     

temp.apply(sigmoid)
     
0    1.000045
1    1.000000
2    1.000000
3    1.000000
4    1.000000
dtype: float64

points_df = pd.DataFrame(
    {
        '1st point':[(3,4),(-6,5),(0,0),(-10,1),(4,5)],
        '2nd point':[(-3,4),(0,0),(2,2),(10,10),(1,1)]
    }
)

points_df
     
1st point	2nd point
0	(3, 4)	(-3, 4)
1	(-6, 5)	(0, 0)
2	(0, 0)	(2, 2)
3	(-10, 1)	(10, 10)
4	(4, 5)	(1, 1)

def euclidean(row):
  pt_A = row['1st point']
  pt_B = row['2nd point']

  return ((pt_A[0] - pt_B[0])**2 + (pt_A[1] - pt_B[1])**2)**0.5
     

points_df['distance'] = points_df.apply(euclidean,axis=1)
points_df
     
1st point	2nd point	distance
0	(3, 4)	(-3, 4)	6.000000
1	(-6, 5)	(0, 0)	7.810250
2	(0, 0)	(2, 2)	2.828427
3	(-10, 1)	(10, 10)	21.931712
4	(4, 5)	(1, 1)	5.000000

# isin(series)
     


     

# corr
     


     


     

# nlargest and nsmallest(series and dataframe)
     

# insert(dataframe)

     


     

# copy(series + dataframe)
     


     
###################################################################################



import pandas as pd
import numpy as np
     

movies = pd.read_csv('/content/imdb-top-1000.csv')
     

movies.head()
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
0	The Shawshank Redemption	1994	142	Drama	9.3	Frank Darabont	Tim Robbins	2343110	28341469.0	80.0
1	The Godfather	1972	175	Crime	9.2	Francis Ford Coppola	Marlon Brando	1620367	134966411.0	100.0
2	The Dark Knight	2008	152	Action	9.0	Christopher Nolan	Christian Bale	2303232	534858444.0	84.0
3	The Godfather: Part II	1974	202	Crime	9.0	Francis Ford Coppola	Al Pacino	1129952	57300000.0	90.0
4	12 Angry Men	1957	96	Crime	9.0	Sidney Lumet	Henry Fonda	689845	4360000.0	96.0

genres = movies.groupby('Genre')
     

# Applying builtin aggregation fuctions on groupby objects
genres.std()
     
Runtime	IMDB_Rating	No_of_Votes	Gross	Metascore
Genre					
Action	28.500706	0.304258	432946.814748	2.256724e+08	12.421252
Adventure	33.317320	0.229781	301188.347642	1.697543e+08	12.345393
Animation	14.530471	0.253221	262173.231571	2.091840e+08	8.813646
Biography	25.514466	0.267140	271284.191372	1.363251e+08	11.028187
Comedy	22.946213	0.228771	188653.570564	1.946513e+08	11.829160
Crime	27.689231	0.335477	373999.730656	1.571191e+08	13.099102
Drama	27.740490	0.267229	305554.162841	2.201164e+08	12.744687
Family	10.606602	0.000000	137008.302816	3.048412e+08	16.970563
Fantasy	12.727922	0.141421	22179.111299	7.606861e+07	NaN
Film-Noir	4.000000	0.152753	54649.083277	7.048472e+07	1.527525
Horror	13.604812	0.311302	234883.508691	9.965017e+07	15.362291
Mystery	14.475423	0.310791	404621.915297	1.567524e+08	18.604435
Thriller	NaN	NaN	NaN	NaN	NaN
Western	17.153717	0.420317	263489.554280	1.230626e+07	9.032349

# find the top 3 genres by total earning
movies.groupby('Genre').sum()['Gross'].sort_values(ascending=False).head(3)
     
Genre
Drama     3.540997e+10
Action    3.263226e+10
Comedy    1.566387e+10
Name: Gross, dtype: float64

movies.groupby('Genre')['Gross'].sum().sort_values(ascending=False).head(3)
     
Genre
Drama     3.540997e+10
Action    3.263226e+10
Comedy    1.566387e+10
Name: Gross, dtype: float64

# find the genre with highest avg IMDB rating
movies.groupby('Genre')['IMDB_Rating'].mean().sort_values(ascending=False).head(1)
     
Genre
Western    8.35
Name: IMDB_Rating, dtype: float64

# find director with most popularity
movies.groupby('Director')['No_of_Votes'].sum().sort_values(ascending=False).head(1)
     
Director
Christopher Nolan    11578345
Name: No_of_Votes, dtype: int64

# find the highest rated movie of each genre
# movies.groupby('Genre')['IMDB_Rating'].max()
     
Genre
Action       9.0
Adventure    8.6
Animation    8.6
Biography    8.9
Comedy       8.6
Crime        9.2
Drama        9.3
Family       7.8
Fantasy      8.1
Film-Noir    8.1
Horror       8.5
Mystery      8.4
Thriller     7.8
Western      8.8
Name: IMDB_Rating, dtype: float64

# find number of movies done by each actor
# movies['Star1'].value_counts()

movies.groupby('Star1')['Series_Title'].count().sort_values(ascending=False)
     
Star1
Tom Hanks             12
Robert De Niro        11
Clint Eastwood        10
Al Pacino             10
Leonardo DiCaprio      9
                      ..
Glen Hansard           1
Giuseppe Battiston     1
Giulietta Masina       1
Gerardo Taracena       1
Ömer Faruk Sorak       1
Name: Series_Title, Length: 660, dtype: int64

# GroupBy Attributes and Methods
# find total number of groups -> len
# find items in each group -> size
# first()/last() -> nth item
# get_group -> vs filtering
# groups
# describe
# sample
# nunique
     

len(movies.groupby('Genre'))
     
14

movies['Genre'].nunique()
     
14

movies.groupby('Genre').size()
     
Genre
Action       172
Adventure     72
Animation     82
Biography     88
Comedy       155
Crime        107
Drama        289
Family         2
Fantasy        2
Film-Noir      3
Horror        11
Mystery       12
Thriller       1
Western        4
dtype: int64

genres = movies.groupby('Genre')
# genres.first()
# genres.last()
genres.nth(6)
     
Series_Title	Released_Year	Runtime	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
Genre									
Action	Star Wars: Episode V - The Empire Strikes Back	1980	124	8.7	Irvin Kershner	Mark Hamill	1159315	290475067.0	82.0
Adventure	North by Northwest	1959	136	8.3	Alfred Hitchcock	Cary Grant	299198	13275000.0	98.0
Animation	WALL·E	2008	98	8.4	Andrew Stanton	Ben Burtt	999790	223808164.0	95.0
Biography	Braveheart	1995	178	8.3	Mel Gibson	Mel Gibson	959181	75600000.0	68.0
Comedy	The Great Dictator	1940	125	8.4	Charles Chaplin	Charles Chaplin	203150	288475.0	NaN
Crime	Se7en	1995	127	8.6	David Fincher	Morgan Freeman	1445096	100125643.0	65.0
Drama	It's a Wonderful Life	1946	130	8.6	Frank Capra	James Stewart	405801	82385199.0	89.0
Horror	Get Out	2017	104	7.7	Jordan Peele	Daniel Kaluuya	492851	176040665.0	85.0
Mystery	Sleuth	1972	138	8.0	Joseph L. Mankiewicz	Laurence Olivier	44748	4081254.0	NaN

movies['Genre'].value_counts()
     
Drama        289
Action       172
Comedy       155
Crime        107
Biography     88
Animation     82
Adventure     72
Mystery       12
Horror        11
Western        4
Film-Noir      3
Fantasy        2
Family         2
Thriller       1
Name: Genre, dtype: int64

genres.get_group('Fantasy')

movies[movies['Genre'] == 'Fantasy']
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
321	Das Cabinet des Dr. Caligari	1920	76	Fantasy	8.1	Robert Wiene	Werner Krauss	57428	337574718.0	NaN
568	Nosferatu	1922	94	Fantasy	7.9	F.W. Murnau	Max Schreck	88794	445151978.0	NaN

genres.groups
     
{'Action': [2, 5, 8, 10, 13, 14, 16, 29, 30, 31, 39, 42, 44, 55, 57, 59, 60, 63, 68, 72, 106, 109, 129, 130, 134, 140, 142, 144, 152, 155, 160, 161, 166, 168, 171, 172, 177, 181, 194, 201, 202, 216, 217, 223, 224, 236, 241, 262, 275, 294, 308, 320, 325, 326, 331, 337, 339, 340, 343, 345, 348, 351, 353, 356, 357, 362, 368, 369, 375, 376, 390, 410, 431, 436, 473, 477, 479, 482, 488, 493, 496, 502, 507, 511, 532, 535, 540, 543, 564, 569, 570, 573, 577, 582, 583, 602, 605, 608, 615, 623, ...], 'Adventure': [21, 47, 93, 110, 114, 116, 118, 137, 178, 179, 191, 193, 209, 226, 231, 247, 267, 273, 281, 300, 301, 304, 306, 323, 329, 361, 366, 377, 402, 406, 415, 426, 458, 470, 497, 498, 506, 513, 514, 537, 549, 552, 553, 566, 576, 604, 609, 618, 638, 647, 675, 681, 686, 692, 711, 713, 739, 755, 781, 797, 798, 851, 873, 884, 912, 919, 947, 957, 964, 966, 984, 991], 'Animation': [23, 43, 46, 56, 58, 61, 66, 70, 101, 135, 146, 151, 158, 170, 197, 205, 211, 213, 219, 229, 230, 242, 245, 246, 270, 330, 332, 358, 367, 378, 386, 389, 394, 395, 399, 401, 405, 409, 469, 499, 510, 516, 518, 522, 578, 586, 592, 595, 596, 599, 633, 640, 643, 651, 665, 672, 694, 728, 740, 741, 744, 756, 758, 761, 771, 783, 796, 799, 822, 828, 843, 875, 891, 892, 902, 906, 920, 956, 971, 976, 986, 992], 'Biography': [7, 15, 18, 35, 38, 54, 102, 107, 131, 139, 147, 157, 159, 173, 176, 212, 215, 218, 228, 235, 243, 263, 276, 282, 290, 298, 317, 328, 338, 342, 346, 359, 360, 365, 372, 373, 385, 411, 416, 418, 424, 429, 484, 525, 536, 542, 545, 575, 579, 587, 600, 606, 614, 622, 632, 635, 644, 649, 650, 657, 671, 673, 684, 729, 748, 753, 757, 759, 766, 770, 779, 809, 810, 815, 820, 831, 849, 858, 877, 882, 897, 910, 915, 923, 940, 949, 952, 987], 'Comedy': [19, 26, 51, 52, 64, 78, 83, 95, 96, 112, 117, 120, 127, 128, 132, 153, 169, 183, 192, 204, 207, 208, 214, 221, 233, 238, 240, 250, 251, 252, 256, 261, 266, 277, 284, 311, 313, 316, 318, 322, 327, 374, 379, 381, 392, 396, 403, 413, 414, 417, 427, 435, 445, 446, 449, 455, 459, 460, 463, 464, 466, 471, 472, 475, 481, 490, 494, 500, 503, 509, 526, 528, 530, 531, 533, 538, 539, 541, 547, 557, 558, 562, 563, 565, 574, 591, 593, 594, 598, 613, 626, 630, 660, 662, 667, 679, 680, 683, 687, 701, ...], 'Crime': [1, 3, 4, 6, 22, 25, 27, 28, 33, 37, 41, 71, 77, 79, 86, 87, 103, 108, 111, 113, 123, 125, 133, 136, 162, 163, 164, 165, 180, 186, 187, 189, 198, 222, 232, 239, 255, 257, 287, 288, 299, 305, 335, 363, 364, 380, 384, 397, 437, 438, 441, 442, 444, 450, 451, 465, 474, 480, 485, 487, 505, 512, 519, 520, 523, 527, 546, 556, 560, 584, 597, 603, 607, 611, 621, 639, 653, 664, 669, 676, 695, 708, 723, 762, 763, 767, 775, 791, 795, 802, 811, 823, 827, 833, 885, 895, 921, 922, 926, 938, ...], 'Drama': [0, 9, 11, 17, 20, 24, 32, 34, 36, 40, 45, 50, 53, 62, 65, 67, 73, 74, 76, 80, 82, 84, 85, 88, 89, 90, 91, 92, 94, 97, 98, 99, 100, 104, 105, 121, 122, 124, 126, 138, 141, 143, 148, 149, 150, 154, 156, 167, 174, 175, 182, 184, 185, 188, 190, 195, 196, 199, 200, 203, 206, 210, 225, 227, 234, 237, 244, 248, 249, 253, 254, 258, 259, 260, 264, 265, 268, 269, 272, 274, 278, 279, 280, 283, 285, 286, 289, 291, 292, 293, 295, 296, 297, 302, 303, 307, 310, 312, 314, 315, ...], 'Family': [688, 698], 'Fantasy': [321, 568], 'Film-Noir': [309, 456, 712], 'Horror': [49, 75, 271, 419, 544, 707, 724, 844, 876, 932, 948], 'Mystery': [69, 81, 119, 145, 220, 393, 420, 714, 829, 899, 959, 961], 'Thriller': [700], 'Western': [12, 48, 115, 691]}

genres.describe()
     
Runtime	IMDB_Rating	...	Gross	Metascore
count	mean	std	min	25%	50%	75%	max	count	mean	...	75%	max	count	mean	std	min	25%	50%	75%	max
Genre																					
Action	172.0	129.046512	28.500706	45.0	110.75	127.5	143.25	321.0	172.0	7.949419	...	2.674437e+08	936662225.0	143.0	73.419580	12.421252	33.0	65.00	74.0	82.00	98.0
Adventure	72.0	134.111111	33.317320	88.0	109.00	127.0	149.00	228.0	72.0	7.937500	...	1.998070e+08	874211619.0	64.0	78.437500	12.345393	41.0	69.75	80.5	87.25	100.0
Animation	82.0	99.585366	14.530471	71.0	90.00	99.5	106.75	137.0	82.0	7.930488	...	2.520612e+08	873839108.0	75.0	81.093333	8.813646	61.0	75.00	82.0	87.50	96.0
Biography	88.0	136.022727	25.514466	93.0	120.00	129.0	146.25	209.0	88.0	7.938636	...	9.829924e+07	753585104.0	79.0	76.240506	11.028187	48.0	70.50	76.0	84.50	97.0
Comedy	155.0	112.129032	22.946213	68.0	96.00	106.0	124.50	188.0	155.0	7.901290	...	8.107809e+07	886752933.0	125.0	78.720000	11.829160	45.0	72.00	79.0	88.00	99.0
Crime	107.0	126.392523	27.689231	80.0	106.50	122.0	141.50	229.0	107.0	8.016822	...	7.102163e+07	790482117.0	87.0	77.080460	13.099102	47.0	69.50	77.0	87.00	100.0
Drama	289.0	124.737024	27.740490	64.0	105.00	121.0	137.00	242.0	289.0	7.957439	...	1.164461e+08	924558264.0	241.0	79.701245	12.744687	28.0	72.00	82.0	89.00	100.0
Family	2.0	107.500000	10.606602	100.0	103.75	107.5	111.25	115.0	2.0	7.800000	...	3.273329e+08	435110554.0	2.0	79.000000	16.970563	67.0	73.00	79.0	85.00	91.0
Fantasy	2.0	85.000000	12.727922	76.0	80.50	85.0	89.50	94.0	2.0	8.000000	...	4.182577e+08	445151978.0	0.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Film-Noir	3.0	104.000000	4.000000	100.0	102.00	104.0	106.00	108.0	3.0	7.966667	...	6.273068e+07	123353292.0	3.0	95.666667	1.527525	94.0	95.00	96.0	96.50	97.0
Horror	11.0	102.090909	13.604812	71.0	98.00	103.0	109.00	122.0	11.0	7.909091	...	1.362817e+08	298791505.0	11.0	80.000000	15.362291	46.0	77.50	87.0	88.50	97.0
Mystery	12.0	119.083333	14.475423	96.0	110.75	117.5	130.25	138.0	12.0	7.975000	...	1.310949e+08	474203697.0	8.0	79.125000	18.604435	52.0	65.25	77.0	98.50	100.0
Thriller	1.0	108.000000	NaN	108.0	108.00	108.0	108.00	108.0	1.0	7.800000	...	1.755074e+07	17550741.0	1.0	81.000000	NaN	81.0	81.00	81.0	81.00	81.0
Western	4.0	148.250000	17.153717	132.0	134.25	148.0	162.00	165.0	4.0	8.350000	...	1.920000e+07	31800000.0	4.0	78.250000	9.032349	69.0	72.75	77.0	82.50	90.0
14 rows × 40 columns


genres.sample(2,replace=True)
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
944	Batoru rowaiaru	2000	114	Action	7.6	Kinji Fukasaku	Tatsuya Fujiwara	169091	195856489.0	81.0
625	Apocalypto	2006	139	Action	7.8	Mel Gibson	Gerardo Taracena	291018	50866635.0	68.0
991	Kelly's Heroes	1970	144	Adventure	7.6	Brian G. Hutton	Clint Eastwood	45338	1378435.0	50.0
300	Ben-Hur	1959	212	Adventure	8.1	William Wyler	Charlton Heston	219466	74700000.0	90.0
891	Incredibles 2	2018	118	Animation	7.6	Brad Bird	Craig T. Nelson	250057	608581744.0	80.0
389	The Iron Giant	1999	86	Animation	8.0	Brad Bird	Eli Marienthal	172083	23159305.0	85.0
536	All the President's Men	1976	138	Biography	7.9	Alan J. Pakula	Dustin Hoffman	103031	70600000.0	84.0
635	Walk the Line	2005	136	Biography	7.8	James Mangold	Joaquin Phoenix	234207	119519402.0	72.0
826	Barton Fink	1991	116	Comedy	7.7	Joel Coen	Ethan Coen	113240	6153939.0	69.0
732	Me and Earl and the Dying Girl	2015	105	Comedy	7.7	Alfonso Gomez-Rejon	Thomas Mann	123210	6743776.0	74.0
438	Touch of Evil	1958	95	Crime	8.0	Orson Welles	Charlton Heston	98431	2237659.0	99.0
222	Prisoners	2013	153	Crime	8.1	Denis Villeneuve	Hugh Jackman	601149	61002302.0	70.0
555	High Noon	1952	85	Drama	7.9	Fred Zinnemann	Gary Cooper	97222	9450000.0	89.0
314	Gone with the Wind	1939	238	Drama	8.1	Victor Fleming	George Cukor	290074	198676459.0	97.0
698	Willy Wonka & the Chocolate Factory	1971	100	Family	7.8	Mel Stuart	Gene Wilder	178731	4000000.0	67.0
698	Willy Wonka & the Chocolate Factory	1971	100	Family	7.8	Mel Stuart	Gene Wilder	178731	4000000.0	67.0
321	Das Cabinet des Dr. Caligari	1920	76	Fantasy	8.1	Robert Wiene	Werner Krauss	57428	337574718.0	NaN
321	Das Cabinet des Dr. Caligari	1920	76	Fantasy	8.1	Robert Wiene	Werner Krauss	57428	337574718.0	NaN
456	The Maltese Falcon	1941	100	Film-Noir	8.0	John Huston	Humphrey Bogart	148928	2108060.0	96.0
456	The Maltese Falcon	1941	100	Film-Noir	8.0	John Huston	Humphrey Bogart	148928	2108060.0	96.0
544	Night of the Living Dead	1968	96	Horror	7.9	George A. Romero	Duane Jones	116557	89029.0	89.0
707	The Innocents	1961	100	Horror	7.8	Jack Clayton	Deborah Kerr	27007	2616000.0	88.0
393	Twelve Monkeys	1995	129	Mystery	8.0	Terry Gilliam	Bruce Willis	578443	57141459.0	74.0
829	Spoorloos	1988	107	Mystery	7.7	George Sluizer	Bernard-Pierre Donnadieu	33982	367916835.0	NaN
700	Wait Until Dark	1967	108	Thriller	7.8	Terence Young	Audrey Hepburn	27733	17550741.0	81.0
700	Wait Until Dark	1967	108	Thriller	7.8	Terence Young	Audrey Hepburn	27733	17550741.0	81.0
12	Il buono, il brutto, il cattivo	1966	161	Western	8.8	Sergio Leone	Clint Eastwood	688390	6100000.0	90.0
115	Per qualche dollaro in più	1965	132	Western	8.3	Sergio Leone	Clint Eastwood	232772	15000000.0	74.0

genres.nunique()
     
Series_Title	Released_Year	Runtime	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
Genre									
Action	172	61	78	15	123	121	172	172	50
Adventure	72	49	58	10	59	59	72	72	33
Animation	82	35	41	11	51	77	82	82	29
Biography	88	44	56	13	76	72	88	88	40
Comedy	155	72	70	11	113	133	155	155	44
Crime	106	56	65	14	86	85	107	107	39
Drama	289	83	95	14	211	250	288	287	52
Family	2	2	2	1	2	2	2	2	2
Fantasy	2	2	2	2	2	2	2	2	0
Film-Noir	3	3	3	3	3	3	3	3	3
Horror	11	11	10	8	10	11	11	11	9
Mystery	12	11	10	8	10	11	12	12	7
Thriller	1	1	1	1	1	1	1	1	1
Western	4	4	4	4	2	2	4	4	4

# agg method
# passing dict
genres.agg(
    {
        'Runtime':'mean',
        'IMDB_Rating':'mean',
        'No_of_Votes':'sum',
        'Gross':'sum',
        'Metascore':'min'
    }
)
     
Runtime	IMDB_Rating	No_of_Votes	Gross	Metascore
Genre					
Action	129.046512	7.949419	72282412	3.263226e+10	33.0
Adventure	134.111111	7.937500	22576163	9.496922e+09	41.0
Animation	99.585366	7.930488	21978630	1.463147e+10	61.0
Biography	136.022727	7.938636	24006844	8.276358e+09	48.0
Comedy	112.129032	7.901290	27620327	1.566387e+10	45.0
Crime	126.392523	8.016822	33533615	8.452632e+09	47.0
Drama	124.737024	7.957439	61367304	3.540997e+10	28.0
Family	107.500000	7.800000	551221	4.391106e+08	67.0
Fantasy	85.000000	8.000000	146222	7.827267e+08	NaN
Film-Noir	104.000000	7.966667	367215	1.259105e+08	94.0
Horror	102.090909	7.909091	3742556	1.034649e+09	46.0
Mystery	119.083333	7.975000	4203004	1.256417e+09	52.0
Thriller	108.000000	7.800000	27733	1.755074e+07	81.0
Western	148.250000	8.350000	1289665	5.822151e+07	69.0

# passing list
genres.agg(['min','max','mean','sum'])
     
Runtime	IMDB_Rating	No_of_Votes	Gross	Metascore
min	max	mean	sum	min	max	mean	sum	min	max	mean	sum	min	max	mean	sum	min	max	mean	sum
Genre																				
Action	45	321	129.046512	22196	7.6	9.0	7.949419	1367.3	25312	2303232	420246.581395	72282412	3296.0	936662225.0	1.897224e+08	3.263226e+10	33.0	98.0	73.419580	10499.0
Adventure	88	228	134.111111	9656	7.6	8.6	7.937500	571.5	29999	1512360	313557.819444	22576163	61001.0	874211619.0	1.319017e+08	9.496922e+09	41.0	100.0	78.437500	5020.0
Animation	71	137	99.585366	8166	7.6	8.6	7.930488	650.3	25229	999790	268032.073171	21978630	128985.0	873839108.0	1.784326e+08	1.463147e+10	61.0	96.0	81.093333	6082.0
Biography	93	209	136.022727	11970	7.6	8.9	7.938636	698.6	27254	1213505	272805.045455	24006844	21877.0	753585104.0	9.404952e+07	8.276358e+09	48.0	97.0	76.240506	6023.0
Comedy	68	188	112.129032	17380	7.6	8.6	7.901290	1224.7	26337	939631	178195.658065	27620327	1305.0	886752933.0	1.010572e+08	1.566387e+10	45.0	99.0	78.720000	9840.0
Crime	80	229	126.392523	13524	7.6	9.2	8.016822	857.8	27712	1826188	313398.271028	33533615	6013.0	790482117.0	7.899656e+07	8.452632e+09	47.0	100.0	77.080460	6706.0
Drama	64	242	124.737024	36049	7.6	9.3	7.957439	2299.7	25088	2343110	212343.612457	61367304	3600.0	924558264.0	1.225259e+08	3.540997e+10	28.0	100.0	79.701245	19208.0
Family	100	115	107.500000	215	7.8	7.8	7.800000	15.6	178731	372490	275610.500000	551221	4000000.0	435110554.0	2.195553e+08	4.391106e+08	67.0	91.0	79.000000	158.0
Fantasy	76	94	85.000000	170	7.9	8.1	8.000000	16.0	57428	88794	73111.000000	146222	337574718.0	445151978.0	3.913633e+08	7.827267e+08	NaN	NaN	NaN	0.0
Film-Noir	100	108	104.000000	312	7.8	8.1	7.966667	23.9	59556	158731	122405.000000	367215	449191.0	123353292.0	4.197018e+07	1.259105e+08	94.0	97.0	95.666667	287.0
Horror	71	122	102.090909	1123	7.6	8.5	7.909091	87.0	27007	787806	340232.363636	3742556	89029.0	298791505.0	9.405902e+07	1.034649e+09	46.0	97.0	80.000000	880.0
Mystery	96	138	119.083333	1429	7.6	8.4	7.975000	95.7	33982	1129894	350250.333333	4203004	1035953.0	474203697.0	1.047014e+08	1.256417e+09	52.0	100.0	79.125000	633.0
Thriller	108	108	108.000000	108	7.8	7.8	7.800000	7.8	27733	27733	27733.000000	27733	17550741.0	17550741.0	1.755074e+07	1.755074e+07	81.0	81.0	81.000000	81.0
Western	132	165	148.250000	593	7.8	8.8	8.350000	33.4	65659	688390	322416.250000	1289665	5321508.0	31800000.0	1.455538e+07	5.822151e+07	69.0	90.0	78.250000	313.0

# Adding both the syntax
genres.agg(
    {
        'Runtime':['min','mean'],
        'IMDB_Rating':'mean',
        'No_of_Votes':['sum','max'],
        'Gross':'sum',
        'Metascore':'min'
    }
)
     
Runtime	IMDB_Rating	No_of_Votes	Gross	Metascore
min	mean	mean	sum	max	sum	min
Genre							
Action	45	129.046512	7.949419	72282412	2303232	3.263226e+10	33.0
Adventure	88	134.111111	7.937500	22576163	1512360	9.496922e+09	41.0
Animation	71	99.585366	7.930488	21978630	999790	1.463147e+10	61.0
Biography	93	136.022727	7.938636	24006844	1213505	8.276358e+09	48.0
Comedy	68	112.129032	7.901290	27620327	939631	1.566387e+10	45.0
Crime	80	126.392523	8.016822	33533615	1826188	8.452632e+09	47.0
Drama	64	124.737024	7.957439	61367304	2343110	3.540997e+10	28.0
Family	100	107.500000	7.800000	551221	372490	4.391106e+08	67.0
Fantasy	76	85.000000	8.000000	146222	88794	7.827267e+08	NaN
Film-Noir	100	104.000000	7.966667	367215	158731	1.259105e+08	94.0
Horror	71	102.090909	7.909091	3742556	787806	1.034649e+09	46.0
Mystery	96	119.083333	7.975000	4203004	1129894	1.256417e+09	52.0
Thriller	108	108.000000	7.800000	27733	27733	1.755074e+07	81.0
Western	132	148.250000	8.350000	1289665	688390	5.822151e+07	69.0

# looping on groups
df = pd.DataFrame(columns=movies.columns)
for group,data in genres:
  df = df.append(data[data['IMDB_Rating'] == data['IMDB_Rating'].max()])

df
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
2	The Dark Knight	2008	152	Action	9.0	Christopher Nolan	Christian Bale	2303232	534858444.0	84.0
21	Interstellar	2014	169	Adventure	8.6	Christopher Nolan	Matthew McConaughey	1512360	188020017.0	74.0
23	Sen to Chihiro no kamikakushi	2001	125	Animation	8.6	Hayao Miyazaki	Daveigh Chase	651376	10055859.0	96.0
7	Schindler's List	1993	195	Biography	8.9	Steven Spielberg	Liam Neeson	1213505	96898818.0	94.0
19	Gisaengchung	2019	132	Comedy	8.6	Bong Joon Ho	Kang-ho Song	552778	53367844.0	96.0
26	La vita è bella	1997	116	Comedy	8.6	Roberto Benigni	Roberto Benigni	623629	57598247.0	59.0
1	The Godfather	1972	175	Crime	9.2	Francis Ford Coppola	Marlon Brando	1620367	134966411.0	100.0
0	The Shawshank Redemption	1994	142	Drama	9.3	Frank Darabont	Tim Robbins	2343110	28341469.0	80.0
688	E.T. the Extra-Terrestrial	1982	115	Family	7.8	Steven Spielberg	Henry Thomas	372490	435110554.0	91.0
698	Willy Wonka & the Chocolate Factory	1971	100	Family	7.8	Mel Stuart	Gene Wilder	178731	4000000.0	67.0
321	Das Cabinet des Dr. Caligari	1920	76	Fantasy	8.1	Robert Wiene	Werner Krauss	57428	337574718.0	NaN
309	The Third Man	1949	104	Film-Noir	8.1	Carol Reed	Orson Welles	158731	449191.0	97.0
49	Psycho	1960	109	Horror	8.5	Alfred Hitchcock	Anthony Perkins	604211	32000000.0	97.0
69	Memento	2000	113	Mystery	8.4	Christopher Nolan	Guy Pearce	1125712	25544867.0	80.0
81	Rear Window	1954	112	Mystery	8.4	Alfred Hitchcock	James Stewart	444074	36764313.0	100.0
700	Wait Until Dark	1967	108	Thriller	7.8	Terence Young	Audrey Hepburn	27733	17550741.0	81.0
12	Il buono, il brutto, il cattivo	1966	161	Western	8.8	Sergio Leone	Clint Eastwood	688390	6100000.0	90.0

# split (apply) combine
# apply -> builtin function

genres.apply(min)
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
Genre										
Action	300	1924	45	Action	7.6	Abhishek Chaubey	Aamir Khan	25312	3296.0	33.0
Adventure	2001: A Space Odyssey	1925	88	Adventure	7.6	Akira Kurosawa	Aamir Khan	29999	61001.0	41.0
Animation	Akira	1940	71	Animation	7.6	Adam Elliot	Adrian Molina	25229	128985.0	61.0
Biography	12 Years a Slave	1928	93	Biography	7.6	Adam McKay	Adrien Brody	27254	21877.0	48.0
Comedy	(500) Days of Summer	1921	68	Comedy	7.6	Alejandro G. Iñárritu	Aamir Khan	26337	1305.0	45.0
Crime	12 Angry Men	1931	80	Crime	7.6	Akira Kurosawa	Ajay Devgn	27712	6013.0	47.0
Drama	1917	1925	64	Drama	7.6	Aamir Khan	Abhay Deol	25088	3600.0	28.0
Family	E.T. the Extra-Terrestrial	1971	100	Family	7.8	Mel Stuart	Gene Wilder	178731	4000000.0	67.0
Fantasy	Das Cabinet des Dr. Caligari	1920	76	Fantasy	7.9	F.W. Murnau	Max Schreck	57428	337574718.0	NaN
Film-Noir	Shadow of a Doubt	1941	100	Film-Noir	7.8	Alfred Hitchcock	Humphrey Bogart	59556	449191.0	94.0
Horror	Alien	1933	71	Horror	7.6	Alejandro Amenábar	Anthony Perkins	27007	89029.0	46.0
Mystery	Dark City	1938	96	Mystery	7.6	Alex Proyas	Bernard-Pierre Donnadieu	33982	1035953.0	52.0
Thriller	Wait Until Dark	1967	108	Thriller	7.8	Terence Young	Audrey Hepburn	27733	17550741.0	81.0
Western	Il buono, il brutto, il cattivo	1965	132	Western	7.8	Clint Eastwood	Clint Eastwood	65659	5321508.0	69.0

# find number of movies starting with A for each group

def foo(group):
  return group['Series_Title'].str.startswith('A').sum()

     

genres.apply(foo)
     
Genre
Action       10
Adventure     2
Animation     2
Biography     9
Comedy       14
Crime         4
Drama        21
Family        0
Fantasy       0
Film-Noir     0
Horror        1
Mystery       0
Thriller      0
Western       0
dtype: int64

# find ranking of each movie in the group according to IMDB score

def rank_movie(group):
  group['genre_rank'] = group['IMDB_Rating'].rank(ascending=False)
  return group
     

genres.apply(rank_movie)
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore	genre_rank
0	The Shawshank Redemption	1994	142	Drama	9.3	Frank Darabont	Tim Robbins	2343110	28341469.0	80.0	1.0
1	The Godfather	1972	175	Crime	9.2	Francis Ford Coppola	Marlon Brando	1620367	134966411.0	100.0	1.0
2	The Dark Knight	2008	152	Action	9.0	Christopher Nolan	Christian Bale	2303232	534858444.0	84.0	1.0
3	The Godfather: Part II	1974	202	Crime	9.0	Francis Ford Coppola	Al Pacino	1129952	57300000.0	90.0	2.5
4	12 Angry Men	1957	96	Crime	9.0	Sidney Lumet	Henry Fonda	689845	4360000.0	96.0	2.5
...	...	...	...	...	...	...	...	...	...	...	...
995	Breakfast at Tiffany's	1961	115	Comedy	7.6	Blake Edwards	Audrey Hepburn	166544	679874270.0	76.0	147.0
996	Giant	1956	201	Drama	7.6	George Stevens	Elizabeth Taylor	34075	195217415.0	84.0	272.5
997	From Here to Eternity	1953	118	Drama	7.6	Fred Zinnemann	Burt Lancaster	43374	30500000.0	85.0	272.5
998	Lifeboat	1944	97	Drama	7.6	Alfred Hitchcock	Tallulah Bankhead	26471	852142728.0	78.0	272.5
999	The 39 Steps	1935	86	Crime	7.6	Alfred Hitchcock	Robert Donat	51853	302787539.0	93.0	101.0
1000 rows × 11 columns


# find normalized IMDB rating group wise

def normal(group):
  group['norm_rating'] = (group['IMDB_Rating'] - group['IMDB_Rating'].min())/(group['IMDB_Rating'].max() - group['IMDB_Rating'].min())
  return group

genres.apply(normal)
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore	norm_rating
0	The Shawshank Redemption	1994	142	Drama	9.3	Frank Darabont	Tim Robbins	2343110	28341469.0	80.0	1.000
1	The Godfather	1972	175	Crime	9.2	Francis Ford Coppola	Marlon Brando	1620367	134966411.0	100.0	1.000
2	The Dark Knight	2008	152	Action	9.0	Christopher Nolan	Christian Bale	2303232	534858444.0	84.0	1.000
3	The Godfather: Part II	1974	202	Crime	9.0	Francis Ford Coppola	Al Pacino	1129952	57300000.0	90.0	0.875
4	12 Angry Men	1957	96	Crime	9.0	Sidney Lumet	Henry Fonda	689845	4360000.0	96.0	0.875
...	...	...	...	...	...	...	...	...	...	...	...
995	Breakfast at Tiffany's	1961	115	Comedy	7.6	Blake Edwards	Audrey Hepburn	166544	679874270.0	76.0	0.000
996	Giant	1956	201	Drama	7.6	George Stevens	Elizabeth Taylor	34075	195217415.0	84.0	0.000
997	From Here to Eternity	1953	118	Drama	7.6	Fred Zinnemann	Burt Lancaster	43374	30500000.0	85.0	0.000
998	Lifeboat	1944	97	Drama	7.6	Alfred Hitchcock	Tallulah Bankhead	26471	852142728.0	78.0	0.000
999	The 39 Steps	1935	86	Crime	7.6	Alfred Hitchcock	Robert Donat	51853	302787539.0	93.0	0.000
1000 rows × 11 columns


# groupby on multiple cols
duo = movies.groupby(['Director','Star1'])
duo
# size
duo.size()
# get_group
duo.get_group(('Aamir Khan','Amole Gupte'))
     
Series_Title	Released_Year	Runtime	Genre	IMDB_Rating	Director	Star1	No_of_Votes	Gross	Metascore
65	Taare Zameen Par	2007	165	Drama	8.4	Aamir Khan	Amole Gupte	168895	1223869.0	NaN

# find the most earning actor->director combo
duo['Gross'].sum().sort_values(ascending=False).head(1)
     
Director        Star1         
Akira Kurosawa  Toshirô Mifune    2.999877e+09
Name: Gross, dtype: float64

# find the best(in-terms of metascore(avg)) actor->genre combo
movies.groupby(['Star1','Genre'])['Metascore'].mean().reset_index().sort_values('Metascore',ascending=False).head(1)
     
Star1	Genre	Metascore
230	Ellar Coltrane	Drama	100.0

# agg on multiple groupby
duo.agg(['min','max','mean'])
     
Runtime	IMDB_Rating	No_of_Votes	Gross	Metascore
min	max	mean	min	max	mean	min	max	mean	min	max	mean	min	max	mean
Director	Star1															
Aamir Khan	Amole Gupte	165	165	165.0	8.4	8.4	8.4	168895	168895	168895.0	1223869.0	1223869.0	1223869.0	NaN	NaN	NaN
Aaron Sorkin	Eddie Redmayne	129	129	129.0	7.8	7.8	7.8	89896	89896	89896.0	853090410.0	853090410.0	853090410.0	77.0	77.0	77.0
Abdellatif Kechiche	Léa Seydoux	180	180	180.0	7.7	7.7	7.7	138741	138741	138741.0	2199675.0	2199675.0	2199675.0	89.0	89.0	89.0
Abhishek Chaubey	Shahid Kapoor	148	148	148.0	7.8	7.8	7.8	27175	27175	27175.0	218428303.0	218428303.0	218428303.0	NaN	NaN	NaN
Abhishek Kapoor	Amit Sadh	130	130	130.0	7.7	7.7	7.7	32628	32628	32628.0	1122527.0	1122527.0	1122527.0	40.0	40.0	40.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
Zaza Urushadze	Lembit Ulfsak	87	87	87.0	8.2	8.2	8.2	40382	40382	40382.0	144501.0	144501.0	144501.0	73.0	73.0	73.0
Zoya Akhtar	Hrithik Roshan	155	155	155.0	8.1	8.1	8.1	67927	67927	67927.0	3108485.0	3108485.0	3108485.0	NaN	NaN	NaN
Vijay Varma	154	154	154.0	8.0	8.0	8.0	31886	31886	31886.0	5566534.0	5566534.0	5566534.0	65.0	65.0	65.0
Çagan Irmak	Çetin Tekindor	112	112	112.0	8.3	8.3	8.3	78925	78925	78925.0	461855363.0	461855363.0	461855363.0	NaN	NaN	NaN
Ömer Faruk Sorak	Cem Yilmaz	127	127	127.0	8.0	8.0	8.0	56960	56960	56960.0	196206077.0	196206077.0	196206077.0	NaN	NaN	NaN
898 rows × 15 columns

Excercise

ipl = pd.read_csv('/content/deliveries.csv')
ipl.head()
     
match_id	inning	batting_team	bowling_team	over	ball	batsman	non_striker	bowler	is_super_over	...	bye_runs	legbye_runs	noball_runs	penalty_runs	batsman_runs	extra_runs	total_runs	player_dismissed	dismissal_kind	fielder
0	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	1	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	0	0	NaN	NaN	NaN
1	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	2	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	0	0	NaN	NaN	NaN
2	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	3	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	4	0	4	NaN	NaN	NaN
3	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	4	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	0	0	NaN	NaN	NaN
4	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	5	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	2	2	NaN	NaN	NaN
5 rows × 21 columns


ipl.shape
     
(179078, 21)

# find the top 10 batsman in terms of runs
ipl.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)
     
batsman
V Kohli           5434
SK Raina          5415
RG Sharma         4914
DA Warner         4741
S Dhawan          4632
CH Gayle          4560
MS Dhoni          4477
RV Uthappa        4446
AB de Villiers    4428
G Gambhir         4223
Name: batsman_runs, dtype: int64

# find the batsman with max no of sixes
six = ipl[ipl['batsman_runs'] == 6]

six.groupby('batsman')['batsman'].count().sort_values(ascending=False).head(1).index[0]
     
'CH Gayle'

# find batsman with most number of 4's and 6's in last 5 overs
temp_df = ipl[ipl['over'] > 15]
temp_df = temp_df[(temp_df['batsman_runs'] == 4) | (temp_df['batsman_runs'] == 6)]
temp_df.groupby('batsman')['batsman'].count().sort_values(ascending=False).head(1).index[0]
     
'MS Dhoni'

# find V Kohli's record against all teams
temp_df = ipl[ipl['batsman'] == 'V Kohli']

temp_df.groupby('bowling_team')['batsman_runs'].sum().reset_index()
     
bowling_team	batsman_runs
0	Chennai Super Kings	749
1	Deccan Chargers	306
2	Delhi Capitals	66
3	Delhi Daredevils	763
4	Gujarat Lions	283
5	Kings XI Punjab	636
6	Kochi Tuskers Kerala	50
7	Kolkata Knight Riders	675
8	Mumbai Indians	628
9	Pune Warriors	128
10	Rajasthan Royals	370
11	Rising Pune Supergiant	83
12	Rising Pune Supergiants	188
13	Sunrisers Hyderabad	509

# Create a function that can return the highest score of any batsman
temp_df = ipl[ipl['batsman'] == 'V Kohli']
temp_df.groupby('match_id')['batsman_runs'].sum().sort_values(ascending=False).head(1).values[0]
     
113

def highest(batsman):
  temp_df = ipl[ipl['batsman'] == batsman]
  return temp_df.groupby('match_id')['batsman_runs'].sum().sort_values(ascending=False).head(1).values[0]

     

highest('DA Warner')
     
126


     
####################################################################################



import pandas as pd
import numpy as np
     

courses = pd.read_csv('/content/courses.csv')
students = pd.read_csv('/content/students.csv')
nov = pd.read_csv('/content/reg-month1.csv')
dec = pd.read_csv('/content/reg-month2.csv')

matches = pd.read_csv('/content/matches.csv')
delivery = pd.read_csv('/content/deliveries.csv')
     

dec
     
student_id	course_id
0	3	5
1	16	7
2	12	10
3	12	1
4	14	9
5	7	7
6	7	2
7	16	3
8	17	10
9	11	8
10	14	6
11	12	5
12	12	7
13	18	8
14	1	10
15	1	9
16	2	5
17	7	6
18	22	5
19	22	6
20	23	9
21	23	5
22	14	4
23	14	1
24	11	10
25	42	9
26	50	8
27	38	1

# pd.concat
# df.concat
# ignore_index
# df.append
# mullitindex -> fetch using iloc
# concat dataframes horizontally
     

regs = pd.concat([nov,dec],ignore_index=True)
regs
     
student_id	course_id
0	23	1
1	15	5
2	18	6
3	23	4
4	16	9
5	18	1
6	1	1
7	7	8
8	22	3
9	15	1
10	19	4
11	1	6
12	7	10
13	11	7
14	13	3
15	24	4
16	21	1
17	16	5
18	23	3
19	17	7
20	23	6
21	25	1
22	19	2
23	25	10
24	3	3
25	3	5
26	16	7
27	12	10
28	12	1
29	14	9
30	7	7
31	7	2
32	16	3
33	17	10
34	11	8
35	14	6
36	12	5
37	12	7
38	18	8
39	1	10
40	1	9
41	2	5
42	7	6
43	22	5
44	22	6
45	23	9
46	23	5
47	14	4
48	14	1
49	11	10
50	42	9
51	50	8
52	38	1

nov.append(dec,ignore_index=True)
     
student_id	course_id
0	23	1
1	15	5
2	18	6
3	23	4
4	16	9
5	18	1
6	1	1
7	7	8
8	22	3
9	15	1
10	19	4
11	1	6
12	7	10
13	11	7
14	13	3
15	24	4
16	21	1
17	16	5
18	23	3
19	17	7
20	23	6
21	25	1
22	19	2
23	25	10
24	3	3
25	3	5
26	16	7
27	12	10
28	12	1
29	14	9
30	7	7
31	7	2
32	16	3
33	17	10
34	11	8
35	14	6
36	12	5
37	12	7
38	18	8
39	1	10
40	1	9
41	2	5
42	7	6
43	22	5
44	22	6
45	23	9
46	23	5
47	14	4
48	14	1
49	11	10
50	42	9
51	50	8
52	38	1

multi = pd.concat([nov,dec],keys=['Nov','Dec'])
# Multiindex DataFrame
multi.loc[('Dec',4)]
     
student_id    14
course_id      9
Name: (Dec, 4), dtype: int64

pd.concat([nov,dec],axis=1)
     
student_id	course_id	student_id	course_id
0	23.0	1.0	3	5
1	15.0	5.0	16	7
2	18.0	6.0	12	10
3	23.0	4.0	12	1
4	16.0	9.0	14	9
5	18.0	1.0	7	7
6	1.0	1.0	7	2
7	7.0	8.0	16	3
8	22.0	3.0	17	10
9	15.0	1.0	11	8
10	19.0	4.0	14	6
11	1.0	6.0	12	5
12	7.0	10.0	12	7
13	11.0	7.0	18	8
14	13.0	3.0	1	10
15	24.0	4.0	1	9
16	21.0	1.0	2	5
17	16.0	5.0	7	6
18	23.0	3.0	22	5
19	17.0	7.0	22	6
20	23.0	6.0	23	9
21	25.0	1.0	23	5
22	19.0	2.0	14	4
23	25.0	10.0	14	1
24	3.0	3.0	11	10
25	NaN	NaN	42	9
26	NaN	NaN	50	8
27	NaN	NaN	38	1

# inner join
students.merge(regs,how='inner',on='student_id')
     
student_id	name	partner	course_id
0	1	Kailash Harjo	23	1
1	1	Kailash Harjo	23	6
2	1	Kailash Harjo	23	10
3	1	Kailash Harjo	23	9
4	2	Esha Butala	1	5
5	3	Parveen Bhalla	3	3
6	3	Parveen Bhalla	3	5
7	7	Tarun Thaker	9	8
8	7	Tarun Thaker	9	10
9	7	Tarun Thaker	9	7
10	7	Tarun Thaker	9	2
11	7	Tarun Thaker	9	6
12	11	David Mukhopadhyay	20	7
13	11	David Mukhopadhyay	20	8
14	11	David Mukhopadhyay	20	10
15	12	Radha Dutt	19	10
16	12	Radha Dutt	19	1
17	12	Radha Dutt	19	5
18	12	Radha Dutt	19	7
19	13	Munni Varghese	24	3
20	14	Pranab Natarajan	22	9
21	14	Pranab Natarajan	22	6
22	14	Pranab Natarajan	22	4
23	14	Pranab Natarajan	22	1
24	15	Preet Sha	16	5
25	15	Preet Sha	16	1
26	16	Elias Dodiya	25	9
27	16	Elias Dodiya	25	5
28	16	Elias Dodiya	25	7
29	16	Elias Dodiya	25	3
30	17	Yasmin Palan	7	7
31	17	Yasmin Palan	7	10
32	18	Fardeen Mahabir	13	6
33	18	Fardeen Mahabir	13	1
34	18	Fardeen Mahabir	13	8
35	19	Qabeel Raman	12	4
36	19	Qabeel Raman	12	2
37	21	Seema Kota	15	1
38	22	Yash Sethi	21	3
39	22	Yash Sethi	21	5
40	22	Yash Sethi	21	6
41	23	Chhavi Lachman	18	1
42	23	Chhavi Lachman	18	4
43	23	Chhavi Lachman	18	3
44	23	Chhavi Lachman	18	6
45	23	Chhavi Lachman	18	9
46	23	Chhavi Lachman	18	5
47	24	Radhika Suri	17	4
48	25	Shashank D’Alia	2	1
49	25	Shashank D’Alia	2	10

# left join
courses.merge(regs,how='left',on='course_id')
     
course_id	course_name	price	student_id
0	1	python	2499	23.0
1	1	python	2499	18.0
2	1	python	2499	1.0
3	1	python	2499	15.0
4	1	python	2499	21.0
5	1	python	2499	25.0
6	1	python	2499	12.0
7	1	python	2499	14.0
8	1	python	2499	38.0
9	2	sql	3499	19.0
10	2	sql	3499	7.0
11	3	data analysis	4999	22.0
12	3	data analysis	4999	13.0
13	3	data analysis	4999	23.0
14	3	data analysis	4999	3.0
15	3	data analysis	4999	16.0
16	4	machine learning	9999	23.0
17	4	machine learning	9999	19.0
18	4	machine learning	9999	24.0
19	4	machine learning	9999	14.0
20	5	tableau	2499	15.0
21	5	tableau	2499	16.0
22	5	tableau	2499	3.0
23	5	tableau	2499	12.0
24	5	tableau	2499	2.0
25	5	tableau	2499	22.0
26	5	tableau	2499	23.0
27	6	power bi	1899	18.0
28	6	power bi	1899	1.0
29	6	power bi	1899	23.0
30	6	power bi	1899	14.0
31	6	power bi	1899	7.0
32	6	power bi	1899	22.0
33	7	ms sxcel	1599	11.0
34	7	ms sxcel	1599	17.0
35	7	ms sxcel	1599	16.0
36	7	ms sxcel	1599	7.0
37	7	ms sxcel	1599	12.0
38	8	pandas	1099	7.0
39	8	pandas	1099	11.0
40	8	pandas	1099	18.0
41	8	pandas	1099	50.0
42	9	plotly	699	16.0
43	9	plotly	699	14.0
44	9	plotly	699	1.0
45	9	plotly	699	23.0
46	9	plotly	699	42.0
47	10	pyspark	2499	7.0
48	10	pyspark	2499	25.0
49	10	pyspark	2499	12.0
50	10	pyspark	2499	17.0
51	10	pyspark	2499	1.0
52	10	pyspark	2499	11.0
53	11	Numpy	699	NaN
54	12	C++	1299	NaN

# right join
temp_df = pd.DataFrame({
    'student_id':[26,27,28],
    'name':['Nitish','Ankit','Rahul'],
    'partner':[28,26,17]
})

students = pd.concat([students,temp_df],ignore_index=True)
     

students.tail()
     
student_id	name	partner
23	24	Radhika Suri	17
24	25	Shashank D’Alia	2
25	26	Nitish	28
26	27	Ankit	26
27	28	Rahul	17

students.merge(regs,how='right',on='student_id')
     
student_id	name	partner	course_id
0	23	Chhavi Lachman	18.0	1
1	15	Preet Sha	16.0	5
2	18	Fardeen Mahabir	13.0	6
3	23	Chhavi Lachman	18.0	4
4	16	Elias Dodiya	25.0	9
5	18	Fardeen Mahabir	13.0	1
6	1	Kailash Harjo	23.0	1
7	7	Tarun Thaker	9.0	8
8	22	Yash Sethi	21.0	3
9	15	Preet Sha	16.0	1
10	19	Qabeel Raman	12.0	4
11	1	Kailash Harjo	23.0	6
12	7	Tarun Thaker	9.0	10
13	11	David Mukhopadhyay	20.0	7
14	13	Munni Varghese	24.0	3
15	24	Radhika Suri	17.0	4
16	21	Seema Kota	15.0	1
17	16	Elias Dodiya	25.0	5
18	23	Chhavi Lachman	18.0	3
19	17	Yasmin Palan	7.0	7
20	23	Chhavi Lachman	18.0	6
21	25	Shashank D’Alia	2.0	1
22	19	Qabeel Raman	12.0	2
23	25	Shashank D’Alia	2.0	10
24	3	Parveen Bhalla	3.0	3
25	3	Parveen Bhalla	3.0	5
26	16	Elias Dodiya	25.0	7
27	12	Radha Dutt	19.0	10
28	12	Radha Dutt	19.0	1
29	14	Pranab Natarajan	22.0	9
30	7	Tarun Thaker	9.0	7
31	7	Tarun Thaker	9.0	2
32	16	Elias Dodiya	25.0	3
33	17	Yasmin Palan	7.0	10
34	11	David Mukhopadhyay	20.0	8
35	14	Pranab Natarajan	22.0	6
36	12	Radha Dutt	19.0	5
37	12	Radha Dutt	19.0	7
38	18	Fardeen Mahabir	13.0	8
39	1	Kailash Harjo	23.0	10
40	1	Kailash Harjo	23.0	9
41	2	Esha Butala	1.0	5
42	7	Tarun Thaker	9.0	6
43	22	Yash Sethi	21.0	5
44	22	Yash Sethi	21.0	6
45	23	Chhavi Lachman	18.0	9
46	23	Chhavi Lachman	18.0	5
47	14	Pranab Natarajan	22.0	4
48	14	Pranab Natarajan	22.0	1
49	11	David Mukhopadhyay	20.0	10
50	42	NaN	NaN	9
51	50	NaN	NaN	8
52	38	NaN	NaN	1

regs.merge(students,how='left',on='student_id')
     
student_id	course_id	name	partner
0	23	1	Chhavi Lachman	18.0
1	15	5	Preet Sha	16.0
2	18	6	Fardeen Mahabir	13.0
3	23	4	Chhavi Lachman	18.0
4	16	9	Elias Dodiya	25.0
5	18	1	Fardeen Mahabir	13.0
6	1	1	Kailash Harjo	23.0
7	7	8	Tarun Thaker	9.0
8	22	3	Yash Sethi	21.0
9	15	1	Preet Sha	16.0
10	19	4	Qabeel Raman	12.0
11	1	6	Kailash Harjo	23.0
12	7	10	Tarun Thaker	9.0
13	11	7	David Mukhopadhyay	20.0
14	13	3	Munni Varghese	24.0
15	24	4	Radhika Suri	17.0
16	21	1	Seema Kota	15.0
17	16	5	Elias Dodiya	25.0
18	23	3	Chhavi Lachman	18.0
19	17	7	Yasmin Palan	7.0
20	23	6	Chhavi Lachman	18.0
21	25	1	Shashank D’Alia	2.0
22	19	2	Qabeel Raman	12.0
23	25	10	Shashank D’Alia	2.0
24	3	3	Parveen Bhalla	3.0
25	3	5	Parveen Bhalla	3.0
26	16	7	Elias Dodiya	25.0
27	12	10	Radha Dutt	19.0
28	12	1	Radha Dutt	19.0
29	14	9	Pranab Natarajan	22.0
30	7	7	Tarun Thaker	9.0
31	7	2	Tarun Thaker	9.0
32	16	3	Elias Dodiya	25.0
33	17	10	Yasmin Palan	7.0
34	11	8	David Mukhopadhyay	20.0
35	14	6	Pranab Natarajan	22.0
36	12	5	Radha Dutt	19.0
37	12	7	Radha Dutt	19.0
38	18	8	Fardeen Mahabir	13.0
39	1	10	Kailash Harjo	23.0
40	1	9	Kailash Harjo	23.0
41	2	5	Esha Butala	1.0
42	7	6	Tarun Thaker	9.0
43	22	5	Yash Sethi	21.0
44	22	6	Yash Sethi	21.0
45	23	9	Chhavi Lachman	18.0
46	23	5	Chhavi Lachman	18.0
47	14	4	Pranab Natarajan	22.0
48	14	1	Pranab Natarajan	22.0
49	11	10	David Mukhopadhyay	20.0
50	42	9	NaN	NaN
51	50	8	NaN	NaN
52	38	1	NaN	NaN

# outer join
students.merge(regs,how='outer',on='student_id').tail(10)
     
student_id	name	partner	course_id
53	23	Chhavi Lachman	18.0	5.0
54	24	Radhika Suri	17.0	4.0
55	25	Shashank D’Alia	2.0	1.0
56	25	Shashank D’Alia	2.0	10.0
57	26	Nitish	28.0	NaN
58	27	Ankit	26.0	NaN
59	28	Rahul	17.0	NaN
60	42	NaN	NaN	9.0
61	50	NaN	NaN	8.0
62	38	NaN	NaN	1.0

# 1. find total revenue generated
total = regs.merge(courses,how='inner',on='course_id')['price'].sum()
total
     
154247

# 2. find month by month revenue
temp_df = pd.concat([nov,dec],keys=['Nov','Dec']).reset_index()
temp_df.merge(courses,on='course_id').groupby('level_0')['price'].sum()
     
level_0
Dec    65072
Nov    89175
Name: price, dtype: int64

# 3. Print the registration table
# cols -> name -> course -> price
regs.merge(students,on='student_id').merge(courses,on='course_id')[['name','course_name','price']]
     
name	course_name	price
0	Chhavi Lachman	python	2499
1	Preet Sha	python	2499
2	Fardeen Mahabir	python	2499
3	Kailash Harjo	python	2499
4	Seema Kota	python	2499
5	Shashank D’Alia	python	2499
6	Radha Dutt	python	2499
7	Pranab Natarajan	python	2499
8	Chhavi Lachman	machine learning	9999
9	Qabeel Raman	machine learning	9999
10	Radhika Suri	machine learning	9999
11	Pranab Natarajan	machine learning	9999
12	Chhavi Lachman	data analysis	4999
13	Elias Dodiya	data analysis	4999
14	Yash Sethi	data analysis	4999
15	Munni Varghese	data analysis	4999
16	Parveen Bhalla	data analysis	4999
17	Chhavi Lachman	power bi	1899
18	Fardeen Mahabir	power bi	1899
19	Kailash Harjo	power bi	1899
20	Tarun Thaker	power bi	1899
21	Yash Sethi	power bi	1899
22	Pranab Natarajan	power bi	1899
23	Chhavi Lachman	plotly	699
24	Elias Dodiya	plotly	699
25	Kailash Harjo	plotly	699
26	Pranab Natarajan	plotly	699
27	Chhavi Lachman	tableau	2499
28	Preet Sha	tableau	2499
29	Elias Dodiya	tableau	2499
30	Yash Sethi	tableau	2499
31	Parveen Bhalla	tableau	2499
32	Radha Dutt	tableau	2499
33	Esha Butala	tableau	2499
34	Fardeen Mahabir	pandas	1099
35	Tarun Thaker	pandas	1099
36	David Mukhopadhyay	pandas	1099
37	Elias Dodiya	ms sxcel	1599
38	Tarun Thaker	ms sxcel	1599
39	David Mukhopadhyay	ms sxcel	1599
40	Yasmin Palan	ms sxcel	1599
41	Radha Dutt	ms sxcel	1599
42	Kailash Harjo	pyspark	2499
43	Tarun Thaker	pyspark	2499
44	David Mukhopadhyay	pyspark	2499
45	Yasmin Palan	pyspark	2499
46	Shashank D’Alia	pyspark	2499
47	Radha Dutt	pyspark	2499
48	Tarun Thaker	sql	3499
49	Qabeel Raman	sql	3499

# 4. Plot bar chart for revenue/course
regs.merge(courses,on='course_id').groupby('course_name')['price'].sum().plot(kind='bar')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f56b73cb2e0>


# 5. find students who enrolled in both the months
common_student_id = np.intersect1d(nov['student_id'],dec['student_id'])
common_student_id
     
array([ 1,  3,  7, 11, 16, 17, 18, 22, 23])

students[students['student_id'].isin(common_student_id)]
     
student_id	name	partner
0	1	Kailash Harjo	23
2	3	Parveen Bhalla	3
6	7	Tarun Thaker	9
10	11	David Mukhopadhyay	20
15	16	Elias Dodiya	25
16	17	Yasmin Palan	7
17	18	Fardeen Mahabir	13
21	22	Yash Sethi	21
22	23	Chhavi Lachman	18

# 6. find course that got no enrollment
# courses['course_id']
# regs['course_id']

course_id_list = np.setdiff1d(courses['course_id'],regs['course_id'])
courses[courses['course_id'].isin(course_id_list)]
     
course_id	course_name	price
10	11	Numpy	699
11	12	C++	1299

# 7. find students who did not enroll into any courses
student_id_list = np.setdiff1d(students['student_id'],regs['student_id'])
students[students['student_id'].isin(student_id_list)].shape[0]

(10/28)*100
     
35.714285714285715

students
     
student_id	name	partner
0	1	Kailash Harjo	23
1	2	Esha Butala	1
2	3	Parveen Bhalla	3
3	4	Marlo Dugal	14
4	5	Kusum Bahri	6
5	6	Lakshmi Contractor	10
6	7	Tarun Thaker	9
7	8	Radheshyam Dey	5
8	9	Nitika Chatterjee	4
9	10	Aayushman Sant	8
10	11	David Mukhopadhyay	20
11	12	Radha Dutt	19
12	13	Munni Varghese	24
13	14	Pranab Natarajan	22
14	15	Preet Sha	16
15	16	Elias Dodiya	25
16	17	Yasmin Palan	7
17	18	Fardeen Mahabir	13
18	19	Qabeel Raman	12
19	20	Hanuman Hegde	11
20	21	Seema Kota	15
21	22	Yash Sethi	21
22	23	Chhavi Lachman	18
23	24	Radhika Suri	17
24	25	Shashank D’Alia	2
25	26	Nitish	28
26	27	Ankit	26
27	28	Rahul	17

# 8. Print student name -> partner name for all enrolled students
# self join
students.merge(students,how='inner',left_on='partner',right_on='student_id')[['name_x','name_y']]
     
name_x	name_y
0	Kailash Harjo	Chhavi Lachman
1	Esha Butala	Kailash Harjo
2	Parveen Bhalla	Parveen Bhalla
3	Marlo Dugal	Pranab Natarajan
4	Kusum Bahri	Lakshmi Contractor
5	Lakshmi Contractor	Aayushman Sant
6	Tarun Thaker	Nitika Chatterjee
7	Radheshyam Dey	Kusum Bahri
8	Nitika Chatterjee	Marlo Dugal
9	Aayushman Sant	Radheshyam Dey
10	David Mukhopadhyay	Hanuman Hegde
11	Radha Dutt	Qabeel Raman
12	Munni Varghese	Radhika Suri
13	Pranab Natarajan	Yash Sethi
14	Preet Sha	Elias Dodiya
15	Elias Dodiya	Shashank D’Alia
16	Yasmin Palan	Tarun Thaker
17	Fardeen Mahabir	Munni Varghese
18	Qabeel Raman	Radha Dutt
19	Hanuman Hegde	David Mukhopadhyay
20	Seema Kota	Preet Sha
21	Yash Sethi	Seema Kota
22	Chhavi Lachman	Fardeen Mahabir
23	Radhika Suri	Yasmin Palan
24	Rahul	Yasmin Palan
25	Shashank D’Alia	Esha Butala
26	Nitish	Rahul
27	Ankit	Nitish

# 9. find top 3 students who did most number enrollments
regs.merge(students,on='student_id').groupby(['student_id','name'])['name'].count().sort_values(ascending=False).head(3)
     
student_id  name          
23          Chhavi Lachman    6
7           Tarun Thaker      5
1           Kailash Harjo     4
Name: name, dtype: int64

# 10. find top 3 students who spent most amount of money on courses
regs.merge(students,on='student_id').merge(courses,on='course_id').groupby(['student_id','name'])['price'].sum().sort_values(ascending=False).head(3)
     
student_id  name            
23          Chhavi Lachman      22594
14          Pranab Natarajan    15096
19          Qabeel Raman        13498
Name: price, dtype: int64

# Alternate syntax for merge
# students.merge(regs)

pd.merge(students,regs,how='inner',on='student_id')
     
student_id	name	partner	course_id
0	1	Kailash Harjo	23	1
1	1	Kailash Harjo	23	6
2	1	Kailash Harjo	23	10
3	1	Kailash Harjo	23	9
4	2	Esha Butala	1	5
5	3	Parveen Bhalla	3	3
6	3	Parveen Bhalla	3	5
7	7	Tarun Thaker	9	8
8	7	Tarun Thaker	9	10
9	7	Tarun Thaker	9	7
10	7	Tarun Thaker	9	2
11	7	Tarun Thaker	9	6
12	11	David Mukhopadhyay	20	7
13	11	David Mukhopadhyay	20	8
14	11	David Mukhopadhyay	20	10
15	12	Radha Dutt	19	10
16	12	Radha Dutt	19	1
17	12	Radha Dutt	19	5
18	12	Radha Dutt	19	7
19	13	Munni Varghese	24	3
20	14	Pranab Natarajan	22	9
21	14	Pranab Natarajan	22	6
22	14	Pranab Natarajan	22	4
23	14	Pranab Natarajan	22	1
24	15	Preet Sha	16	5
25	15	Preet Sha	16	1
26	16	Elias Dodiya	25	9
27	16	Elias Dodiya	25	5
28	16	Elias Dodiya	25	7
29	16	Elias Dodiya	25	3
30	17	Yasmin Palan	7	7
31	17	Yasmin Palan	7	10
32	18	Fardeen Mahabir	13	6
33	18	Fardeen Mahabir	13	1
34	18	Fardeen Mahabir	13	8
35	19	Qabeel Raman	12	4
36	19	Qabeel Raman	12	2
37	21	Seema Kota	15	1
38	22	Yash Sethi	21	3
39	22	Yash Sethi	21	5
40	22	Yash Sethi	21	6
41	23	Chhavi Lachman	18	1
42	23	Chhavi Lachman	18	4
43	23	Chhavi Lachman	18	3
44	23	Chhavi Lachman	18	6
45	23	Chhavi Lachman	18	9
46	23	Chhavi Lachman	18	5
47	24	Radhika Suri	17	4
48	25	Shashank D’Alia	2	1
49	25	Shashank D’Alia	2	10

# IPL Problems

# find top 3 studiums with highest sixes/match ratio
# find orange cap holder of all the seasons
     

matches
     
id	season	city	date	team1	team2	toss_winner	toss_decision	result	dl_applied	winner	win_by_runs	win_by_wickets	player_of_match	venue	umpire1	umpire2	umpire3
0	1	2017	Hyderabad	2017-04-05	Sunrisers Hyderabad	Royal Challengers Bangalore	Royal Challengers Bangalore	field	normal	0	Sunrisers Hyderabad	35	0	Yuvraj Singh	Rajiv Gandhi International Stadium, Uppal	AY Dandekar	NJ Llong	NaN
1	2	2017	Pune	2017-04-06	Mumbai Indians	Rising Pune Supergiant	Rising Pune Supergiant	field	normal	0	Rising Pune Supergiant	0	7	SPD Smith	Maharashtra Cricket Association Stadium	A Nand Kishore	S Ravi	NaN
2	3	2017	Rajkot	2017-04-07	Gujarat Lions	Kolkata Knight Riders	Kolkata Knight Riders	field	normal	0	Kolkata Knight Riders	0	10	CA Lynn	Saurashtra Cricket Association Stadium	Nitin Menon	CK Nandan	NaN
3	4	2017	Indore	2017-04-08	Rising Pune Supergiant	Kings XI Punjab	Kings XI Punjab	field	normal	0	Kings XI Punjab	0	6	GJ Maxwell	Holkar Cricket Stadium	AK Chaudhary	C Shamshuddin	NaN
4	5	2017	Bangalore	2017-04-08	Royal Challengers Bangalore	Delhi Daredevils	Royal Challengers Bangalore	bat	normal	0	Royal Challengers Bangalore	15	0	KM Jadhav	M Chinnaswamy Stadium	NaN	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
631	632	2016	Raipur	2016-05-22	Delhi Daredevils	Royal Challengers Bangalore	Royal Challengers Bangalore	field	normal	0	Royal Challengers Bangalore	0	6	V Kohli	Shaheed Veer Narayan Singh International Stadium	A Nand Kishore	BNJ Oxenford	NaN
632	633	2016	Bangalore	2016-05-24	Gujarat Lions	Royal Challengers Bangalore	Royal Challengers Bangalore	field	normal	0	Royal Challengers Bangalore	0	4	AB de Villiers	M Chinnaswamy Stadium	AK Chaudhary	HDPK Dharmasena	NaN
633	634	2016	Delhi	2016-05-25	Sunrisers Hyderabad	Kolkata Knight Riders	Kolkata Knight Riders	field	normal	0	Sunrisers Hyderabad	22	0	MC Henriques	Feroz Shah Kotla	M Erasmus	C Shamshuddin	NaN
634	635	2016	Delhi	2016-05-27	Gujarat Lions	Sunrisers Hyderabad	Sunrisers Hyderabad	field	normal	0	Sunrisers Hyderabad	0	4	DA Warner	Feroz Shah Kotla	M Erasmus	CK Nandan	NaN
635	636	2016	Bangalore	2016-05-29	Sunrisers Hyderabad	Royal Challengers Bangalore	Sunrisers Hyderabad	bat	normal	0	Sunrisers Hyderabad	8	0	BCJ Cutting	M Chinnaswamy Stadium	HDPK Dharmasena	BNJ Oxenford	NaN
636 rows × 18 columns


delivery
     
match_id	inning	batting_team	bowling_team	over	ball	batsman	non_striker	bowler	is_super_over	...	bye_runs	legbye_runs	noball_runs	penalty_runs	batsman_runs	extra_runs	total_runs	player_dismissed	dismissal_kind	fielder
0	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	1	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	0	0	NaN	NaN	NaN
1	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	2	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	0	0	NaN	NaN	NaN
2	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	3	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	4	0	4	NaN	NaN	NaN
3	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	4	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	0	0	NaN	NaN	NaN
4	1	1	Sunrisers Hyderabad	Royal Challengers Bangalore	1	5	DA Warner	S Dhawan	TS Mills	0	...	0	0	0	0	0	2	2	NaN	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
150455	636	2	Royal Challengers Bangalore	Sunrisers Hyderabad	20	2	Sachin Baby	CJ Jordan	B Kumar	0	...	0	0	0	0	2	0	2	NaN	NaN	NaN
150456	636	2	Royal Challengers Bangalore	Sunrisers Hyderabad	20	3	Sachin Baby	CJ Jordan	B Kumar	0	...	0	0	0	0	0	0	0	CJ Jordan	run out	NV Ojha
150457	636	2	Royal Challengers Bangalore	Sunrisers Hyderabad	20	4	Iqbal Abdulla	Sachin Baby	B Kumar	0	...	0	1	0	0	0	1	1	NaN	NaN	NaN
150458	636	2	Royal Challengers Bangalore	Sunrisers Hyderabad	20	5	Sachin Baby	Iqbal Abdulla	B Kumar	0	...	0	0	0	0	1	0	1	NaN	NaN	NaN
150459	636	2	Royal Challengers Bangalore	Sunrisers Hyderabad	20	6	Iqbal Abdulla	Sachin Baby	B Kumar	0	...	0	0	0	0	4	0	4	NaN	NaN	NaN
150460 rows × 21 columns


temp_df = delivery.merge(matches,left_on='match_id',right_on='id')
     

six_df = temp_df[temp_df['batsman_runs'] == 6]
     

# stadium -> sixes
num_sixes = six_df.groupby('venue')['venue'].count()
     

num_matches = matches['venue'].value_counts()
     

(num_sixes/num_matches).sort_values(ascending=False).head(10)
     
Holkar Cricket Stadium                                 17.600000
M Chinnaswamy Stadium                                  13.227273
Sharjah Cricket Stadium                                12.666667
Himachal Pradesh Cricket Association Stadium           12.000000
Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium    11.727273
Wankhede Stadium                                       11.526316
De Beers Diamond Oval                                  11.333333
Maharashtra Cricket Association Stadium                11.266667
JSCA International Stadium Complex                     10.857143
Sardar Patel Stadium, Motera                           10.833333
Name: venue, dtype: float64

matches
     
id	season	city	date	team1	team2	toss_winner	toss_decision	result	dl_applied	winner	win_by_runs	win_by_wickets	player_of_match	venue	umpire1	umpire2	umpire3
0	1	2017	Hyderabad	2017-04-05	Sunrisers Hyderabad	Royal Challengers Bangalore	Royal Challengers Bangalore	field	normal	0	Sunrisers Hyderabad	35	0	Yuvraj Singh	Rajiv Gandhi International Stadium, Uppal	AY Dandekar	NJ Llong	NaN
1	2	2017	Pune	2017-04-06	Mumbai Indians	Rising Pune Supergiant	Rising Pune Supergiant	field	normal	0	Rising Pune Supergiant	0	7	SPD Smith	Maharashtra Cricket Association Stadium	A Nand Kishore	S Ravi	NaN
2	3	2017	Rajkot	2017-04-07	Gujarat Lions	Kolkata Knight Riders	Kolkata Knight Riders	field	normal	0	Kolkata Knight Riders	0	10	CA Lynn	Saurashtra Cricket Association Stadium	Nitin Menon	CK Nandan	NaN
3	4	2017	Indore	2017-04-08	Rising Pune Supergiant	Kings XI Punjab	Kings XI Punjab	field	normal	0	Kings XI Punjab	0	6	GJ Maxwell	Holkar Cricket Stadium	AK Chaudhary	C Shamshuddin	NaN
4	5	2017	Bangalore	2017-04-08	Royal Challengers Bangalore	Delhi Daredevils	Royal Challengers Bangalore	bat	normal	0	Royal Challengers Bangalore	15	0	KM Jadhav	M Chinnaswamy Stadium	NaN	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
631	632	2016	Raipur	2016-05-22	Delhi Daredevils	Royal Challengers Bangalore	Royal Challengers Bangalore	field	normal	0	Royal Challengers Bangalore	0	6	V Kohli	Shaheed Veer Narayan Singh International Stadium	A Nand Kishore	BNJ Oxenford	NaN
632	633	2016	Bangalore	2016-05-24	Gujarat Lions	Royal Challengers Bangalore	Royal Challengers Bangalore	field	normal	0	Royal Challengers Bangalore	0	4	AB de Villiers	M Chinnaswamy Stadium	AK Chaudhary	HDPK Dharmasena	NaN
633	634	2016	Delhi	2016-05-25	Sunrisers Hyderabad	Kolkata Knight Riders	Kolkata Knight Riders	field	normal	0	Sunrisers Hyderabad	22	0	MC Henriques	Feroz Shah Kotla	M Erasmus	C Shamshuddin	NaN
634	635	2016	Delhi	2016-05-27	Gujarat Lions	Sunrisers Hyderabad	Sunrisers Hyderabad	field	normal	0	Sunrisers Hyderabad	0	4	DA Warner	Feroz Shah Kotla	M Erasmus	CK Nandan	NaN
635	636	2016	Bangalore	2016-05-29	Sunrisers Hyderabad	Royal Challengers Bangalore	Sunrisers Hyderabad	bat	normal	0	Sunrisers Hyderabad	8	0	BCJ Cutting	M Chinnaswamy Stadium	HDPK Dharmasena	BNJ Oxenford	NaN
636 rows × 18 columns


temp_df.groupby(['season','batsman'])['batsman_runs'].sum().reset_index().sort_values('batsman_runs',ascending=False).drop_duplicates(subset=['season'],keep='first').sort_values('season')
     
season	batsman	batsman_runs
115	2008	SE Marsh	616
229	2009	ML Hayden	572
446	2010	SR Tendulkar	618
502	2011	CH Gayle	608
684	2012	CH Gayle	733
910	2013	MEK Hussey	733
1088	2014	RV Uthappa	660
1148	2015	DA Warner	562
1383	2016	V Kohli	973
1422	2017	DA Warner	641

temp_df.groupby(['season','batsman'])['batsman_runs'].sum().reset_index().sort_values('batsman_runs',ascending=False)
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-214-5a08c989d1a3> in <module>
----> 1 temp_df.groupby(['season','batsman'])['batsman_runs'].sum().reset_index().sort_values('batsman_runs',ascending=False).first('season')

/usr/local/lib/python3.8/dist-packages/pandas/core/generic.py in first(self, offset)
   8193         """ """
   8194         if not isinstance(self.index, DatetimeIndex):
-> 8195             raise TypeError("'first' only supports a DatetimeIndex index")
   8196 
   8197         if len(self.index) == 0:

TypeError: 'first' only supports a DatetimeIndex index


     
#################################################################################




import numpy as np
import pandas as pd
     
Series is 1D and DataFrames are 2D objects
But why?
And what exactly is index?

# can we have multiple index? Let's try
index_val = [('cse',2019),('cse',2020),('cse',2021),('cse',2022),('ece',2019),('ece',2020),('ece',2021),('ece',2022)]
a = pd.Series([1,2,3,4,5,6,7,8],index=index_val)
a
     
(cse, 2019)    1
(cse, 2020)    2
(cse, 2021)    3
(cse, 2022)    4
(ece, 2019)    5
(ece, 2020)    6
(ece, 2021)    7
(ece, 2022)    8
dtype: int64

# The problem?
a['cse']
     
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.8/dist-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   3360             try:
-> 3361                 return self._engine.get_loc(casted_key)
   3362             except KeyError as err:

/usr/local/lib/python3.8/dist-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()

/usr/local/lib/python3.8/dist-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'cse'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
<ipython-input-126-be5a0fa56305> in <module>
      1 # The problem?
----> 2 a['cse']

/usr/local/lib/python3.8/dist-packages/pandas/core/series.py in __getitem__(self, key)
    940 
    941         elif key_is_scalar:
--> 942             return self._get_value(key)
    943 
    944         if is_hashable(key):

/usr/local/lib/python3.8/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1049 
   1050         # Similar to Index.get_value, but we do not fall back to positional
-> 1051         loc = self.index.get_loc(label)
   1052         return self.index._get_values_for_loc(self, loc, label)
   1053 

/usr/local/lib/python3.8/dist-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   3361                 return self._engine.get_loc(casted_key)
   3362             except KeyError as err:
-> 3363                 raise KeyError(key) from err
   3364 
   3365         if is_scalar(key) and isna(key) and not self.hasnans:

KeyError: 'cse'

# The solution -> multiindex series(also known as Hierarchical Indexing)
# multiple index levels within a single index
     

# how to create multiindex object
# 1. pd.MultiIndex.from_tuples()
index_val = [('cse',2019),('cse',2020),('cse',2021),('cse',2022),('ece',2019),('ece',2020),('ece',2021),('ece',2022)]
multiindex = pd.MultiIndex.from_tuples(index_val)
multiindex.levels[1]
# 2. pd.MultiIndex.from_product()
pd.MultiIndex.from_product([['cse','ece'],[2019,2020,2021,2022]])
     
MultiIndex([('cse', 2019),
            ('cse', 2020),
            ('cse', 2021),
            ('cse', 2022),
            ('ece', 2019),
            ('ece', 2020),
            ('ece', 2021),
            ('ece', 2022)],
           )

# level inside multiindex object
     

# creating a series with multiindex object
s = pd.Series([1,2,3,4,5,6,7,8],index=multiindex)
s
     
cse  2019    1
     2020    2
     2021    3
     2022    4
ece  2019    5
     2020    6
     2021    7
     2022    8
dtype: int64

# how to fetch items from such a series
s['cse']
     
2019    1
2020    2
2021    3
2022    4
dtype: int64

# a logical question to ask
     

# unstack
temp = s.unstack()
temp
     
2019	2020	2021	2022
cse	1	2	3	4
ece	5	6	7	8

# stack
temp.stack()
     
cse  2019    1
     2020    2
     2021    3
     2022    4
ece  2019    5
     2020    6
     2021    7
     2022    8
dtype: int64

# Then what was the point of multiindex series?
     

# multiindex dataframe
     

branch_df1 = pd.DataFrame(
    [
        [1,2],
        [3,4],
        [5,6],
        [7,8],
        [9,10],
        [11,12],
        [13,14],
        [15,16],
    ],
    index = multiindex,
    columns = ['avg_package','students']
)

branch_df1
     
avg_package	students
cse	2019	1	2
2020	3	4
2021	5	6
2022	7	8
ece	2019	9	10
2020	11	12
2021	13	14
2022	15	16

branch_df1['students']
     
cse  2019     2
     2020     4
     2021     6
     2022     8
ece  2019    10
     2020    12
     2021    14
     2022    16
Name: students, dtype: int64

# Are columns really different from index?
     

# multiindex df from columns perspective
branch_df2 = pd.DataFrame(
    [
        [1,2,0,0],
        [3,4,0,0],
        [5,6,0,0],
        [7,8,0,0],
    ],
    index = [2019,2020,2021,2022],
    columns = pd.MultiIndex.from_product([['delhi','mumbai'],['avg_package','students']])
)

branch_df2
     
delhi	mumbai
avg_package	students	avg_package	students
2019	1	2	0	0
2020	3	4	0	0
2021	5	6	0	0
2022	7	8	0	0

branch_df2.loc[2019]
     
delhi   avg_package    1
        students       2
mumbai  avg_package    0
        students       0
Name: 2019, dtype: int64

# Multiindex df in terms of both cols and index

branch_df3 = pd.DataFrame(
    [
        [1,2,0,0],
        [3,4,0,0],
        [5,6,0,0],
        [7,8,0,0],
        [9,10,0,0],
        [11,12,0,0],
        [13,14,0,0],
        [15,16,0,0],
    ],
    index = multiindex,
    columns = pd.MultiIndex.from_product([['delhi','mumbai'],['avg_package','students']])
)

branch_df3
     
delhi	mumbai
avg_package	students	avg_package	students
cse	2019	1	2	0	0
2020	3	4	0	0
2021	5	6	0	0
2022	7	8	0	0
ece	2019	9	10	0	0
2020	11	12	0	0
2021	13	14	0	0
2022	15	16	0	0
Stacking and Unstacking

branch_df3.stack().stack()
     
cse  2019  avg_package  delhi      1
                        mumbai     0
           students     delhi      2
                        mumbai     0
     2020  avg_package  delhi      3
                        mumbai     0
           students     delhi      4
                        mumbai     0
     2021  avg_package  delhi      5
                        mumbai     0
           students     delhi      6
                        mumbai     0
     2022  avg_package  delhi      7
                        mumbai     0
           students     delhi      8
                        mumbai     0
ece  2019  avg_package  delhi      9
                        mumbai     0
           students     delhi     10
                        mumbai     0
     2020  avg_package  delhi     11
                        mumbai     0
           students     delhi     12
                        mumbai     0
     2021  avg_package  delhi     13
                        mumbai     0
           students     delhi     14
                        mumbai     0
     2022  avg_package  delhi     15
                        mumbai     0
           students     delhi     16
                        mumbai     0
dtype: int64
Working with multiindex dataframes

# head and tail
branch_df3.head()
# shape
branch_df3.shape
# info
branch_df3.info()
# duplicated -> isnull
branch_df3.duplicated()
branch_df3.isnull()
     
<class 'pandas.core.frame.DataFrame'>
MultiIndex: 8 entries, ('cse', 2019) to ('ece', 2022)
Data columns (total 4 columns):
 #   Column                 Non-Null Count  Dtype
---  ------                 --------------  -----
 0   (delhi, avg_package)   8 non-null      int64
 1   (delhi, students)      8 non-null      int64
 2   (mumbai, avg_package)  8 non-null      int64
 3   (mumbai, students)     8 non-null      int64
dtypes: int64(4)
memory usage: 932.0+ bytes
delhi	mumbai
avg_package	students	avg_package	students
cse	2019	False	False	False	False
2020	False	False	False	False
2021	False	False	False	False
2022	False	False	False	False
ece	2019	False	False	False	False
2020	False	False	False	False
2021	False	False	False	False
2022	False	False	False	False

# Extracting rows single
branch_df3.loc[('cse',2022)]
     
delhi   avg_package    7
        students       8
mumbai  avg_package    0
        students       0
Name: (cse, 2022), dtype: int64

# multiple
branch_df3.loc[('cse',2019):('ece',2020):2]
     
delhi	mumbai
avg_package	students	avg_package	students
cse	2019	1	2	0	0
2021	5	6	0	0
ece	2019	9	10	0	0

# using iloc
branch_df3.iloc[0:5:2]
     
delhi	mumbai
avg_package	students	avg_package	students
cse	2019	1	2	0	0
2021	5	6	0	0
ece	2019	9	10	0	0

# Extracting cols
branch_df3['delhi']['students']
     
cse  2019     2
     2020     4
     2021     6
     2022     8
ece  2019    10
     2020    12
     2021    14
     2022    16
Name: students, dtype: int64

branch_df3.iloc[:,1:3]
     
delhi	mumbai
students	avg_package
cse	2019	2	0
2020	4	0
2021	6	0
2022	8	0
ece	2019	10	0
2020	12	0
2021	14	0
2022	16	0

# Extracting both
branch_df3.iloc[[0,4],[1,2]]
     
delhi	mumbai
students	avg_package
cse	2019	2	0
ece	2019	10	0

# sort index
# both -> descending -> diff order
# based on one level
branch_df3.sort_index(ascending=False)
branch_df3.sort_index(ascending=[False,True])
branch_df3.sort_index(level=0,ascending=[False])
     
delhi	mumbai
avg_package	students	avg_package	students
ece	2019	9	10	0	0
2020	11	12	0	0
2021	13	14	0	0
2022	15	16	0	0
cse	2019	1	2	0	0
2020	3	4	0	0
2021	5	6	0	0
2022	7	8	0	0

# multiindex dataframe(col) -> transpose
branch_df3.transpose()
     
cse	ece
2019	2020	2021	2022	2019	2020	2021	2022
delhi	avg_package	1	3	5	7	9	11	13	15
students	2	4	6	8	10	12	14	16
mumbai	avg_package	0	0	0	0	0	0	0	0
students	0	0	0	0	0	0	0	0

# swaplevel
branch_df3.swaplevel(axis=1)
     
avg_package	students	avg_package	students
delhi	delhi	mumbai	mumbai
cse	2019	1	2	0	0
2020	3	4	0	0
2021	5	6	0	0
2022	7	8	0	0
ece	2019	9	10	0	0
2020	11	12	0	0
2021	13	14	0	0
2022	15	16	0	0
Long Vs Wide Data
image.png

Wide format is where we have a single row for every data point with multiple columns to hold the values of various attributes.

Long format is where, for each data point we have as many rows as the number of attributes and each row contains the value of a particular attribute for a given data point.


# melt -> simple example branch
# wide to long
pd.DataFrame({'cse':[120]}).melt()
     
variable	value
0	cse	120

# melt -> branch with year
pd.DataFrame({'cse':[120],'ece':[100],'mech':[50]}).melt(var_name='branch',value_name='num_students')
     
branch	num_students
0	cse	120
1	ece	100
2	mech	50

pd.DataFrame(
    {
        'branch':['cse','ece','mech'],
        '2020':[100,150,60],
        '2021':[120,130,80],
        '2022':[150,140,70]
    }
).melt(id_vars=['branch'],var_name='year',value_name='students')
     
branch	year	students
0	cse	2020	100
1	ece	2020	150
2	mech	2020	60
3	cse	2021	120
4	ece	2021	130
5	mech	2021	80
6	cse	2022	150
7	ece	2022	140
8	mech	2022	70

# melt -> real world example
death = pd.read_csv('/content/time_series_covid19_deaths_global.csv')
confirm = pd.read_csv('/content/time_series_covid19_confirmed_global.csv')
     

death.head()
     
Province/State	Country/Region	Lat	Long	1/22/20	1/23/20	1/24/20	1/25/20	1/26/20	1/27/20	...	12/24/22	12/25/22	12/26/22	12/27/22	12/28/22	12/29/22	12/30/22	12/31/22	1/1/23	1/2/23
0	NaN	Afghanistan	33.93911	67.709953	0	0	0	0	0	0	...	7845	7846	7846	7846	7846	7847	7847	7849	7849	7849
1	NaN	Albania	41.15330	20.168300	0	0	0	0	0	0	...	3595	3595	3595	3595	3595	3595	3595	3595	3595	3595
2	NaN	Algeria	28.03390	1.659600	0	0	0	0	0	0	...	6881	6881	6881	6881	6881	6881	6881	6881	6881	6881
3	NaN	Andorra	42.50630	1.521800	0	0	0	0	0	0	...	165	165	165	165	165	165	165	165	165	165
4	NaN	Angola	-11.20270	17.873900	0	0	0	0	0	0	...	1928	1928	1928	1930	1930	1930	1930	1930	1930	1930
5 rows × 1081 columns


confirm.head()
     
Province/State	Country/Region	Lat	Long	1/22/20	1/23/20	1/24/20	1/25/20	1/26/20	1/27/20	...	12/24/22	12/25/22	12/26/22	12/27/22	12/28/22	12/29/22	12/30/22	12/31/22	1/1/23	1/2/23
0	NaN	Afghanistan	33.93911	67.709953	0	0	0	0	0	0	...	207310	207399	207438	207460	207493	207511	207550	207559	207616	207627
1	NaN	Albania	41.15330	20.168300	0	0	0	0	0	0	...	333749	333749	333751	333751	333776	333776	333806	333806	333811	333812
2	NaN	Algeria	28.03390	1.659600	0	0	0	0	0	0	...	271194	271198	271198	271202	271208	271217	271223	271228	271229	271229
3	NaN	Andorra	42.50630	1.521800	0	0	0	0	0	0	...	47686	47686	47686	47686	47751	47751	47751	47751	47751	47751
4	NaN	Angola	-11.20270	17.873900	0	0	0	0	0	0	...	104973	104973	104973	105095	105095	105095	105095	105095	105095	105095
5 rows × 1081 columns


death = death.melt(id_vars=['Province/State','Country/Region','Lat','Long'],var_name='date',value_name='num_deaths')
confirm = confirm.melt(id_vars=['Province/State','Country/Region','Lat','Long'],var_name='date',value_name='num_cases')
     

death.head()
     
Province/State	Country/Region	Lat	Long	date	num_deaths
0	NaN	Afghanistan	33.93911	67.709953	1/22/20	0
1	NaN	Albania	41.15330	20.168300	1/22/20	0
2	NaN	Algeria	28.03390	1.659600	1/22/20	0
3	NaN	Andorra	42.50630	1.521800	1/22/20	0
4	NaN	Angola	-11.20270	17.873900	1/22/20	0

confirm.merge(death,on=['Province/State','Country/Region','Lat','Long','date'])[['Country/Region','date','num_cases','num_deaths']]
     
Country/Region	date	num_cases	num_deaths
0	Afghanistan	1/22/20	0	0
1	Albania	1/22/20	0	0
2	Algeria	1/22/20	0	0
3	Andorra	1/22/20	0	0
4	Angola	1/22/20	0	0
...	...	...	...	...
311248	West Bank and Gaza	1/2/23	703228	5708
311249	Winter Olympics 2022	1/2/23	535	0
311250	Yemen	1/2/23	11945	2159
311251	Zambia	1/2/23	334661	4024
311252	Zimbabwe	1/2/23	259981	5637
311253 rows × 4 columns



     


     
Pivot Table
The pivot table takes simple column-wise data as input, and groups the entries into a two-dimensional table that provides a multidimensional summarization of the data.


import numpy as np
import pandas as pd
import seaborn as sns
     

df = sns.load_dataset('tips')
df.head()
     
total_bill	tip	sex	smoker	day	time	size
0	16.99	1.01	Female	No	Sun	Dinner	2
1	10.34	1.66	Male	No	Sun	Dinner	3
2	21.01	3.50	Male	No	Sun	Dinner	3
3	23.68	3.31	Male	No	Sun	Dinner	2
4	24.59	3.61	Female	No	Sun	Dinner	4

df.groupby('sex')[['total_bill']].mean()
     
total_bill
sex	
Male	20.744076
Female	18.056897

df.groupby(['sex','smoker'])[['total_bill']].mean().unstack()
     
total_bill
smoker	Yes	No
sex		
Male	22.284500	19.791237
Female	17.977879	18.105185

df.pivot_table(index='sex',columns='smoker',values='total_bill')
     
smoker	Yes	No
sex		
Male	22.284500	19.791237
Female	17.977879	18.105185


     


     

# aggfunc
df.pivot_table(index='sex',columns='smoker',values='total_bill',aggfunc='std')
     
smoker	Yes	No
sex		
Male	9.911845	8.726566
Female	9.189751	7.286455

# all cols together
df.pivot_table(index='sex',columns='smoker')['size']
     
smoker	Yes	No
sex		
Male	2.500000	2.711340
Female	2.242424	2.592593

# multidimensional
df.pivot_table(index=['sex','smoker'],columns=['day','time'],aggfunc={'size':'mean','tip':'max','total_bill':'sum'},margins=True)
     
size	tip	total_bill
day	Thur	Fri	Sat	Sun	All	Thur	Fri	...	All	Thur	Fri	Sat	Sun	All
time	Lunch	Dinner	Lunch	Dinner	Dinner	Dinner		Lunch	Dinner	Lunch	...		Lunch	Dinner	Lunch	Dinner	Lunch	Dinner	Lunch	Dinner	
sex	smoker																					
Male	Yes	2.300000	NaN	1.666667	2.400000	2.629630	2.600000	2.500000	5.00	NaN	2.20	...	10.0	191.71	0.00	34.16	129.46	0.0	589.62	0.0	392.12	1337.07
No	2.500000	NaN	NaN	2.000000	2.656250	2.883721	2.711340	6.70	NaN	NaN	...	9.0	369.73	0.00	0.00	34.95	0.0	637.73	0.0	877.34	1919.75
Female	Yes	2.428571	NaN	2.000000	2.000000	2.200000	2.500000	2.242424	5.00	NaN	3.48	...	6.5	134.53	0.00	39.78	48.80	0.0	304.00	0.0	66.16	593.27
No	2.500000	2.0	3.000000	2.000000	2.307692	3.071429	2.592593	5.17	3.0	3.00	...	5.2	381.58	18.78	15.98	22.75	0.0	247.05	0.0	291.54	977.68
All		2.459016	2.0	2.000000	2.166667	2.517241	2.842105	2.569672	6.70	3.0	3.48	...	10.0	1077.55	18.78	89.92	235.96	NaN	1778.40	NaN	1627.16	4827.77
5 rows × 23 columns


# margins
df.pivot_table(index='sex',columns='smoker',values='total_bill',aggfunc='sum',margins=True)
     
smoker	Yes	No	All
sex			
Male	1337.07	1919.75	3256.82
Female	593.27	977.68	1570.95
All	1930.34	2897.43	4827.77

# plotting graphs
df = pd.read_csv('/content/expense_data.csv')
     

df.head()
     
Date	Account	Category	Subcategory	Note	INR	Income/Expense	Note.1	Amount	Currency	Account.1
0	3/2/2022 10:11	CUB - online payment	Food	NaN	Brownie	50.0	Expense	NaN	50.0	INR	50.0
1	3/2/2022 10:11	CUB - online payment	Other	NaN	To lended people	300.0	Expense	NaN	300.0	INR	300.0
2	3/1/2022 19:50	CUB - online payment	Food	NaN	Dinner	78.0	Expense	NaN	78.0	INR	78.0
3	3/1/2022 18:56	CUB - online payment	Transportation	NaN	Metro	30.0	Expense	NaN	30.0	INR	30.0
4	3/1/2022 18:22	CUB - online payment	Food	NaN	Snacks	67.0	Expense	NaN	67.0	INR	67.0

df['Category'].value_counts()
     
Food                156
Other                60
Transportation       31
Apparel               7
Household             6
Allowance             6
Social Life           5
Education             1
Salary                1
Self-development      1
Beauty                1
Gift                  1
Petty cash            1
Name: Category, dtype: int64

df.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 277 entries, 0 to 276
Data columns (total 11 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Date            277 non-null    object 
 1   Account         277 non-null    object 
 2   Category        277 non-null    object 
 3   Subcategory     0 non-null      float64
 4   Note            273 non-null    object 
 5   INR             277 non-null    float64
 6   Income/Expense  277 non-null    object 
 7   Note.1          0 non-null      float64
 8   Amount          277 non-null    float64
 9   Currency        277 non-null    object 
 10  Account.1       277 non-null    float64
dtypes: float64(5), object(6)
memory usage: 23.9+ KB

df['Date'] = pd.to_datetime(df['Date'])
     

df.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 277 entries, 0 to 276
Data columns (total 11 columns):
 #   Column          Non-Null Count  Dtype         
---  ------          --------------  -----         
 0   Date            277 non-null    datetime64[ns]
 1   Account         277 non-null    object        
 2   Category        277 non-null    object        
 3   Subcategory     0 non-null      float64       
 4   Note            273 non-null    object        
 5   INR             277 non-null    float64       
 6   Income/Expense  277 non-null    object        
 7   Note.1          0 non-null      float64       
 8   Amount          277 non-null    float64       
 9   Currency        277 non-null    object        
 10  Account.1       277 non-null    float64       
dtypes: datetime64[ns](1), float64(5), object(5)
memory usage: 23.9+ KB

df['month'] = df['Date'].dt.month_name()
     

df.head()
     
Date	Account	Category	Subcategory	Note	INR	Income/Expense	Note.1	Amount	Currency	Account.1	month
0	2022-03-02 10:11:00	CUB - online payment	Food	NaN	Brownie	50.0	Expense	NaN	50.0	INR	50.0	March
1	2022-03-02 10:11:00	CUB - online payment	Other	NaN	To lended people	300.0	Expense	NaN	300.0	INR	300.0	March
2	2022-03-01 19:50:00	CUB - online payment	Food	NaN	Dinner	78.0	Expense	NaN	78.0	INR	78.0	March
3	2022-03-01 18:56:00	CUB - online payment	Transportation	NaN	Metro	30.0	Expense	NaN	30.0	INR	30.0	March
4	2022-03-01 18:22:00	CUB - online payment	Food	NaN	Snacks	67.0	Expense	NaN	67.0	INR	67.0	March

df.pivot_table(index='month',columns='Category',values='INR',aggfunc='sum',fill_value=0).plot()
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f8976423df0>


df.pivot_table(index='month',columns='Income/Expense',values='INR',aggfunc='sum',fill_value=0).plot()
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f8976118f70>


df.pivot_table(index='month',columns='Account',values='INR',aggfunc='sum',fill_value=0).plot()
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f89760dbe50>



     
###################################################################################




import numpy as np
import pandas as pd
     
Timestamp Object
Time stamps reference particular moments in time (e.g., Oct 24th, 2022 at 7:00pm)

Creating Timestamp objects

# creating a timestamp
type(pd.Timestamp('2023/1/5'))
     
pandas._libs.tslibs.timestamps.Timestamp

# variations
pd.Timestamp('2023-1-5')
pd.Timestamp('2023, 1, 5')
     
Timestamp('2023-01-05 00:00:00')

# only year
pd.Timestamp('2023')
     
Timestamp('2023-01-01 00:00:00')

# using text
pd.Timestamp('5th January 2023')
     
Timestamp('2023-01-05 00:00:00')

# providing time also
pd.Timestamp('5th January 2023 9:21AM')
# pd.Timestamp('2023/1/5/9/21')
     
---------------------------------------------------------------------------
ParserError                               Traceback (most recent call last)
/usr/local/lib/python3.8/dist-packages/pandas/_libs/tslibs/conversion.pyx in pandas._libs.tslibs.conversion._convert_str_to_tsobject()

/usr/local/lib/python3.8/dist-packages/pandas/_libs/tslibs/parsing.pyx in pandas._libs.tslibs.parsing.parse_datetime_string()

/usr/local/lib/python3.8/dist-packages/dateutil/parser/_parser.py in parse(timestr, parserinfo, **kwargs)
   1367     else:
-> 1368         return DEFAULTPARSER.parse(timestr, **kwargs)
   1369 

/usr/local/lib/python3.8/dist-packages/dateutil/parser/_parser.py in parse(self, timestr, default, ignoretz, tzinfos, **kwargs)
    642         if res is None:
--> 643             raise ParserError("Unknown string format: %s", timestr)
    644 

ParserError: Unknown string format: 2023/1/5/9/21

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
<ipython-input-16-d14ab29f95a1> in <module>
      1 # providing time also
      2 pd.Timestamp('5th January 2023 9:21AM')
----> 3 pd.Timestamp('2023/1/5/9/21')

/usr/local/lib/python3.8/dist-packages/pandas/_libs/tslibs/timestamps.pyx in pandas._libs.tslibs.timestamps.Timestamp.__new__()

/usr/local/lib/python3.8/dist-packages/pandas/_libs/tslibs/conversion.pyx in pandas._libs.tslibs.conversion.convert_to_tsobject()

/usr/local/lib/python3.8/dist-packages/pandas/_libs/tslibs/conversion.pyx in pandas._libs.tslibs.conversion._convert_str_to_tsobject()

ValueError: could not convert string to Timestamp

# AM and PM
     

# using datetime.datetime object
import datetime as dt

x = pd.Timestamp(dt.datetime(2023,1,5,9,21,56))
x
     
Timestamp('2023-01-05 09:21:56')

# fetching attributes
x.year
x.month
x.day
x.hour
x.minute
x.second
     
56

# why separate objects to handle data and time when python already has datetime functionality?
     
syntax wise datetime is very convenient
But the performance takes a hit while working with huge data. List vs Numpy Array
The weaknesses of Python's datetime format inspired the NumPy team to add a set of native time series data type to NumPy.
The datetime64 dtype encodes dates as 64-bit integers, and thus allows arrays of dates to be represented very compactly.

import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64)
date
     
array('2015-07-04', dtype='datetime64[D]')

date + np.arange(12)
     
array(['2015-07-04', '2015-07-05', '2015-07-06', '2015-07-07',
       '2015-07-08', '2015-07-09', '2015-07-10', '2015-07-11',
       '2015-07-12', '2015-07-13', '2015-07-14', '2015-07-15'],
      dtype='datetime64[D]')
Because of the uniform type in NumPy datetime64 arrays, this type of operation can be accomplished much more quickly than if we were working directly with Python's datetime objects, especially as arrays get large

Pandas Timestamp object combines the ease-of-use of python datetime with the efficient storage and vectorized interface of numpy.datetime64

From a group of these Timestamp objects, Pandas can construct a DatetimeIndex that can be used to index data in a Series or DataFrame

DatetimeIndex Object
A collection of pandas timestamp


# from strings
type(pd.DatetimeIndex(['2023/1/1','2022/1/1','2021/1/1']))
     
pandas.core.indexes.datetimes.DatetimeIndex

# using python datetime object
pd.DatetimeIndex([dt.datetime(2023,1,1),dt.datetime(2022,1,1),dt.datetime(2021,1,1)])
     
DatetimeIndex(['2023-01-01', '2022-01-01', '2021-01-01'], dtype='datetime64[ns]', freq=None)

# using pd.timestamps
dt_index = pd.DatetimeIndex([pd.Timestamp(2023,1,1),pd.Timestamp(2022,1,1),pd.Timestamp(2021,1,1)])
     

# using datatimeindex as series index

pd.Series([1,2,3],index=dt_index)
     
2023-01-01    1
2022-01-01    2
2021-01-01    3
dtype: int64
date_range function

# generate daily dates in a given range
pd.date_range(start='2023/1/5',end='2023/2/28',freq='3D')
     
DatetimeIndex(['2023-01-05', '2023-01-08', '2023-01-11', '2023-01-14',
               '2023-01-17', '2023-01-20', '2023-01-23', '2023-01-26',
               '2023-01-29', '2023-02-01', '2023-02-04', '2023-02-07',
               '2023-02-10', '2023-02-13', '2023-02-16', '2023-02-19',
               '2023-02-22', '2023-02-25', '2023-02-28'],
              dtype='datetime64[ns]', freq='3D')

# alternate days in a given range
pd.date_range(start='2023/1/5',end='2023/2/28',freq='3D')
     

# B -> business days
pd.date_range(start='2023/1/5',end='2023/2/28',freq='B')
     
DatetimeIndex(['2023-01-05', '2023-01-06', '2023-01-09', '2023-01-10',
               '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-16',
               '2023-01-17', '2023-01-18', '2023-01-19', '2023-01-20',
               '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26',
               '2023-01-27', '2023-01-30', '2023-01-31', '2023-02-01',
               '2023-02-02', '2023-02-03', '2023-02-06', '2023-02-07',
               '2023-02-08', '2023-02-09', '2023-02-10', '2023-02-13',
               '2023-02-14', '2023-02-15', '2023-02-16', '2023-02-17',
               '2023-02-20', '2023-02-21', '2023-02-22', '2023-02-23',
               '2023-02-24', '2023-02-27', '2023-02-28'],
              dtype='datetime64[ns]', freq='B')

# W -> one week per day
pd.date_range(start='2023/1/5',end='2023/2/28',freq='W-THU')
     
DatetimeIndex(['2023-01-05', '2023-01-12', '2023-01-19', '2023-01-26',
               '2023-02-02', '2023-02-09', '2023-02-16', '2023-02-23'],
              dtype='datetime64[ns]', freq='W-THU')

# H -> Hourly data(factor)
pd.date_range(start='2023/1/5',end='2023/2/28',freq='6H')
     
DatetimeIndex(['2023-01-05 00:00:00', '2023-01-05 06:00:00',
               '2023-01-05 12:00:00', '2023-01-05 18:00:00',
               '2023-01-06 00:00:00', '2023-01-06 06:00:00',
               '2023-01-06 12:00:00', '2023-01-06 18:00:00',
               '2023-01-07 00:00:00', '2023-01-07 06:00:00',
               ...
               '2023-02-25 18:00:00', '2023-02-26 00:00:00',
               '2023-02-26 06:00:00', '2023-02-26 12:00:00',
               '2023-02-26 18:00:00', '2023-02-27 00:00:00',
               '2023-02-27 06:00:00', '2023-02-27 12:00:00',
               '2023-02-27 18:00:00', '2023-02-28 00:00:00'],
              dtype='datetime64[ns]', length=217, freq='6H')

# M -> Month end
pd.date_range(start='2023/1/5',end='2023/2/28',freq='M')
     
DatetimeIndex(['2023-01-31', '2023-02-28'], dtype='datetime64[ns]', freq='M')

# MS -> Month start
pd.date_range(start='2023/1/5',end='2023/2/28',freq='MS')
     
DatetimeIndex(['2023-02-01'], dtype='datetime64[ns]', freq='MS')

# A -> Year end
pd.date_range(start='2023/1/5',end='2030/2/28',freq='A')
     
DatetimeIndex(['2023-12-31', '2024-12-31', '2025-12-31', '2026-12-31',
               '2027-12-31', '2028-12-31', '2029-12-31'],
              dtype='datetime64[ns]', freq='A-DEC')

# using periods(number of results)
pd.date_range(start='2023/1/5',periods=25,freq='M')
     
DatetimeIndex(['2023-01-31', '2023-02-28', '2023-03-31', '2023-04-30',
               '2023-05-31', '2023-06-30', '2023-07-31', '2023-08-31',
               '2023-09-30', '2023-10-31', '2023-11-30', '2023-12-31',
               '2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30',
               '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31',
               '2024-09-30', '2024-10-31', '2024-11-30', '2024-12-31',
               '2025-01-31'],
              dtype='datetime64[ns]', freq='M')
to_datetime function
converts an existing objects to pandas timestamp/datetimeindex object


# simple series example

s = pd.Series(['2023/1/1','2022/1/1','2021/1/1'])
pd.to_datetime(s).dt.day_name()
     
0      Sunday
1    Saturday
2      Friday
dtype: object

# with errors
s = pd.Series(['2023/1/1','2022/1/1','2021/130/1'])
pd.to_datetime(s,errors='coerce').dt.month_name()
     
0    January
1    January
2        NaN
dtype: object

df = pd.read_csv('/content/expense_data.csv')
df.shape
     
(277, 11)

df.head()
     
Date	Account	Category	Subcategory	Note	INR	Income/Expense	Note.1	Amount	Currency	Account.1
0	3/2/2022 10:11	CUB - online payment	Food	NaN	Brownie	50.0	Expense	NaN	50.0	INR	50.0
1	3/2/2022 10:11	CUB - online payment	Other	NaN	To lended people	300.0	Expense	NaN	300.0	INR	300.0
2	3/1/2022 19:50	CUB - online payment	Food	NaN	Dinner	78.0	Expense	NaN	78.0	INR	78.0
3	3/1/2022 18:56	CUB - online payment	Transportation	NaN	Metro	30.0	Expense	NaN	30.0	INR	30.0
4	3/1/2022 18:22	CUB - online payment	Food	NaN	Snacks	67.0	Expense	NaN	67.0	INR	67.0

df['Date'] = pd.to_datetime(df['Date'])
     

df.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 277 entries, 0 to 276
Data columns (total 11 columns):
 #   Column          Non-Null Count  Dtype         
---  ------          --------------  -----         
 0   Date            277 non-null    datetime64[ns]
 1   Account         277 non-null    object        
 2   Category        277 non-null    object        
 3   Subcategory     0 non-null      float64       
 4   Note            273 non-null    object        
 5   INR             277 non-null    float64       
 6   Income/Expense  277 non-null    object        
 7   Note.1          0 non-null      float64       
 8   Amount          277 non-null    float64       
 9   Currency        277 non-null    object        
 10  Account.1       277 non-null    float64       
dtypes: datetime64[ns](1), float64(5), object(5)
memory usage: 23.9+ KB
dt accessor
Accessor object for datetimelike properties of the Series values.


df['Date'].dt.is_quarter_start
     
0      False
1      False
2      False
3      False
4      False
       ...  
272    False
273    False
274    False
275    False
276    False
Name: Date, Length: 277, dtype: bool

# plot graph
import matplotlib.pyplot as plt
plt.plot(df['Date'],df['INR'])
     
[<matplotlib.lines.Line2D at 0x7f89b2206880>]


# day name wise bar chart/month wise bar chart

df['day_name'] = df['Date'].dt.day_name()
     

df.head()
     
Date	Account	Category	Subcategory	Note	INR	Income/Expense	Note.1	Amount	Currency	Account.1	day_name
0	2022-03-02 10:11:00	CUB - online payment	Food	NaN	Brownie	50.0	Expense	NaN	50.0	INR	50.0	Wednesday
1	2022-03-02 10:11:00	CUB - online payment	Other	NaN	To lended people	300.0	Expense	NaN	300.0	INR	300.0	Wednesday
2	2022-03-01 19:50:00	CUB - online payment	Food	NaN	Dinner	78.0	Expense	NaN	78.0	INR	78.0	Tuesday
3	2022-03-01 18:56:00	CUB - online payment	Transportation	NaN	Metro	30.0	Expense	NaN	30.0	INR	30.0	Tuesday
4	2022-03-01 18:22:00	CUB - online payment	Food	NaN	Snacks	67.0	Expense	NaN	67.0	INR	67.0	Tuesday

df.groupby('day_name')['INR'].mean().plot(kind='bar')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f89b18629a0>


df['month_name'] = df['Date'].dt.month_name()
     

df.head()
     
Date	Account	Category	Subcategory	Note	INR	Income/Expense	Note.1	Amount	Currency	Account.1	day_name	month_name
0	2022-03-02 10:11:00	CUB - online payment	Food	NaN	Brownie	50.0	Expense	NaN	50.0	INR	50.0	Wednesday	March
1	2022-03-02 10:11:00	CUB - online payment	Other	NaN	To lended people	300.0	Expense	NaN	300.0	INR	300.0	Wednesday	March
2	2022-03-01 19:50:00	CUB - online payment	Food	NaN	Dinner	78.0	Expense	NaN	78.0	INR	78.0	Tuesday	March
3	2022-03-01 18:56:00	CUB - online payment	Transportation	NaN	Metro	30.0	Expense	NaN	30.0	INR	30.0	Tuesday	March
4	2022-03-01 18:22:00	CUB - online payment	Food	NaN	Snacks	67.0	Expense	NaN	67.0	INR	67.0	Tuesday	March

df.groupby('month_name')['INR'].sum().plot(kind='bar')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f89b1905c40>


df[df['Date'].dt.is_month_end]
     
Date	Account	Category	Subcategory	Note	INR	Income/Expense	Note.1	Amount	Currency	Account.1	day_name	month_name
7	2022-02-28 11:56:00	CUB - online payment	Food	NaN	Pizza	339.15	Expense	NaN	339.15	INR	339.15	Monday	February
8	2022-02-28 11:45:00	CUB - online payment	Other	NaN	From kumara	200.00	Income	NaN	200.00	INR	200.00	Monday	February
61	2022-01-31 08:44:00	CUB - online payment	Transportation	NaN	Vnr to apk	50.00	Expense	NaN	50.00	INR	50.00	Monday	January
62	2022-01-31 08:27:00	CUB - online payment	Other	NaN	To vicky	200.00	Expense	NaN	200.00	INR	200.00	Monday	January
63	2022-01-31 08:26:00	CUB - online payment	Transportation	NaN	To ksr station	153.00	Expense	NaN	153.00	INR	153.00	Monday	January
242	2021-11-30 14:24:00	CUB - online payment	Gift	NaN	Bharath birthday	115.00	Expense	NaN	115.00	INR	115.00	Tuesday	November
243	2021-11-30 14:17:00	CUB - online payment	Food	NaN	Lunch with company	128.00	Expense	NaN	128.00	INR	128.00	Tuesday	November
244	2021-11-30 10:11:00	CUB - online payment	Food	NaN	Breakfast	70.00	Expense	NaN	70.00	INR	70.00	Tuesday	November


   
####################################################################################



import pandas as pd
import numpy as np
     

# What are vectorized operations
a = np.array([1,2,3,4])
a * 4
     
array([ 4,  8, 12, 16])

# problem in vectorized opertions in vanilla python
s = ['cat','mat',None,'rat']

[i.startswith('c') for i in s]
     
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-19-3fe713c7ebb8> in <module>
      2 s = ['cat','mat',None,'rat']
      3 
----> 4 [i.startswith('c') for i in s]

<ipython-input-19-3fe713c7ebb8> in <listcomp>(.0)
      2 s = ['cat','mat',None,'rat']
      3 
----> 4 [i.startswith('c') for i in s]

AttributeError: 'NoneType' object has no attribute 'startswith'

# How pandas solves this issue?

s = pd.Series(['cat','mat',None,'rat'])
# string accessor
s.str.startswith('c')

# fast and optimized
     
0     True
1    False
2     None
3    False
dtype: object

# import titanic
df = pd.read_csv('/content/titanic.csv')
df['Name']
     
0                                Braund, Mr. Owen Harris
1      Cumings, Mrs. John Bradley (Florence Briggs Th...
2                                 Heikkinen, Miss. Laina
3           Futrelle, Mrs. Jacques Heath (Lily May Peel)
4                               Allen, Mr. William Henry
                             ...                        
886                                Montvila, Rev. Juozas
887                         Graham, Miss. Margaret Edith
888             Johnston, Miss. Catherine Helen "Carrie"
889                                Behr, Mr. Karl Howell
890                                  Dooley, Mr. Patrick
Name: Name, Length: 891, dtype: object

# Common Functions
# lower/upper/capitalize/title
df['Name'].str.upper()
df['Name'].str.capitalize()
df['Name'].str.title()
# len
df['Name'][df['Name'].str.len() == 82].values[0]
# strip
"                   nitish                              ".strip()
df['Name'].str.strip()
     
0                                Braund, Mr. Owen Harris
1      Cumings, Mrs. John Bradley (Florence Briggs Th...
2                                 Heikkinen, Miss. Laina
3           Futrelle, Mrs. Jacques Heath (Lily May Peel)
4                               Allen, Mr. William Henry
                             ...                        
886                                Montvila, Rev. Juozas
887                         Graham, Miss. Margaret Edith
888             Johnston, Miss. Catherine Helen "Carrie"
889                                Behr, Mr. Karl Howell
890                                  Dooley, Mr. Patrick
Name: Name, Length: 891, dtype: object

# split -> get
df['lastname'] = df['Name'].str.split(',').str.get(0)
df.head()
     
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	lastname
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S	Braund
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C	Cumings
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S	Heikkinen
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S	Futrelle
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S	Allen

df[['title','firstname']] = df['Name'].str.split(',').str.get(1).str.strip().str.split(' ', n=1, expand=True)
df.head()

df['title'].value_counts()
     
Mr.          517
Miss.        182
Mrs.         125
Master.       40
Dr.            7
Rev.           6
Mlle.          2
Major.         2
Col.           2
the            1
Capt.          1
Ms.            1
Sir.           1
Lady.          1
Mme.           1
Don.           1
Jonkheer.      1
Name: title, dtype: int64

# replace
df['title'] = df['title'].str.replace('Ms.','Miss.')
df['title'] = df['title'].str.replace('Mlle.','Miss.')
     
<ipython-input-60-9403e5934d1f>:2: FutureWarning: The default value of regex will change from True to False in a future version.
  df['title'] = df['title'].str.replace('Ms.','Miss.')
<ipython-input-60-9403e5934d1f>:3: FutureWarning: The default value of regex will change from True to False in a future version.
  df['title'] = df['title'].str.replace('Mlle.','Miss.')

df['title'].value_counts()
     
Mr.          517
Miss.        185
Mrs.         125
Master.       40
Dr.            7
Rev.           6
Major.         2
Col.           2
Don.           1
Mme.           1
Lady.          1
Sir.           1
Capt.          1
the            1
Jonkheer.      1
Name: title, dtype: int64

# filtering
# startswith/endswith
df[df['firstname'].str.endswith('A')]
# isdigit/isalpha...
df[df['firstname'].str.isdigit()]
     
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	lastname	title	firstname

# applying regex
# contains
# search john -> both case
df[df['firstname'].str.contains('john',case=False)]
# find lastnames with start and end char vowel
df[df['lastname'].str.contains('^[^aeiouAEIOU].+[^aeiouAEIOU]$')]
     
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	lastname	title	firstname
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S	Braund	Mr.	Owen Harris
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C	Cumings	Mrs.	John Bradley (Florence Briggs Thayer)
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S	Heikkinen	Miss.	Laina
5	6	0	3	Moran, Mr. James	male	NaN	0	0	330877	8.4583	NaN	Q	Moran	Mr.	James
6	7	0	1	McCarthy, Mr. Timothy J	male	54.0	0	0	17463	51.8625	E46	S	McCarthy	Mr.	Timothy J
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
884	885	0	3	Sutehall, Mr. Henry Jr	male	25.0	0	0	SOTON/OQ 392076	7.0500	NaN	S	Sutehall	Mr.	Henry Jr
887	888	1	1	Graham, Miss. Margaret Edith	female	19.0	0	0	112053	30.0000	B42	S	Graham	Miss.	Margaret Edith
888	889	0	3	Johnston, Miss. Catherine Helen "Carrie"	female	NaN	1	2	W./C. 6607	23.4500	NaN	S	Johnston	Miss.	Catherine Helen "Carrie"
889	890	1	1	Behr, Mr. Karl Howell	male	26.0	0	0	111369	30.0000	C148	C	Behr	Mr.	Karl Howell
890	891	0	3	Dooley, Mr. Patrick	male	32.0	0	0	370376	7.7500	NaN	Q	Dooley	Mr.	Patrick
671 rows × 15 columns


# slicing
df['Name'].str[::-1]
     
0                                sirraH newO .rM ,dnuarB
1      )reyahT sggirB ecnerolF( yeldarB nhoJ .srM ,sg...
2                                 aniaL .ssiM ,nenikkieH
3           )leeP yaM yliL( htaeH seuqcaJ .srM ,ellertuF
4                               yrneH mailliW .rM ,nellA
                             ...                        
886                                sazouJ .veR ,alivtnoM
887                         htidE teragraM .ssiM ,maharG
888             "eirraC" neleH enirehtaC .ssiM ,notsnhoJ
889                                llewoH lraK .rM ,rheB
890                                  kcirtaP .rM ,yelooD
Name: Name, Length: 891, dtype: object


       