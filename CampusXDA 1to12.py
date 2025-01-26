# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:30:53 2025

@author: Radha Sharma
"""


1. Python Output

# Python is a case sensitive language
print('Hello World')
     
Hello World

print('salman khan')
     
salman khan

print(salman khan)
     
  File "<ipython-input-3-0713073d8d88>", line 1
    print(salman khan)
                    ^
SyntaxError: invalid syntax

print(7)
     
7

print(7.7)
     
7.7

print(True)
     
True

print('Hello',1,4.5,True)
     
Hello 1 4.5 True

print('Hello',1,4.5,True,sep='/')
     
Hello/1/4.5/True

print('hello')
print('world')
     
hello
world

print('hello',end='-')
print('world')
     
hello-world
2. Data Types

# Integer
print(8)
# 1*10^308
print(1e309)
     
8
inf

# Decimal/Float
print(8.55)
print(1.7e309)
     
8.55
inf

# Boolean
print(True)
print(False)
     
True
False

# Text/String
print('Hello World')
     
Hello World

# complex
print(5+6j)
     
(5+6j)

# List-> C-> Array
print([1,2,3,4,5])
     
[1, 2, 3, 4, 5]

# Tuple
print((1,2,3,4,5))
     
(1, 2, 3, 4, 5)

# Sets
print({1,2,3,4,5})
     
{1, 2, 3, 4, 5}

# Dictionary
print({'name':'Nitish','gender':'Male','weight':70})
     
{'name': 'Nitish', 'gender': 'Male', 'weight': 70}

# type
type([1,2,3])
     
list
3. Variables

# Static Vs Dynamic Typing
# Static Vs Dynamic Binding
# stylish declaration techniques
     

# C/C++
name = 'nitish'
print(name)

a = 5
b = 6

print(a + b)
     
nitish
11

# Dynamic Typing
a = 5
# Static Typing
int a = 5
     

# Dynamic Binding
a = 5
print(a)
a = 'nitish'
print(a)

# Static Binding
int a = 5

     
5
nitish

a = 1
b = 2
c = 3
print(a,b,c)
     
1 2 3

a,b,c = 1,2,3
print(a,b,c)
     
1 2 3

a=b=c= 5
print(a,b,c)
     
5 5 5
Comments

# this is a comment
# second line
a = 4
b = 6 # like this
# second comment
print(a+b)
     
10
4. Keywords & Identifiers

# Keywords
     

# Identifiers
# You can't start with a digit
name1 = 'Nitish'
print(name1)
# You can use special chars -> _
_ = 'ntiish'
print(_)
# identiers can not be keyword
     
Nitish
ntiish
Temp Heading
5. User Input

# Static Vs Dynamic
input('Enter Email')
     
Enter Emailnitish@gmail.com
'nitish@gmail.com'

# take input from users and store them in a variable
fnum = int(input('enter first number'))
snum = int(input('enter second number'))
#print(type(fnum),type(snum))
# add the 2 variables
result = fnum + snum
# print the result
print(result)
print(type(fnum))
     
enter first number56
enter second number67
123
<class 'int'>
6. Type Conversion

# Implicit Vs Explicit
print(5+5.6)
print(type(5),type(5.6))

print(4 + '4')
     
10.6
<class 'int'> <class 'float'>
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-57-72e5c45cdb6f> in <module>
      3 print(type(5),type(5.6))
      4 
----> 5 print(4 + '4')

TypeError: unsupported operand type(s) for +: 'int' and 'str'

# Explicit
# str -> int
#int(4+5j)

# int to str
str(5)

# float
float(4)
     
4.0
7. Literals

a = 0b1010 #Binary Literals
b = 100 #Decimal Literal
c = 0o310 #Octal Literal
d = 0x12c #Hexadecimal Literal

#Float Literal
float_1 = 10.5
float_2 = 1.5e2 # 1.5 * 10^2
float_3 = 1.5e-3 # 1.5 * 10^-3

#Complex Literal
x = 3.14j

print(a, b, c, d)
print(float_1, float_2,float_3)
print(x, x.imag, x.real)
     

# binary
x = 3.14j
print(x.imag)
     
3.14

string = 'This is Python'
strings = "This is Python"
char = "C"
multiline_str = """This is a multiline string with more than one line code."""
unicode = u"\U0001f600\U0001F606\U0001F923"
raw_str = r"raw \n string"

print(string)
print(strings)
print(char)
print(multiline_str)
print(unicode)
print(raw_str)
     
This is Python
This is Python
C
This is a multiline string with more than one line code.
😀😆🤣
raw \n string

a = True + 4
b = False + 10

print("a:", a)
print("b:", b)
     
a: 5
b: 10

k = None
a = 5
b = 6
print('Program exe')
     
Program exe
8. Operators

# Arithmetic
# Relational
# Logical
# Bitwise
# Assignment
# Membership
     
9. If-Else


################################################################################  



What is an Iteration
Iteration is a general term for taking each item of something, one after another. Any time you use a loop, explicit or implicit, to go over a group of items, that is iteration.

# Example
num = [1,2,3]

for i in num:
    print(i)
1
2
3
What is Iterator
An Iterator is an object that allows the programmer to traverse through a sequence of data without having to store the entire data in the memory

# Example
L = [x for x in range(1,10000)]

#for i in L:
    #print(i*2)
    
import sys

print(sys.getsizeof(L)/64)

x = range(1,10000000000)

#for i in x:
    #print(i*2)
    
print(sys.getsizeof(x)/64)
1369.0
0.75
What is Iterable
Iterable is an object, which one can iterate over

It generates an Iterator when passed to iter() method.

# Example

L = [1,2,3]
type(L)


# L is an iterable
type(iter(L))

# iter(L) --> iterator
list_iterator
Point to remember
Every Iterator is also and Iterable
Not all Iterables are Iterators
Trick
Every Iterable has an iter function
Every Iterator has both iter function as well as a next function
a = 2
a

#for i in a:
    #print(i)
    
dir(a)
['__abs__',
 '__add__',
 '__and__',
 '__bool__',
 '__ceil__',
 '__class__',
 '__delattr__',
 '__dir__',
 '__divmod__',
 '__doc__',
 '__eq__',
 '__float__',
 '__floor__',
 '__floordiv__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__getnewargs__',
 '__gt__',
 '__hash__',
 '__index__',
 '__init__',
 '__init_subclass__',
 '__int__',
 '__invert__',
 '__le__',
 '__lshift__',
 '__lt__',
 '__mod__',
 '__mul__',
 '__ne__',
 '__neg__',
 '__new__',
 '__or__',
 '__pos__',
 '__pow__',
 '__radd__',
 '__rand__',
 '__rdivmod__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__rfloordiv__',
 '__rlshift__',
 '__rmod__',
 '__rmul__',
 '__ror__',
 '__round__',
 '__rpow__',
 '__rrshift__',
 '__rshift__',
 '__rsub__',
 '__rtruediv__',
 '__rxor__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__sub__',
 '__subclasshook__',
 '__truediv__',
 '__trunc__',
 '__xor__',
 'as_integer_ratio',
 'bit_length',
 'conjugate',
 'denominator',
 'from_bytes',
 'imag',
 'numerator',
 'real',
 'to_bytes']
T = {1:2,3:4}
dir(T)
['__class__',
 '__contains__',
 '__delattr__',
 '__delitem__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__getitem__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__iter__',
 '__le__',
 '__len__',
 '__lt__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__reversed__',
 '__setattr__',
 '__setitem__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 'clear',
 'copy',
 'fromkeys',
 'get',
 'items',
 'keys',
 'pop',
 'popitem',
 'setdefault',
 'update',
 'values']
L = [1,2,3]

# L is not an iterator
iter_L = iter(L)

# iter_L is an iterator
['__class__',
 '__delattr__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__iter__',
 '__le__',
 '__length_hint__',
 '__lt__',
 '__ne__',
 '__new__',
 '__next__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__setstate__',
 '__sizeof__',
 '__str__',
 '__subclasshook__']
Understanding how for loop works
num = [1,2,3]

for i in num:
    print(i)
1
2
3
num = [1,2,3]

# fetch the iterator
iter_num = iter(num)

# step2 --> next
next(iter_num)
next(iter_num)
next(iter_num)
next(iter_num)
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-47-0793ebf651f8> in <module>
      8 next(iter_num)
      9 next(iter_num)
---> 10 next(iter_num)

StopIteration: 
Making our own for loop
def mera_khudka_for_loop(iterable):
    
    iterator = iter(iterable)
    
    while True:
        
        try:
            print(next(iterator))
        except StopIteration:
            break           
a = [1,2,3]
b = range(1,11)
c = (1,2,3)
d = {1,2,3}
e = {0:1,1:1}

mera_khudka_for_loop(e)
0
1
A confusing point
num = [1,2,3]
iter_obj = iter(num)

print(id(iter_obj),'Address of iterator 1')

iter_obj2 = iter(iter_obj)
print(id(iter_obj2),'Address of iterator 2')
2280889893936 Address of iterator 1
2280889893936 Address of iterator 2
Let's create our own range() function
class mera_range:
    
    def __init__(self,start,end):
        self.start = start
        self.end = end
        
    def __iter__(self):
        return mera_range_iterator(self)
class mera_range_iterator:
    
    def __init__(self,iterable_obj):
        self.iterable = iterable_obj
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.iterable.start >= self.iterable.end:
            raise StopIteration
            
        current = self.iterable.start
        self.iterable.start+=1
        return current
x = mera_range(1,11)
type(x)
__main__.mera_range
iter(x)
<__main__.mera_range_iterator at 0x2130fd362b0>
 


##################################################################################   




What is a Generator
Python generators are a simple way of creating iterators.

# iterable
class mera_range:
    
    def __init__(self,start,end):
        self.start = start
        self.end = end
        
    def __iter__(self):
        return mera_iterator(self)
    

# iterator
class mera_iterator:
    
    def __init__(self,iterable_obj):
        self.iterable = iterable_obj
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.iterable.start >= self.iterable.end:
            raise StopIteration
        
        current = self.iterable.start
        self.iterable.start+=1
        return current
The Why
L = [x for x in range(100000)]

#for i in L:
    #print(i**2)
    
import sys
sys.getsizeof(L)

x = range(10000000)

#for i in x:
    #print(i**2)
sys.getsizeof(x)
48
A Simple Example
def gen_demo():
    
    yield "first statement"
    yield "second statement"
    yield "third statement"
gen = gen_demo()

for i in gen:
    print(i)
first statement
second statement
third statement
Python Tutor Demo (yield vs return)
 
Example 2
def square(num):
    for i in range(1,num+1):
        yield i**2
gen = square(10)

print(next(gen))
print(next(gen))
print(next(gen))

for i in gen:
    print(i)
1
4
9
16
25
36
49
64
81
100
Range Function using Generator
def mera_range(start,end):
    
    for i in range(start,end):
        yield i
for i in mera_range(15,26):
    print(i)
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
Generator Expression
# list comprehension
L = [i**2 for i in range(1,101)]
gen = (i**2 for i in range(1,101))

for i in gen:
    print(i)
1
4
9
16
25
36
49
64
81
100
121
144
169
196
225
256
289
324
361
400
441
484
529
576
625
676
729
784
841
900
961
1024
1089
1156
1225
1296
1369
1444
1521
1600
1681
1764
1849
1936
2025
2116
2209
2304
2401
2500
2601
2704
2809
2916
3025
3136
3249
3364
3481
3600
3721
3844
3969
4096
4225
4356
4489
4624
4761
4900
5041
5184
5329
5476
5625
5776
5929
6084
6241
6400
6561
6724
6889
7056
7225
7396
7569
7744
7921
8100
8281
8464
8649
8836
9025
9216
9409
9604
9801
10000
Practical Example
import os
import cv2

def image_data_reader(folder_path):

    for file in os.listdir(folder_path):
        f_array = cv2.imread(os.path.join(folder_path,file))
        yield f_array
    
gen = image_data_reader('C:/Users/91842/emotion-detector/train/Sad')

next(gen)
next(gen)

next(gen)
next(gen)
array([[[ 38,  38,  38],
        [ 26,  26,  26],
        [ 23,  23,  23],
        ...,
        [198, 198, 198],
        [196, 196, 196],
        [167, 167, 167]],

       [[ 32,  32,  32],
        [ 25,  25,  25],
        [ 26,  26,  26],
        ...,
        [194, 194, 194],
        [204, 204, 204],
        [181, 181, 181]],

       [[ 44,  44,  44],
        [ 42,  42,  42],
        [ 38,  38,  38],
        ...,
        [156, 156, 156],
        [214, 214, 214],
        [199, 199, 199]],

       ...,

       [[150, 150, 150],
        [165, 165, 165],
        [186, 186, 186],
        ...,
        [229, 229, 229],
        [226, 226, 226],
        [239, 239, 239]],

       [[145, 145, 145],
        [156, 156, 156],
        [180, 180, 180],
        ...,
        [227, 227, 227],
        [228, 228, 228],
        [221, 221, 221]],

       [[144, 144, 144],
        [150, 150, 150],
        [172, 172, 172],
        ...,
        [211, 211, 211],
        [189, 189, 189],
        [217, 217, 217]]], dtype=uint8)
Benefits of using a Generator
1. Ease of Implementation
class mera_range:
    
    def __init__(self,start,end):
        self.start = start
        self.end = end
        
    def __iter__(self):
        return mera_iterator(self)
# iterator
class mera_iterator:
    
    def __init__(self,iterable_obj):
        self.iterable = iterable_obj
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.iterable.start >= self.iterable.end:
            raise StopIteration
        
        current = self.iterable.start
        self.iterable.start+=1
        return current
def mera_range(start,end):
    
    for i in range(start,end):
        yield i
2. Memory Efficient
L = [x for x in range(100000)]
gen = (x for x in range(100000))

import sys

print('Size of L in memory',sys.getsizeof(L))
print('Size of gen in memory',sys.getsizeof(gen))
Size of L in memory 824456
Size of gen in memory 112
3. Representing Infinite Streams
def all_even():
    n = 0
    while True:
        yield n
        n += 2
even_num_gen = all_even()
next(even_num_gen)
next(even_num_gen)
2
4. Chaining Generators
def fibonacci_numbers(nums):
    x, y = 0, 1
    for _ in range(nums):
        x, y = y, x+y
        yield x

def square(nums):
    for num in nums:
        yield num**2

print(sum(square(fibonacci_numbers(10))))
4895
 





###################################################################################





Operators in Python
Arithmetic Operators
Relational Operators
Logical Operators
Bitwise Operators
Assignment Operators
Membership Operators

# Arithmetric Operators
print(5+6)

print(5-6)

print(5*6)

print(5/2)

print(5//2)

print(5%2)

print(5**2)
     
11
-1
30
2.5
2
1
25

# Relational Operators
print(4>5)

print(4<5)

print(4>=4)

print(4<=4)

print(4==4)

print(4!=4)
     
False
True
True
True
True
False

# Logical Operators
print(1 and 0)

print(1 or 0)

print(not 1)
     
0
1
False

# Bitwise Operators

# bitwise and
print(2 & 3)

# bitwise or
print(2 | 3)

# bitwise xor
print(2 ^ 3)

print(~3)

print(4 >> 2)

print(5 << 2)
     
2
3
1
-4
1
20

# Assignment Operators

# =
# a = 2

a = 2

# a = a % 2
a %= 2

# a++ ++a

print(a)
     
4

# Membership Operators

# in/not in

print('D' not in 'Delhi')

print(1 in [2,3,4,5,6])
     
False
False

# Program - Find the sum of a 3 digit number entered by the user

number = int(input('Enter a 3 digit number'))

# 345%10 -> 5
a = number%10

number = number//10

# 34%10 -> 4
b = number % 10

number = number//10
# 3 % 10 -> 3
c = number % 10

print(a + b + c)
     
Enter a 3 digit number666
18
If-else in Python

# login program and indentation
# email -> nitish.campusx@gmail.com
# password -> 1234

email = input('enter email')
password = input('enter password')

if email == 'nitish.campusx@gmail.com' and password == '1234':
  print('Welcome')
elif email == 'nitish.campusx@gmail.com' and password != '1234':
  # tell the user
  print('Incorrect password')
  password = input('enter password again')
  if password == '1234':
    print('Welcome,finally!')
  else:
    print('beta tumse na ho paayega!')
else:
  print('Not correct')
     
enter emailsrhreh
enter passworderhetjh
Not correct

# if-else examples
# 1. Find the min of 3 given numbers
# 2. Menu Driven Program
     
first num4
second num1
third num10
smallest is 1

# menu driven calculator
menu = input("""
Hi! how can I help you.
1. Enter 1 for pin change
2. Enter 2 for balance check
3. Enter 3 for withdrawl
4. Enter 4 for exit
""")

if menu == '1':
  print('pin change')
elif menu == '2':
  print('balance')
else:
  print('exit')
     
Hi! how can I help you.
1. Enter 1 for pin change
2. Enter 2 for balance check
3. Enter 3 for withdrawl
4. Enter 4 for exit
2
balance
Modules in Python
math
keywords
random
datetime

# math
import math

math.sqrt(196)
     
14.0

# keyword
import keyword
print(keyword.kwlist)
     
['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

# random
import random
print(random.randint(1,100))
     
88

# datetime
import datetime
print(datetime.datetime.now())
     
2022-11-08 15:50:21.228643

help('modules')
     
Please wait a moment while I gather a list of all available modules...

/usr/local/lib/python3.7/dist-packages/caffe2/proto/__init__.py:17: UserWarning: Caffe2 support is not enabled in this PyTorch build. Please enable Caffe2 by building PyTorch from source with `BUILD_CAFFE2=1` flag.
/usr/local/lib/python3.7/dist-packages/caffe2/proto/__init__.py:17: UserWarning: Caffe2 support is not enabled in this PyTorch build. Please enable Caffe2 by building PyTorch from source with `BUILD_CAFFE2=1` flag.
/usr/local/lib/python3.7/dist-packages/caffe2/python/__init__.py:9: UserWarning: Caffe2 support is not enabled in this PyTorch build. Please enable Caffe2 by building PyTorch from source with `BUILD_CAFFE2=1` flag.
Cython              collections         kaggle              requests_oauthlib
IPython             colorcet            kanren              resampy
OpenGL              colorlover          kapre               resource
PIL                 colorsys            keras               rlcompleter
ScreenResolution    community           keras_preprocessing rmagic
__future__          compileall          keyword             rpy2
_abc                concurrent          kiwisolver          rsa
_ast                confection          korean_lunar_calendar runpy
_asyncio            configparser        langcodes           samples
_bisect             cons                lib2to3             sched
_blake2             contextlib          libfuturize         scipy
_bootlocale         contextlib2         libpasteurize       scs
_bz2                contextvars         librosa             seaborn
_cffi_backend       convertdate         lightgbm            secrets
_codecs             copy                linecache           select
_codecs_cn          copyreg             llvmlite            selectors
_codecs_hk          crashtest           lmdb                send2trash
_codecs_iso2022     crcmod              locale              setuptools
_codecs_jp          crypt               locket              setuptools_git
_codecs_kr          csimdjson           logging             shapely
_codecs_tw          csv                 lsb_release         shelve
_collections        ctypes              lunarcalendar       shlex
_collections_abc    cufflinks           lxml                shutil
_compat_pickle      curses              lzma                signal
_compression        cv2                 macpath             simdjson
_contextvars        cvxopt              mailbox             site
_crypt              cvxpy               mailcap             sitecustomize
_csv                cycler              markdown            six
_ctypes             cymem               markupsafe          skimage
_ctypes_test        cython              marshal             sklearn
_curses             cythonmagic         marshmallow         sklearn_pandas
_curses_panel       daft                math                slugify
_cvxcore            dask                matplotlib          smart_open
_datetime           dataclasses         matplotlib_venn     smtpd
_dbm                datascience         mimetypes           smtplib
_decimal            datetime            missingno           sndhdr
_distutils_hack     dateutil            mistune             snowballstemmer
_dlib_pybind11      dbm                 mizani              socket
_dummy_thread       dbus                mlxtend             socketserver
_ecos               debugpy             mmap                socks
_elementtree        decimal             modulefinder        sockshandler
_functools          decorator           more_itertools      softwareproperties
_hashlib            defusedxml          moviepy             sortedcontainers
_heapq              descartes           mpmath              soundfile
_imp                difflib             msgpack             spacy
_io                 dill                multidict           spacy_legacy
_json               dis                 multipledispatch    spacy_loggers
_locale             distributed         multiprocessing     sphinx
_lsprof             distutils           multitasking        spwd
_lzma               dlib                murmurhash          sql
_markupbase         dns                 music21             sqlalchemy
_md5                docs                natsort             sqlite3
_multibytecodec     doctest             nbconvert           sqlparse
_multiprocessing    docutils            nbformat            sre_compile
_opcode             dopamine            netCDF4             sre_constants
_operator           dot_parser          netrc               sre_parse
_osx_support        dummy_threading     networkx            srsly
_pickle             easydict            nibabel             ssl
_plotly_future_     ecos                nis                 stan
_plotly_utils       editdistance        nisext              stat
_posixsubprocess    ee                  nltk                statistics
_py_abc             email               nntplib             statsmodels
_pydecimal          en_core_web_sm      notebook            storemagic
_pyio               encodings           ntpath              string
_pyrsistent_version entrypoints         nturl2path          stringprep
_pytest             enum                numba               struct
_queue              ephem               numbergen           subprocess
_random             erfa                numbers             sunau
_remote_module_non_scriptable errno               numexpr             symbol
_rinterface_cffi_abi et_xmlfile          numpy               sympy
_rinterface_cffi_api etils               oauth2client        sympyprinting
_scs_direct         etuples             oauthlib            symtable
_scs_indirect       fa2                 ogr                 sys
_sha1               fastai              okgrade             sysconfig
_sha256             fastcore            opcode              syslog
_sha3               fastdownload        openpyxl            tables
_sha512             fastdtw             operator            tabnanny
_signal             fastjsonschema      opt_einsum          tabulate
_sitebuiltins       fastprogress        optparse            tarfile
_socket             fastrlock           os                  tblib
_soundfile          faulthandler        osgeo               telnetlib
_sqlite3            fcntl               osqp                tempfile
_sre                feather             osqppurepy          tenacity
_ssl                filecmp             osr                 tensorboard
_stat               fileinput           ossaudiodev         tensorboard_data_server
_string             filelock            packaging           tensorboard_plugin_wit
_strptime           firebase_admin      palettable          tensorflow
_struct             fix_yahoo_finance   pandas              tensorflow_datasets
_symtable           flask               pandas_datareader   tensorflow_estimator
_sysconfigdata_m_linux_x86_64-linux-gnu flatbuffers         pandas_gbq          tensorflow_gcs_config
_sysconfigdata_m_x86_64-linux-gnu fnmatch             pandas_profiling    tensorflow_hub
_testbuffer         folium              pandocfilters       tensorflow_io_gcs_filesystem
_testcapi           formatter           panel               tensorflow_metadata
_testimportmultiple fractions           param               tensorflow_probability
_testmultiphase     frozenlist          parser              termcolor
_thread             fsspec              parso               terminado
_threading_local    ftplib              partd               termios
_tkinter            functools           past                test
_tracemalloc        future              pasta               testpath
_warnings           gast                pastel              tests
_weakref            gc                  pathlib             text_unidecode
_weakrefset         gdal                pathy               textblob
_xxtestfuzz         gdalconst           patsy               textwrap
_yaml               gdalnumeric         pdb                 thinc
abc                 gdown               pep517              this
absl                genericpath         pexpect             threading
aeppl               gensim              pickle              threadpoolctl
aesara              geographiclib       pickleshare         tifffile
aifc                geopy               pickletools         time
aiohttp             getopt              pip                 timeit
aiosignal           getpass             pipes               tkinter
alabaster           gettext             piptools            tlz
albumentations      gi                  pkg_resources       token
altair              gin                 pkgutil             tokenize
antigravity         glob                platform            toml
apiclient           glob2               plistlib            tomli
appdirs             gnm                 plotly              toolz
apt                 google_auth_httplib2 plotlywidget        torch
apt_inst            google_auth_oauthlib plotnine            torchaudio
apt_pkg             google_drive_downloader pluggy              torchgen
aptsources          googleapiclient     pooch               torchsummary
argparse            googlesearch        poplib              torchtext
array               graphviz            portpicker          torchvision
arviz               greenlet            posix               tornado
ast                 gridfs              posixpath           tqdm
astor               grp                 pprint              trace
astropy             grpc                prefetch_generator  traceback
astunparse          gspread             preshed             tracemalloc
async_timeout       gspread_dataframe   prettytable         traitlets
asynchat            gym                 profile             tree
asyncio             gym_notices         progressbar         tty
asyncore            gzip                promise             turtle
asynctest           h5py                prompt_toolkit      tweepy
atari_py            hashlib             prophet             typeguard
atexit              heapdict            pstats              typer
atomicwrites        heapq               psutil              types
attr                hijri_converter     psycopg2            typing
attrs               hmac                pty                 typing_extensions
audioop             holidays            ptyprocess          tzlocal
audioread           holoviews           pvectorc            unicodedata
autograd            html                pwd                 unification
autoreload          html5lib            py                  unittest
babel               http                py_compile          uritemplate
backcall            httpimport          pyarrow             urllib
base64              httplib2            pyasn1              urllib3
bdb                 httplib2shim        pyasn1_modules      uu
bin                 httpstan            pyclbr              uuid
binascii            humanize            pycocotools         vega_datasets
binhex              hyperopt            pycparser           venv
bisect              idna                pyct                vis
bleach              imageio             pydantic            warnings
blis                imagesize           pydata_google_auth  wasabi
bokeh               imaplib             pydoc               wave
boost               imblearn            pydoc_data          wcwidth
branca              imgaug              pydot               weakref
bs4                 imghdr              pydot_ng            webargs
bson                imp                 pydotplus           webbrowser
builtins            importlib           pydrive             webencodings
bz2                 importlib_metadata  pyemd               werkzeug
cProfile            importlib_resources pyexpat             wheel
cachecontrol        imutils             pygments            widgetsnbextension
cached_property     inflect             pygtkcompat         wordcloud
cachetools          inspect             pylab               wrapt
caffe2              intervaltree        pylev               wsgiref
calendar            io                  pymc                xarray
catalogue           ipaddress           pymeeus             xarray_einstats
certifi             ipykernel           pymongo             xdrlib
cffi                ipykernel_launcher  pymystem3           xgboost
cftime              ipython_genutils    pyparsing           xkit
cgi                 ipywidgets          pyrsistent          xlrd
cgitb               isympy              pysndfile           xlwt
chardet             itertools           pytest              xml
charset_normalizer  itsdangerous        python_utils        xmlrpc
chunk               jax                 pytz                xxlimited
clang               jaxlib              pyviz_comms         xxsubtype
click               jieba               pywt                yaml
client              jinja2              pyximport           yarl
clikit              joblib              qdldl               yellowbrick
cloudpickle         jpeg4py             qudida              zict
cmake               json                queue               zipapp
cmath               jsonschema          quopri              zipfile
cmd                 jupyter             random              zipimport
cmdstanpy           jupyter_client      re                  zipp
code                jupyter_console     readline            zlib
codecs              jupyter_core        regex               zmq
codeop              jupyterlab_plotly   reprlib             
colab               jupyterlab_widgets  requests            

Enter any module name to get more help.  Or, type "modules spam" to search
for modules whose name or summary contain the string "spam".

Loops in Python
Need for loops
While Loop
For Loop

# While loop example -> program to print the table
# Program -> Sum of all digits of a given number
# Program -> keep accepting numbers from users till he/she enters a 0 and then find the avg
     

number = int(input('enter the number'))

i = 1

while i<11:
  print(number,'*',i,'=',number * i)
  i += 1
     
enter the number12
12 * 1 = 12
12 * 2 = 24
12 * 3 = 36
12 * 4 = 48
12 * 5 = 60
12 * 6 = 72
12 * 7 = 84
12 * 8 = 96
12 * 9 = 108
12 * 10 = 120

# while loop with else

x = 1

while x < 3:
  print(x)
  x += 1

else:
  print('limit crossed')
     
1
2
limit crossed

# Guessing game

# generate a random integer between 1 and 100
import random
jackpot = random.randint(1,100)

guess = int(input('guess karo'))
counter = 1
while guess != jackpot:
  if guess < jackpot:
    print('galat!guess higher')
  else:
    print('galat!guess lower')

  guess = int(input('guess karo'))
  counter += 1

else:
  print('correct guess')
  print('attempts',counter)



     
guess karo7
galat!guess higher
guess karo50
galat!guess lower
guess karo30
galat!guess higher
guess karo40
galat!guess lower
guess karo35
galat!guess lower
guess karo32
galat!guess higher
guess karo33
correct guess
attempts 7

# For loop demo

for i in {1,2,3,4,5}:
  print(i)
     
1
2
3
4
5

# For loop examples
     
Program - The current population of a town is 10000. The population of the town is increasing at the rate of 10% per year. You have to write a program to find out the population at the end of each of the last 10 years.

curr_pop = 10000

for i in range(10,0,-1):
  print(i,curr_pop)
  curr_pop = curr_pop - 0.1*curr_pop

     
10 10000
9 9000.0
8 8100.0
7 7290.0
6 6561.0
5 5904.9
4 5314.41
3 4782.969
2 4304.6721
1 3874.20489
Sequence sum
1/1! + 2/2! + 3/3! + ...


# code here
     

# For loop vs While loops (When to use what?)
     
Nested Loops

# Examples
     

# Program - Unique combination of 1,2,3,4
# Program - Pattern 1 and 2
     
Pattern 1
***
****
***



     
Pattern 2
1
121
12321
1234321



     
Loop Control Statement
Break
Continue
Pass

# Break demo
     

# Break example (Linear Search) -> Prime number in a given range
     

# Continue demo
     

# Continue Example (Ecommerce)
     

# Pass demo





################################################################################


Program - The current population of a town is 10000. The population of the town is increasing at the rate of 10% per year. You have to write a program to find out the population at the end of each of the last 10 years.

# Code here
curr_pop = 10000

for i in range(10,0,-1):
  print(i,curr_pop)
  curr_pop = curr_pop/1.1

     
10 10000
9 9090.90909090909
8 8264.462809917353
7 7513.148009015775
6 6830.134553650703
5 6209.213230591548
4 5644.739300537771
3 5131.5811823070635
2 4665.07380209733
1 4240.976183724845
Sequence sum
1/1! + 2/2! + 3/3! + ...


# Code here

n = int(input('enter n'))

result = 0
fact = 1

for i in range(1,n+1):
  fact = fact * i
  result = result + i/fact

print(result)



     
enter n2
2.0
Nested Loops

# Examples -> unique pairs

for i in range(1,5):
  for j in range(1,5):
    print(i,j)
     
1 1
1 2
1 3
1 4
2 1
2 2
2 3
2 4
3 1
3 2
3 3
3 4
4 1
4 2
4 3
4 4
Pattern 1
***
****
***


# code here

rows = int(input('enter number of rows'))

for i in range(1,rows+1):
  for j in range(1,i+1):
    print('*',end='')
  print()

     
enter number of rows10
*
**
***
****
*****
******
*******
********
*********
**********
Pattern 2
1
121
12321
1234321


# Code here
rows = int(input('enter number of rows'))

for i in range(1,rows+1):
  for j in range(1,i+1):
    print(j,end='')
  for k in range(i-1,0,-1):
    print(k,end='')

  print()

     
enter number of rows4
1
121
12321
1234321
Loop Control Statement
Break
Continue
Pass

for i in range(1,10):
  if i == 5:
    break
  print(i)
     
1
2
3
4

lower = int(input('enter lower range'))
upper = int(input('enter upper range'))

for i in range(lower,upper+1):
  for j in range(2,i):
    if i%j == 0:
      break
  else:
    print(i)
     
enter lower range10
enter upper range100
11
13
17
19
23
29
31
37
41
43
47
53
59
61
67
71
73
79
83
89
97

# Continue
for i in range(1,10):
  if i == 5:
    continue
  print(i)
     
1
2
3
4
6
7
8
9

for i in range(1,10):
  pass

     
Strings are sequence of Characters

In Python specifically, strings are a sequence of Unicode Characters

Creating Strings
Accessing Strings
Adding Chars to Strings
Editing Strings
Deleting Strings
Operations on Strings
String Functions
Creating Stings

s = 'hello'
s = "hello"
# multiline strings
s = '''hello'''
s = """hello"""
s = str('hello')
print(s)
     
hello

"it's raining outside"
     
"it's raining outside"
Accessing Substrings from a String

# Positive Indexing
s = 'hello world'
print(s[41])
     
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-61-633ba99ed6e5> in <module>
      1 # Positive Indexing
      2 s = 'hello world'
----> 3 print(s[41])

IndexError: string index out of range

# Negative Indexing
s = 'hello world'
print(s[-3])
     
r

# Slicing
s = 'hello world'
print(s[6:0:-2])
     
wol

print(s[::-1])
     
dlrow olleh

s = 'hello world'
print(s[-1:-6:-1])
     
dlrow
Editing and Deleting in Strings

s = 'hello world'
s[0] = 'H'

# Python strings are immutable
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-80-0c8a824e3b73> in <module>
      1 s = 'hello world'
----> 2 s[0] = 'H'

TypeError: 'str' object does not support item assignment

s = 'hello world'
del s
print(s)
     
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-81-9ae37fbf1c6c> in <module>
      1 s = 'hello world'
      2 del s
----> 3 print(s)

NameError: name 's' is not defined

s = 'hello world'
del s[-1:-5:2]
print(s)
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-82-d0d823eafb6b> in <module>
      1 s = 'hello world'
----> 2 del s[-1:-5:2]
      3 print(s)

TypeError: 'str' object does not support item deletion
Operations on Strings
Arithmetic Operations
Relational Operations
Logical Operations
Loops on Strings
Membership Operations

print('delhi' + ' ' + 'mumbai')
     
delhi mumbai

print('delhi'*5)
     
delhidelhidelhidelhidelhi

print("*"*50)
     
**************************************************

'delhi' != 'delhi'
     
False

'mumbai' > 'pune'
# lexiographically
     
False

'Pune' > 'pune'
     
False

'hello' and 'world'
     
'world'

'hello' or 'world'
     
'hello'

'' and 'world'
     
''

'' or 'world'
     
'world'

'hello' or 'world'
     
'hello'

'hello' and 'world'
     
'world'

not 'hello'
     
False

for i in 'hello':
  print(i)
     
h
e
l
l
o

for i in 'delhi':
  print('pune')
     
pune
pune
pune
pune
pune

'D' in 'delhi'
     
False


     


     


     
Common Functions
len
max
min
sorted

len('hello world')
     
11

max('hello world')
     
'w'

min('hello world')
     
' '

sorted('hello world',reverse=True)
     
['w', 'r', 'o', 'o', 'l', 'l', 'l', 'h', 'e', 'd', ' ']


     
Capitalize/Title/Upper/Lower/Swapcase

s = 'hello world'
print(s.capitalize())
print(s)
     
Hello world
hello world

s.title()
     
'Hello World'

s.upper()
     
'HELLO WORLD'

'Hello Wolrd'.lower()
     
'hello wolrd'

'HeLlO WorLD'.swapcase()
     
'hElLo wORld'
Count/Find/Index

'my name is nitish'.count('i')
     
3

'my name is nitish'.find('x')
     
-1

'my name is nitish'.index('x')
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-121-12e2ad5b75e9> in <module>
----> 1 'my name is nitish'.index('x')

ValueError: substring not found


     
endswith/startswith

'my name is nitish'.endswith('sho')
     
False

'my name is nitish'.startswith('1my')
     
False
format

name = 'nitish'
gender = 'male'

'Hi my name is {1} and I am a {0}'.format(gender,name)
     
'Hi my name is nitish and I am a male'
isalnum/ isalpha/ isdigit/ isidentifier

'nitish1234%'.isalnum()
     
False

'nitish'.isalpha()
     
True

'123abc'.isdigit()
     
False

'first-name'.isidentifier()
     
False


     
Split/Join

'hi my name is nitish'.split()
     
['hi', 'my', 'name', 'is', 'nitish']

" ".join(['hi', 'my', 'name', 'is', 'nitish'])
     
'hi my name is nitish'
Replace

'hi my name is nitish'.replace('nitisrgewrhgh','campusx')
     
'hi my name is nitish'
Strip

'nitish                           '.strip()
     
'nitish'
Example Programs

# Find the length of a given string without using the len() function

s = input('enter the string')

counter = 0

for i in s:
  counter += 1

print('length of string is',counter)
     
enter the stringnitish
length of string is 6

# Extract username from a given email.
# Eg if the email is nitish24singh@gmail.com
# then the username should be nitish24singh

s = input('enter the email')

pos = s.index('@')
print(s[0:pos])


     
enter the emailsupport@campusx.in
support

# Count the frequency of a particular character in a provided string.
# Eg 'hello how are you' is the string, the frequency of h in this string is 2.

s = input('enter the email')
term = input('what would like to search for')

counter = 0
for i in s:
  if i == term:
    counter += 1

print('frequency',counter)

     
enter the emailhi how are you
what would like to search foro
frequency 2

# Write a program which can remove a particular character from a string.
s = input('enter the string')
term = input('what would like to remove')

result = ''

for i in s:
  if i != term:
    result = result + i

print(result)
     
enter the stringnitish
what would like to removei
ntsh

# Write a program that can check whether a given string is palindrome or not.
# abba
# malayalam

s = input('enter the string')
flag = True
for i in range(0,len(s)//2):
  if s[i] != s[len(s) - i -1]:
    flag = False
    print('Not a Palindrome')
    break

if flag:
  print('Palindrome')


     
enter the stringpython
Not a Palindrome

# Write a program to count the number of words in a string without split()

s = input('enter the string')
L = []
temp = ''
for i in s:

  if i != ' ':
    temp = temp + i
  else:
    L.append(temp)
    temp = ''

L.append(temp)
print(L)


     
enter the stringhi how are you
['hi', 'how', 'are', 'you']

# Write a python program to convert a string to title case without using the title()
s = input('enter the string')

L = []
for i in s.split():
  L.append(i[0].upper() + i[1:].lower())

print(" ".join(L))
     
enter the stringhi my namE iS NitiSh
Hi My Name Is Nitish

# Write a program that can convert an integer to string.

number = int(input('enter the number'))

digits = '0123456789'
result = ''
while number != 0:
  result = digits[number % 10] + result
  number = number//10

print(result)
print(type(result))
     
enter the number345
345
<class 'str'>


     
###############################################################################



1. Lists
What are Lists?
Lists Vs Arrays
Characterstics of a List
How to create a list
Access items from a List
Editing items in a List
Deleting items from a List
Operations on Lists
Functions on Lists
What are Lists
List is a data type where you can store multiple items under 1 name. More technically, lists act like dynamic arrays which means you can add more items on the fly.

image.png

Why Lists are required in programming?
Array Vs Lists
Fixed Vs Dynamic Size
Convenience -> Hetrogeneous
Speed of Execution
Memory

L = [1,2,3]

print(id(L))
print(id(L[0]))
print(id(L[1]))
print(id(L[2]))
print(id(1))
print(id(2))
print(id(3))
     
140163201133376
11126688
11126720
11126752
11126688
11126720
11126752
How lists are stored in memory
Characterstics of a List
Ordered
Changeble/Mutable
Hetrogeneous
Can have duplicates
are dynamic
can be nested
items can be accessed
can contain any kind of objects in python

L = [1,2,3,1]
L1 = [3,2,1]

L == L1
     
False
Creating a List

# Empty
print([])
# 1D -> Homo
print([1,2,3,4,5])
# 2D
print([1,2,3,[4,5]])
# 3D
print([[[1,2],[3,4]],[[5,6],[7,8]]])
# Hetrogenous
print([1,True,5.6,5+6j,'Hello'])
# Using Type conversion
print(list('hello'))
     
[]
[1, 2, 3, 4, 5]
[1, 2, 3, [4, 5]]
[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
[1, True, 5.6, (5+6j), 'Hello']
['h', 'e', 'l', 'l', 'o']
Accessing Items from a List

# Indexing
L = [[[1,2],[3,4]],[[5,6],[7,8]]]
#positive
#print(L[0][0][1])

# Slicing
L = [1,2,3,4,5,6]

print(L[::-1])
     
[6, 5, 4, 3, 2, 1]
Adding Items to a List

# append
L = [1,2,3,4,5]
L.append(True)
print(L)
     
[1, 2, 3, 4, 5, True]

# extend
L = [1,2,3,4,5]
L.extend([6,7,8])
print(L)
     
[1, 2, 3, 4, 5, 6, 7, 8]

L = [1,2,3,4,5]
L.append([6,7,8])
print(L)
     
[1, 2, 3, 4, 5, [6, 7, 8]]

L = [1,2,3,4,5]
L.extend('delhi')
print(L)
     
[1, 2, 3, 4, 5, 'd', 'e', 'l', 'h', 'i']

# insert
L = [1,2,3,4,5]

L.insert(1,100)
print(L)
     
[1, 100, 2, 3, 4, 5]
Editing items in a List

L = [1,2,3,4,5]

# editing with indexing
L[-1] = 500

# editing with slicing
L[1:4] = [200,300,400]

print(L)
     
[1, 200, 300, 400, 500]
Deleting items from a List

# del
L = [1,2,3,4,5]

# indexing
del L[-1]

# slicing
del L[1:3]
print(L)
     
[1, 4]

# remove

L = [1,2,3,4,5]

L.remove(5)

print(L)

     
[1, 2, 3, 4]

# pop
L = [1,2,3,4,5]

L.pop()

print(L)
     
[1, 2, 3, 4]

# clear
L = [1,2,3,4,5]

L.clear()

print(L)
     
[]
Operations on Lists
Arithmetic
Membership
Loop

# Arithmetic (+ ,*)

L1 = [1,2,3,4]
L2 = [5,6,7,8]

# Concatenation/Merge
print(L1 + L2)
     
[1, 2, 3, 4, 5, 6, 7, 8]

print(L1*3)
     
[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

L1 = [1,2,3,4,5]
L2 = [1,2,3,4,[5,6]]

print(5 not in L1)
print([5,6] in L2)
     
False
True

# Loops
L1 = [1,2,3,4,5]
L2 = [1,2,3,4,[5,6]]
L3 = [[[1,2],[3,4]],[[5,6],[7,8]]]

for i in L3:
  print(i)
     
[[1, 2], [3, 4]]
[[5, 6], [7, 8]]
List Functions

# len/min/max/sorted
L = [2,1,5,7,0]

print(len(L))
print(min(L))
print(max(L))
print(sorted(L,reverse=True))
     
5
0
7
[7, 5, 2, 1, 0]

# count
L = [1,2,1,3,4,1,5]
L.count(5)
     
1

# index
L = [1,2,1,3,4,1,5]
L.index(1)
     
0

# reverse
L = [2,1,5,7,0]
# permanently reverses the list
L.reverse()
print(L)
     
[0, 7, 5, 1, 2]

# sort (vs sorted)
L = [2,1,5,7,0]
print(L)
print(sorted(L))
print(L)
L.sort()
print(L)
     
[2, 1, 5, 7, 0]
[0, 1, 2, 5, 7]
[2, 1, 5, 7, 0]
[0, 1, 2, 5, 7]

# copy -> shallow
L = [2,1,5,7,0]
print(L)
print(id(L))
L1 = L.copy()
print(L1)
print(id(L1))
     
[2, 1, 5, 7, 0]
140163201056112
[2, 1, 5, 7, 0]
140163201128800
List Comprehension
List Comprehension provides a concise way of creating lists.

newlist = [expression for item in iterable if condition == True]image.png

Advantages of List Comprehension

More time-efficient and space-efficient than loops.
Require fewer lines of code.
Transforms iterative statement into a formula.

# Add 1 to 10 numbers to a list
L = []

for i in range(1,11):
  L.append(i)

print(L)
     
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

L = [i for i in range(1,11)]
print(L)
     
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# scalar multiplication on a vector
v = [2,3,4]
s = -3
# [-6,-9,-12]

[s*i for i in v]
     
[-6, -9, -12]

# Add squares
L = [1,2,3,4,5]

[i**2 for i in L]

     
[1, 4, 9, 16, 25]

# Print all numbers divisible by 5 in the range of 1 to 50

[i for i in range(1,51) if i%5 == 0]
     
[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# find languages which start with letter p
languages = ['java','python','php','c','javascript']

[language for language in languages if language.startswith('p')]
     
['python', 'php']

# Nested if with List Comprehension
basket = ['apple','guava','cherry','banana']
my_fruits = ['apple','kiwi','grapes','banana']

# add new list from my_fruits and items if the fruit exists in basket and also starts with 'a'

[fruit for fruit in my_fruits if fruit in basket if fruit.startswith('a')]
     
['apple']

# Print a (3,3) matrix using list comprehension -> Nested List comprehension
[[i*j for i in range(1,4)] for j in range(1,4)]
     
[[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# cartesian products -> List comprehension on 2 lists together
L1 = [1,2,3,4]
L2 = [5,6,7,8]

[i*j for i in L1 for j in L2]
     
[5, 6, 7, 8, 10, 12, 14, 16, 15, 18, 21, 24, 20, 24, 28, 32]
2 ways to traverse a list
itemwise
indexwise

# itemwise
L = [1,2,3,4]

for i in L:
  print(i)
     
1
2
3
4

# indexwise
L = [1,2,3,4]

for i in range(0,len(L)):
  print(L[i])

     
1
2
3
4
Zip
The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed iterator is paired together, and then the second item in each passed iterator are paired together.

If the passed iterators have different lengths, the iterator with the least items decides the length of the new iterator.


# Write a program to add items of 2 lists indexwise

L1 = [1,2,3,4]
L2 = [-1,-2,-3,-4]

list(zip(L1,L2))

[i+j for i,j in zip(L1,L2)]
     
[0, 0, 0, 0]

L = [1,2,print,type,input]

print(L)
     
[1, 2, <built-in function print>, <class 'type'>, <bound method Kernel.raw_input of <google.colab._kernel.Kernel object at 0x7f7a67452a90>>]
Disadvantages of Python Lists
Slow
Risky usage
eats up more memory

a = [1,2,3]
b = a.copy()

print(a)
print(b)

a.append(4)
print(a)
print(b)

# lists are mutable
     
[1, 2, 3]
[1, 2, 3]
[1, 2, 3, 4]
[1, 2, 3]
List Programs

# Create 2 lists from a given list where
# 1st list will contain all the odd numbers from the original list and
# the 2nd one will contain all the even numbers

L = [1,2,3,4,5,6]
     

# How to take list as input from user
     

# Write a program to merge 2 list without using the + operator
L1 = [1,2,3,4]
L2 = [5,6,7,8]

     

# Write a program to replace an item with a different item if found in the list
L = [1,2,3,4,5,3]
# replace 3 with 300
     

# Write a program that can convert a 2D list to 1D list
     

# Write a program to remove duplicate items from a list

L = [1,2,1,2,3,4,5,3,4]
     

# Write a program to check if a list is in ascending order or not
     



################################################################################



Tuples
A tuple in Python is similar to a list. The difference between the two is that we cannot change the elements of a tuple once it is assigned whereas we can change the elements of a list.

In short, a tuple is an immutable list. A tuple can not be changed in any way once it is created.

Characterstics

Ordered
Unchangeble
Allows duplicate
Plan of attack
Creating a Tuple
Accessing items
Editing items
Adding items
Deleting items
Operations on Tuples
Tuple Functions
Creating Tuples

# empty
t1 = ()
print(t1)
# create a tuple with a single item
t2 = ('hello',)
print(t2)
print(type(t2))
# homo
t3 = (1,2,3,4)
print(t3)
# hetro
t4 = (1,2.5,True,[1,2,3])
print(t4)
# tuple
t5 = (1,2,3,(4,5))
print(t5)
# using type conversion
t6 = tuple('hello')
print(t6)
     
()
('hello',)
<class 'tuple'>
(1, 2, 3, 4)
(1, 2.5, True, [1, 2, 3])
(1, 2, 3, (4, 5))
('h', 'e', 'l', 'l', 'o')
Accessing Items
Indexing
Slicing

print(t3)
print(t3[0])
print(t3[-1])
     
(1, 2, 3, 4)
1
4

t5[-1][0]
     
4
Editing items

print(t3)
t3[0] = 100
# immutable just like strings
     
(1, 2, 3, 4)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-30-49d9e1416ccf> in <module>
      1 print(t3)
----> 2 t3[0] = 100

TypeError: 'tuple' object does not support item assignment
Adding items

print(t3)
# not possible
     
(1, 2, 3, 4)
Deleting items

print(t3)
del t3
print(t3)
     
(1, 2, 3, 4)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-33-0a67b29ad777> in <module>
      1 print(t3)
      2 del t3
----> 3 print(t3)

NameError: name 't3' is not defined

t = (1,2,3,4,5)

t[-1:-4:-1]
     
(5, 4, 3)

print(t5)
del t5[-1]
     
(1, 2, 3, (4, 5))
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-35-2b39d140e8ae> in <module>
      1 print(t5)
----> 2 del t5[-1]

TypeError: 'tuple' object doesn't support item deletion
Operations on Tuples

# + and *
t1 = (1,2,3,4)
t2 = (5,6,7,8)

print(t1 + t2)

print(t1*3)
# membership
1 in t1
# iteration
for i in t1:
  print(i)
     
(1, 2, 3, 4, 5, 6, 7, 8)
(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)
1
2
3
4
Tuple Functions

# len/sum/min/max/sorted
t = (1,2,3,4)
len(t)

sum(t)

min(t)

max(t)

sorted(t,reverse=True)
     
[4, 3, 2, 1]

# count

t = (1,2,3,4,5)

t.count(50)
     
0

# index
t.index(50)
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-51-cae2b6ba49a8> in <module>
      1 # index
----> 2 t.index(50)

ValueError: tuple.index(x): x not in tuple
Difference between Lists and Tuples
Syntax
Mutability
Speed
Memory
Built in functionality
Error prone
Usability

import time

L = list(range(100000000))
T = tuple(range(100000000))

start = time.time()
for i in L:
  i*5
print('List time',time.time()-start)

start = time.time()
for i in T:
  i*5
print('Tuple time',time.time()-start)
     
List time 9.853569507598877
Tuple time 8.347511053085327

import sys

L = list(range(1000))
T = tuple(range(1000))

print('List size',sys.getsizeof(L))
print('Tuple size',sys.getsizeof(T))

     
List size 9120
Tuple size 8056

a = [1,2,3]
b = a

a.append(4)
print(a)
print(b)
     
[1, 2, 3, 4]
[1, 2, 3, 4]

a = (1,2,3)
b = a

a = a + (4,)
print(a)
print(b)
     
(1, 2, 3, 4)
(1, 2, 3)
Why use tuple?
Special Syntax

# tuple unpacking
a,b,c = (1,2,3)
print(a,b,c)
     
1 2 3

a,b = (1,2,3)
print(a,b)
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-55-22f327f11d4b> in <module>
----> 1 a,b = (1,2,3)
      2 print(a,b)

ValueError: too many values to unpack (expected 2)

a = 1
b = 2
a,b = b,a

print(a,b)
     
2 1

a,b,*others = (1,2,3,4)
print(a,b)
print(others)
     
1 2
[3, 4]

# zipping tuples
a = (1,2,3,4)
b = (5,6,7,8)

tuple(zip(a,b))
     
((1, 5), (2, 6), (3, 7), (4, 8))
Sets
A set is an unordered collection of items. Every set element is unique (no duplicates) and must be immutable (cannot be changed).

However, a set itself is mutable. We can add or remove items from it.

Sets can also be used to perform mathematical set operations like union, intersection, symmetric difference, etc.

Characterstics:

Unordered
Mutable
No Duplicates
Can't contain mutable data types
Creating Sets

# empty
s = set()
print(s)
print(type(s))
# 1D and 2D
s1 = {1,2,3}
print(s1)
#s2 = {1,2,3,{4,5}}
#print(s2)
# homo and hetro
s3 = {1,'hello',4.5,(1,2,3)}
print(s3)
# using type conversion

s4 = set([1,2,3])
print(s4)
# duplicates not allowed
s5 = {1,1,2,2,3,3}
print(s5)
# set can't have mutable items
s6 = {1,2,[3,4]}
print(s6)
     
set()
<class 'set'>
{1, 2, 3}
{1, 4.5, (1, 2, 3), 'hello'}
{1, 2, 3}
{1, 2, 3}
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-71-ab3c7dde6aed> in <module>
     19 print(s5)
     20 # set can't have mutable items
---> 21 s6 = {1,2,[3,4]}
     22 print(s6)

TypeError: unhashable type: 'list'

s1 = {1,2,3}
s2 = {3,2,1}

print(s1 == s2)
     
True


     
Accessing Items

s1 = {1,2,3,4}
s1[0:3]
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-75-4c49b6b6050d> in <module>
      1 s1 = {1,2,3,4}
----> 2 s1[0:3]

TypeError: 'set' object is not subscriptable
Editing Items

s1 = {1,2,3,4}
s1[0] = 100
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-76-bd617ce25076> in <module>
      1 s1 = {1,2,3,4}
----> 2 s1[0] = 100

TypeError: 'set' object does not support item assignment
Adding Items

S = {1,2,3,4}
# add
# S.add(5)
# print(S)
# update
S.update([5,6,7])
print(S)
     
{1, 2, 3, 4, 5, 6, 7}
Deleting Items

# del
s = {1,2,3,4,5}
# print(s)
# del s[0]
# print(s)
# discard
# s.discard(50)
# print(s)
# remove
# s.remove(50)
# print(s)
# pop
# s.pop()
# clear
s.clear()
print(s)
     
set()
Set Operation

s1 = {1,2,3,4,5}
s2 = {4,5,6,7,8}
s1 | s2
# Union(|)
# Intersection(&)
s1 & s2
# Difference(-)
s1 - s2
s2 - s1
# Symmetric Difference(^)
s1 ^ s2
# Membership Test
1 not in s1
# Iteration
for i in s1:
  print(i)
     
1
2
3
4
5
Set Functions

# len/sum/min/max/sorted
s = {3,1,4,5,2,7}
len(s)
sum(s)
min(s)
max(s)
sorted(s,reverse=True)
     
[7, 5, 4, 3, 2, 1]

# union/update
s1 = {1,2,3,4,5}
s2 = {4,5,6,7,8}

# s1 | s2
s1.union(s1)

s1.update(s2)
print(s1)
print(s2)
     
{1, 2, 3, 4, 5, 6, 7, 8}
{4, 5, 6, 7, 8}

# intersection/intersection_update
s1 = {1,2,3,4,5}
s2 = {4,5,6,7,8}

s1.intersection(s2)

s1.intersection_update(s2)
print(s1)
print(s2)
     
{4, 5}
{4, 5, 6, 7, 8}

# difference/difference_update
s1 = {1,2,3,4,5}
s2 = {4,5,6,7,8}

s1.difference(s2)

s1.difference_update(s2)
print(s1)
print(s2)
     
{1, 2, 3}
{4, 5, 6, 7, 8}

# symmetric_difference/symmetric_difference_update
s1 = {1,2,3,4,5}
s2 = {4,5,6,7,8}

s1.symmetric_difference(s2)

s1.symmetric_difference_update(s2)
print(s1)
print(s2)
     
{1, 2, 3, 6, 7, 8}
{4, 5, 6, 7, 8}

# isdisjoint/issubset/issuperset
s1 = {1,2,3,4}
s2 = {7,8,5,6}

s1.isdisjoint(s2)
     
True

s1 = {1,2,3,4,5}
s2 = {3,4,5}

s1.issuperset(s2)
     
True

# copy
s1 = {1,2,3}
s2 = s1.copy()

print(s1)
print(s2)
     
{1, 2, 3}
{1, 2, 3}
Frozenset
Frozen set is just an immutable version of a Python set object


# create frozenset
fs1 = frozenset([1,2,3])
fs2 = frozenset([3,4,5])

fs1 | fs2
     
frozenset({1, 2, 3, 4, 5})

# what works and what does not
# works -> all read functions
# does't work -> write operations
     

# When to use
# 2D sets
fs = frozenset([1,2,frozenset([3,4])])
fs
     
frozenset({1, 2, frozenset({3, 4})})
Set Comprehension

# examples

{i**2 for i in range(1,11) if i>5}
     
{36, 49, 64, 81, 100}


     
Dictionary
Dictionary in Python is a collection of keys values, used to store data values like a map, which, unlike other data types which hold only a single value as an element.

In some languages it is known as map or assosiative arrays.

dict = { 'name' : 'nitish' , 'age' : 33 , 'gender' : 'male' }

Characterstics:

Mutable
Indexing has no meaning
keys can't be duplicated
keys can't be mutable items
Create Dictionary

# empty dictionary
d = {}
d
# 1D dictionary
d1 = { 'name' : 'nitish' ,'gender' : 'male' }
d1
# with mixed keys
d2 = {(1,2,3):1,'hello':'world'}
d2
# 2D dictionary -> JSON
s = {
    'name':'nitish',
     'college':'bit',
     'sem':4,
     'subjects':{
         'dsa':50,
         'maths':67,
         'english':34
     }
}
s
# using sequence and dict function
d4 = dict([('name','nitish'),('age',32),(3,3)])
d4
# duplicate keys
d5 = {'name':'nitish','name':'rahul'}
d5
# mutable items as keys
d6 = {'name':'nitish',(1,2,3):2}
print(d6)
     
{'name': 'nitish', (1, 2, 3): 2}
Accessing items

my_dict = {'name': 'Jack', 'age': 26}
# []
my_dict['age']
# get
my_dict.get('age')

s['subjects']['maths']
     
67
Adding key-value pair

d4['gender'] = 'male'
d4
d4['weight'] = 72
d4

s['subjects']['ds'] = 75
s
     
{'name': 'nitish',
 'college': 'bit',
 'sem': 4,
 'subjects': {'dsa': 50, 'maths': 67, 'english': 34, 'ds': 75}}
Remove key-value pair

d = {'name': 'nitish', 'age': 32, 3: 3, 'gender': 'male', 'weight': 72}
# pop
#d.pop(3)
#print(d)
# popitem
#d.popitem()
# d.popitem()
# print(d)
# del
#del d['name']
#print(d)
# clear
d.clear()
print(d)

del s['subjects']['maths']
s
     
{}
{'name': 'nitish',
 'college': 'bit',
 'sem': 4,
 'subjects': {'dsa': 50, 'english': 34, 'ds': 75}}
Editing key-value pair

s['subjects']['dsa'] = 80
s
     
{'name': 'nitish',
 'college': 'bit',
 'sem': 5,
 'subjects': {'dsa': 80, 'english': 34, 'ds': 75}}
Dictionary Operations
Membership
Iteration

print(s)

'name' in s
     
{'name': 'nitish', 'college': 'bit', 'sem': 5, 'subjects': {'dsa': 80, 'english': 34, 'ds': 75}}
True

d = {'name':'nitish','gender':'male','age':33}

for i in d:
  print(i,d[i])
     
name nitish
gender male
age 33
Dictionary Functions

# len/sorted
len(d)
print(d)
sorted(d,reverse=True)
max(d)
     
{'name': 'nitish', 'gender': 'male', 'age': 33}
'name'

# items/keys/values
print(d)

print(d.items())
print(d.keys())
print(d.values())
     
{'name': 'nitish', 'gender': 'male', 'age': 33}
dict_items([('name', 'nitish'), ('gender', 'male'), ('age', 33)])
dict_keys(['name', 'gender', 'age'])
dict_values(['nitish', 'male', 33])

# update
d1 = {1:2,3:4,4:5}
d2 = {4:7,6:8}

d1.update(d2)
print(d1)
     
{1: 2, 3: 4, 4: 7, 6: 8}
Dictionary Comprehension
image.png


# print 1st 10 numbers and their squares
{i:i**2 for i in range(1,11)}
     
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100}

distances = {'delhi':1000,'mumbai':2000,'bangalore':3000}
print(distances.items())
     
dict_items([('delhi', 1000), ('mumbai', 2000), ('bangalore', 3000)])

# using existing dict
distances = {'delhi':1000,'mumbai':2000,'bangalore':3000}
{key:value*0.62 for (key,value) in distances.items()}
     
{'delhi': 620.0, 'mumbai': 1240.0, 'bangalore': 1860.0}

# using zip
days = ["Sunday", "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
temp_C = [30.5,32.6,31.8,33.4,29.8,30.2,29.9]

{i:j for (i,j) in zip(days,temp_C)}
     
{'Sunday': 30.5,
 'Monday': 32.6,
 'Tuesday': 31.8,
 'Wednesday': 33.4,
 'Thursday': 29.8,
 'Friday': 30.2,
 'Saturday': 29.9}

# using if condition
products = {'phone':10,'laptop':0,'charger':32,'tablet':0}

{key:value for (key,value) in products.items() if value>0}
     
{'phone': 10, 'charger': 32}

# Nested Comprehension
# print tables of number from 2 to 4
{i:{j:i*j for j in range(1,11)} for i in range(2,5)}
     
{2: {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16, 9: 18, 10: 20},
 3: {1: 3, 2: 6, 3: 9, 4: 12, 5: 15, 6: 18, 7: 21, 8: 24, 9: 27, 10: 30},
 4: {1: 4, 2: 8, 3: 12, 4: 16, 5: 20, 6: 24, 7: 28, 8: 32, 9: 36, 10: 40}}

{
    2:{1:2,2:4,3:6,4:8},
    3:{1:3,2:6,3:9,4:12},
    4:{1:4,2:8,3:12,4:16}
}
     


##################################################################################




Let's create a function(with docstring)

def is_even(num):
  """
  This function returns if a given number is odd or even
  input - any valid integer
  output - odd/even
  created on - 16th Nov 2022
  """
  if type(num) == int:
    if num % 2 == 0:
      return 'even'
    else:
      return 'odd'
  else:
    return 'pagal hai kya?'
     

# function
# function_name(input)
for i in range(1,11):
  x = is_even(i)
  print(x)
     
odd
even
odd
even
odd
even
odd
even
odd
even

print(type.__doc__)
     
type(object_or_name, bases, dict)
type(object) -> the object's type
type(name, bases, dict) -> a new type
2 Point of views

is_even('hello')
     
'pagal hai kya?'
Parameters Vs Arguments
Types of Arguments
Default Argument
Positional Argument
Keyword Argument

def power(a=1,b=1):
  return a**b
     

power()
     
1

# positional argument
power(2,3)
     
8

# keyword argument
power(b=3,a=2)
     
8
*args and **kwargs
*args and **kwargs are special Python keywords that are used to pass the variable length of arguments to a function


# *args
# allows us to pass a variable number of non-keyword arguments to a function.

def multiply(*kwargs):
  product = 1

  for i in kwargs:
    product = product * i

  print(kwargs)
  return product
     

multiply(1,2,3,4,5,6,7,8,9,10,12)
     
(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
43545600

# **kwargs
# **kwargs allows us to pass any number of keyword arguments.
# Keyword arguments mean that they contain a key-value pair, like a Python dictionary.

def display(**salman):

  for (key,value) in salman.items():
    print(key,'->',value)

     

display(india='delhi',srilanka='colombo',nepal='kathmandu',pakistan='islamabad')
     
india -> delhi
srilanka -> colombo
nepal -> kathmandu
pakistan -> islamabad
Points to remember while using *args and **kwargs
order of the arguments matter(normal -> *args -> **kwargs)
The words “args” and “kwargs” are only a convention, you can use any name of your choice


     
How Functions are executed in memory?


     
Without return statement

L = [1,2,3]
print(L.append(4))
print(L)
     
None
[1, 2, 3, 4]
Variable Scope

def g(y):
    print(x)
    print(x+1)
x = 5
g(x)
print(x)
     

def f(y):
    x = 1
    x += 1
    print(x)
x = 5
f(x)
print(x)
     

def h(y):
    x += 1
x = 5
h(x)
print(x)
     

def f(x):
   x = x + 1
   print('in f(x): x =', x)
   return x

x = 3
z = f(x)
print('in main program scope: z =', z)
print('in main program scope: x =', x)
     
Nested Functions

def f():
  def g():
    print('inside function g')
    f()
  g()
  print('inside function f')
     

f()
     
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
inside function g
---------------------------------------------------------------------------
RecursionError                            Traceback (most recent call last)
<ipython-input-92-c43e34e6d405> in <module>
----> 1 f()

<ipython-input-91-374a68ddd49e> in f()
      3     print('inside function g')
      4     f()
----> 5   g()
      6   print('inside function f')

<ipython-input-91-374a68ddd49e> in g()
      2   def g():
      3     print('inside function g')
----> 4     f()
      5   g()
      6   print('inside function f')

... last 2 frames repeated, from the frame below ...

<ipython-input-91-374a68ddd49e> in f()
      3     print('inside function g')
      4     f()
----> 5   g()
      6   print('inside function f')

RecursionError: maximum recursion depth exceeded

def g(x):
    def h():
        x = 'abc'
    x = x + 1
    print('in g(x): x =', x)
    h()
    return x

x = 3
z = g(x)
     

def g(x):
    def h(x):
        x = x+1
        print("in h(x): x = ", x)
    x = x + 1
    print('in g(x): x = ', x)
    h(x)
    return x

x = 3
z = g(x)
print('in main program scope: x = ', x)
print('in main program scope: z = ', z)
     


     
Functions are 1st class citizens

# type and id
def square(num):
  return num**2

type(square)

id(square)
     
140471717004784

# reassign
x = square
id(x)
x(3)
     
9

a = 2
b = a
b
     
2

# deleting a function
del square
     

square(3)
     
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-104-2cfd8bba3a88> in <module>
----> 1 square(3)

NameError: name 'square' is not defined

# storing
L = [1,2,3,4,square]
L[-1](3)
     
9

s = {square}
s
     
{<function __main__.square(num)>}

# returning a function

     


     

def f():
    def x(a, b):
        return a+b
    return x

val = f()(3,4)
print(val)
     

# function as argument
     

def func_a():
    print('inside func_a')

def func_b(z):
    print('inside func_c')
    return z()

print(func_b(func_a))
     
Benefits of using a Function
Code Modularity
Code Readibility
Code Reusability
Lambda Function
A lambda function is a small anonymous function.

A lambda function can take any number of arguments, but can only have one expression.

image.png


# x -> x^2
lambda x:x**2
     
<function __main__.<lambda>(x)>

# x,y -> x+y
a = lambda x,y:x+y
a(5,2)
     
7
Diff between lambda vs Normal Function
No name
lambda has no return value(infact,returns a function)
lambda is written in 1 line
not reusable
Then why use lambda functions?
They are used with HOF


# check if a string has 'a'
a = lambda s:'a' in s
a('hello')
     
False

# odd or even
a = lambda x:'even' if x%2 == 0 else 'odd'
a(6)
     
'even'
Higher Order Functions

# Example

def square(x):
  return x**2

def cube(x):
  return x**3

# HOF
def transform(f,L):
  output = []
  for i in L:
    output.append(f(i))

  print(output)

L = [1,2,3,4,5]

transform(lambda x:x**3,L)
     
[1, 8, 27, 64, 125]
Map

# square the items of a list
list(map(lambda x:x**2,[1,2,3,4,5]))
     
[1, 4, 9, 16, 25]

# odd/even labelling of list items
L = [1,2,3,4,5]
list(map(lambda x:'even' if x%2 == 0 else 'odd',L))
     
['odd', 'even', 'odd', 'even', 'odd']

# fetch names from a list of dict

users = [
    {
        'name':'Rahul',
        'age':45,
        'gender':'male'
    },
    {
        'name':'Nitish',
        'age':33,
        'gender':'male'
    },
    {
        'name':'Ankita',
        'age':50,
        'gender':'female'
    }
]

list(map(lambda users:users['gender'],users))
     
['male', 'male', 'female']
Filter

# numbers greater than 5
L = [3,4,5,6,7]

list(filter(lambda x:x>5,L))
     
[6, 7]

# fetch fruits starting with 'a'
fruits = ['apple','guava','cherry']

list(filter(lambda x:x.startswith('a'),fruits))
     
['apple']
Reduce

# sum of all item
import functools

functools.reduce(lambda x,y:x+y,[1,2,3,4,5])
     
15

# find min
functools.reduce(lambda x,y:x if x>y else y,[23,11,45,10,1])
     
45


     
##################################################################################





L = [1,2,3]

L.upper()
     
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-1-af1f83522ab7> in <module>
      1 L = [1,2,3]
      2 
----> 3 L.upper()

AttributeError: 'list' object has no attribute 'upper'

s = 'hello'
s.append('x')
     
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-2-2cb7c5babec0> in <module>
      1 s = 'hello'
----> 2 s.append('x')

AttributeError: 'str' object has no attribute 'append'

L = [1,2,3]
print(type(L))
     
<class 'list'>

s = [1,2,3]
     

# syntax to create an object

#objectname = classname()
     

# object literal
L = [1,2,3]
     

L = list()
L
     
[]

s = str()
s
     
''

# Pascal Case

HelloWorld
     

class Atm:

  # constructor(special function)->superpower ->
  def __init__(self):
    print(id(self))
    self.pin = ''
    self.balance = 0
    #self.menu()

  def menu(self):
    user_input = input("""
    Hi how can I help you?
    1. Press 1 to create pin
    2. Press 2 to change pin
    3. Press 3 to check balance
    4. Press 4 to withdraw
    5. Anything else to exit
    """)

    if user_input == '1':
      self.create_pin()
    elif user_input == '2':
      self.change_pin()
    elif user_input == '3':
      self.check_balance()
    elif user_input == '4':
      self.withdraw()
    else:
      exit()

  def create_pin(self):
    user_pin = input('enter your pin')
    self.pin = user_pin

    user_balance = int(input('enter balance'))
    self.balance = user_balance

    print('pin created successfully')
    self.menu()

  def change_pin():
    old_pin = input('enter old pin')

    if old_pin == self.pin:
      # let him change the pin
      new_pin = input('enter new pin')
      self.pin = new_pin
      print('pin change successful')
      self.menu()
    else:
      print('nai karne de sakta re baba')
      self.menu()

  def check_balance(self):
    user_pin = input('enter your pin')
    if user_pin == self.pin:
      print('your balance is ',self.balance)
    else:
      print('chal nikal yahan se')

  def withdraw(self):
    user_pin = input('enter the pin')
    if user_pin == self.pin:
      # allow to withdraw
      amount = int(input('enter the amount'))
      if amount <= self.balance:
        self.balance = self.balance - amount
        print('withdrawl successful.balance is',self.balance)
      else:
        print('abe garib')
    else:
      print('sale chor')
    self.menu()



     

obj1 = Atm()
     
140289660099024

id(obj1)
     
140289660099024

obj2 = Atm()
     
140289660586384

id(obj2)
     
140289660586384

L = [1,2,3]
len(L) # function ->bcos it is outside the list class
L.append()# method -> bcos it is inside the list class
     

class Temp:

  def __init__(self):
    print('hello')

obj = Temp()
     
hello

3/4*1/2
     
0.375

class Fraction:

  # parameterized constructor
  def __init__(self,x,y):
    self.num = x
    self.den = y

  def __str__(self):
    return '{}/{}'.format(self.num,self.den)

  def __add__(self,other):
    new_num = self.num*other.den + other.num*self.den
    new_den = self.den*other.den

    return '{}/{}'.format(new_num,new_den)

  def __sub__(self,other):
    new_num = self.num*other.den - other.num*self.den
    new_den = self.den*other.den

    return '{}/{}'.format(new_num,new_den)

  def __mul__(self,other):
    new_num = self.num*other.num
    new_den = self.den*other.den

    return '{}/{}'.format(new_num,new_den)

  def __truediv__(self,other):
    new_num = self.num*other.den
    new_den = self.den*other.num

    return '{}/{}'.format(new_num,new_den)

  def convert_to_decimal(self):
    return self.num/self.den





     

fr1 = Fraction(3,4)
fr2 = Fraction(1,2)
     

fr1.convert_to_decimal()
# 3/4
     
0.75

print(fr1 + fr2)
print(fr1 - fr2)
print(fr1 * fr2)
print(fr1 / fr2)
     
10/8
2/8
3/8
6/4

s1={1,2,3}
s2={3,4,5}

s1 + s2
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-32-3a417afc75fb> in <module>
      2 s2={3,4,5}
      3 
----> 4 s1 + s2

TypeError: unsupported operand type(s) for +: 'set' and 'set'

print(fr1 - fr2)
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-39-929bcd8b32dc> in <module>
----> 1 print(fr1 - fr2)

TypeError: unsupported operand type(s) for -: 'Fraction' and 'Fraction'


     
###################################################################################



Write OOP classes to handle the following scenarios:
A user can create and view 2D coordinates
A user can find out the distance between 2 coordinates
A user can find find the distance of a coordinate from origin
A user can check if a point lies on a given line
A user can find the distance between a given 2D point and a given line

class Point:

  def __init__(self,x,y):
    self.x_cod = x
    self.y_cod = y

  def __str__(self):
    return '<{},{}>'.format(self.x_cod,self.y_cod)

  def euclidean_distance(self,other):
    return ((self.x_cod - other.x_cod)**2 + (self.y_cod - other.y_cod)**2)**0.5

  def distance_from_origin(self):
    return (self.x_cod**2 + self.y_cod**2)**0.5
    # return self.euclidean_distance(Point(0,0))


class Line:

  def __init__(self,A,B,C):
    self.A = A
    self.B = B
    self.C = C

  def __str__(self):
    return '{}x + {}y + {} = 0'.format(self.A,self.B,self.C)

  def point_on_line(line,point):
    if line.A*point.x_cod + line.B*point.y_cod + line.C == 0:
      return "lies on the line"
    else:
      return "does not lie on the line"

  def shortest_distance(line,point):
    return abs(line.A*point.x_cod + line.B*point.y_cod + line.C)/(line.A**2 + line.B**2)**0.5

     

l1 = Line(1,1,-2)
p1 = Point(1,10)
print(l1)
print(p1)

l1.shortest_distance(p1)
     
1x + 1y + -2 = 0
<1,10>
6.363961030678928
How objects access attributes

class Person:

  def __init__(self,name_input,country_input):
    self.name = name_input
    self.country = country_input

  def greet(self):
    if self.country == 'india':
      print('Namaste',self.name)
    else:
      print('Hello',self.name)

     

# how to access attributes
p = Person('nitish','india')
     

p.name
     
'nitish'

# how to access methods
p.greet()
     
Namaste nitish

# what if i try to access non-existent attributes
p.gender
     
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-49-39388d77d830> in <module>
      1 # what if i try to access non-existent attributes
----> 2 p.gender

AttributeError: 'Person' object has no attribute 'gender'
Attribute creation from outside of the class

p.gender = 'male'
     

p.gender
     
'male'
Reference Variables
Reference variables hold the objects
We can create objects without reference variable as well
An object can have multiple reference variables
Assigning a new reference variable to an existing object does not create a new object

# object without a reference
class Person:

  def __init__(self):
    self.name = 'nitish'
    self.gender = 'male'

p = Person()
q = p
     

# Multiple ref
print(id(p))
print(id(q))
     
140655538334992
140655538334992

# change attribute value with the help of 2nd object
     

print(p.name)
print(q.name)
q.name = 'ankit'
print(q.name)
print(p.name)
     
nitish
nitish
ankit
ankit
Pass by reference

class Person:

  def __init__(self,name,gender):
    self.name = name
    self.gender = gender

# outside the class -> function
def greet(person):
  print('Hi my name is',person.name,'and I am a',person.gender)
  p1 = Person('ankit','male')
  return p1

p = Person('nitish','male')
x = greet(p)
print(x.name)
print(x.gender)
     
Hi my name is nitish and I am a male
ankit
male

class Person:

  def __init__(self,name,gender):
    self.name = name
    self.gender = gender

# outside the class -> function
def greet(person):
  print(id(person))
  person.name = 'ankit'
  print(person.name)

p = Person('nitish','male')
print(id(p))
greet(p)
print(p.name)
     
140655538334288
140655538334288
ankit
ankit
Object ki mutability

class Person:

  def __init__(self,name,gender):
    self.name = name
    self.gender = gender

# outside the class -> function
def greet(person):
  person.name = 'ankit'
  return person

p = Person('nitish','male')
print(id(p))
p1 = greet(p)
print(id(p1))
     
140655555218960
140655555218960
Encapsulation

# instance var -> python tutor
class Person:

  def __init__(self,name_input,country_input):
    self.name = name_input
    self.country = country_input

p1 = Person('nitish','india')
p2 = Person('steve','australia')
     

p2.name
     
'steve'

class Atm:

  # constructor(special function)->superpower ->
  def __init__(self):
    print(id(self))
    self.pin = ''
    self.__balance = 0
    #self.menu()

  def get_balance(self):
    return self.__balance

  def set_balance(self,new_value):
    if type(new_value) == int:
      self.__balance = new_value
    else:
      print('beta bahot maarenge')

  def __menu(self):
    user_input = input("""
    Hi how can I help you?
    1. Press 1 to create pin
    2. Press 2 to change pin
    3. Press 3 to check balance
    4. Press 4 to withdraw
    5. Anything else to exit
    """)

    if user_input == '1':
      self.create_pin()
    elif user_input == '2':
      self.change_pin()
    elif user_input == '3':
      self.check_balance()
    elif user_input == '4':
      self.withdraw()
    else:
      exit()

  def create_pin(self):
    user_pin = input('enter your pin')
    self.pin = user_pin

    user_balance = int(input('enter balance'))
    self.__balance = user_balance

    print('pin created successfully')

  def change_pin(self):
    old_pin = input('enter old pin')

    if old_pin == self.pin:
      # let him change the pin
      new_pin = input('enter new pin')
      self.pin = new_pin
      print('pin change successful')
    else:
      print('nai karne de sakta re baba')

  def check_balance(self):
    user_pin = input('enter your pin')
    if user_pin == self.pin:
      print('your balance is ',self.__balance)
    else:
      print('chal nikal yahan se')

  def withdraw(self):
    user_pin = input('enter the pin')
    if user_pin == self.pin:
      # allow to withdraw
      amount = int(input('enter the amount'))
      if amount <= self.__balance:
        self.__balance = self.__balance - amount
        print('withdrawl successful.balance is',self.__balance)
      else:
        print('abe garib')
    else:
      print('sale chor')
     

obj = Atm()
     
140655538526416

obj.get_balance()
     
1000

obj.set_balance(1000)
     

obj.withdraw()
     
enter the pin
enter the amount5000
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-93-826ea677aa70> in <module>
----> 1 obj.withdraw()

<ipython-input-86-f5bffac7e2a0> in withdraw(self)
     67       # allow to withdraw
     68       amount = int(input('enter the amount'))
---> 69       if amount <= self.__balance:
     70         self.__balance = self.__balance - amount
     71         print('withdrawl successful.balance is',self.__balance)

TypeError: '<=' not supported between instances of 'int' and 'str'
Collection of objects


     

# list of objects
class Person:

  def __init__(self,name,gender):
    self.name = name
    self.gender = gender

p1 = Person('nitish','male')
p2 = Person('ankit','male')
p3 = Person('ankita','female')

L = [p1,p2,p3]

for i in L:
  print(i.name,i.gender)
     
nitish male
ankit male
ankita female

# dict of objects
# list of objects
class Person:

  def __init__(self,name,gender):
    self.name = name
    self.gender = gender

p1 = Person('nitish','male')
p2 = Person('ankit','male')
p3 = Person('ankita','female')

d = {'p1':p1,'p2':p2,'p3':p3}

for i in d:
  print(d[i].gender)
     
male
male
female
Static Variables(Vs Instance variables)

# need for static vars
     

class Atm:

  __counter = 1

  # constructor(special function)->superpower ->
  def __init__(self):
    print(id(self))
    self.pin = ''
    self.__balance = 0
    self.cid = Atm.__counter
    Atm.__counter = Atm.__counter + 1
    #self.menu()

  # utility functions
  @staticmethod
  def get_counter():
    return Atm.__counter


  def get_balance(self):
    return self.__balance

  def set_balance(self,new_value):
    if type(new_value) == int:
      self.__balance = new_value
    else:
      print('beta bahot maarenge')

  def __menu(self):
    user_input = input("""
    Hi how can I help you?
    1. Press 1 to create pin
    2. Press 2 to change pin
    3. Press 3 to check balance
    4. Press 4 to withdraw
    5. Anything else to exit
    """)

    if user_input == '1':
      self.create_pin()
    elif user_input == '2':
      self.change_pin()
    elif user_input == '3':
      self.check_balance()
    elif user_input == '4':
      self.withdraw()
    else:
      exit()

  def create_pin(self):
    user_pin = input('enter your pin')
    self.pin = user_pin

    user_balance = int(input('enter balance'))
    self.__balance = user_balance

    print('pin created successfully')

  def change_pin(self):
    old_pin = input('enter old pin')

    if old_pin == self.pin:
      # let him change the pin
      new_pin = input('enter new pin')
      self.pin = new_pin
      print('pin change successful')
    else:
      print('nai karne de sakta re baba')

  def check_balance(self):
    user_pin = input('enter your pin')
    if user_pin == self.pin:
      print('your balance is ',self.__balance)
    else:
      print('chal nikal yahan se')

  def withdraw(self):
    user_pin = input('enter the pin')
    if user_pin == self.pin:
      # allow to withdraw
      amount = int(input('enter the amount'))
      if amount <= self.__balance:
        self.__balance = self.__balance - amount
        print('withdrawl successful.balance is',self.__balance)
      else:
        print('abe garib')
    else:
      print('sale chor')
     

c1 = Atm()
     
140655538287248

Atm.get_counter()
     
2

c3 = Atm()
     
140655538226704

c3.cid
     
3

Atm.counter
     
4
Static methods


     
Points to remember about static
Static attributes are created at class level.
Static attributes are accessed using ClassName.
Static attributes are object independent. We can access them without creating instance (object) of the class in which they are defined.
The value stored in static attribute is shared between all instances(objects) of the class in which the static attribute is defined.

class Lion:
  __water_source="well in the circus"

  def __init__(self,name, gender):
      self.__name=name
      self.__gender=gender

  def drinks_water(self):
      print(self.__name,
      "drinks water from the",Lion.__water_source)

  @staticmethod
  def get_water_source():
      return Lion.__water_source

simba=Lion("Simba","Male")
simba.drinks_water()
print( "Water source of lions:",Lion.get_water_source())
     

##################################################################################




Class Relationships
Aggregation
Inheritance
Aggregation(Has-A relationship)

# example
class Customer:

  def __init__(self,name,gender,address):
    self.name = name
    self.gender = gender
    self.address = address

  def print_address(self):
    print(self.address._Address__city,self.address.pin,self.address.state)

  def edit_profile(self,new_name,new_city,new_pin,new_state):
    self.name = new_name
    self.address.edit_address(new_city,new_pin,new_state)

class Address:

  def __init__(self,city,pin,state):
      self.__city = city
      self.pin = pin
      self.state = state

  def get_city(self):
    return self.__city

  def edit_address(self,new_city,new_pin,new_state):
    self.__city = new_city
    self.pin = new_pin
    self.state = new_state

add1 = Address('gurgaon',122011,'haryana')
cust = Customer('nitish','male',add1)

cust.print_address()

cust.edit_profile('ankit','mumbai',111111,'maharastra')
cust.print_address()
# method example
# what about private attribute
     
gurgaon 122011 haryana
mumbai 111111 maharastra
Aggregation class diagram


     
Inheritance
What is inheritance
Example
What gets inherited?

# Inheritance and it's benefits
     

# Example

# parent
class User:

  def __init__(self):
    self.name = 'nitish'
    self.gender = 'male'

  def login(self):
    print('login')

# child
class Student(User):

  def __init__(self):
    self.rollno = 100

  def enroll(self):
    print('enroll into the course')

u = User()
s = Student()

print(s.name)
s.login()
s.enroll()
     
nitish
login
enroll into the course

# Class diagram
     
What gets inherited?
Constructor
Non Private Attributes
Non Private Methods

# constructor example

class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    pass

s=SmartPhone(20000, "Apple", 13)
s.buy()
     
Inside phone constructor
Buying a phone

# constructor example 2

class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

class SmartPhone(Phone):
    def __init__(self, os, ram):
        self.os = os
        self.ram = ram
        print ("Inside SmartPhone constructor")

s=SmartPhone("Android", 2)
s.brand
     
Inside SmartPhone constructor
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-27-fff5c9f9674f> in <module>
     15 
     16 s=SmartPhone("Android", 2)
---> 17 s.brand

AttributeError: 'SmartPhone' object has no attribute 'brand'

# child can't access private members of the class

class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    #getter
    def show(self):
        print (self.__price)

class SmartPhone(Phone):
    def check(self):
        print(self.__price)

s=SmartPhone(20000, "Apple", 13)
s.show()
     
Inside phone constructor
20000

class Parent:

    def __init__(self,num):
        self.__num=num

    def get_num(self):
        return self.__num

class Child(Parent):

    def show(self):
        print("This is in child class")

son=Child(100)
print(son.get_num())
son.show()
     
100
This is in child class

class Parent:

    def __init__(self,num):
        self.__num=num

    def get_num(self):
        return self.__num

class Child(Parent):

    def __init__(self,val,num):
        self.__val=val

    def get_val(self):
        return self.__val

son=Child(100,10)
print("Parent: Num:",son.get_num())
print("Child: Val:",son.get_val())
     
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-35-5a17300f6fc7> in <module>
     16 
     17 son=Child(100,10)
---> 18 print("Parent: Num:",son.get_num())
     19 print("Child: Val:",son.get_val())

<ipython-input-35-5a17300f6fc7> in get_num(self)
      5 
      6     def get_num(self):
----> 7         return self.__num
      8 
      9 class Child(Parent):

AttributeError: 'Child' object has no attribute '_Parent__num'

class A:
    def __init__(self):
        self.var1=100

    def display1(self,var1):
        print("class A :", self.var1)
class B(A):

    def display2(self,var1):
        print("class B :", self.var1)

obj=B()
obj.display1(200)
     
class A : 200

# Method Overriding
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    def buy(self):
        print ("Buying a smartphone")

s=SmartPhone(20000, "Apple", 13)

s.buy()
     
Inside phone constructor
Buying a smartphone
Super Keyword

class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    def buy(self):
        print ("Buying a smartphone")
        # syntax to call parent ka buy method
        super().buy()

s=SmartPhone(20000, "Apple", 13)

s.buy()
     
Inside phone constructor
Buying a smartphone
Buying a phone

# using super outside the class
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    def buy(self):
        print ("Buying a smartphone")
        # syntax to call parent ka buy method
        super().buy()

s=SmartPhone(20000, "Apple", 13)

s.buy()
     
Inside phone constructor
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-42-b20080504d0e> in <module>
     17 s=SmartPhone(20000, "Apple", 13)
     18 
---> 19 super().buy()

RuntimeError: super(): no arguments

# can super access parent ka data?
# using super outside the class
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    def buy(self):
        print ("Buying a smartphone")
        # syntax to call parent ka buy method
        print(super().brand)

s=SmartPhone(20000, "Apple", 13)

s.buy()
     
Inside phone constructor
Buying a smartphone
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-43-87cd65570d46> in <module>
     19 s=SmartPhone(20000, "Apple", 13)
     20 
---> 21 s.buy()

<ipython-input-43-87cd65570d46> in buy(self)
     15         print ("Buying a smartphone")
     16         # syntax to call parent ka buy method
---> 17         print(super().brand)
     18 
     19 s=SmartPhone(20000, "Apple", 13)

AttributeError: 'super' object has no attribute 'brand'

# super -> constuctor
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

class SmartPhone(Phone):
    def __init__(self, price, brand, camera, os, ram):
        print('Inside smartphone constructor')
        super().__init__(price, brand, camera)
        self.os = os
        self.ram = ram
        print ("Inside smartphone constructor")

s=SmartPhone(20000, "Samsung", 12, "Android", 2)

print(s.os)
print(s.brand)
     
Inside smartphone constructor
Inside phone constructor
Inside smartphone constructor
Android
Samsung
Inheritance in summary
A class can inherit from another class.

Inheritance improves code reuse

Constructor, attributes, methods get inherited to the child class

The parent has no access to the child class

Private properties of parent are not accessible directly in child class

Child class can override the attributes or methods. This is called method overriding

super() is an inbuilt function which is used to invoke the parent class methods and constructor


class Parent:

    def __init__(self,num):
      self.__num=num

    def get_num(self):
      return self.__num

class Child(Parent):

    def __init__(self,num,val):
      super().__init__(num)
      self.__val=val

    def get_val(self):
      return self.__val

son=Child(100,200)
print(son.get_num())
print(son.get_val())
     
100
200

class Parent:
    def __init__(self):
        self.num=100

class Child(Parent):

    def __init__(self):
        super().__init__()
        self.var=200

    def show(self):
        print(self.num)
        print(self.var)

son=Child()
son.show()
     
100
200

class Parent:
    def __init__(self):
        self.__num=100

    def show(self):
        print("Parent:",self.__num)

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.__var=10

    def show(self):
        print("Child:",self.__var)

obj=Child()
obj.show()
     
Child: 10

class Parent:
    def __init__(self):
        self.__num=100

    def show(self):
        print("Parent:",self.__num)

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.__var=10

    def show(self):
        print("Child:",self.__var)

obj=Child()
obj.show()
     
Child: 10
Types of Inheritance
Single Inheritance
Multilevel Inheritance
Hierarchical Inheritance
Multiple Inheritance(Diamond Problem)
Hybrid Inheritance

# single inheritance
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    pass

SmartPhone(1000,"Apple","13px").buy()
     
Inside phone constructor
Buying a phone

# multilevel
class Product:
    def review(self):
        print ("Product customer review")

class Phone(Product):
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    pass

s=SmartPhone(20000, "Apple", 12)

s.buy()
s.review()
     
Inside phone constructor
Buying a phone
Product customer review

# Hierarchical
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class SmartPhone(Phone):
    pass

class FeaturePhone(Phone):
    pass

SmartPhone(1000,"Apple","13px").buy()
FeaturePhone(10,"Lava","1px").buy()
     
Inside phone constructor
Buying a phone
Inside phone constructor
Buying a phone

# Multiple
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class Product:
    def review(self):
        print ("Customer review")

class SmartPhone(Phone, Product):
    pass

s=SmartPhone(20000, "Apple", 12)

s.buy()
s.review()

     
Inside phone constructor
Buying a phone
Customer review

# the diamond problem
# https://stackoverflow.com/questions/56361048/what-is-the-diamond-problem-in-python-and-why-its-not-appear-in-python2
class Phone:
    def __init__(self, price, brand, camera):
        print ("Inside phone constructor")
        self.__price = price
        self.brand = brand
        self.camera = camera

    def buy(self):
        print ("Buying a phone")

class Product:
    def buy(self):
        print ("Product buy method")

# Method resolution order
class SmartPhone(Phone,Product):
    pass

s=SmartPhone(20000, "Apple", 12)

s.buy()
     
Inside phone constructor
Buying a phone

class A:

    def m1(self):
        return 20

class B(A):

    def m1(self):
        return 30

    def m2(self):
        return 40

class C(B):

    def m2(self):
        return 20
obj1=A()
obj2=B()
obj3=C()
print(obj1.m1() + obj3.m1()+ obj3.m2())
     
70

class A:

    def m1(self):
        return 20

class B(A):

    def m1(self):
        val=super().m1()+30
        return val

class C(B):

    def m1(self):
        val=self.m1()+20
        return val
obj=C()
print(obj.m1())
     
---------------------------------------------------------------------------
RecursionError                            Traceback (most recent call last)
<ipython-input-56-bb3659d52487> in <module>
     16         return val
     17 obj=C()
---> 18 print(obj.m1())

<ipython-input-56-bb3659d52487> in m1(self)
     13 
     14     def m1(self):
---> 15         val=self.m1()+20
     16         return val
     17 obj=C()

... last 1 frames repeated, from the frame below ...

<ipython-input-56-bb3659d52487> in m1(self)
     13 
     14     def m1(self):
---> 15         val=self.m1()+20
     16         return val
     17 obj=C()

RecursionError: maximum recursion depth exceeded
Polymorphism
Method Overriding
Method Overloading
Operator Overloading

class Shape:

  def area(self,a,b=0):
    if b == 0:
      return 3.14*a*a
    else:
      return a*b

s = Shape()

print(s.area(2))
print(s.area(3,4))
     
12.56
12

'hello' + 'world'
     
'helloworld'

4 + 5
     
9

[1,2,3] + [4,5]
     
[1, 2, 3, 4, 5]
Abstraction

from abc import ABC,abstractmethod
class BankApp(ABC):

  def database(self):
    print('connected to database')

  @abstractmethod
  def security(self):
    pass

  @abstractmethod
  def display(self):
    pass

     

class MobileApp(BankApp):

  def mobile_login(self):
    print('login into mobile')

  def security(self):
    print('mobile security')

  def display(self):
    print('display')
     

mob = MobileApp()
     

mob.security()
     
mobile security

obj = BankApp()
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-24-0aa75fd04378> in <module>
----> 1 obj = BankApp()

TypeError: Can't instantiate abstract class BankApp with abstract methods display, security


     
##################################################################################



Some Theory
Types of data used for I/O:
Text - '12345' as a sequence of unicode chars
Binary - 12345 as a sequence of bytes of its binary equivalent
Hence there are 2 file types to deal with
Text files - All program files are text files
Binary Files - Images,music,video,exe files
How File I/O is done in most programming languages
Open a file
Read/Write data
Close the file
Writing to a file

# case 1 - if the file is not present
f = open('sample.txt','w')
f.write('Hello world')
f.close()
# since file is closed hence this will not work
f.write('hello')
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-109-c02a4a856526> in <module>
      3 f.write('Hello world')
      4 f.close()
----> 5 f.write('hello')

ValueError: I/O operation on closed file.

# write multiline strings
f = open('sample1.txt','w')
f.write('hello world')
f.write('\nhow are you?')
f.close()
     

# case 2 - if the file is already present
f = open('sample.txt','w')
f.write('salman khan')
f.close()
     

# how exactly open() works?
     

# Problem with w mode
# introducing append mode
f = open('/content/sample1.txt','a')
f.write('\nI am fine')
f.close()
     

# write lines
L = ['hello\n','hi\n','how are you\n','I am fine']

f = open('/content/temp/sample.txt','w')
f.writelines(L)
f.close()
     

# reading from files
# -> using read()
f = open('/content/sample.txt','r')
s = f.read()
print(s)
f.close()
     
hello
hi
how are you
I am fine

# reading upto n chars
f = open('/content/sample.txt','r')
s = f.read(10)
print(s)
f.close()
     
hello
hi
h

# readline() -> to read line by line
f = open('/content/sample.txt','r')
print(f.readline(),end='')
print(f.readline(),end='')
f.close()
     
hello
hi

# reading entire using readline
f = open('/content/sample.txt','r')

while True:

  data = f.readline()

  if data == '':
    break
  else:
    print(data,end='')

f.close()
     
hello
hi
how are you
I am fine
Using Context Manager (With)
It's a good idea to close a file after usage as it will free up the resources
If we dont close it, garbage collector would close it
with keyword closes the file as soon as the usage is over

# with
with open('/content/sample1.txt','w') as f:
  f.write('selmon bhai')
     

f.write('hello')
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-4-00cba062fa3d> in <module>
----> 1 f.write('hello')

ValueError: I/O operation on closed file.

# try f.read() now
with open('/content/sample.txt','r') as f:
  print(f.readline())
     
hello


# moving within a file -> 10 char then 10 char
with open('sample.txt','r') as f:
  print(f.read(10))
  print(f.read(10))
  print(f.read(10))
  print(f.read(10))
     
hello
hi
h
ow are you

I am fine


# benefit? -> to load a big file in memory
big_L = ['hello world ' for i in range(1000)]

with open('big.txt','w') as f:
  f.writelines(big_L)

     

with open('big.txt','r') as f:

  chunk_size = 10

  while len(f.read(chunk_size)) > 0:
    print(f.read(chunk_size),end='***')
    f.read(chunk_size)
     
d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***

# seek and tell function
with open('sample.txt','r') as f:
  f.seek(15)
  print(f.read(10))
  print(f.tell())

  print(f.read(10))
  print(f.tell())
     
e you
I am
25
 fine
30

# seek during write
with open('sample.txt','w') as f:
  f.write('Hello')
  f.seek(0)
  f.write('Xa')
     
Problems with working in text mode
can't work with binary files like images
not good for other data types like int/float/list/tuples

# working with binary file
with open('screenshot1.png','r') as f:
  f.read()
     
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
<ipython-input-23-b662b4ad1a91> in <module>
      1 # working with binary file
      2 with open('screenshot1.png','r') as f:
----> 3   f.read()

/usr/lib/python3.7/codecs.py in decode(self, input, final)
    320         # decode input (taking the buffer into account)
    321         data = self.buffer + input
--> 322         (result, consumed) = self._buffer_decode(data, self.errors, final)
    323         # keep undecoded input until the next call
    324         self.buffer = data[consumed:]

UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte

# working with binary file
with open('screenshot1.png','rb') as f:
  with open('screenshot_copy.png','wb') as wf:
    wf.write(f.read())
     

# working with a big binary file
     

# working with other data types
with open('sample.txt','w') as f:
  f.write(5)
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-26-a8e7a73b1431> in <module>
      1 # working with other data types
      2 with open('sample.txt','w') as f:
----> 3   f.write(5)

TypeError: write() argument must be str, not int

with open('sample.txt','w') as f:
  f.write('5')
     

with open('sample.txt','r') as f:
  print(int(f.read()) + 5)
     
10


     

# more complex data
d = {
    'name':'nitish',
     'age':33,
     'gender':'male'
}

with open('sample.txt','w') as f:
  f.write(str(d))
     

with open('sample.txt','r') as f:
  print(dict(f.read()))
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-34-949b64f1fbe0> in <module>
      1 with open('sample.txt','r') as f:
----> 2   print(dict(f.read()))

ValueError: dictionary update sequence element #0 has length 1; 2 is required
Serialization and Deserialization
Serialization - process of converting python data types to JSON format
Deserialization - process of converting JSON to python data types
What is JSON?
image.png


# serialization using json module
# list
import json

L = [1,2,3,4]

with open('demo.json','w') as f:
  json.dump(L,f)

     

# dict
d = {
    'name':'nitish',
     'age':33,
     'gender':'male'
}

with open('demo.json','w') as f:
  json.dump(d,f,indent=4)
     

# deserialization
import json

with open('demo.json','r') as f:
  d = json.load(f)
  print(d)
  print(type(d))
     
{'name': 'nitish', 'age': 33, 'gender': 'male'}
<class 'dict'>

# serialize and deserialize tuple
import json

t = (1,2,3,4,5)

with open('demo.json','w') as f:
  json.dump(t,f)
     

# serialize and deserialize a nested dict

d = {
    'student':'nitish',
     'marks':[23,14,34,45,56]
}

with open('demo.json','w') as f:
  json.dump(d,f)
     
Serializing and Deserializing custom objects

class Person:

  def __init__(self,fname,lname,age,gender):
    self.fname = fname
    self.lname = lname
    self.age = age
    self.gender = gender

# format to printed in
# -> Nitish Singh age -> 33 gender -> male
     

person = Person('Nitish','Singh',33,'male')
     

# As a string
import json

def show_object(person):
  if isinstance(person,Person):
    return "{} {} age -> {} gender -> {}".format(person.fname,person.lname,person.age,person.gender)

with open('demo.json','w') as f:
  json.dump(person,f,default=show_object)
     

# As a dict
import json

def show_object(person):
  if isinstance(person,Person):
    return {'name':person.fname + ' ' + person.lname,'age':person.age,'gender':person.gender}

with open('demo.json','w') as f:
  json.dump(person,f,default=show_object,indent=4)
     

# indent arrtribute
# As a dict
     

# deserializing
import json

with open('demo.json','r') as f:
  d = json.load(f)
  print(d)
  print(type(d))
     
{'name': 'Nitish Singh', 'age': 33, 'gender': 'male'}
<class 'dict'>
Pickling
Pickling is the process whereby a Python object hierarchy is converted into a byte stream, and unpickling is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.


class Person:

  def __init__(self,name,age):
    self.name = name
    self.age = age

  def display_info(self):
    print('Hi my name is',self.name,'and I am ',self.age,'years old')
     

p = Person('nitish',33)

     

# pickle dump
import pickle
with open('person.pkl','wb') as f:
  pickle.dump(p,f)
     

# pickle load
import pickle
with open('person.pkl','rb') as f:
  p = pickle.load(f)

p.display_info()
     
Hi my name is nitish and I am  33 years old
Pickle Vs Json
Pickle lets the user to store data in binary format. JSON lets the user store data in a human-readable text format.


     


     
###################################################################################



Some Theory
Types of data used for I/O:
Text - '12345' as a sequence of unicode chars
Binary - 12345 as a sequence of bytes of its binary equivalent
Hence there are 2 file types to deal with
Text files - All program files are text files
Binary Files - Images,music,video,exe files
How File I/O is done in most programming languages
Open a file
Read/Write data
Close the file
Writing to a file

# case 1 - if the file is not present
f = open('sample.txt','w')
f.write('Hello world')
f.close()
# since file is closed hence this will not work
f.write('hello')
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-109-c02a4a856526> in <module>
      3 f.write('Hello world')
      4 f.close()
----> 5 f.write('hello')

ValueError: I/O operation on closed file.

# write multiline strings
f = open('sample1.txt','w')
f.write('hello world')
f.write('\nhow are you?')
f.close()
     

# case 2 - if the file is already present
f = open('sample.txt','w')
f.write('salman khan')
f.close()
     

# how exactly open() works?
     

# Problem with w mode
# introducing append mode
f = open('/content/sample1.txt','a')
f.write('\nI am fine')
f.close()
     

# write lines
L = ['hello\n','hi\n','how are you\n','I am fine']

f = open('/content/temp/sample.txt','w')
f.writelines(L)
f.close()
     

# reading from files
# -> using read()
f = open('/content/sample.txt','r')
s = f.read()
print(s)
f.close()
     
hello
hi
how are you
I am fine

# reading upto n chars
f = open('/content/sample.txt','r')
s = f.read(10)
print(s)
f.close()
     
hello
hi
h

# readline() -> to read line by line
f = open('/content/sample.txt','r')
print(f.readline(),end='')
print(f.readline(),end='')
f.close()
     
hello
hi

# reading entire using readline
f = open('/content/sample.txt','r')

while True:

  data = f.readline()

  if data == '':
    break
  else:
    print(data,end='')

f.close()
     
hello
hi
how are you
I am fine
Using Context Manager (With)
It's a good idea to close a file after usage as it will free up the resources
If we dont close it, garbage collector would close it
with keyword closes the file as soon as the usage is over

# with
with open('/content/sample1.txt','w') as f:
  f.write('selmon bhai')
     

f.write('hello')
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-4-00cba062fa3d> in <module>
----> 1 f.write('hello')

ValueError: I/O operation on closed file.

# try f.read() now
with open('/content/sample.txt','r') as f:
  print(f.readline())
     
hello


# moving within a file -> 10 char then 10 char
with open('sample.txt','r') as f:
  print(f.read(10))
  print(f.read(10))
  print(f.read(10))
  print(f.read(10))
     
hello
hi
h
ow are you

I am fine


# benefit? -> to load a big file in memory
big_L = ['hello world ' for i in range(1000)]

with open('big.txt','w') as f:
  f.writelines(big_L)

     

with open('big.txt','r') as f:

  chunk_size = 10

  while len(f.read(chunk_size)) > 0:
    print(f.read(chunk_size),end='***')
    f.read(chunk_size)
     
d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***d hello wo***o world he***

# seek and tell function
with open('sample.txt','r') as f:
  f.seek(15)
  print(f.read(10))
  print(f.tell())

  print(f.read(10))
  print(f.tell())
     
e you
I am
25
 fine
30

# seek during write
with open('sample.txt','w') as f:
  f.write('Hello')
  f.seek(0)
  f.write('Xa')
     
Problems with working in text mode
can't work with binary files like images
not good for other data types like int/float/list/tuples

# working with binary file
with open('screenshot1.png','r') as f:
  f.read()
     
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
<ipython-input-23-b662b4ad1a91> in <module>
      1 # working with binary file
      2 with open('screenshot1.png','r') as f:
----> 3   f.read()

/usr/lib/python3.7/codecs.py in decode(self, input, final)
    320         # decode input (taking the buffer into account)
    321         data = self.buffer + input
--> 322         (result, consumed) = self._buffer_decode(data, self.errors, final)
    323         # keep undecoded input until the next call
    324         self.buffer = data[consumed:]

UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte

# working with binary file
with open('screenshot1.png','rb') as f:
  with open('screenshot_copy.png','wb') as wf:
    wf.write(f.read())
     

# working with a big binary file
     

# working with other data types
with open('sample.txt','w') as f:
  f.write(5)
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-26-a8e7a73b1431> in <module>
      1 # working with other data types
      2 with open('sample.txt','w') as f:
----> 3   f.write(5)

TypeError: write() argument must be str, not int

with open('sample.txt','w') as f:
  f.write('5')
     

with open('sample.txt','r') as f:
  print(int(f.read()) + 5)
     
10


     

# more complex data
d = {
    'name':'nitish',
     'age':33,
     'gender':'male'
}

with open('sample.txt','w') as f:
  f.write(str(d))
     

with open('sample.txt','r') as f:
  print(dict(f.read()))
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-34-949b64f1fbe0> in <module>
      1 with open('sample.txt','r') as f:
----> 2   print(dict(f.read()))

ValueError: dictionary update sequence element #0 has length 1; 2 is required
Serialization and Deserialization
Serialization - process of converting python data types to JSON format
Deserialization - process of converting JSON to python data types
What is JSON?
image.png


# serialization using json module
# list
import json

L = [1,2,3,4]

with open('demo.json','w') as f:
  json.dump(L,f)

     

# dict
d = {
    'name':'nitish',
     'age':33,
     'gender':'male'
}

with open('demo.json','w') as f:
  json.dump(d,f,indent=4)
     

# deserialization
import json

with open('demo.json','r') as f:
  d = json.load(f)
  print(d)
  print(type(d))
     
{'name': 'nitish', 'age': 33, 'gender': 'male'}
<class 'dict'>

# serialize and deserialize tuple
import json

t = (1,2,3,4,5)

with open('demo.json','w') as f:
  json.dump(t,f)
     

# serialize and deserialize a nested dict

d = {
    'student':'nitish',
     'marks':[23,14,34,45,56]
}

with open('demo.json','w') as f:
  json.dump(d,f)
     
Serializing and Deserializing custom objects

class Person:

  def __init__(self,fname,lname,age,gender):
    self.fname = fname
    self.lname = lname
    self.age = age
    self.gender = gender

# format to printed in
# -> Nitish Singh age -> 33 gender -> male
     

person = Person('Nitish','Singh',33,'male')
     

# As a string
import json

def show_object(person):
  if isinstance(person,Person):
    return "{} {} age -> {} gender -> {}".format(person.fname,person.lname,person.age,person.gender)

with open('demo.json','w') as f:
  json.dump(person,f,default=show_object)
     

# As a dict
import json

def show_object(person):
  if isinstance(person,Person):
    return {'name':person.fname + ' ' + person.lname,'age':person.age,'gender':person.gender}

with open('demo.json','w') as f:
  json.dump(person,f,default=show_object,indent=4)
     

# indent arrtribute
# As a dict
     

# deserializing
import json

with open('demo.json','r') as f:
  d = json.load(f)
  print(d)
  print(type(d))
     
{'name': 'Nitish Singh', 'age': 33, 'gender': 'male'}
<class 'dict'>
Pickling
Pickling is the process whereby a Python object hierarchy is converted into a byte stream, and unpickling is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.


class Person:

  def __init__(self,name,age):
    self.name = name
    self.age = age

  def display_info(self):
    print('Hi my name is',self.name,'and I am ',self.age,'years old')
     

p = Person('nitish',33)

     

# pickle dump
import pickle
with open('person.pkl','wb') as f:
  pickle.dump(p,f)
     

# pickle load
import pickle
with open('person.pkl','rb') as f:
  p = pickle.load(f)

p.display_info()
     
Hi my name is nitish and I am  33 years old
Pickle Vs Json
Pickle lets the user to store data in binary format. JSON lets the user store data in a human-readable text format.


     


     
###################################################################################



There are 2 stages where error may happen in a program

During compilation -> Syntax Error
During execution -> Exceptions
Syntax Error
Something in the program is not written according to the program grammar.
Error is raised by the interpreter/compiler
You can solve it by rectifying the program

# Examples of syntax error
print 'hello world'
     
  File "<ipython-input-3-4655b84ba7b7>", line 2
    print 'hello world'
                      ^
SyntaxError: Missing parentheses in call to 'print'. Did you mean print('hello world')?
Other examples of syntax error
Leaving symbols like colon,brackets
Misspelling a keyword
Incorrect indentation
empty if/else/loops/class/functions

a = 5
if a==3
  print('hello')
     
  File "<ipython-input-68-efc58c10458d>", line 2
    if a==3
           ^
SyntaxError: invalid syntax

a = 5
iff a==3:
  print('hello')
     
  File "<ipython-input-69-d1e6fae154d5>", line 2
    iff a==3:
        ^
SyntaxError: invalid syntax

a = 5
if a==3:
print('hello')
     
  File "<ipython-input-70-ccc702dc036c>", line 3
    print('hello')
        ^
IndentationError: expected an indented block

# IndexError
# The IndexError is thrown when trying to access an item at an invalid index.
L = [1,2,3]
L[100]
     
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-71-c90668d2b194> in <module>
      2 # The IndexError is thrown when trying to access an item at an invalid index.
      3 L = [1,2,3]
----> 4 L[100]

IndexError: list index out of range

# ModuleNotFoundError
# The ModuleNotFoundError is thrown when a module could not be found.
import mathi
math.floor(5.3)
     
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-73-cbdaf00191df> in <module>
      1 # ModuleNotFoundError
      2 # The ModuleNotFoundError is thrown when a module could not be found.
----> 3 import mathi
      4 math.floor(5.3)

ModuleNotFoundError: No module named 'mathi'

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
---------------------------------------------------------------------------

# KeyError
# The KeyError is thrown when a key is not found

d = {'name':'nitish'}
d['age']
     
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-74-453afa1c9765> in <module>
      3 
      4 d = {'name':'nitish'}
----> 5 d['age']

KeyError: 'age'

# TypeError
# The TypeError is thrown when an operation or function is applied to an object of an inappropriate type.
1 + 'a'
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-78-2a3eb3f5bb0a> in <module>
      1 # TypeError
      2 # The TypeError is thrown when an operation or function is applied to an object of an inappropriate type.
----> 3 1 + 'a'

TypeError: unsupported operand type(s) for +: 'int' and 'str'

# ValueError
# The ValueError is thrown when a function's argument is of an inappropriate type.
int('a')
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-76-e419d2a084b4> in <module>
      1 # ValueError
      2 # The ValueError is thrown when a function's argument is of an inappropriate type.
----> 3 int('a')

ValueError: invalid literal for int() with base 10: 'a'

# NameError
# The NameError is thrown when an object could not be found.
print(k)
     
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-79-e3e8aaa4ec45> in <module>
      1 # NameError
      2 # The NameError is thrown when an object could not be found.
----> 3 print(k)

NameError: name 'k' is not defined

# AttributeError
L = [1,2,3]
L.upper()

# Stacktrace
     
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-80-dd5a29625ddc> in <module>
      1 # AttributeError
      2 L = [1,2,3]
----> 3 L.upper()

AttributeError: 'list' object has no attribute 'upper'
Exceptions
If things go wrong during the execution of the program(runtime). It generally happens when something unforeseen has happened.

Exceptions are raised by python runtime
You have to takle is on the fly
Examples
Memory overflow
Divide by 0 -> logical error
Database error

# Why is it important to handle exceptions
# how to handle exceptions
# -> Try except block
     

# let's create a file
with open('sample.txt','w') as f:
  f.write('hello world')
     

# try catch demo
try:
  with open('sample1.txt','r') as f:
    print(f.read())
except:
  print('sorry file not found')
     
sorry file not found

# catching specific exception
try:
  m=5
  f = open('sample1.txt','r')
  print(f.read())
  print(m)
  print(5/2)
  L = [1,2,3]
  L[100]
except FileNotFoundError:
  print('file not found')
except NameError:
  print('variable not defined')
except ZeroDivisionError:
  print("can't divide by 0")
except Exception as e:
  print(e)
     
[Errno 2] No such file or directory: 'sample1.txt'

# else
try:
  f = open('sample1.txt','r')
except FileNotFoundError:
  print('file nai mili')
except Exception:
  print('kuch to lafda hai')
else:
  print(f.read())


     
file nai mili

# finally
# else
try:
  f = open('sample1.txt','r')
except FileNotFoundError:
  print('file nai mili')
except Exception:
  print('kuch to lafda hai')
else:
  print(f.read())
finally:
  print('ye to print hoga hi')
     
file nai mili
ye to print hoga hi

# raise Exception
# In Python programming, exceptions are raised when errors occur at runtime.
# We can also manually raise exceptions using the raise keyword.

# We can optionally pass values to the exception to clarify why that exception was raised
     

raise ZeroDivisionError('aise hi try kar raha hu')
# Java
# try -> try
# except -> catch
# raise -> throw
     
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-106-5a07d7d89433> in <module>
----> 1 raise ZeroDivisionError('aise hi try kar raha hu')

ZeroDivisionError: aise hi try kar raha hu

class Bank:

  def __init__(self,balance):
    self.balance = balance

  def withdraw(self,amount):
    if amount < 0:
      raise Exception('amount cannot be -ve')
    if self.balance < amount:
      raise Exception('paise nai hai tere paas')
    self.balance = self.balance - amount

obj = Bank(10000)
try:
  obj.withdraw(15000)
except Exception as e:
  print(e)
else:
  print(obj.balance)
     
paise nai hai tere paas

class MyException(Exception):
  def __init__(self,message):
    print(message)

class Bank:

  def __init__(self,balance):
    self.balance = balance

  def withdraw(self,amount):
    if amount < 0:
      raise MyException('amount cannot be -ve')
    if self.balance < amount:
      raise MyException('paise nai hai tere paas')
    self.balance = self.balance - amount

obj = Bank(10000)
try:
  obj.withdraw(5000)
except MyException as e:
  pass
else:
  print(obj.balance)
     
5000



# creating custom exceptions
# exception hierarchy in python
     

# simple example
     

class SecurityError(Exception):

  def __init__(self,message):
    print(message)

  def logout(self):
    print('logout')

class Google:

  def __init__(self,name,email,password,device):
    self.name = name
    self.email = email
    self.password = password
    self.device = device

  def login(self,email,password,device):
    if device != self.device:
      raise SecurityError('bhai teri to lag gayi')
    if email == self.email and password == self.password:
      print('welcome')
    else:
      print('login error')



obj = Google('nitish','nitish@gmail.com','1234','android')

try:
  obj.login('nitish@gmail.com','1234','windows')
except SecurityError as e:
  e.logout()
else:
  print(obj.name)
finally:
  print('database connection closed')


     
bhai teri to lag gayi
logout
database connection closed


     
######################################################################################




Namespaces
A namespace is a space that holds names(identifiers).Programmatically speaking, namespaces are dictionary of identifiers(keys) and their objects(values)

There are 4 types of namespaces:

Builtin Namespace
Global Namespace
Enclosing Namespace
Local Namespace
Scope and LEGB Rule
A scope is a textual region of a Python program where a namespace is directly accessible.

The interpreter searches for a name from the inside out, looking in the local, enclosing, global, and finally the built-in scope. If the interpreter doesn’t find the name in any of these locations, then Python raises a NameError exception.


# local and global
# global var
a = 2

def temp():
  # local var
  b = 3
  print(b)

temp()
print(a)
     
3
2

# local and global -> same name
a = 2

def temp():
  # local var
  a = 3
  print(b)

temp()
print(a)
     

# local and global -> local does not have but global has
a = 2

def temp():
  # local var
  print(a)

temp()
print(a)

     
2
2

# local and global -> editing global
a = 2

def temp():
  # local var
  a += 1
  print(a)

temp()
print(a)
     
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
<ipython-input-49-0bff4ae6448f> in <module>
      7   print(a)
      8 
----> 9 temp()
     10 print(a)

<ipython-input-49-0bff4ae6448f> in temp()
      4 def temp():
      5   # local var
----> 6   a += 1
      7   print(a)
      8 

UnboundLocalError: local variable 'a' referenced before assignment

a = 2

def temp():
  # local var
  global a
  a += 1
  print(a)

temp()
print(a)
     
3
3

# local and global -> global created inside local
def temp():
  # local var
  global a
  a = 1
  print(a)

temp()
print(a)
     

# local and global -> function parameter is local
def temp(z):
  # local var
  print(z)

a = 5
temp(5)
print(a)
print(z)
     
5
5
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-51-aac3f4d9657f> in <module>
      7 temp(5)
      8 print(a)
----> 9 print(z)

NameError: name 'z' is not defined

# built-in scope
import builtins
print(dir(builtins))
     
['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning', 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning', 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FileExistsError', 'FileNotFoundError', 'FloatingPointError', 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt', 'LookupError', 'MemoryError', 'ModuleNotFoundError', 'NameError', 'None', 'NotADirectoryError', 'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError', 'RecursionError', 'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning', 'StopAsyncIteration', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'TimeoutError', 'True', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning', 'ZeroDivisionError', '__IPYTHON__', '__build_class__', '__debug__', '__doc__', '__import__', '__loader__', '__name__', '__package__', '__spec__', 'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'copyright', 'credits', 'delattr', 'dict', 'dir', 'display', 'divmod', 'enumerate', 'eval', 'exec', 'execfile', 'filter', 'float', 'format', 'frozenset', 'get_ipython', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'runfile', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip']

# how to see all the built-ins
     

# renaming built-ins
L = [1,2,3]
print(max(L))
def max():
  print('hello')

print(max(L))
     
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-68-c19f3451a38f> in <module>
      1 # renaming built-ins
      2 L = [1,2,3]
----> 3 print(max(L))
      4 def max():
      5   print('hello')

TypeError: max() takes 0 positional arguments but 1 was given

# Enclosing scope
def outer():
  def inner():
    print(a)
  inner()
  print('outer function')


outer()
print('main program')
     
1
outer function
main program

# nonlocal keyword
def outer():
  a = 1
  def inner():
    nonlocal a
    a += 1
    print('inner',a)
  inner()
  print('outer',a)


outer()
print('main program')
     
inner 2
outer 2
main program

# Summary
     
Decorators
A decorator in python is a function that receives another function as input and adds some functionality(decoration) to and it and returns it.

This can happen only because python functions are 1st class citizens.

There are 2 types of decorators available in python

Built in decorators like @staticmethod, @classmethod, @abstractmethod and @property etc
User defined decorators that we programmers can create according to our needs

# Python are 1st class function

def modify(func,num):
  return func(num)

def square(num):
  return num**2

modify(square,2)
     
4

# simple example

def my_decorator(func):
  def wrapper():
    print('***********************')
    func()
    print('***********************')
  return wrapper

def hello():
  print('hello')

def display():
  print('hello nitish')

a = my_decorator(hello)
a()

b = my_decorator(display)
b()
     
***********************
hello
***********************
***********************
hello nitish
***********************

# more functions
     

# how this works -> closure?
     

# python tutor
     

# Better syntax?
# simple example

def my_decorator(func):
  def wrapper():
    print('***********************')
    func()
    print('***********************')
  return wrapper

@my_decorator
def hello():
  print('hello')

hello()
     
***********************
hello
***********************

# anything meaningful?
import time

def timer(func):
  def wrapper(*args):
    start = time.time()
    func(*args)
    print('time taken by',func.__name__,time.time()-start,'secs')
  return wrapper

@timer
def hello():
  print('hello wolrd')
  time.sleep(2)

@timer
def square(num):
  time.sleep(1)
  print(num**2)

@timer
def power(a,b):
  print(a**b)

hello()
square(2)
power(2,3)

     
hello wolrd
time taken by hello 2.003671884536743 secs
4
time taken by square 1.0009939670562744 secs
8
time taken by power 2.1696090698242188e-05 secs

# A big problem
     

# One last example -> decorators with arguments

     

@checkdt(int)
def square(num):
  print(num**2)
     

def sanity_check(data_type):
  def outer_wrapper(func):
    def inner_wrapper(*args):
      if type(*args) == data_type:
        func(*args)
      else:
        raise TypeError('Ye datatype nai chalega')
    return inner_wrapper
  return outer_wrapper

@sanity_check(int)
def square(num):
  print(num**2)

@sanity_check(str)
def greet(name):
  print('hello',name)

square(2)
     
4


     


     
##################################################################################