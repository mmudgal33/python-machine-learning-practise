# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:17:24 2025

@author: Radha Sharma
"""


Types of Data
Numerical Data
Categorical Data

# import the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
     
2D Line plot
image.png

Bivariate Analysis
categorical -> numerical and numerical -> numerical
Use case - Time series data

# plotting a simple function
price = [48000,54000,57000,49000,47000,45000]
year = [2015,2016,2017,2018,2019,2020]

plt.plot(year,price)
     
[<matplotlib.lines.Line2D at 0x7fb5d79e54f0>]


# from a pandas dataframe
batsman = pd.read_csv('/content/sharma-kohli.csv')
batsman

plt.plot(batsman['index'],batsman['V Kohli'])
     
[<matplotlib.lines.Line2D at 0x7fb5d682a220>]


# plotting multiple plots
plt.plot(batsman['index'],batsman['V Kohli'])
plt.plot(batsman['index'],batsman['RG Sharma'])
     
[<matplotlib.lines.Line2D at 0x7fb5d66f6fa0>]


# labels title
plt.plot(batsman['index'],batsman['V Kohli'])
plt.plot(batsman['index'],batsman['RG Sharma'])

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')
     
Text(0, 0.5, 'Runs Scored')


# colors(hex) and line(width and style) and marker(size)
plt.plot(batsman['index'],batsman['V Kohli'],color='#D9F10F')
plt.plot(batsman['index'],batsman['RG Sharma'],color='#FC00D6')

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')
     
Text(0, 0.5, 'Runs Scored')


plt.plot(batsman['index'],batsman['V Kohli'],color='#D9F10F',linestyle='solid',linewidth=3)
plt.plot(batsman['index'],batsman['RG Sharma'],color='#FC00D6',linestyle='dashdot',linewidth=2)

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')
     
Text(0, 0.5, 'Runs Scored')


plt.plot(batsman['index'],batsman['V Kohli'],color='#D9F10F',linestyle='solid',linewidth=3,marker='D',markersize=10)
plt.plot(batsman['index'],batsman['RG Sharma'],color='#FC00D6',linestyle='dashdot',linewidth=2,marker='o')

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')
     
Text(0, 0.5, 'Runs Scored')


# legend -> location
plt.plot(batsman['index'],batsman['V Kohli'],color='#D9F10F',linestyle='solid',linewidth=3,marker='D',markersize=10,label='Virat')
plt.plot(batsman['index'],batsman['RG Sharma'],color='#FC00D6',linestyle='dashdot',linewidth=2,marker='o',label='Rohit')

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')

plt.legend(loc='upper right')
     
<matplotlib.legend.Legend at 0x7fb5d60124f0>


# limiting axes
price = [48000,54000,57000,49000,47000,45000,4500000]
year = [2015,2016,2017,2018,2019,2020,2021]

plt.plot(year,price)
plt.ylim(0,75000)
plt.xlim(2017,2019)
     
(2017.0, 2019.0)


# grid
plt.plot(batsman['index'],batsman['V Kohli'],color='#D9F10F',linestyle='solid',linewidth=3,marker='D',markersize=10)
plt.plot(batsman['index'],batsman['RG Sharma'],color='#FC00D6',linestyle='dashdot',linewidth=2,marker='o')

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')

plt.grid()
     


# show
plt.plot(batsman['index'],batsman['V Kohli'],color='#D9F10F',linestyle='solid',linewidth=3,marker='D',markersize=10)
plt.plot(batsman['index'],batsman['RG Sharma'],color='#FC00D6',linestyle='dashdot',linewidth=2,marker='o')

plt.title('Rohit Sharma Vs Virat Kohli Career Comparison')
plt.xlabel('Season')
plt.ylabel('Runs Scored')

plt.grid()

plt.show()
     

Scatter Plots
image.png

Bivariate Analysis
numerical vs numerical
Use case - Finding correlation

# plt.scatter simple function
x = np.linspace(-10,10,50)

y = 10*x + 3 + np.random.randint(0,300,50)
y
     
array([199.        ,  70.08163265,  13.16326531,  25.24489796,
       198.32653061,  40.40816327, -64.51020408, 206.57142857,
       -60.34693878, -28.26530612,  23.81632653,  29.89795918,
         6.97959184, 166.06122449, 136.14285714, 156.2244898 ,
        -8.69387755, 204.3877551 ,  66.46938776,  85.55102041,
       203.63265306, 182.71428571, 139.79591837, 164.87755102,
        67.95918367,  57.04081633, 190.12244898,  51.20408163,
       101.28571429,  84.36734694,  31.44897959,  47.53061224,
       223.6122449 , 145.69387755, 278.7755102 , 122.85714286,
       258.93877551, 174.02040816, 315.10204082, 338.18367347,
       363.26530612, 242.34693878, 342.42857143, 376.51020408,
        98.59183673, 376.67346939,  95.75510204, 268.83673469,
       309.91836735, 324.        ])

plt.scatter(x,y)
     
<matplotlib.collections.PathCollection at 0x7fb5d5da8850>


# plt.scatter on pandas data
df = pd.read_csv('/content/batter.csv')
df = df.head(50)
df
     
batter	runs	avg	strike_rate
0	V Kohli	6634	36.251366	125.977972
1	S Dhawan	6244	34.882682	122.840842
2	DA Warner	5883	41.429577	136.401577
3	RG Sharma	5881	30.314433	126.964594
4	SK Raina	5536	32.374269	132.535312
5	AB de Villiers	5181	39.853846	148.580442
6	CH Gayle	4997	39.658730	142.121729
7	MS Dhoni	4978	39.196850	130.931089
8	RV Uthappa	4954	27.522222	126.152279
9	KD Karthik	4377	26.852761	129.267572
10	G Gambhir	4217	31.007353	119.665153
11	AT Rayudu	4190	28.896552	124.148148
12	AM Rahane	4074	30.863636	117.575758
13	KL Rahul	3895	46.927711	132.799182
14	SR Watson	3880	30.793651	134.163209
15	MK Pandey	3657	29.731707	117.739858
16	SV Samson	3526	29.140496	132.407060
17	KA Pollard	3437	28.404959	140.457703
18	F du Plessis	3403	34.373737	127.167414
19	YK Pathan	3222	29.290909	138.046272
20	BB McCullum	2882	27.711538	126.848592
21	RR Pant	2851	34.768293	142.550000
22	PA Patel	2848	22.603175	116.625717
23	JC Buttler	2832	39.333333	144.859335
24	SS Iyer	2780	31.235955	121.132898
25	Q de Kock	2767	31.804598	130.951254
26	Yuvraj Singh	2754	24.810811	124.784776
27	V Sehwag	2728	27.555556	148.827059
28	SA Yadav	2644	29.707865	134.009123
29	M Vijay	2619	25.930693	118.614130
30	RA Jadeja	2502	26.617021	122.108346
31	SPD Smith	2495	34.652778	124.812406
32	SE Marsh	2489	39.507937	130.109775
33	DA Miller	2455	36.102941	133.569097
34	JH Kallis	2427	28.552941	105.936272
35	WP Saha	2427	25.281250	124.397745
36	DR Smith	2385	28.392857	132.279534
37	MA Agarwal	2335	22.669903	129.506378
38	SR Tendulkar	2334	33.826087	114.187867
39	GJ Maxwell	2320	25.494505	147.676639
40	N Rana	2181	27.961538	130.053667
41	R Dravid	2174	28.233766	113.347237
42	KS Williamson	2105	36.293103	123.315759
43	AJ Finch	2092	24.904762	123.349057
44	AC Gilchrist	2069	27.223684	133.054662
45	AD Russell	2039	29.985294	168.234323
46	JP Duminy	2029	39.784314	120.773810
47	MEK Hussey	1977	38.764706	119.963592
48	HH Pandya	1972	29.878788	140.256046
49	Shubman Gill	1900	32.203390	122.186495

plt.scatter(df['avg'],df['strike_rate'],color='red',marker='+')
plt.title('Avg and SR analysis of Top 50 Batsman')
plt.xlabel('Average')
plt.ylabel('SR')
     
Text(0, 0.5, 'SR')


# marker
     

# size
tips = sns.load_dataset('tips')


# slower
plt.scatter(tips['total_bill'],tips['tip'],s=tips['size']*20)
     
<matplotlib.collections.PathCollection at 0x7fb5d597f550>


# scatterplot using plt.plot
# faster
plt.plot(tips['total_bill'],tips['tip'],'o')
     
[<matplotlib.lines.Line2D at 0x7fb5d591ac10>]


# plt.plot vs plt.scatter
     
Bar chart
image.png

Bivariate Analysis
Numerical vs Categorical
Use case - Aggregate analysis of groups

# simple bar chart
children = [10,20,40,10,30]
colors = ['red','blue','green','yellow','pink']

plt.bar(colors,children,color='black')
     
<BarContainer object of 5 artists>


# bar chart using data
     

# horizontal bar chart
plt.barh(colors,children,color='black')
     
<BarContainer object of 5 artists>


# color and label
df = pd.read_csv('/content/batsman_season_record.csv')
df
     
batsman	2015	2016	2017
0	AB de Villiers	513	687	216
1	DA Warner	562	848	641
2	MS Dhoni	372	284	290
3	RG Sharma	482	489	333
4	V Kohli	505	973	308

plt.bar(np.arange(df.shape[0]) - 0.2,df['2015'],width=0.2,color='yellow')
plt.bar(np.arange(df.shape[0]),df['2016'],width=0.2,color='red')
plt.bar(np.arange(df.shape[0]) + 0.2,df['2017'],width=0.2,color='blue')

plt.xticks(np.arange(df.shape[0]), df['batsman'])

plt.show()
     


np.arange(df.shape[0])
     
array([0, 1, 2, 3, 4])

# Multiple Bar charts
     

# xticks
     

# a problem
children = [10,20,40,10,30]
colors = ['red red red red red red','blue blue blue blue','green green green green green','yellow yellow yellow yellow ','pink pinkpinkpink']

plt.bar(colors,children,color='black')
plt.xticks(rotation='vertical')
     
([0, 1, 2, 3, 4], <a list of 5 Text major ticklabel objects>)


# Stacked Bar chart
plt.bar(df['batsman'],df['2017'],label='2017')
plt.bar(df['batsman'],df['2016'],bottom=df['2017'],label='2016')
plt.bar(df['batsman'],df['2015'],bottom=(df['2016'] + df['2017']),label='2015')

plt.legend()
plt.show()
     
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

KeyError: '2017'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
<ipython-input-280-5c1da7ceedc3> in <module>
      1 # Stacked Bar chart
----> 2 plt.bar(df['batsman'],df['2017'],label='2017')
      3 plt.bar(df['batsman'],df['2016'],bottom=df['2017'],label='2016')
      4 plt.bar(df['batsman'],df['2015'],bottom=(df['2016'] + df['2017']),label='2015')
      5 

/usr/local/lib/python3.8/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   3456             if self.columns.nlevels > 1:
   3457                 return self._getitem_multilevel(key)
-> 3458             indexer = self.columns.get_loc(key)
   3459             if is_integer(indexer):
   3460                 indexer = [indexer]

/usr/local/lib/python3.8/dist-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
   3361                 return self._engine.get_loc(casted_key)
   3362             except KeyError as err:
-> 3363                 raise KeyError(key) from err
   3364 
   3365         if is_scalar(key) and isna(key) and not self.hasnans:

KeyError: '2017'
Histogram
image.png

Univariate Analysis
Numerical col
Use case - Frequency Count

# simple data

data = [32,45,56,10,15,27,61]

plt.hist(data,bins=[10,25,40,55,70])
     
(array([2., 2., 1., 2.]),
 array([10, 25, 40, 55, 70]),
 <a list of 4 Patch objects>)


# on some data
df = pd.read_csv('/content/vk.csv')
df
     
match_id	batsman_runs
0	12	62
1	17	28
2	20	64
3	27	0
4	30	10
...	...	...
136	624	75
137	626	113
138	632	54
139	633	0
140	636	54
141 rows × 2 columns


plt.hist(df['batsman_runs'],bins=[0,10,20,30,40,50,60,70,80,90,100,110,120])
plt.show()
     


# handling bins
     

# logarithmic scale
arr = np.load('/content/big-array.npy')
plt.hist(arr,bins=[10,20,30,40,50,60,70],log=True)
plt.show()
     

Pie Chart
image.png

Univariate/Bivariate Analysis
Categorical vs numerical
Use case - To find contibution on a standard scale

# simple data
data = [23,45,100,20,49]
subjects = ['eng','science','maths','sst','hindi']
plt.pie(data,labels=subjects)

plt.show()
     


# dataset
df = pd.read_csv('/content/gayle-175.csv')
df
     
batsman	batsman_runs
0	AB de Villiers	31
1	CH Gayle	175
2	R Rampaul	0
3	SS Tiwary	2
4	TM Dilshan	33
5	V Kohli	11

plt.pie(df['batsman_runs'],labels=df['batsman'],autopct='%0.1f%%')
plt.show()
     


# percentage and colors
plt.pie(df['batsman_runs'],labels=df['batsman'],autopct='%0.1f%%',colors=['blue','green','yellow','pink','cyan','brown'])
plt.show()
     


# explode shadow
plt.pie(df['batsman_runs'],labels=df['batsman'],autopct='%0.1f%%',explode=[0.3,0,0,0,0,0.1],shadow=True)
plt.show()
     

Changing styles

plt.style.available
     
['Solarize_Light2',
 '_classic_test_patch',
 'bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn',
 'seaborn-bright',
 'seaborn-colorblind',
 'seaborn-dark',
 'seaborn-dark-palette',
 'seaborn-darkgrid',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-notebook',
 'seaborn-paper',
 'seaborn-pastel',
 'seaborn-poster',
 'seaborn-talk',
 'seaborn-ticks',
 'seaborn-white',
 'seaborn-whitegrid',
 'tableau-colorblind10']

plt.style.use('dark_background')
     

arr = np.load('/content/big-array.npy')
plt.hist(arr,bins=[10,20,30,40,50,60,70],log=True)
plt.show()
     

Save figure

arr = np.load('/content/big-array.npy')
plt.hist(arr,bins=[10,20,30,40,50,60,70],log=True)

plt.savefig('sample.png')
     

Checkout Doc on website


     
#################################################################################



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
     
Colored Scatterplots

iris = pd.read_csv('iris.csv')
iris.sample(5)
     
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
9	10	4.9	3.1	1.5	0.1	Iris-setosa
73	74	6.1	2.8	4.7	1.2	Iris-versicolor
44	45	5.1	3.8	1.9	0.4	Iris-setosa
51	52	6.4	3.2	4.5	1.5	Iris-versicolor
104	105	6.5	3.0	5.8	2.2	Iris-virginica

iris['Species'] = iris['Species'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
iris.sample(5)
     
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
87	88	6.3	2.3	4.4	1.3	1
20	21	5.4	3.4	1.7	0.2	0
56	57	6.3	3.3	4.7	1.6	1
140	141	6.7	3.1	5.6	2.4	2
141	142	6.9	3.1	5.1	2.3	2

plt.scatter(iris['SepalLengthCm'],iris['PetalLengthCm'],c=iris['Species'],cmap='jet',alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.colorbar()
     
<matplotlib.colorbar.Colorbar at 0x7f5e17170bb0>


# cmap and alpha
     
Plot size

plt.figure(figsize=(15,7))

plt.scatter(iris['SepalLengthCm'],iris['PetalLengthCm'],c=iris['Species'],cmap='jet',alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.colorbar()
     
<matplotlib.colorbar.Colorbar at 0x7f5e16ee4430>

Annotations

batters = pd.read_csv('batter.csv')
     

batters.shape
     
(605, 4)

sample_df = df.head(100).sample(25,random_state=5)
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-137-839dfd0bcf32> in <module>
----> 1 sample_df = df.head(100).sample(25,random_state=5)

/usr/local/lib/python3.8/dist-packages/pandas/core/generic.py in sample(self, n, frac, replace, weights, random_state, axis, ignore_index)
   5363             )
   5364 
-> 5365         locs = rs.choice(axis_length, size=n, replace=replace, p=weights)
   5366         result = self.take(locs, axis=axis)
   5367         if ignore_index:

mtrand.pyx in numpy.random.mtrand.RandomState.choice()

ValueError: Cannot take a larger sample than population when 'replace=False'

sample_df
     
batter	runs	avg	strike_rate
66	KH Pandya	1326	22.100000	132.203390
32	SE Marsh	2489	39.507937	130.109775
46	JP Duminy	2029	39.784314	120.773810
28	SA Yadav	2644	29.707865	134.009123
74	IK Pathan	1150	21.698113	116.751269
23	JC Buttler	2832	39.333333	144.859335
10	G Gambhir	4217	31.007353	119.665153
20	BB McCullum	2882	27.711538	126.848592
17	KA Pollard	3437	28.404959	140.457703
35	WP Saha	2427	25.281250	124.397745
97	ST Jayasuriya	768	27.428571	134.031414
37	MA Agarwal	2335	22.669903	129.506378
70	DJ Hooda	1237	20.278689	127.525773
40	N Rana	2181	27.961538	130.053667
60	SS Tiwary	1494	28.730769	115.724245
34	JH Kallis	2427	28.552941	105.936272
42	KS Williamson	2105	36.293103	123.315759
57	DJ Bravo	1560	22.608696	125.100241
12	AM Rahane	4074	30.863636	117.575758
69	D Padikkal	1260	28.000000	119.205298
94	SO Hetmyer	831	30.777778	144.020797
56	PP Shaw	1588	25.206349	143.580470
22	PA Patel	2848	22.603175	116.625717
39	GJ Maxwell	2320	25.494505	147.676639
24	SS Iyer	2780	31.235955	121.132898

plt.figure(figsize=(18,10))
plt.scatter(sample_df['avg'],sample_df['strike_rate'],s=sample_df['runs'])

for i in range(sample_df.shape[0]):
  plt.text(sample_df['avg'].values[i],sample_df['strike_rate'].values[i],sample_df['batter'].values[i])
     


x = [1,2,3,4]
y = [5,6,7,8]

plt.scatter(x,y)
plt.text(1,5,'Point 1')
plt.text(2,6,'Point 2')
plt.text(3,7,'Point 3')
plt.text(4,8,'Point 4',fontdict={'size':12,'color':'brown'})
     
Text(4, 8, 'Point 4')

Horizontal and Vertical lines

plt.figure(figsize=(18,10))
plt.scatter(sample_df['avg'],sample_df['strike_rate'],s=sample_df['runs'])

plt.axhline(130,color='red')
plt.axhline(140,color='green')
plt.axvline(30,color='red')

for i in range(sample_df.shape[0]):
  plt.text(sample_df['avg'].values[i],sample_df['strike_rate'].values[i],sample_df['batter'].values[i])
     

Subplots

# A diff way to plot graphs
batters.head()
     
batter	runs	avg	strike_rate
0	V Kohli	6634	36.251366	125.977972
1	S Dhawan	6244	34.882682	122.840842
2	DA Warner	5883	41.429577	136.401577
3	RG Sharma	5881	30.314433	126.964594
4	SK Raina	5536	32.374269	132.535312

plt.figure(figsize=(15,6))
plt.scatter(batters['avg'],batters['strike_rate'])
plt.title('Something')
plt.xlabel('Avg')
plt.ylabel('Strike Rate')

plt.show()
     


fig,ax = plt.subplots(figsize=(15,6))

ax.scatter(batters['avg'],batters['strike_rate'],color='red',marker='+')
ax.set_title('Something')
ax.set_xlabel('Avg')
ax.set_ylabel('Strike Rate')

fig.show()
     


# batter dataset
     

fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(10,6))

ax[0].scatter(batters['avg'],batters['strike_rate'],color='red')
ax[1].scatter(batters['avg'],batters['runs'])

ax[0].set_title('Avg Vs Strike Rate')
ax[0].set_ylabel('Strike Rate')


ax[1].set_title('Avg Vs Runs')
ax[1].set_ylabel('Runs')
ax[1].set_xlabel('Avg')
     
Text(0.5, 0, 'Avg')


fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10))

ax[0,0].
ax[0,1].scatter(batters['avg'],batters['runs'])
ax[1,0].hist(batters['avg'])
ax[1,1].hist(batters['runs'])
     
(array([499.,  40.,  19.,  19.,   9.,   6.,   4.,   4.,   3.,   2.]),
 array([   0. ,  663.4, 1326.8, 1990.2, 2653.6, 3317. , 3980.4, 4643.8,
        5307.2, 5970.6, 6634. ]),
 <a list of 10 Patch objects>)


fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.scatter(batters['avg'],batters['strike_rate'],color='red')

ax2 = fig.add_subplot(2,2,2)
ax2.hist(batters['runs'])

ax3 = fig.add_subplot(2,2,3)
ax3.hist(batters['avg'])
     
(array([102., 125., 103.,  82.,  78.,  43.,  22.,  14.,   2.,   1.]),
 array([ 0.        ,  5.56666667, 11.13333333, 16.7       , 22.26666667,
        27.83333333, 33.4       , 38.96666667, 44.53333333, 50.1       ,
        55.66666667]),
 <a list of 10 Patch objects>)


fig, ax = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=(10,10))

ax[1,1]
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e15913c10>

3D Scatter Plots

batters

fig = plt.figure()

ax = plt.subplot(projection='3d')

ax.scatter3D(batters['runs'],batters['avg'],batters['strike_rate'],marker='+')
ax.set_title('IPL batsman analysis')

ax.set_xlabel('Runs')
ax.set_ylabel('Avg')
ax.set_zlabel('SR')
     
Text(0.5, 0, 'SR')

3D Line Plot

x = [0,1,5,25]
y = [0,10,13,0]
z = [0,13,20,9]

fig = plt.figure()

ax = plt.subplot(projection='3d')

ax.scatter3D(x,y,z,s=[100,100,100,100])
ax.plot3D(x,y,z,color='red')
     
[<mpl_toolkits.mplot3d.art3d.Line3D at 0x7f5e14d13f10>]

3D Surface Plots

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
     

xx, yy = np.meshgrid(x,y)
     
(100, 100)

z = xx**2 + yy**2
z.shape
     
(100, 100)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot(projection='3d')

p = ax.plot_surface(xx,yy,z,cmap='viridis')
fig.colorbar(p)
     
<matplotlib.colorbar.Colorbar at 0x7f5e141ac970>


z = np.sin(xx) + np.cos(yy)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot(projection='3d')

p = ax.plot_surface(xx,yy,z,cmap='viridis')
fig.colorbar(p)
     
<matplotlib.colorbar.Colorbar at 0x7f5e14076be0>


z = np.sin(xx) + np.log(xx)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot(projection='3d')

p = ax.plot_surface(xx,yy,z,cmap='viridis')
fig.colorbar(p)
     
<ipython-input-229-bbcd37ea4152>:1: RuntimeWarning: invalid value encountered in log
  z = np.sin(xx) + np.log(xx)
<ipython-input-229-bbcd37ea4152>:7: UserWarning: Z contains NaN values. This may result in rendering artifacts.
  p = ax.plot_surface(xx,yy,z,cmap='viridis')
<matplotlib.colorbar.Colorbar at 0x7f5e139a4a00>


fig = plt.figure(figsize=(12,8))

ax = plt.subplot(projection='3d')

p = ax.plot_surface(xx,yy,z,cmap='viridis')
fig.colorbar(p)
     
<matplotlib.colorbar.Colorbar at 0x7f5e136f8970>

Contour Plots

fig = plt.figure(figsize=(12,8))

ax = plt.subplot()

p = ax.contour(xx,yy,z,cmap='viridis')
fig.colorbar(p)
     
<matplotlib.colorbar.Colorbar at 0x7f5e13580a30>


fig = plt.figure(figsize=(12,8))

ax = plt.subplot()

p = ax.contourf(xx,yy,z,cmap='viridis')
fig.colorbar(p)
     
<matplotlib.colorbar.Colorbar at 0x7f5e14f202b0>


z = np.sin(xx) + np.cos(yy)

fig = plt.figure(figsize=(12,8))

ax = plt.subplot()

p = ax.contourf(xx,yy,z,cmap='viridis')
fig.colorbar(p)
     
<matplotlib.colorbar.Colorbar at 0x7f5e14d5a2e0>

Heatmap

delivery = pd.read_csv('/content/IPL_Ball_by_Ball_2008_2022.csv')
delivery.head()
     
ID	innings	overs	ballnumber	batter	bowler	non-striker	extra_type	batsman_run	extras_run	total_run	non_boundary	isWicketDelivery	player_out	kind	fielders_involved	BattingTeam
0	1312200	1	0	1	YBK Jaiswal	Mohammed Shami	JC Buttler	NaN	0	0	0	0	0	NaN	NaN	NaN	Rajasthan Royals
1	1312200	1	0	2	YBK Jaiswal	Mohammed Shami	JC Buttler	legbyes	0	1	1	0	0	NaN	NaN	NaN	Rajasthan Royals
2	1312200	1	0	3	JC Buttler	Mohammed Shami	YBK Jaiswal	NaN	1	0	1	0	0	NaN	NaN	NaN	Rajasthan Royals
3	1312200	1	0	4	YBK Jaiswal	Mohammed Shami	JC Buttler	NaN	0	0	0	0	0	NaN	NaN	NaN	Rajasthan Royals
4	1312200	1	0	5	YBK Jaiswal	Mohammed Shami	JC Buttler	NaN	0	0	0	0	0	NaN	NaN	NaN	Rajasthan Royals

temp_df = delivery[(delivery['ballnumber'].isin([1,2,3,4,5,6])) & (delivery['batsman_run']==6)]
     

grid = temp_df.pivot_table(index='overs',columns='ballnumber',values='batsman_run',aggfunc='count')
     

plt.figure(figsize=(20,10))
plt.imshow(grid)
plt.yticks(delivery['overs'].unique(), list(range(1,21)))
plt.xticks(np.arange(0,6), list(range(1,7)))
plt.colorbar()
     
<matplotlib.colorbar.Colorbar at 0x7f5e12f98cd0>



     
Pandas Plot()

# on a series

s = pd.Series([1,2,3,4,5,6,7])
s.plot(kind='pie')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e12f0a070>


# can be used on a dataframe as well
     

import seaborn as sns
tips = sns.load_dataset('tips')
     

tips['size'] = tips['size'] * 100
     


     

tips.head()
     
total_bill	tip	sex	smoker	day	time	size
0	16.99	1.01	Female	No	Sun	Dinner	200
1	10.34	1.66	Male	No	Sun	Dinner	300
2	21.01	3.50	Male	No	Sun	Dinner	300
3	23.68	3.31	Male	No	Sun	Dinner	200
4	24.59	3.61	Female	No	Sun	Dinner	400

# Scatter plot -> labels -> markers -> figsize -> color -> cmap
tips.plot(kind='scatter',x='total_bill',y='tip',title='Cost Analysis',marker='+',figsize=(10,6),s='size',c='sex',cmap='viridis')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e12b4d760>


# 2d plot
# dataset = 'https://raw.githubusercontent.com/m-mehdi/pandas_tutorials/main/weekly_stocks.csv'

stocks = pd.read_csv('https://raw.githubusercontent.com/m-mehdi/pandas_tutorials/main/weekly_stocks.csv')
stocks.head()
     
Date	MSFT	FB	AAPL
0	2021-05-24	249.679993	328.730011	124.610001
1	2021-05-31	250.789993	330.350006	125.889999
2	2021-06-07	257.890015	331.260010	127.349998
3	2021-06-14	259.429993	329.660004	130.460007
4	2021-06-21	265.019989	341.369995	133.110001

# line plot
stocks['MSFT'].plot(kind='line')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e12a55730>


stocks.plot(kind='line',x='Date')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e129f15e0>


stocks[['Date','AAPL','FB']].plot(kind='line',x='Date')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e12950fa0>


# bar chart -> single -> horizontal -> multiple
# using tips
temp = pd.read_csv('/content/batsman_season_record.csv')
temp.head()
     
batsman	2015	2016	2017
0	AB de Villiers	513	687	216
1	DA Warner	562	848	641
2	MS Dhoni	372	284	290
3	RG Sharma	482	489	333
4	V Kohli	505	973	308

tips.groupby('sex')['total_bill'].mean().plot(kind='bar')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e12350550>


temp['2015'].plot(kind='bar')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e123ceaf0>


temp.plot(kind='bar')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e1228fac0>


# stacked bar chart
temp.plot(kind='bar',stacked=True)
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e12216e50>


# histogram
# using stocks

stocks[['MSFT','FB']].plot(kind='hist',bins=40)
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e150247f0>


# pie -> single and multiple
df = pd.DataFrame(
    {
        'batsman':['Dhawan','Rohit','Kohli','SKY','Pandya','Pant'],
        'match1':[120,90,35,45,12,10],
        'match2':[0,1,123,130,34,45],
        'match3':[50,24,145,45,10,90]
    }
)

df.head()
     
batsman	match1	match2	match3
0	Dhawan	120	0	50
1	Rohit	90	1	24
2	Kohli	35	123	145
3	SKY	45	130	45
4	Pandya	12	34	10

df['match1'].plot(kind='pie',labels=df['batsman'].values,autopct='%0.1f%%')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f5e11e50790>


# multiple pie charts

df[['match1','match2','match3']].plot(kind='pie',subplots=True,figsize=(15,8))
     
array([<matplotlib.axes._subplots.AxesSubplot object at 0x7f5e11cc1b50>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x7f5e11c628b0>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x7f5e11c7ff10>],
      dtype=object)


# multiple separate graphs together
# using stocks

stocks.plot(kind='line',subplots=True)
     
array([<matplotlib.axes._subplots.AxesSubplot object at 0x7f5e11a2f7f0>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x7f5e11a50640>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x7f5e119f0d00>],
      dtype=object)


# on multiindex dataframes
# using tips
     

tips.pivot_table(index=['day','time'],columns=['sex','smoker'],values='total_bill',aggfunc='mean').plot(kind='pie',subplots=True,figsize=(20,10))
     
array([<matplotlib.axes._subplots.AxesSubplot object at 0x7f5e116a8cd0>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x7f5e115363d0>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x7f5e114d2a90>,
       <matplotlib.axes._subplots.AxesSubplot object at 0x7f5e1150b040>],
      dtype=object)


tips
     
total_bill	tip	sex	smoker	day	time	size
0	16.99	1.01	Female	No	Sun	Dinner	200
1	10.34	1.66	Male	No	Sun	Dinner	300
2	21.01	3.50	Male	No	Sun	Dinner	300
3	23.68	3.31	Male	No	Sun	Dinner	200
4	24.59	3.61	Female	No	Sun	Dinner	400
...	...	...	...	...	...	...	...
239	29.03	5.92	Male	No	Sat	Dinner	300
240	27.18	2.00	Female	Yes	Sat	Dinner	200
241	22.67	2.00	Male	Yes	Sat	Dinner	200
242	17.82	1.75	Male	No	Sat	Dinner	200
243	18.78	3.00	Female	No	Thur	Dinner	200
244 rows × 7 columns


stocks.plot(kind='scatter3D')
     
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-321-4e91fa40f850> in <module>
----> 1 stocks.plot(kind='scatter3D')

/usr/local/lib/python3.8/dist-packages/pandas/plotting/_core.py in __call__(self, *args, **kwargs)
    903 
    904         if kind not in self._all_kinds:
--> 905             raise ValueError(f"{kind} is not a valid plot kind")
    906 
    907         # The original data structured can be transformed before passed to the

ValueError: scatter3D is not a valid plot kind


     
#################################################################################


Why Seaborn?
provides a layer of abstraction hence simpler to use
better aesthetics
more graphs included
Seaborn Roadmap
Types of Functions

Figure Level
Axis Level
Main Classification

Relational Plot
Distribution Plot
Categorical Plot
Regression Plot
Matrix Plot
Multiplots
https://seaborn.pydata.org/api.html

1. Relational Plot
to see the statistical relation between 2 or more variables.
Bivariate Analysis
Plots under this section

scatterplot
lineplot

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
     

tips = sns.load_dataset('tips')
tips
     
total_bill	tip	sex	smoker	day	time	size
0	16.99	1.01	Female	No	Sun	Dinner	2
1	10.34	1.66	Male	No	Sun	Dinner	3
2	21.01	3.50	Male	No	Sun	Dinner	3
3	23.68	3.31	Male	No	Sun	Dinner	2
4	24.59	3.61	Female	No	Sun	Dinner	4
...	...	...	...	...	...	...	...
239	29.03	5.92	Male	No	Sat	Dinner	3
240	27.18	2.00	Female	Yes	Sat	Dinner	2
241	22.67	2.00	Male	Yes	Sat	Dinner	2
242	17.82	1.75	Male	No	Sat	Dinner	2
243	18.78	3.00	Female	No	Thur	Dinner	2
244 rows × 7 columns


# scatter plot -> axes level function
sns.scatterplot(data=tips, x='total_bill', y='tip',hue='sex',style='time',size='size')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f9585634670>


# relplot -> figure level -> square shape
sns.relplot(data=tips, x='total_bill', y='tip', kind='scatter',hue='sex',style='time',size='size')
     
<seaborn.axisgrid.FacetGrid at 0x7f9585625820>


# scatter using relplot -> size and hue
     

# style semantics
     

# line plot
gap = px.data.gapminder()
temp_df = gap[gap['country'] == 'India']
temp_df
     
country	continent	year	lifeExp	pop	gdpPercap	iso_alpha	iso_num
696	India	Asia	1952	37.373	372000000	546.565749	IND	356
697	India	Asia	1957	40.249	409000000	590.061996	IND	356
698	India	Asia	1962	43.605	454000000	658.347151	IND	356
699	India	Asia	1967	47.193	506000000	700.770611	IND	356
700	India	Asia	1972	50.651	567000000	724.032527	IND	356
701	India	Asia	1977	54.208	634000000	813.337323	IND	356
702	India	Asia	1982	56.596	708000000	855.723538	IND	356
703	India	Asia	1987	58.553	788000000	976.512676	IND	356
704	India	Asia	1992	60.223	872000000	1164.406809	IND	356
705	India	Asia	1997	61.765	959000000	1458.817442	IND	356
706	India	Asia	2002	62.879	1034172547	1746.769454	IND	356
707	India	Asia	2007	64.698	1110396331	2452.210407	IND	356

# axes level function
sns.lineplot(data=temp_df, x='year', y='lifeExp')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f95854b7c70>


# using relpplot
sns.relplot(data=temp_df, x='year', y='lifeExp', kind='line')
     
<seaborn.axisgrid.FacetGrid at 0x7f9585427a60>


# hue -> style
temp_df = gap[gap['country'].isin(['India','Brazil','Germany'])]
temp_df
     
country	continent	year	lifeExp	pop	gdpPercap	iso_alpha	iso_num
168	Brazil	Americas	1952	50.917	56602560	2108.944355	BRA	76
169	Brazil	Americas	1957	53.285	65551171	2487.365989	BRA	76
170	Brazil	Americas	1962	55.665	76039390	3336.585802	BRA	76
171	Brazil	Americas	1967	57.632	88049823	3429.864357	BRA	76
172	Brazil	Americas	1972	59.504	100840058	4985.711467	BRA	76
173	Brazil	Americas	1977	61.489	114313951	6660.118654	BRA	76
174	Brazil	Americas	1982	63.336	128962939	7030.835878	BRA	76
175	Brazil	Americas	1987	65.205	142938076	7807.095818	BRA	76
176	Brazil	Americas	1992	67.057	155975974	6950.283021	BRA	76
177	Brazil	Americas	1997	69.388	168546719	7957.980824	BRA	76
178	Brazil	Americas	2002	71.006	179914212	8131.212843	BRA	76
179	Brazil	Americas	2007	72.390	190010647	9065.800825	BRA	76
564	Germany	Europe	1952	67.500	69145952	7144.114393	DEU	276
565	Germany	Europe	1957	69.100	71019069	10187.826650	DEU	276
566	Germany	Europe	1962	70.300	73739117	12902.462910	DEU	276
567	Germany	Europe	1967	70.800	76368453	14745.625610	DEU	276
568	Germany	Europe	1972	71.000	78717088	18016.180270	DEU	276
569	Germany	Europe	1977	72.500	78160773	20512.921230	DEU	276
570	Germany	Europe	1982	73.800	78335266	22031.532740	DEU	276
571	Germany	Europe	1987	74.847	77718298	24639.185660	DEU	276
572	Germany	Europe	1992	76.070	80597764	26505.303170	DEU	276
573	Germany	Europe	1997	77.340	82011073	27788.884160	DEU	276
574	Germany	Europe	2002	78.670	82350671	30035.801980	DEU	276
575	Germany	Europe	2007	79.406	82400996	32170.374420	DEU	276
696	India	Asia	1952	37.373	372000000	546.565749	IND	356
697	India	Asia	1957	40.249	409000000	590.061996	IND	356
698	India	Asia	1962	43.605	454000000	658.347151	IND	356
699	India	Asia	1967	47.193	506000000	700.770611	IND	356
700	India	Asia	1972	50.651	567000000	724.032527	IND	356
701	India	Asia	1977	54.208	634000000	813.337323	IND	356
702	India	Asia	1982	56.596	708000000	855.723538	IND	356
703	India	Asia	1987	58.553	788000000	976.512676	IND	356
704	India	Asia	1992	60.223	872000000	1164.406809	IND	356
705	India	Asia	1997	61.765	959000000	1458.817442	IND	356
706	India	Asia	2002	62.879	1034172547	1746.769454	IND	356
707	India	Asia	2007	64.698	1110396331	2452.210407	IND	356

sns.relplot(kind='line', data=temp_df, x='year', y='lifeExp', hue='country', style='continent', size='continent')
     
<seaborn.axisgrid.FacetGrid at 0x7f9585258c70>


sns.lineplot(data=temp_df, x='year', y='lifeExp', hue='country')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f95853837c0>


# facet plot -> figure level function -> work with relplot
# it will not work with scatterplot and lineplot
sns.relplot(data=tips, x='total_bill', y='tip', kind='line', col='sex', row='day')
     
<seaborn.axisgrid.FacetGrid at 0x7f9584b8f8b0>


# col wrap
sns.relplot(data=gap, x='lifeExp', y='gdpPercap', kind='scatter', col='year', col_wrap=3)
     
<seaborn.axisgrid.FacetGrid at 0x7f95844efb80>


sns.scatterplot(data=tips, x='total_bill', y='tip', col='sex', row='day')
     
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-199-13fa0b1b528e> in <module>
----> 1 sns.scatterplot(data=tips, x='total_bill', y='tip', col='sex', row='day')

/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py in inner_f(*args, **kwargs)
     44             )
     45         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
---> 46         return f(**kwargs)
     47     return inner_f
     48 

/usr/local/lib/python3.8/dist-packages/seaborn/relational.py in scatterplot(x, y, hue, style, size, data, palette, hue_order, hue_norm, sizes, size_order, size_norm, markers, style_order, x_bins, y_bins, units, estimator, ci, n_boot, alpha, x_jitter, y_jitter, legend, ax, **kwargs)
    825     p._attach(ax)
    826 
--> 827     p.plot(ax, kwargs)
    828 
    829     return ax

/usr/local/lib/python3.8/dist-packages/seaborn/relational.py in plot(self, ax, kws)
    606         )
    607         scout_x = scout_y = np.full(scout_size, np.nan)
--> 608         scout = ax.scatter(scout_x, scout_y, **kws)
    609         s = kws.pop("s", scout.get_sizes())
    610         c = kws.pop("c", scout.get_facecolors())

/usr/local/lib/python3.8/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1563     def inner(ax, *args, data=None, **kwargs):
   1564         if data is None:
-> 1565             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1566 
   1567         bound = new_sig.bind(ax, *args, **kwargs)

/usr/local/lib/python3.8/dist-packages/matplotlib/cbook/deprecation.py in wrapper(*args, **kwargs)
    356                 f"%(removal)s.  If any parameter follows {name!r}, they "
    357                 f"should be pass as keyword, not positionally.")
--> 358         return func(*args, **kwargs)
    359 
    360     return wrapper

/usr/local/lib/python3.8/dist-packages/matplotlib/axes/_axes.py in scatter(self, x, y, s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths, verts, edgecolors, plotnonfinite, **kwargs)
   4441                 )
   4442         collection.set_transform(mtransforms.IdentityTransform())
-> 4443         collection.update(kwargs)
   4444 
   4445         if colors is None:

/usr/local/lib/python3.8/dist-packages/matplotlib/artist.py in update(self, props)
   1004 
   1005         with cbook._setattr_cm(self, eventson=False):
-> 1006             ret = [_update_property(self, k, v) for k, v in props.items()]
   1007 
   1008         if len(ret):

/usr/local/lib/python3.8/dist-packages/matplotlib/artist.py in <listcomp>(.0)
   1004 
   1005         with cbook._setattr_cm(self, eventson=False):
-> 1006             ret = [_update_property(self, k, v) for k, v in props.items()]
   1007 
   1008         if len(ret):

/usr/local/lib/python3.8/dist-packages/matplotlib/artist.py in _update_property(self, k, v)
    999                 func = getattr(self, 'set_' + k, None)
   1000                 if not callable(func):
-> 1001                     raise AttributeError('{!r} object has no property {!r}'
   1002                                          .format(type(self).__name__, k))
   1003                 return func(v)

AttributeError: 'PathCollection' object has no property 'col'

2. Distribution Plots
used for univariate analysis
used to find out the distribution
Range of the observation
Central Tendency
is the data bimodal?
Are there outliers?
Plots under distribution plot

histplot
kdeplot
rugplot

# figure level -> displot
# axes level -> histplot -> kdeplot -> rugplot
     

# plotting univariate histogram
sns.histplot(data=tips, x='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f9583d7dbb0>


sns.displot(data=tips, x='total_bill', kind='hist')
     
<seaborn.axisgrid.FacetGrid at 0x7f9583ebc7f0>


# bins parameter
sns.displot(data=tips, x='total_bill', kind='hist',bins=2)
     
<seaborn.axisgrid.FacetGrid at 0x7f9583c93280>


# It’s also possible to visualize the distribution of a categorical variable using the logic of a histogram.
# Discrete bins are automatically set for categorical variables

# countplot
sns.displot(data=tips, x='day', kind='hist')
     
<seaborn.axisgrid.FacetGrid at 0x7f958517d9d0>


# hue parameter
sns.displot(data=tips, x='tip', kind='hist',hue='sex')
     
<seaborn.axisgrid.FacetGrid at 0x7f9583d05280>


# element -> step
sns.displot(data=tips, x='tip', kind='hist',hue='sex',element='step')
     
<seaborn.axisgrid.FacetGrid at 0x7f9583c2cfa0>


titanic = sns.load_dataset('titanic')
titanic
     
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone
0	0	3	male	22.0	1	0	7.2500	S	Third	man	True	NaN	Southampton	no	False
1	1	1	female	38.0	1	0	71.2833	C	First	woman	False	C	Cherbourg	yes	False
2	1	3	female	26.0	0	0	7.9250	S	Third	woman	False	NaN	Southampton	yes	True
3	1	1	female	35.0	1	0	53.1000	S	First	woman	False	C	Southampton	yes	False
4	0	3	male	35.0	0	0	8.0500	S	Third	man	True	NaN	Southampton	no	True
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
886	0	2	male	27.0	0	0	13.0000	S	Second	man	True	NaN	Southampton	no	True
887	1	1	female	19.0	0	0	30.0000	S	First	woman	False	B	Southampton	yes	True
888	0	3	female	NaN	1	2	23.4500	S	Third	woman	False	NaN	Southampton	no	False
889	1	1	male	26.0	0	0	30.0000	C	First	man	True	C	Cherbourg	yes	True
890	0	3	male	32.0	0	0	7.7500	Q	Third	man	True	NaN	Queenstown	no	True
891 rows × 15 columns


sns.displot(data=titanic, x='age', kind='hist',element='step',hue='sex')
     
<seaborn.axisgrid.FacetGrid at 0x7f9583c1b9d0>


# faceting using col and row -> not work on histplot function
sns.displot(data=tips, x='tip', kind='hist',col='sex',element='step')
     
<seaborn.axisgrid.FacetGrid at 0x7f9583904850>


# kdeplot
# Rather than using discrete bins, a KDE plot smooths the observations with a Gaussian kernel, producing a continuous density estimate
sns.kdeplot(data=tips,x='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f958384a160>


sns.displot(data=tips,x='total_bill',kind='kde')
     
<seaborn.axisgrid.FacetGrid at 0x7f95838c2790>


# hue -> fill
sns.displot(data=tips,x='total_bill',kind='kde',hue='sex',fill=True,height=10,aspect=2)
     
<seaborn.axisgrid.FacetGrid at 0x7f95821a35b0>


# Rugplot

# Plot marginal distributions by drawing ticks along the x and y axes.

# This function is intended to complement other plots by showing the location of individual observations in an unobtrusive way.
sns.kdeplot(data=tips,x='total_bill')
sns.rugplot(data=tips,x='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f95836ff0d0>


# Bivariate histogram
# A bivariate histogram bins the data within rectangles that tile the plot
# and then shows the count of observations within each rectangle with the fill color

# sns.histplot(data=tips, x='total_bill', y='tip')
sns.displot(data=tips, x='total_bill', y='tip',kind='hist')
     
<seaborn.axisgrid.FacetGrid at 0x7f958362fc10>


# Bivariate Kdeplot
# a bivariate KDE plot smoothes the (x, y) observations with a 2D Gaussian
sns.kdeplot(data=tips, x='total_bill', y='tip')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f95835b60d0>

2. Matrix Plot
Heatmap
Clustermap

# Heatmap

# Plot rectangular data as a color-encoded matrix
temp_df = gap.pivot(index='country',columns='year',values='lifeExp')

# axes level function
plt.figure(figsize=(15,15))
sns.heatmap(temp_df)
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f958387d910>


# annot
temp_df = gap[gap['continent'] == 'Europe'].pivot(index='country',columns='year',values='lifeExp')

plt.figure(figsize=(15,15))
sns.heatmap(temp_df,annot=True,linewidth=0.5, cmap='summer')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7f9584de7af0>


# linewidth
     

# cmap
     

# Clustermap

# Plot a matrix dataset as a hierarchically-clustered heatmap.

# This function requires scipy to be available.

iris = px.data.iris()
iris
     
sepal_length	sepal_width	petal_length	petal_width	species	species_id
0	5.1	3.5	1.4	0.2	setosa	1
1	4.9	3.0	1.4	0.2	setosa	1
2	4.7	3.2	1.3	0.2	setosa	1
3	4.6	3.1	1.5	0.2	setosa	1
4	5.0	3.6	1.4	0.2	setosa	1
...	...	...	...	...	...	...
145	6.7	3.0	5.2	2.3	virginica	3
146	6.3	2.5	5.0	1.9	virginica	3
147	6.5	3.0	5.2	2.0	virginica	3
148	6.2	3.4	5.4	2.3	virginica	3
149	5.9	3.0	5.1	1.8	virginica	3
150 rows × 6 columns


sns.clustermap(iris.iloc[:,[0,1,2,3]])
     
<seaborn.matrix.ClusterGrid at 0x7f958226c580>



     
#################################################################################



import seaborn as sns
     

# import datasets
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
     
Categorical Plots
Categorical Scatter Plot
Stripplot
Swarmplot
Categorical Distribution Plots
Boxplot
Violinplot
Categorical Estimate Plot -> for central tendency
Barplot
Pointplot
Countplot
Figure level function -> catplot

sns.scatterplot(data=tips, x='total_bill',y='tip')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc63bca430>


# strip plot
# axes level function
sns.stripplot(data=tips,x='day',y='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc63ae0790>


# using catplot
# figure level function
sns.catplot(data=tips, x='day',y='total_bill',kind='strip')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc63ab2610>


# jitter
sns.catplot(data=tips, x='day',y='total_bill',kind='strip',jitter=0.2,hue='sex')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc63f700a0>


# swarmplot
sns.catplot(data=tips, x='day',y='total_bill',kind='swarm')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc63808670>


sns.swarmplot(data=tips, x='day',y='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc637d6520>


# hue
sns.swarmplot(data=tips, x='day',y='total_bill',hue='sex')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc63748dc0>

Boxplot
A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile [Q1], median, third quartile [Q3] and “maximum”). It can tell you about your outliers and what their values are. Boxplots can also tell you if your data is symmetrical, how tightly your data is grouped and if and how your data is skewed.

image.png


# Box plot
sns.boxplot(data=tips,x='day',y='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc6369a1c0>


# Using catplot
sns.catplot(data=tips,x='day',y='total_bill',kind='box')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc638849a0>


# hue
sns.boxplot(data=tips,x='day',y='total_bill',hue='sex')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc64d03490>


# single boxplot -> numerical col
sns.boxplot(data=tips,y='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc63588280>

Violinplot = (Boxplot + KDEplot)

# violinplot
sns.violinplot(data=tips,x='day',y='total_bill')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc63576f40>


sns.catplot(data=tips,x='day',y='total_bill',kind='violin')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc635084f0>


# hue

sns.catplot(data=tips,x='day',y='total_bill',kind='violin',hue='sex',split=True)
     
<seaborn.axisgrid.FacetGrid at 0x7fcc635085e0>


# barplot
# some issue with errorbar
import numpy as np
sns.barplot(data=tips, x='sex', y='total_bill',hue='smoker',estimator=np.min)
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc62f23a00>


sns.barplot(data=tips, x='sex', y='total_bill',ci=None)
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc634468e0>


# point plot
sns.pointplot(data=tips, x='sex', y='total_bill',hue='smoker',ci=None)
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc634a3100>

When there are multiple observations in each category, it also uses bootstrapping to compute a confidence interval around the estimate, which is plotted using error bars


# countplot
sns.countplot(data=tips,x='sex',hue='day')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc62c552b0>

A special case for the bar plot is when you want to show the number of observations in each category rather than computing a statistic for a second variable. This is similar to a histogram over a categorical, rather than quantitative, variable


# pointplot
     

# faceting using catplot
sns.catplot(data=tips, x='sex',y='total_bill',col='smoker',kind='box',row='time')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc62b8ee80>

Regression Plots
regplot
lmplot
In the simplest invocation, both functions draw a scatterplot of two variables, x and y, and then fit the regression model y ~ x and plot the resulting regression line and a 95% confidence interval for that regression.


# axes level
# hue parameter is not available
sns.regplot(data=tips,x='total_bill',y='tip')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc62829550>


sns.lmplot(data=tips,x='total_bill',y='tip',hue='sex')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc627f1250>


# residplot
sns.residplot(data=tips,x='total_bill',y='tip')
     
<matplotlib.axes._subplots.AxesSubplot at 0x7fcc62755d30>



     


     
A second way to plot Facet plots -> FacetGrid

# figure level -> relplot -> displot -> catplot -> lmplot
sns.catplot(data=tips,x='sex',y='total_bill',kind='violin',col='day',row='time')
     
<seaborn.axisgrid.FacetGrid at 0x7fcc62538970>


g = sns.FacetGrid(data=tips,col='day',row='time',hue='smoker')
g.map(sns.boxplot,'sex','total_bill')
g.add_legend()
     
/usr/local/lib/python3.8/dist-packages/seaborn/axisgrid.py:670: UserWarning: Using the boxplot function without specifying `order` is likely to produce an incorrect plot.
  warnings.warn(warning)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-134-983b773fc0d8> in <module>
      1 g = sns.FacetGrid(data=tips,col='day',row='time',hue='smoker')
----> 2 g.map(sns.boxplot,'sex','total_bill')
      3 g.add_legend()

/usr/local/lib/python3.8/dist-packages/seaborn/axisgrid.py in map(self, func, *args, **kwargs)
    708 
    709             # Draw the plot
--> 710             self._facet_plot(func, ax, plot_args, kwargs)
    711 
    712         # Finalize the annotations and layout

/usr/local/lib/python3.8/dist-packages/seaborn/axisgrid.py in _facet_plot(self, func, ax, plot_args, plot_kwargs)
    804             plot_args = []
    805             plot_kwargs["ax"] = ax
--> 806         func(*plot_args, **plot_kwargs)
    807 
    808         # Sort out the supporting information

/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py in inner_f(*args, **kwargs)
     44             )
     45         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
---> 46         return f(**kwargs)
     47     return inner_f
     48 

/usr/local/lib/python3.8/dist-packages/seaborn/categorical.py in boxplot(x, y, hue, data, order, hue_order, orient, color, palette, saturation, width, dodge, fliersize, linewidth, whis, ax, **kwargs)
   2249     kwargs.update(dict(whis=whis))
   2250 
-> 2251     plotter.plot(ax, kwargs)
   2252     return ax
   2253 

/usr/local/lib/python3.8/dist-packages/seaborn/categorical.py in plot(self, ax, boxplot_kws)
    507     def plot(self, ax, boxplot_kws):
    508         """Make the plot."""
--> 509         self.draw_boxplot(ax, boxplot_kws)
    510         self.annotate_axes(ax)
    511         if self.orient == "h":

/usr/local/lib/python3.8/dist-packages/seaborn/categorical.py in draw_boxplot(self, ax, kws)
    439                     continue
    440 
--> 441                 artist_dict = ax.boxplot(box_data,
    442                                          vert=vert,
    443                                          patch_artist=True,

/usr/local/lib/python3.8/dist-packages/matplotlib/cbook/deprecation.py in wrapper(*args, **kwargs)
    294                 f"for the old name will be dropped %(removal)s.")
    295             kwargs[new] = kwargs.pop(old)
--> 296         return func(*args, **kwargs)
    297 
    298     # wrapper() must keep the same documented signature as func(): if we

/usr/local/lib/python3.8/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1563     def inner(ax, *args, data=None, **kwargs):
   1564         if data is None:
-> 1565             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1566 
   1567         bound = new_sig.bind(ax, *args, **kwargs)

TypeError: boxplot() got an unexpected keyword argument 'label'


# height and aspect
     
<seaborn.axisgrid.FacetGrid at 0x7f8b2a2f3070>


     


     
Plotting Pairwise Relationship (PairGrid Vs Pairplot)

sns.pairplot(iris,hue='species')
     
<seaborn.axisgrid.PairGrid at 0x7fcc60bdcac0>



     

# pair grid
g = sns.PairGrid(data=iris,hue='species')
# g.map
g.map(sns.scatterplot)
     
<seaborn.axisgrid.PairGrid at 0x7fcc5fd0d700>


# map_diag -> map_offdiag
g = sns.PairGrid(data=iris,hue='species')
g.map_diag(sns.boxplot)
g.map_offdiag(sns.kdeplot)
     
<seaborn.axisgrid.PairGrid at 0x7fcc5e28da60>


# map_diag -> map_upper -> map_lower
g = sns.PairGrid(data=iris,hue='species')
g.map_diag(sns.histplot)
g.map_upper(sns.kdeplot)
g.map_lower(sns.scatterplot)
     
<seaborn.axisgrid.PairGrid at 0x7fcc5daaa880>


# vars
g = sns.PairGrid(data=iris,hue='species',vars=['sepal_width','petal_width'])
g.map_diag(sns.histplot)
g.map_upper(sns.kdeplot)
g.map_lower(sns.scatterplot)
     
<seaborn.axisgrid.PairGrid at 0x7fcc5ea01790>

JointGrid Vs Jointplot

sns.jointplot(data=tips,x='total_bill',y='tip',kind='hist',hue='sex')
     
<seaborn.axisgrid.JointGrid at 0x7fcc5c8c6070>


g = sns.JointGrid(data=tips,x='total_bill',y='tip')
g.plot(sns.kdeplot,sns.violinplot)
     
<seaborn.axisgrid.JointGrid at 0x7fcc5bf817f0>



     


     


     


     


     


     
Utility Functions

# get dataset names
sns.get_dataset_names()
     
['anagrams',
 'anscombe',
 'attention',
 'brain_networks',
 'car_crashes',
 'diamonds',
 'dots',
 'dowjones',
 'exercise',
 'flights',
 'fmri',
 'geyser',
 'glue',
 'healthexp',
 'iris',
 'mpg',
 'penguins',
 'planets',
 'seaice',
 'taxis',
 'tips',
 'titanic']

# load dataset
sns.load_dataset('planets')
     
method	number	orbital_period	mass	distance	year
0	Radial Velocity	1	269.300000	7.10	77.40	2006
1	Radial Velocity	1	874.774000	2.21	56.95	2008
2	Radial Velocity	1	763.000000	2.60	19.84	2011
3	Radial Velocity	1	326.030000	19.40	110.62	2007
4	Radial Velocity	1	516.220000	10.50	119.47	2009
...	...	...	...	...	...	...
1030	Transit	1	3.941507	NaN	172.00	2006
1031	Transit	1	2.615864	NaN	148.00	2007
1032	Transit	1	3.191524	NaN	174.00	2007
1033	Transit	1	4.125083	NaN	293.00	2008
1034	Transit	1	4.187757	NaN	260.00	2008
1035 rows × 6 columns



     