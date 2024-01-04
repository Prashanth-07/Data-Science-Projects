#!/usr/bin/env python
# coding: utf-8

# # Data Structures

# In[3]:


L=[1,2,3,4,4.5,"name",56]
T=(1,2,3,4,4.5,"name",56)
S={1,2,3,4,4.5,"name",56}
D={23:"twothree",'B':43,'C':"ddss"}


# In[4]:


print("the type of L is :",type(L))
print("the type of T is :",type(T))
print("the type of S is :",type(S))
print("the type of D is :",type(D))


# # how to access elements inside these ?

# In[5]:


print(L[1])
print(T[1])
print(3 in S)
print(D[23])


# In[6]:


print(D['B'])


# In[7]:


L


# In[8]:


L[1:3]


# In[9]:


L[::-1]


# In[10]:


T[1:3]


# In[11]:


L = L+["added"]
print(L)


# In[12]:


L = L + ["hi",9]


# In[13]:


print(L)


# In[16]:


L.append(6.7)


# In[17]:


print(L)


# In[15]:


T2 = ('a','b',56)
T3 = T+T2
print(T3)


# In[18]:


T3


# In[21]:


S


# In[24]:


S.add(45)
S


# In[25]:


S.update({23,"game"})
S


# In[26]:


D


# In[28]:


D['newKey'] = "newValue"
D


# In[29]:


D2 = {"Y":"hi","Z":10}
D3=D+D2
D3 # doesnt support concnatination 


# In[30]:


L


# In[31]:


del L[3]


# In[32]:


L


# In[33]:


S.remove('game')
S


# In[34]:


D


# In[35]:


del D['C']
D


# In[36]:


L


# In[37]:


L2 = L
L2


# In[38]:


L2[2] = "four point"
L2


# In[39]:


L


# both value changes as it creates reference where both lists are pointing to same memory

# In[40]:


L2=L.copy() # using this it wont reference to same memory where both have diff memory
L2


# In[43]:


L2=L2+[2]
L2


# In[44]:


L


# In[45]:


L3 = L[1:5]
L3


# In[46]:


L3[0] = "three"


# In[47]:


L3


# In[48]:


L


# In[49]:


get_ipython().run_line_magic('pinfo', 'L.append')


# In[50]:


get_ipython().run_line_magic('pinfo', 'L.pop')


# In[54]:


L.reverse()


# In[55]:


L


# In[57]:


D.items()


# In[58]:


L


# In[59]:


T


# In[60]:


S


# In[61]:


D


# In[62]:


D2 = {'a':L,'b':T,'c':S,'d':D}


# In[63]:


D2


# In[64]:


D2['a']


# In[65]:


D2['b']


# In[66]:


D2['a'][3] # gives element in third index if that List


# In[67]:


k = D2['d']


# In[68]:


k


# In[69]:


for x in k:
    print(x,k[x])


# In[70]:


L3 = [L,T,D,23,"hi"]


# In[71]:


L3


# In[72]:


type(L3[2])


# In[73]:


L3 = [x**2 for x in range(10)]


# In[74]:


L3


# In[75]:


S3={x**2 for x in range(2,20,3)}


# In[76]:


S3


# In[81]:


"""let say you are a teacher and you have diff student records 
containing id of the student and marks list in each subject where 
diff students have taken diff number of subjects.
All these records are in hard copy and you want to  eneter all the data
in computer and want to compute avg marks of each student and display""" 


def getDataFromUser():
    D={}
    while True:
        studentId = input("enter student Id: ")
        marksList = input("Enter the marks by comma seperated values : ")
        moreStudents = input('enter "no" to quit insertion: ')
        if studentId in D:
            print(studentId , " is already inserted")
        else:
            D[studentId] = marksList.split(",")
        if moreStudents.lower() == "no":
            return D


# In[83]:


studentData = getDataFromUser()


# In[84]:


studentData


# In[98]:


def getAvgMarks(D):
    avgMarks = {}
    for x in D:
        L = D[x]
        s = 0
        for marks in L:
            s += int(marks)
        avgMarks[x] = s/len(L)
    return avgMarks


# In[102]:


avgM = getAvgMarks(studentData)


# In[103]:


avgM


# In[104]:


for x in avgM:
    print("Student :",x,"got avg marks as :",avgM[x])


# # numpy

# In[105]:


import numpy as np


# In[110]:


a = np.array([1,2,3,44,5,66],dtype='i')
b = np.array((3,4,4,3,2,44),dtype='f')


# In[111]:


print(a)


# In[112]:


type(a)


# In[115]:


a.dtype


# In[116]:


a = np.array([[1,2],[3,4],[5,6]])


# In[117]:


a.ndim


# In[118]:


a[0,1]


# In[124]:


b = np.array([[2,3,9],[5,6,7,8]]) # no of elements should be consistent ie no of elemnts should be same


# In[125]:


b.ndim


# In[126]:


b = np.array([[1,2,3,9],[5,6,7,8]])


# In[127]:


b.ndim


# In[128]:


b[1,2]


# In[129]:


c = np.array([[[1,2,3],[4,5,6],[0,0,-1]],[[-1,-2,-3],[-4,-5,-6],[0,0,1]]]) # array of two dimentional array is three dimensional array


# In[130]:


c.ndim


# In[131]:


c[1,0,2]


# In[132]:


type(c)


# In[133]:


c.shape # 2 - tells that there are 2 2d arrays and 3 - tells tht there are 3 1d arrays and last 3 - tells that there are 3 elements in that 1d array


# In[134]:


A = np.array([2])


# In[135]:


A.ndim


# In[136]:


B = np.array(2)


# In[137]:


B.ndim


# In[138]:


c.size


# In[139]:


c.nbytes


# In[140]:


A = np.arange(100)


# In[141]:


print(A)


# In[142]:


A = np.arange(20,100)
print(A)


# In[143]:


A = np.arange(20,100,3)
print(A)


# In[144]:


print(range(10))


# In[145]:


print(list(range(10)))


# In[147]:


A = np.random.permutation(np.arange(10))
print(A)


# In[153]:


v = np.random.randint(20,30)
print(v)


# In[154]:


type(v)


# In[1]:


import numpy as np


# In[4]:


A = np.random.rand(1000) # btw 0 and 1


# In[5]:


A


# In[6]:


import matplotlib.pyplot as plt


# In[8]:


plt.hist(A,bins=100)


# In[9]:


B = np.random.randn(10000)
plt.hist(B,bins = 200)


# In[12]:


c = np.random.rand(2,3)


# In[14]:


c


# In[15]:


c= np.random.rand(2,3,4,3)


# In[16]:


c.ndim


# In[17]:


c


# In[21]:


d = np.arange(100).reshape(4,25)


# In[22]:


d.shape


# In[23]:


get_ipython().run_line_magic('pinfo', 'np.zeros')


# # slicing 

# In[25]:


A = np.arange(100)


# In[26]:


B = A[3:10]
print(B)


# In[28]:


B[0]=-200


# In[29]:


B


# In[30]:


A


# In[31]:


B = A[3:10].copy()


# In[34]:


A[::5]


# In[35]:


A[::-5] # starts from end


# In[36]:


A[::-1]


# In[37]:


A


# In[43]:


idx = np.argwhere(A == -200)[0][0]
idx


# In[44]:


A = np.round(10*np.random.rand(5,4))


# In[45]:


A


# In[46]:


A[1,2]


# In[47]:


A[1,:]


# In[48]:


A[:,1]


# In[49]:


A[1:3,2:4]


# In[50]:


A.T


# In[51]:


import numpy.linalg as la


# In[52]:


la.inv(np.random.rand(3,3))


# In[53]:


A


# In[54]:


A.sort(axis=0) # sorts all columns individually


# In[55]:


A


# In[56]:


A.sort(axis=1) # sorts all rows individually
A


# # masking

# In[57]:


A = np.arange(100)


# In[59]:


B = A[[3,5,6]] # this directly creates a copy
B


# In[60]:


B[0] = 9
B


# In[61]:


A


# In[62]:


B = A[A<40]


# In[63]:


B


# In[64]:


B = A[(A<40) & (A>30)] # used when lest and right side are arrays and each element can be true or false


# In[65]:


B


# In[72]:


A = np.round(10*np.random.rand(2,3))


# In[73]:


A


# In[74]:


A+3 # broadcasting


# In[75]:


A+(np.arange(2).reshape(2,1))


# In[76]:


A


# In[77]:


B =np.round(10*np.random.rand(2,2))


# In[78]:


A


# In[79]:


B


# In[80]:


C = np.hstack((A,B)) # arrays to be concatenated should be given inside a tuple


# In[81]:


C


# In[82]:


A = np.random.permutation(np.arange(10))


# In[83]:


A


# In[84]:


A.sort()


# In[85]:


A


# In[86]:


A= A[::-1] # sorting in desc ording


# In[87]:


A


# In[88]:


A=np.array(["abc","hi","helow"])


# In[89]:


A.sort() # sorting will be done based on alphabetical order


# In[90]:


A


# In[91]:


B = np.random.rand(1000000)
get_ipython().run_line_magic('timeit', 'sum(B) # python func')
get_ipython().run_line_magic('timeit', 'np.sum(B)# numpy func')


# In[92]:


def mySum(G):
    s=0
    for x in G:
        s+=x
    return s


# In[94]:


get_ipython().run_line_magic('timeit', 'mySum(B)')


# # pandas

# In[2]:


import pandas as pd


# In[3]:


print(pd.__version__)


# In[4]:


A = pd.Series([1,3,5,7],index=['a','b','c','d']) # just like dict


# In[5]:


type(A.values)


# In[6]:


type(A)


# In[7]:


A.index


# In[8]:


A['a']


# In[9]:


A['a':'c'] # here final index is included in pandas


# In[10]:


grades_dict = {'A':4,'B':3.5,'C':3,'D':2.5}
grades = pd.Series(grades_dict)


# In[11]:


grades.values


# In[12]:


marks_dict={'A':85,'B':75,'C':65}
marks=pd.Series(marks_dict) # A,B,C,D are explicit indecies


# In[13]:


marks


# In[14]:


marks['A']


# In[15]:


marks[0:2] # implicite indecies are 0,1,2....


# In[16]:


marks


# In[18]:


grades


# In[19]:


D = pd.DataFrame({'marks':marks,'grades':grades})
D # each row is 1 record mostly great using with files


# In[20]:


D.T # transpose 


# In[21]:


D.values


# In[22]:


D.values[2,0] # to access each value


# In[23]:


D.columns


# In[24]:


D['scaled_marks'] = 100*(D['marks']/90) # add column
D


# In[27]:


D['new column'] = 1,2,3,4
D


# In[28]:


del D['new column']


# In[29]:


D


# In[30]:


G =D[D['marks']>70]


# In[31]:


G


# In[32]:


S = pd.DataFrame([{'a':1,'b':2},{'b':3,'c':4}])
S


# In[33]:


S.fillna(0)


# In[34]:


S.fillna('blank')


# In[35]:


get_ipython().run_line_magic('pinfo', 'A.dropna')


# In[36]:


A = pd.Series(['a','b','c'],index=[1,3,5])


# In[37]:


A[1]


# In[38]:


A[1:3]


# In[39]:


A.loc[1:3] # using explicit indecies


# In[40]:


A.iloc[1:3]


# In[41]:


D


# In[42]:


D.iloc[2,:]


# In[43]:


D.iloc[::-1,:]


#  # matplotlib

# In[46]:


import matplotlib.pyplot as plt
import numpy as np


# In[48]:


x = np.linspace(0,10,1000)
y = np.sin(x)
plt.plot(x,y)


# In[49]:


plt.scatter(x[::10],y[::10],color='red')


# In[50]:


plt.plot(x,y,color='b')
plt.plot(x,np.cos(x),color='g')


# In[ ]:




