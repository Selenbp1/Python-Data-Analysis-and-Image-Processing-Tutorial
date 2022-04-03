#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Python의 Numpy 라이브러리는 List와 상호 변환이 가능합니다.
import numpy as np

list_data = [1,2,3]
array = np.array(list_data)

print(array.size) # 배열의 크기
print(array.dtype) # 배열 원소의 타입
print(array[2]) # 인덱스 2의 원소


# In[3]:


#Python의 Numpy 라이브러리는 다양한 형태의 배열을 초기화 할 수 있습니다.
import numpy as np

# 0부터 3까지의 배열 만들기
array1 = np.arange(4)
print(array1)

# 0으로 초기화
array2 = np.zeros((4, 4), dtype=float)
print(array2)

# 1로 초기화
array3 = np.ones((3, 3), dtype=str)
print(array3)

# 0부터 9까지 랜덤하게 초기화 된 배열 만들기
array4 = np.random.randint(0, 10, (3, 3))
print(array4)

# 평균이 0이고 표준편차가 1인 표준 정규를 띄는 배열
array5 = np.random.normal(0, 1, (3, 3))
print(array5)


# In[3]:


#Python의 Numpy 라이브러리는 다양한 형태의 배열을 초기화 할 수 있습니다.
import numpy as np

# 0부터 3까지의 배열 만들기
array1 = np.arange(4)
print(array1)

# 0으로 초기화
array2 = np.zeros((4, 4), dtype=float)
print(array2)

# 1로 초기화
array3 = np.ones((3, 3), dtype=str)
print(array3)

# 0부터 9까지 랜덤하게 초기화 된 배열 만들기
array4 = np.random.randint(0, 10, (3, 3))
print(array4)

# 평균이 0이고 표준편차가 1인 표준 정규를 띄는 배열
array5 = np.random.normal(0, 1, (3, 3))
print(array5)


# In[4]:


#Numpy는 다양한 형태로 합치기가 가능 있습니다.
import numpy as np

array1 = np.array([1, 2, 3]) 
array2 = np.array([4, 5, 6])
array3 = np.concatenate([array1, array2])

print(array3.shape)
print(array3)


# In[8]:


import numpy as np
#Numpy를 위 아래로 합칠 수 있습니다.
array1 = np.arange(4).reshape(1, 4)
array2 = np.arange(8).reshape(2, 4)
array3 = np.concatenate([array1, array2], axis=0)

print(array3.shape)
print(array3)


# In[7]:


#Numpy의 형태를 변경할 수 있습니다.
import numpy as np

array1 = np.array([1, 2, 3, 4])
array2 = array1.reshape((2, 2))
print(array2.shape)


# In[10]:


#Numpy의 형태를 나누기 할 수 있습니다.
import numpy as np

array = np.arange(8).reshape(2, 4)
left, right = np.split(array, [2], axis=1)

print(left.shape)
print(right.shape)
print(array)
print(left)
print(right[1][1])


# In[ ]:




