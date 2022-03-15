#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# # Data Collection and PreProcessing

# In[2]:


# loading the data from the csv file to apandas dataframe
movies_info = pd.read_csv('/Users/saurabh453/Desktop/MRS/movies_data.csv')
movies_info


# In[3]:


# printing the first 5 rows of the dataframe
movies_info.head()


# In[4]:


# number of rows and columns in the data frame

movies_info.shape


# In[5]:


# selecting the relevant features for recommendation

columns_selected = ['genres','keywords','tagline','cast','director']
print(columns_selected)


# In[6]:


# replacing the null valuess with null string

for feature in columns_selected:
  movies_info[feature] = movies_info[feature].fillna('')


# In[7]:


# combining all the 5 selected features

combined_columns = movies_info['genres']+' '+movies_info['keywords']+' '+movies_info['tagline']+' '+movies_info['cast']+' '+movies_info['director']


# In[8]:


print(combined_columns)


# In[9]:


# converting the text data to feature vectors

vectorizer = TfidfVectorizer()


# In[10]:


column_vectors = vectorizer.fit_transform(combined_columns)


# In[11]:


print(column_vectors)


# # Cosine Similarity

# In[12]:


# getting the similarity scores using cosine similarity

cosinesimilarity = cosine_similarity(column_vectors)


# In[13]:


print(cosinesimilarity)


# In[14]:


print(cosinesimilarity.shape)


# # Getting movie name from user  

# In[17]:


# getting the movie name from the user

movie_name = input('Favorite movie name : ')


# In[18]:


# creating a list with all the movie names given in the dataset

titles_list = movies_info['title'].tolist()
print(titles_list)


# In[19]:


# finding the close match for the movie name given by the user

getting_close_match = difflib.get_close_matches(movie_name, titles_list)
print(getting_close_match)


# In[20]:


close_match = getting_close_match[0]
print(close_match)


# In[21]:


# finding the index of the movie with title

movie_index = movies_info[movies_info.title == close_match]['index'].values[0]
print(movie_index)


# In[22]:


# getting a list of similar movies

similarity_score = list(enumerate(cosinesimilarity[movie_index]))
print(similarity_score)


# In[24]:


len(similarity_score)


# In[25]:


# sorting the movies based on their similarity score

similar_movies_sorted = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(similar_movies_sorted)


# In[ ]:


# print the name of similar movies based on the index

print('Suggested Movies: \n')

i = 1

for movie in similar_movies_sorted:
  index = movie[0]
  title_index = movies_info[movies_info.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_index)
    i+=1


# # Movie Recommendation System

# In[ ]:



movie_name = input('Favorite movie name : ')

titles_list = movies_info['title'].tolist()

getting_close_match = difflib.get_close_matches(movie_name, titles_list)

close_match = getting_close_match[0]

movie_index = movies_info[movies_info.title == close_match]['index'].values[0]

similarity_score = list(enumerate(cosinesimilarity[movie_index]))

similar_movies_sorted = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Suggested Movies: \n')

i = 1

for movie in similar_movies_sorted:
  index = movie[0]
  title_index = movies_info[movies_info.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_index)
    i+=1


# In[ ]:





# In[ ]:





# In[ ]:




