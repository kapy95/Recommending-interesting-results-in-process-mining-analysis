#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pm4py
import time
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np


# ### Filtrar actividades por frecuencia

# In[3]:


def filterActivitiesByFrequency(dataframe, activities, frequency):    
    """
    This function filters the log, eliminating the activities whose frequency is less than the given value
    
    Inputs:
    dataframe: dataframe which represents the log
    activities: name of the column of the activities in the dataframe
    frequency: value by which activities whose frequency exceeds this value will be filtered 
    
    Output:
    The dataframe resulting from filtering the activities by frequency
    
    """
    freq = dataframe[activities].value_counts()
    dataframe_filter = dataframe[dataframe[activities].isin(freq[freq > frequency].index)]
    return(dataframe_filter) 


# ### Filtrar dataframe por frecuencia de paths

# In[5]:


def filterPathsByFrequency(dataframe, case_id, activities, frequency):
    """
    This function filters the log, eliminating the paths o transitions whose frequency is less than the given value
    
    Inputs:
    dataframe: dataframe which represents the log
    case_id: name of the column of the activities in the dataframe
    activities: name of the column with the IDs of the traces in the dataframe
    frequency: value by which paths whose frequency exceeds this value will be filtered 
    
    Output:
    The dataframe resulting from filtering the paths by frequency
    
    """
    path_df=dataframe
    tuples1=[]
    tuples2=[]
    tuples2.append("-")
    
    for i in range(len(path_df)-1): 
        fila=path_df.iloc[i]
        fila_siguiente=path_df.iloc[i+1]
        fila_anterior=path_df.iloc[i+-1]
 
        if fila[case_id]==fila_siguiente[case_id]:
            tuples1.append((fila[activities],fila_siguiente[activities]))
            tuples2.append((fila[activities],fila_siguiente[activities]))
                    
        else:
            tuples1.append("-")
            tuples2.append("-")
            
    tuples1.append("-")
    
    path_df['Transition1']=tuples1
    path_df['Transition2']=tuples2
    
    freq = path_df['Transition1'].value_counts()
    path_df_filter = path_df[path_df['Transition1'].isin(freq[freq > frequency].index)]
    path_df_filter = path_df_filter[path_df_filter['Transition2'].isin(freq[freq > frequency].index)]
    path_df_filter
    
    return (path_df_filter)


# ### Filtrar dataframe por frecuencia de actividades y paths

# In[7]:


def filterActivitiesAndPathsByFrequency(dataframe, case_id, activities, freq_activities, freq_paths):
    """
    This function filters the log, eliminating the activities and the paths whose frequencies is less than the given values
    
    Inputs:
    dataframe: dataframe which represents the log
    case_id: name of the column of the activities in the dataframe
    activities: name of the column with the IDs of the traces in the dataframe
    freq_activities: value by which activities whose frequency exceeds this value will be filtered 
    freq_paths: value by which paths whose frequency exceeds this value will be filtered 
    
    Output:
    The dataframe resulting from filtering the activities and the paths by frequencies
    """
    
    freq = dataframe[activities].value_counts()
    dataframe_filter = dataframe[dataframe[activities].isin(freq[freq > freq_activities].index)]
    
    tuples1=[]
    tuples2=[]
    tuples2.append("-")
    
    for i in range(len(dataframe_filter)-1): 
        fila=dataframe_filter.iloc[i]
        fila_siguiente=dataframe_filter.iloc[i+1]
        fila_anterior=dataframe_filter.iloc[i+-1]
 
        if fila[case_id]==fila_siguiente[case_id]:
            tuples1.append((fila[activities],fila_siguiente[activities]))
            tuples2.append((fila[activities],fila_siguiente[activities]))
                    
        else:
            tuples1.append("-")
            tuples2.append("-")
            
    tuples1.append("-")
    
    dataframe_filter['Transition1']=tuples1
    dataframe_filter['Transition2']=tuples2
    
    freq = dataframe_filter['Transition1'].value_counts()
    path_df_filter = dataframe_filter[dataframe_filter['Transition1'].isin(freq[freq > freq_paths].index)]
    path_df_filter = path_df_filter[path_df_filter['Transition2'].isin(freq[freq > freq_paths].index)]
    path_df_filter
    
    return (path_df_filter)


# ### Filtrar dataframe por orden eventual de actividades

# In[9]:


def eventualOrderOfActivities(dataframe, case_id, activities, activity_1, activity_2, order):
    """
    This function filters the log depending on the eventual order of two given activities
    
    Inputs:
    dataframe: dataframe which represents the log
    case_id: name of the column of the activities in the dataframe
    activities: name of the column with the IDs of the traces in the dataframe
    activity_1: name of the first activity of interest 
    activity_1: name of the second activity of interest 
    order: it indicates whether or not the first activity should be followed eventually (at any time) by the second activity
        order=='TRUE': the first activity should eventually be followed by the second
        order=='FALSE': the first activity should not eventually be followed by the second
    
    Output:
    The dataframe resulting from filtering the traces in which the order condition is not included
    
    """
    lista=[]
    lista2=[]

    for case in (dataframe[case_id].unique()):
        traza = dataframe[dataframe[case_id]==case]
        for evento in range(len(traza)):
            if(traza.iloc[evento][activities]==activity_1):
                evento2=evento+1
                subtraza = traza[evento2:len(traza)]
                for evento3 in range(len(subtraza)):
                    if(subtraza.iloc[evento3][activities]==activity_2):
                        lista.append(case)
                        break
                    else:
                        lista2.append(case)
                        
    d_orden_eventual = dataframe[dataframe[case_id].isin(lista)]  
    d_no_orden_eventual = dataframe[dataframe[case_id].isin(lista2)]
    
    if(order=='TRUE'):   
        return (d_orden_eventual)
    elif(order=='FALSE'):
        return(d_no_orden_eventual)


# In[27]:


def filterDirectTransition(df,case_id,act1,act2):
    """
    This function filters the cases of a log where an activity is directly followed by another activity
    
    Inputs:
    df: dataframe which represents the log
    case_id: name of the column of the activities in the dataframe
    act1: activity that happens first.
    act2: activity that must occur immediately after the act1 
    
    Output:
    Cases which have that pattern
    
    """
    #df['Transition']->column where the sucession of activities is registered in form of tuples
    #Example: act1->act2->ac3 (there will be two tuples representing the sucessions (act1,act2),(act2,act3))
    #df['Transition'].isin([(act1,act2)]->checks which rows  are contained that pattern
    #df[df['Transition'].isin([(act1,act2)])]-> filter the rows contained in that pattern
    #df[df['Transition'].isin([(act1,act2)])][case_id].unique()-> get the case_ids (without repetition) of the filtered rows
    case_id_validos=df[df['Transition'].isin([(act1,act2)])][case_id].unique()
    
    
    #get all the rows that are in these cases
    df_filtrado=df[df[case_id].isin(case_id_validos)]
    
    return df_filtrado


# In[30]:


def filterNotDirectTransition(df,case_id,act1,act2):
    """
    This function filters the cases of a log where an activity is never directly followed by another activity
    
    Inputs:
    df: dataframe which represents the log
    case_id: name of the column of the activities in the dataframe
    act1: activity that happens first.
    act2: activity that must occur immediately after the act1 
    
    Output:
    Cases which do not have that pattern
    
    """
    #df['Transition']->column where the sucession of activities is registered in form of tuples
    #Example: act1->act2->ac3 (there will be two tuples representing the sucessions (act1,act2),(act2,act3))
    #df['Transition'].isin([(act1,act2)]->checks which rows contain that pattern
    #df[df['Transition'].isin([(act1,act2)])]-> filter the rows contained in that pattern
    #df[df['Transition'].isin([(act1,act2)])][case_id].unique()-> get the case_ids (without repetition) of the filtered rows
    case_id_no_validos = df[df['Transition'].isin([(act1,act2)])][case_id].unique()
    
    #df[~df[case_id].isin(case_id_no_validos)]->filter the rows which do not have that cases 
    #df[~df[case_id].isin(case_id_no_validos)][case_id].unique()->get the case IDs of the rows that do not contain that pattern
    case_id_validos=df[~df[case_id].isin(case_id_no_validos)][case_id].unique()
    df_filtrado2=df[df["case:id"].isin(case_id_validos)]#filter the rows of the cases that do not contain that pattern
    
    return df_filtrado2


# In[1]:


def filterCasesByActivities(df,activities,case_id,acts):
    """
    This function filters the cases of a log, which exist an activity or multiple activities
    
    Inputs:
    df: dataframe which represents the log
    activities: name of the column of the dataframe which contain the activities
    case_id: name of the column of the activities in the dataframe
    acts: activity or activities that must occur
    
    Output:
    Cases which have that pattern
    
    """
    #df[activities]->column which represents the activities
    #df[activities].isin(acts)->checks which rows contained that activity
    #df[df[activities].isin(acts)]-> filter the rows contained in that pattern
    #df[df[activities].isin(acts)][case_id].unique()-> get the case_ids (without repetition) of the filtered rows
    case_id_validos3=df[df[activities].isin(acts)][case_id].unique()
    
    
    #get all the rows that are in these cases
    df_filtrado3=df[df[case_id].isin(case_id_validos3)]
    
    return df_filtrado3


# In[25]:


def filterCasesWithoutActivities(df,activities,case_id,acts):
    """
    This function filters the cases of a log, which do not exist an activity or multiple activities
    
    Inputs:
    df: dataframe which represents the log
    activities: column of the dataframe which contain the activities
    case_id: name of the column of the activities in the dataframe
    acts: activity or activities that must not occur
    
    Output:
    Cases which do not have that pattern
    
    """
    
    #df[activities]->column which represents the activities
    #df[activities].isin(acts)->checks which values are contained in the array of activities that must not occur
    #df[df[activities].isin(acts)]-> filter the rows which accomplished the previous condition 
    #df[df[activities].isin(acts)][case_id].unique()-> get the case_ids (without repetition) of the filtered rows
    case_id_no_validos4=df[df[activities].isin(acts)][case_id].unique()
    
    #filter the rows whose case IDs are not in the invalid case IDs
    case_id_validos4=df[~df[case_id].isin(case_id_no_validos4)][case_id].unique()
    df_filtrado4=df[df[case_id].isin(case_id_validos4)]#filter the rows of the cases that do not contain that activities
    
    
    
    return df_filtrado4
    
