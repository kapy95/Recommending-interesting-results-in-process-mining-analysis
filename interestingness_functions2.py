#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pm4py
from scipy.stats import entropy
from collections import Counter
from operator import itemgetter
from scipy import stats
import numpy as np
import math
import time
import numbers


# In[2]:


# log = pm4py.read_xes('DomesticDeclarations.xes_')
# dataframe = pm4py.convert_to_dataframe(log)


# In[3]:


#Counter(dataframe['org:role'].dropna().values)


# In[ ]:


def indexNumericalValue(dataframe,attributes):
    labels={}
    bins={}
    
    for attribute in attributes:
        nbinsCase=round(freedman_diaconis(dataframe[attribute],"number"))
        #widthCaseAmount=round(freedman_diaconis(dataframe["case:Amount"],"width"))
        discretized_case=pd.cut(dataframe[attribute].values,nbinsCase).value_counts().sort_values(ascending=False)
        bins[attribute]=nbinsCase
        labels[attribute]=discretized_case.index.categories
    
    return labels,bins


# In[1]:


def freedman_diaconis(data, returnas):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``return`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returns: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    elif returnas=="number":
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
        
    return result


# In[4]:


def normBoxScore(data):

    #first the boxcox function tries to normalize the data
    boxcox=stats.boxcox(data)#boxcox returns an array (position 0) and a value related to the boxcox transformation (position 1)    
    #after that the zscore function is also applied to reduce the scale of the data 
    results_normalized=stats.zscore(boxcox[0])
    
    return results_normalized


# In[ ]:


def discretizeContinousVariable(data,nbins,labels):
    #nbins=round(freedman_diaconis(data,"number"))
    discretized_data=pd.cut(data,nbins,labels=labels)
    df_vc=discretized_data.value_counts()
    df_vc=df_vc.replace(0,np.nan).dropna()
    df_vc=pd.Series(df_vc)
    return df_vc


# In[5]:


def compute_interestingness_kl_divergence(df_new,df_prev,columns,group):
        '''
        #Returns 1-log(3)^max_{KL_div_attr for each attribute in the current dataframe}(-KL_div_attr)
        Returns sigmoid(max_{KL_div_attr for each attribute in the current datafame}/2-3)
        :param dfs:
        :param state:
        :return:
        '''
        kl_distances = []
        
        if group==True:
            aggregate_attributes_list = get_aggregate_attributes(state)
            kl_attrs = aggregate_attributes_list
            
        else:
            KL_DIV_EPSILON = 2 / len(df_prev) * 0.1#valor que se utilizara para sustituir valores nulos
            kl_attrs = columns#saca los atributos del dataframe

        # compute KL_divergence for each attribute
        for attr in kl_attrs:
            # attr_value_count1 = df_D[attr].value_counts().to_dict()
            # attr_value_count2 = df_dt[attr].value_counts().to_dict()
            variable = 5

            if isinstance(df_new[attr][0], numbers.Number):
                #nbins_new=freedman_diaconis(df_new[attr])
                attr_value_count1 = Counter(df_new[attr].dropna().values)
                attr_value_count2 = Counter(df_prev[attr].dropna().values)
            else:
                attr_value_count1 = Counter(df_new[attr].dropna().values)
                attr_value_count2 = Counter(df_prev[attr].dropna().values)

#             attr_value_count1 = CounterWithoutNanKeys(df_D[attr].values)
#             attr_value_count2 = CounterWithoutNanKeys(df_dt[attr].values)

            if group==True:
                KL_DIV_EPSILON = 2 / sum(attr_value_count1.elements()) * 0.1

            '''if not is_grouping:
                num_of_NaNs_1 = len(df_D) - sum(attr_value_count1.values())
                num_of_NaNs_2 = len(df_dt) - sum(attr_value_count2.values())'''

            pk1 = []
            pk2 = []
            for key in attr_value_count1:
                pk1.append(attr_value_count1[key])
                if key in attr_value_count2:
                    pk2.append(attr_value_count2[key])
                else:
                    pk2.append(KL_DIV_EPSILON)

            # add the rest of attributes not in attr_value_count1
            for key in attr_value_count2:
                if key not in attr_value_count1:
                    pk2.append(attr_value_count2[key])
                    pk1.append(KL_DIV_EPSILON)

            # add NaNs number for non-grouping case
            '''if not is_grouping:
                if num_of_NaNs_1 != 0 or num_of_NaNs_2 != 0:
                    num_of_NaNs_1 = num_of_NaNs_1 if num_of_NaNs_1 != 0 else KL_DIV_EPSILON
                    num_of_NaNs_2 = num_of_NaNs_2 if num_of_NaNs_2 != 0 else KL_DIV_EPSILON
                    pk1.append(num_of_NaNs_1)
                    pk2.append(num_of_NaNs_2)'''

            attr_kl_div = entropy(pk1, pk2)
            kl_distances.append((attr,attr_kl_div,"kullback"))
        # return 1-math.log(3)**(-max(kl_distances))
        #kl_distances.sort(key=itemgetter(1))
        return kl_distances
        #return max(kl_distances,key=itemgetter(1)) #1 / (1 + math.exp(-(max(kl_distances) / 2 - 3)))-> versión sigmoide


# In[ ]:


def compute_interestingness_kl_divergence2(df_new,df_prev,columns,group,nbinsVar,allLabels):
        '''
        #Returns 1-log(3)^max_{KL_div_attr for each attribute in the current dataframe}(-KL_div_attr)
        Returns sigmoid(max_{KL_div_attr for each attribute in the current datafame}/2-3)
        :param dfs:
        :param state:
        :return:
        '''
        kl_distances = []
        
        if group==True:
            aggregate_attributes_list = get_aggregate_attributes(state)
            kl_attrs = aggregate_attributes_list
            
        else:
            KL_DIV_EPSILON = 2 / len(df_prev) * 0.1#valor que se utilizara para sustituir valores nulos
            kl_attrs = columns#saca los atributos del dataframe

        # compute KL_divergence for each attribute
        for attr in kl_attrs:
            # attr_value_count1 = df_D[attr].value_counts().to_dict()
            # attr_value_count2 = df_dt[attr].value_counts().to_dict()
            variable = 5

            if isinstance(df_new[attr][0], numbers.Number):
                labels=allLabels[attr]
                nbins=nbinsVar[attr]
                attr_value_count1 = discretizeContinousVariable(df_new[attr],nbins,labels)
                attr_value_count2 = discretizeContinousVariable(df_prev[attr],nbins,labels)
            else:
                attr_value_count1 = Counter(df_new[attr].dropna().values)
                attr_value_count2 = Counter(df_prev[attr].dropna().values)

#             attr_value_count1 = CounterWithoutNanKeys(df_D[attr].values)
#             attr_value_count2 = CounterWithoutNanKeys(df_dt[attr].values)

            if group==True:
                KL_DIV_EPSILON = 2 / sum(attr_value_count1.elements()) * 0.1

            '''if not is_grouping:
                num_of_NaNs_1 = len(df_D) - sum(attr_value_count1.values())
                num_of_NaNs_2 = len(df_dt) - sum(attr_value_count2.values())'''

            pk1 = []
            pk2 = []
            for key in attr_value_count1:
                pk1.append(attr_value_count1[key])
                if key in attr_value_count2:
                    pk2.append(attr_value_count2[key])
                else:
                    pk2.append(KL_DIV_EPSILON)

            # add the rest of attributes not in attr_value_count1
            for key in attr_value_count2:
                if key not in attr_value_count1:
                    pk2.append(attr_value_count2[key])
                    pk1.append(KL_DIV_EPSILON)

            # add NaNs number for non-grouping case
            '''if not is_grouping:
                if num_of_NaNs_1 != 0 or num_of_NaNs_2 != 0:
                    num_of_NaNs_1 = num_of_NaNs_1 if num_of_NaNs_1 != 0 else KL_DIV_EPSILON
                    num_of_NaNs_2 = num_of_NaNs_2 if num_of_NaNs_2 != 0 else KL_DIV_EPSILON
                    pk1.append(num_of_NaNs_1)
                    pk2.append(num_of_NaNs_2)'''

            attr_kl_div = entropy(pk1, pk2)
            kl_distances.append((attr,attr_kl_div,"kullback"))
        # return 1-math.log(3)**(-max(kl_distances))
        #kl_distances.sort(key=itemgetter(1))
        return kl_distances


# In[ ]:


def discretizeContinousVariable(data,nbins,label):
    #nbins=round(freedman_diaconis(data,"number"))
    discretized_data=pd.cut(data,nbins,labels=label)
    df_vc=discretized_data.value_counts().replace(0,np.nan).dropna().to_dict()
    return df_vc


# In[6]:


# df_not_system=dataframe[dataframe["org:resource"]!="SYSTEM"]

# compute_interestingness_kl_divergence(df_not_system,dataframe,["org:resource","case:Amount"],False)


# In[7]:


#def compute_interestingness_variance(df,columns, indexes):
    
#     listVars=[]
    
#     for column in columns:
        
#         listValues=df[column]
#         print(column)
#         if isinstance(listValues[0],str)==True:
#             frec=df.groupby(column).apply(lambda x: len(x))
#             listVars.append((column,np.var(frec),"variance"))
#         else:
#             listVars.append((column,np.var(listValues),"variance"))
        
#     for ind in indexes:
#         listValues=df.index.get_level_values(ind).unique()
        
#         if isinstance(listValues[0],str)==True:
#             frec=df.groupby(ind).apply(lambda x: len(x))
#             listVars.append((ind,np.var(frec),"variance"))
#         else:
#             listVars.append((ind,np.var(listValues),"variance"))
        
        
#     #return max(listaVars,key=itemgetter(1)) returns only one
#     return listVars.sort(key=itemgetter(1))


# In[8]:


# def compute_interestingness_variance(df,columns, indexes):
    
#     listVars=[]
#     listColumnsIndex=[]
    
#     for column in columns:
        
#         listValues=df[column]
#         print(column)
#         if isinstance(listValues[0],str)==True:
#             frec=df.groupby(column).apply(lambda x: len(x))
#             listVars.append(np.var(frec))
#             listColumnsIndex.append((column,"variance"))
#         else:
#             listVars.append(np.var(listValues))
#             listColumnsIndex.append((column,"variance"))
            
        
#     for ind in indexes:
#         listValues=df.index.get_level_values(ind).unique()
        
#         if isinstance(listValues[0],str)==True:
#             frec=df.groupby(ind).apply(lambda x: len(x))
#             listVars.append(np.var(frec))
#             listColumnsIndex.append((ind,"variance"))

#         else:
#             listVars.append(np.var(listValues))
#             listColumnsIndex.append((ind,"variance"))
        
#     listVars=normBoxScore(listVars)
#     df = pd.DataFrame({'interestValues':listVars,'columnMeasure':listColumnsIndex})
    
#     return df


# In[9]:


def variance(frec):
    valVariance=np.var(frec)
#     if isinstance(df[field][0],str)==True:
#         frec=df[field].value_counts()
#         valVariance=np.var(frec)

#     else:
#         valVariance=np.var(df[field])
    #print(valVariance)
    return valVariance
    


# In[10]:


import itertools

def shapleyFormula(fun,field,dataframe):
    
    args=dataframe[field].unique()
    N=np.math.factorial(len(args))
    shapleyValues={}
    valuesFunc={} #first all values are calculated depending on the possible combinations
    
    #for example if we have {A,B,C}
    for j in range(1,len(args)+1):#counter to generate all possible combinations
        #1->{A}, 2->{A,B}, 3->{A,B,C}
        #if you want to check what itertools does, execute this code:
#         for i in itertools.combinations(['A','B','C','D'], 2):
#             print(i)
        for i in itertools.combinations(args, j):#generate all possible combinations of size i
            #print(i)
            #if i is 2-> {A,B}, {B,C}, {C,A}
            dataframeComb=dataframe[dataframe[field].isin(i)].reset_index()#filter the rows that are included in the combination
            joined_string = "".join(i)#concatenation to generate the key
            valuesFunc[joined_string]=fun(dataframeComb,field)#calculate the value for that rows and addition of the key 
    
    for val in args:#for each value (for example A)
        #print(val)
        print()
        keys=valuesFunc.keys()#A,B,C,D,AB,CD...ABC,
        valKeys=[key for key in keys if val in key]#get the combinations where the value is involved
        #for instance, in the case of A: A, AB,CA, ABC
        values=[]

        for key in valKeys:#for each key (combination)
            comb=key.replace(val,"")#get the case where A is not involved in S-> AB would be B, or ABC->BC
            
#             print(key)
#             print(comb)
            if comb!="":#if the other case is not null (A->"")
                value2=valuesFunc[comb] #val(BC)
            else: 
                value2=0
            #get the other combination
            #valuesShapley1[key]->val(Pr u Xi), valuesShapley1[comb]-> Val(Pr)
            #valuesFunc[key]-> val(ABC)
            values.append(valuesFunc[key]-value2)#val(Pr u Xi) - Val(Pr)
            
        shapleyValues[val]=sum(values)/N
        
    return shapleyValues
                
#args array de valores unicos    
#     #args=log[field].unique()#{a,b,c}
#     valShapleys=[]
#     for val in args:#{a}
#         print(val)
#         args2=args
#         args2.remove(val)

#         for z in range(0,len(args2)):
#             #print("z="+str(z))
            
#             #valShapleys.append(fun(arg3))#int(A)
            
#             for j in range(z+1,len(args2)+1):
#                 #print("j="+str(j))
#                 arg3=[]#{}
#                 arg3.append(val)#{a}
#                 arg4=arg3+args2[z:j]
#                 print(arg4)
#                 #log[log[field]].value_counts/len(filtrado)
    


# In[11]:


#shapleyFormula(variance,"org:resource",dataframe)


# In[12]:


# dataframe["org:role"].value_counts()


# In[13]:


# import time
# start_time = time.time()
# dic=shapleyFormula(variance,"org:role",dataframe)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[14]:


# start_time = time.time()
# dic2=shapleyFormula(variance,"concept:name",dataframe)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[15]:


#dispersion
def schutzFormula(counts):
    
    #it receives value_counts
    if sum(counts)!=0:
        probs=counts/sum(counts)
        numberTuples=len(probs)
        q=1/numberTuples

    #     if np.isfinite(probs)==False:
    #         print("infinite")
        #print(counts)
        #schutz=float(1-sum([prob-q for prob in probs])/(2*numberTuples*q))
        schutz=sum([abs(prob-q) for prob in probs])/(2*numberTuples*q)
    else:    
        schutz=0
        
    return schutz 


# In[16]:


def compute_interestingness_formula(df,fun,columns,indexes,name,inverted=False):
    #inverted is a boolen stating that the function assess low values better than high values (i.e, the order is inverted)
    listVars=[]
    listColumnsIndex=[]
    listVariables=[]
    listFun=[]
    
    for column in columns:
        
        listValues=df[column]
        #print(column)
        if isinstance(listValues[0],str)==True:
            frec=df.groupby(column).apply(lambda x: len(x))
            listVars.append(fun(frec))
            listFun.append(name)
            listColumnsIndex.append(column)

            
        else:
            listVars.append(fun(listValues))
            listFun.append(name)
            listColumnsIndex.append(column)

            
        
    for ind in indexes:
        listValues=df.index.get_level_values(ind).unique()
        
        if isinstance(listValues[0],str)==True:
            frec=df.groupby(ind).apply(lambda x: len(x))
            listVars.append(fun(frec))
            listFun.append(name)
            listColumnsIndex.append(ind)


        else:
            listVars.append(fun(listValues))
            listFun.append(name)
            listColumnsIndex.append(ind)

    listVars=normBoxScore(listVars)
    
    if inverted==True:
        listVars=listVars*-1
        
    df = pd.DataFrame({'interestValue':listVars,'column':listColumnsIndex,'function':listFun})
    
    return df


# In[ ]:


def compute_interestingness_formula2(df,fun,columns,indexes,name,allLabels,nbinsVar,inverted=False,freq=False):
    #inverted is a boolen stating that the function assess low values better than high values (i.e, the order is inverted)
    listVars=[]
    listColumnsIndex=[]
    listVariables=[]
    listFun=[]
    
    for column in columns:
        
        listValues=df[column]
        #print(column)
        if isinstance(listValues[0],str)==True:
            frec=df.groupby(column).apply(lambda x: len(x))
            listVars.append(fun(frec))
            listFun.append(name)
            listColumnsIndex.append(column)
            
        elif freq==True:
            labels=allLabels[column]
            nbins=nbinsVar[column]
            numeric_freq = discretizeContinousVariable(df[column],nbins,labels)
            #print(numeric_freq)
            listVars.append(fun(numeric_freq))
            listFun.append(name)
            listColumnsIndex.append(column)
            
        else:
            listVars.append(fun(listValues))
            listFun.append(name)
            listColumnsIndex.append(column)

            
        
    for ind in indexes:
        listValues=df.index.get_level_values(ind).unique()
        
        if isinstance(listValues[0],str)==True:
            frec=df.groupby(ind).apply(lambda x: len(x))
            listVars.append(fun(frec))
            listFun.append(name)
            listColumnsIndex.append(ind)


        else:
            listVars.append(fun(listValues))
            listFun.append(name)
            listColumnsIndex.append(ind)

    listVars=normBoxScore(listVars)
    
    if inverted==True:
        listVars=listVars*-1
        
    df = pd.DataFrame({'interestValue':listVars,'column':listColumnsIndex,'function':listFun})
    
    return df


# In[17]:


# counts_res_act=dataframe.groupby(["org:resource"])["concept:name"].value_counts()


# In[18]:


#schutzFormula(counts_res_act)


# In[19]:


def support(df,columns):
    
    #dataframe.drop(["id","time:timestamp","case:id","case:Amount"],axis=1).apply(pd.Series.value_counts)
    counts = []
    results=[]
    listColumns=[]
    listFun=[]
    
    for column in columns:

        max_val=df[column].value_counts().sort_values(ascending=False)[[0]]
        counts.append(max_val[[0]].values[0])
        results.append(str(max_val.index[0]))
        listColumns.append(column)
        listFun.append("support")
       
    counts=normBoxScore(counts)
    df = pd.DataFrame({'interestValue':counts,'column':listColumns,'function':listFun,"concreteValue":results})
        
    return df


# In[20]:


# res_support=support(dataframe.drop(["id","time:timestamp","case:id","case:Amount"],axis=1))


# In[21]:


# normBoxScore(res_support)

