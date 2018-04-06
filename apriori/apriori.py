# -*- coding:utf-8 -*-

import pandas as pd 
import numpy as np
from collections import Counter
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dir = "E:/Python/MachineLearning/data/"


def itemSet_count(trade_list,sub_itemset):
   
    cnt_item = 0
    for item in trade_list:
        if set(sub_itemset).issubset(set(item)) == True:
            cnt_item = cnt_item +1
    return cnt_item


def item_join(item_set1,item_set2,n):
    if item_set1[:n-1] == item_set2[:n-1]:
       
        item_list1 = list(item_set1)
        item_list1.append(item_set2[-1])
        return tuple(item_list1)
    else:
        return ([])



if __name__ == "__main__":
    data_df = pd.read_csv(os.path.join(dir+"apriori_set.csv"),header=0,index_col="tradeID")
    #print(data_df)
    trade_dict={}
    for index in data_df.index:
        #print(data_df.loc[index])
        trade_dict[index] = data_df.loc[index].itemIDList.split(u'，') 
       
    #print(trade_dict)
    trade_list = list(trade_dict.values())
    trade_flat_list = [item for sublist in trade_list for item in sublist]

    THRSHOLD = 2


    """
    Caculate C1 and L1 for frequent itemset 1
    """
    #STEP1: static the frequence of C1
    c1_l1 = Counter(trade_flat_list)

    #STEP2: Select the frequent itemset 1
    c1_fre = {k: v for k, v in c1_l1.items() if v >= 2}
    print("Frequent itemset1 is: ")
    print(c1_fre)

    """
    Caculate C2 and L2 for frequent itemset 2
    """
    #STEP1: JOIN
    #c1_fre_sorted = sorted(c1_fre.items(),key=lambda item:item[1],reverse=True)
    c2_l2 = {}
    
    c1_fre_keys = list(c1_fre.keys())
    
    for i in range(len(c1_fre_keys) - 1):
        for j in range(len(c1_fre_keys) - 1 - i):
            c2 = (c1_fre_keys[i],c1_fre_keys[i+j+1]) ##tuple
            c2_l2[c2] = itemSet_count(trade_list, c2)
    
    #STEP2: Select the frequent itemset 2
    c2_fre = {k: v for k, v in c2_l2.items() if v >= 2}
    print("Frequent itemset2 is: ")
    print(c2_fre)

    """
    Caculate C3 and L3 for frequent itemset 3
    """
    #STEP1: JOIN
    #c1_fre_sorted = sorted(c1_fre.items(),key=lambda item:item[1],reverse=True)
    c3_l3 = {}
    
    c2_fre_keys = list(c2_fre.keys()) ##list of list
    
    for i in range(len(c2_fre_keys) - 1):
        for j in range(len(c2_fre_keys) - 1 - i):
            c3 = item_join(c2_fre_keys[i],c2_fre_keys[i+j+1],n=2) ##tuple
            if c3:
                c3_l3[c3] = itemSet_count(trade_list, c3)
    
    #STEP2: Select the frequent itemset 2
    c3_fre = {k: v for k, v in c3_l3.items() if v >= 2}
    print("Frequent itemset3 is: ")
    print(c3_fre)



