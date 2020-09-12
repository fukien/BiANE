#! ~/conda3/bin/python3
# -*- coding:utf-8 -*-

import datetime

def get_batch_data(list_of_lists, batch_idx, batch_size):
    list_of_returned_lists = []
    for l in list_of_lists:
        list_of_returned_lists.append(l[batch_idx * batch_size: (batch_idx+1) * batch_size])
    return list_of_returned_lists

# get the fromatted string of time
def date_time_format():
    return datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')

# round function, saving 5 digits
# sometimes need to add one bit 0 
# to the end of the float
def round_up_4(x):
    return round(x*10000)/10000.0