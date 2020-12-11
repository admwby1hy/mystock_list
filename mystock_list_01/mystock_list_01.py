
from mystock_list import *
import time
from numba import jit
import numpy as np
#starttime = time.process_time()

#pd.set_option('display.width', 1000)  # 设置字符显示宽度
#pd.set_option('display.max_rows', None)  # 设置显示最大行






shanghailist = get_shanghai_from_tushare()
shanghaiarray = np.asarray(shanghailist, dtype=np.float32)
#a = np.array(shanghailist)
#print(a[1])
starttime = time.process_time()
for i in range(1000):
    get_ndays_average_shanghai_pb_dif1(shanghaiarray,1220-i)
endtime = time.process_time()
print (endtime - starttime)
"""
starttime = time.process_time()
for i in range(1000):
    get_ndays_average_shanghai_pb_dif(shanghailist,1220)
endtime = time.process_time()
print (endtime - starttime)

stockcode  = input("请输入证券代码(601398):")
if len(stockcode) == 0:
    stockcode = '601398'

strndays = input("请输入ndays(1220):")
if len(strndays) == 0:
    ndays = 244 * 5
else:
    ndays = int(strndays)

strjustvalue = input("请输入justvalue(0):")
if len(strjustvalue) == 0:
    justvalue = 0.0
else:
    justvalue = round(float(strjustvalue),2)

"""