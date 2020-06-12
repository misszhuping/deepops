from hmmlearn.hmm import GaussianHMM
import datetime
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import seaborn as sns
sns.set_style('white')

#data process
beginDate = '20100401'
endDate = '20160317'
data = DataAPI.MktIdxdGet(ticker='000001',beginDate=beginDate,endDate=endDate,field=['tradeDate','closeIndex','lowestIndex','highestIndex','turnoverVol'],pandas="1")
data1 = DataAPI.FstTotalGet(exchangeCD=u"XSHE",beginDate=beginDate,endDate=endDate,field=['tradeVal'],pandas="1")
data2 = DataAPI.FstTotalGet(exchangeCD=u"XSHG",beginDate=beginDate,endDate=endDate,field=['tradeVal'],pandas="1")
tradeVal = data1 + data2
tradeDate = pd.to_datetime(data['tradeDate'][5:])
volume = data['turnoverVol'][5:]
closeIndex = data['closeIndex']
deltaIndex = np.log(np.array(data['highestIndex'])) - np.log(np.array(data['lowestIndex']))
deltaIndex = deltaIndex[5:]
logReturn1 = np.array(np.diff(np.log(closeIndex)))
logReturn1 = logReturn1[4:]
logReturn5 = np.log(np.array(closeIndex[5:])) - np.log(np.array(closeIndex[:-5]))
logReturnFst = np.array(np.diff(np.log(tradeVal['tradeVal'])))[4:]
closeIndex = closeIndex[5:]
X = np.column_stack([logReturn1,logReturn5,deltaIndex,volume,logReturnFst])


#build model
start = '2012-01-01'                       # 回测起始时间
end = '2016-10-21'                         # 回测结束时间
ticker_name='000001'
data_sz=DataAPI.MktIdxdGet(ticker=ticker_name,beginDate=start,endDate=end,field=u"",pandas="1")
data=data_sz[['tradeDate','preCloseIndex','openIndex','lowestIndex','highestIndex','closeIndex','turnoverVol','turnoverValue']]
print data[0:5]
volume=data['turnoverVol']
close=data['closeIndex']
close2=data['preCloseIndex']
logDel = np.log(np.array(data['highestIndex'])) - np.log(np.array(data['lowestIndex']))
logRet_1 = np.array(np.diff(np.log(close2)))                         #这个作为后面计算收益使用
logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))#5日指数对数收益差
logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))
logDel = logDel[5:]
logRet_1 = logRet_1[4:]
close = close[5:]
Date = pd.to_datetime(data['tradeDate'][5:])
A = np.column_stack([logDel,logRet_5,logVol_5])#3个特征 理解成3维数据
print A[0:2]                                   #格式注意

#build model
n = 3 #6个隐藏状态
model = GaussianHMM(n_components= n, covariance_type="full", n_iter=2000).fit([A])
hidden_states = model.predict(A)
hidden_states[0:10]

plt.figure(figsize=(14, 6)) 
for i in range(model.n_components):
    pos = (hidden_states==i)
    plt.plot_date(Date[pos],close[pos],'o',label='hidden state %d'%i,lw=3)
    plt.legend(loc="left")
