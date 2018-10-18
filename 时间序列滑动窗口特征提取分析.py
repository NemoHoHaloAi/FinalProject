def gen_filter_loan_feat():
    dump_path = './tmp/train_filter_loan_feat.pkl'
    if os.path.exists(dump_path):
        df_filter_loan = pickle.load(open(dump_path,'rb'))
    else:
        df_filter_loan = pd.read_csv(t_loan_file,header=0)
        df_filter_loan['month'] = df_filter_loan['loan_time'].map(lambda x: conver_time(x))
        df_filter_loan['loan_amount'] = df_filter_loan['loan_amount'].map(lambda x: round(change_data(x)))
        df_filter_loan = df_filter_loan[df_filter_loan['month'] != 11]
        del df_filter_loan['month']

        # 贷款行为在滑动时间窗口内的贷款统计特征
        df_filter_loan['days'] = df_filter_loan['loan_time'].map(lambda x: datetime.strptime('2016-11-1 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_filter_loan['days'] = df_filter_loan['days'].map(lambda x: int(x.days))

        uid = df_filter_loan['uid'].unique()
        exclu = [1]*len(uid) 
        days_df = pd.DataFrame({'uid':uid,'exclu':exclu})
        day_list = [0,3,7,14,21,28,35,42,49,56,63,70,77,84]
        for i in range(len(day_list)-1):
            days1 = day_list[i]
            days2 = day_list[i+1]
            df = df_filter_loan[['uid','days','loan_amount']].copy()
            day_df = get_stat_feat(df,'loan_amount', 'loan', days1,days2)
            days_df = pd.merge(days_df,day_df,how='left',on='uid')

        days_df = days_df.fillna(0.)
        del days_df['exclu']

        df_filter_loan = days_df
        pickle.dump(df_filter_loan, open(dump_path, 'wb'))
    return df_filter_loan
    


def get_stat_feat(df,values,action,days1,days2):
    df = df[df['days'] > days1]
    df = df[df['days'] <= days2]
    stat_feat = ['min','mean','max','median','count','sum','std']
    df = df.groupby('uid')[values].agg(stat_feat).reset_index()
    df.columns = ['uid'] + ['%s_%s_' % (action,days2) + col for col in stat_feat] #loan_7_min,loan_7_max

    return df
    
    
    
'''
假设场景：    
公司给出了这样的数据：
	以天为单位；每天有N维特征；一共是最近1000天的数据
项目要求：    
	预测往后20天的数据。
一：该怎么做比例合适的测试集和训练集？ 
	1. 对于非平稳序列，通过差分预处理获得平稳时间序列。 
	2. 按照时间先后顺序（有头有尾）；以【1】天为步长，依次把第【J】天的数据做为X，把第【J+1】天的数据做为Y，形成一个embedding；那么有999个embedding。
	3. 按照时间先后顺序（有头有尾）；以【m】天为步长，依次把【（J*m：J*m+N】天拓展数据做为X，把第【J*m+N+1】天作为Y，进行大跨度切片，形成一个embedding；那么有（1000-N）% m 个embedding。
	4. 把时间序列首尾相连形成循环队列状（无头无尾），按照2的规则进行切片。不同的是，有的切片会切到时间序列上收尾相接的部分。
	5. 在3或4的规则下，以适当小的比例打乱单个embedding的X内部元素的顺序，也添加到训练集中。
	6. 在3或3的规则下，协同过滤方法为训练集增加特征：去除样本中的时间维度，计算每个样本和其他样本之间的相似度，用相似度乘以其他样本的【某一个维度A】进行加权平均，得到纯基于特征相似度的【A】特征，加入到样本特征中。
	7. 在5规则下，各种组合。
	8. 去除异常点（四种方法，先mark）

二：该怎么安排训练过程？
	1. 完全按照时间序列的前后顺序进行训练，更新rnn的隐藏层输出值，这是利用网络之前的输出会影响之后的输入的特性。
	2. 不完全按照时间序列的前后顺序进行训练，而是打乱shuffle顺序或者随机抽取测试样本进行训练，kaggle竞赛有些是这么处理的（这个过程有点粗暴，相当于用着lstm/rnn网络，却丢掉了时间维度，只是做着传统cnn的特征提取工作。所以我写了“不完全按照时间序列”，这样处理可以泛化除时间维度之外的特征提取能力，同时在数据量小的情况下也可以增加样本，适当地防止过拟合，算是个小窍门），预测准确性获得提升。
'''









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 15

data = pd.read_csv('./train.csv')[['Store', 'Date','Sales']]
data = data[['Date','Sales']][data['Store']==1]
data = data.reset_index(drop=True)
data.Date = pd.to_datetime(data.Date)
plt.title('Every day`s sales by 7 step')
plt.plot(data.Date[::7], data.Sales[::7])

#这里以一个月为一个窗口，每一个时间t的值由它前面30天（包括自己）的均值代替，标准差同理。
from statsmodels.tsa.stattools import adfuller
def test_stationarity(ts):
	rolmean = pd.rolling_mean(ts,window=30)
	rolstd = pd.rolling_std(ts, window=30)
	#plot rolling statistics:
	plt.figure(figsize=(10,10))
	plt.legend(loc = 'best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.plot(ts, color = 'blue',label='Original')
	plt.plot(rolmean , color = 'red',label = 'rolling mean')
	plt.plot(rolstd, color = 'black', label= 'Rolling standard deviation')
	
	print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(ts,autolag = 'AIC')
    #dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)' %key] = value
    print dfoutput

test_stationarity(data.Sales)

看出数据有季节性相关，因此也是不稳定的（所谓稳定应该是时间独立的）

让数据变得不稳定的原因主要有俩：
1. 趋势（trend）-数据随着时间变化。比如说升高或者降低。
2. 季节性(seasonality)-数据在特定的时间段内变动。比如说节假日，或者活动导致数据的异常。

检测和去除趋势，通常有三种方法：
1. 聚合 : 将时间轴缩短，以一段时间内星期/月/年的均值作为数据值。使不同时间段内的值差距缩小。
2. 平滑： 以一个滑动窗口内的均值代替原来的值，为了使值之间的差距缩小
3. 多项式过滤：用一个回归模型来拟合现有数据，使得数据更平滑。

该项目使用平滑窗口的方式使数据稳定。


ts_log = np.log(data.Sales[data['Sales']>0][:])
moving_avg = pd.rolling_mean(ts_log,window=90) # 窗口大小为半个月，即15天
plt.figure(figsize=(8,8))
plt.plot(ts_log, color='blue')
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log-moving_avg
ts_log_moving_avg_diff.dropna(inplace = True)
plt.figure(figsize=(8,8))
plt.plot(ts_log, color='blue')
plt.plot(ts_log_moving_avg_diff, color='red')


'''
Results of Dickey-Fuller Test:
Test Statistic                  -4.838566
p-value                          0.000046
#Lags Used                      13.000000
Number of Observations Used    668.000000
Critical value (5%)             -2.865876
Critical value (1%)             -3.440177
Critical value (10%)            -2.569079
'''







# 两种方式：
# 1. 简单：忽略测试数据无Sales的问题，直接当做NaN处理。
# 2. 复杂：一个数据一个数据的预测，计算各特征值，再预测，反复进行。
# 当前选择方式1，方式2的话测试数据的相应特征值计算就要放到预测的时候了，因为每个值都依赖于前一个的Sales值
train_all.sort_values(by=['Date'], inplace=True)
test_all.sort_values(by=['Date'], inplace=True)
test_all['Sales'] = [np.nan]*len(test_all)
roll_7s_train = pd.Series([])
roll_7s_test = pd.Series([])
roll_14s_train = pd.Series([])
roll_14s_test = pd.Series([])
roll_30s_train = pd.Series([])
roll_30s_test = pd.Series([])
roll_90s_train = pd.Series([])
roll_90s_test = pd.Series([])
roll_180s_train = pd.Series([])
roll_180s_test = pd.Series([])
roll_360s_train = pd.Series([])
roll_360s_test = pd.Series([])

for i in range(train_all.Store.min(), train_all.Store.max()+1, 1):
    sales_all = pd.concat([train_all[train_all['Store']==i][['Sales']], 
                           test_all[test_all['Store']==i][['Sales']]], ignore_index=True)['Sales']
    roll_7 = sales_all.ewm(span=7,min_periods=1, ignore_na=True).mean()
    roll_14 = sales_all.ewm(span=14,min_periods=1, ignore_na=True).mean()
    roll_30 = sales_all.ewm(span=30,min_periods=1, ignore_na=True).mean()
    roll_90 = sales_all.ewm(span=90,min_periods=1, ignore_na=True).mean()
    roll_180 = sales_all.ewm(span=180,min_periods=1, ignore_na=True).mean()
    roll_360 = sales_all.ewm(span=360,min_periods=1, ignore_na=True).mean()
    
    train_len = len(train_all[train_all['Store']==i])
    
    roll_7s_train = pd.concat([roll_7s_train, roll_7[:train_len]], ignore_index=True)
    roll_7s_test = pd.concat([roll_7s_test, roll_7[train_len:]], ignore_index=True)
    roll_14s_train = pd.concat([roll_14s_train, roll_14[:train_len]], ignore_index=True)
    roll_14s_test = pd.concat([roll_14s_test, roll_14[train_len:]], ignore_index=True)
    roll_30s_train = pd.concat([roll_30s_train, roll_30[:train_len]], ignore_index=True)
    roll_30s_test = pd.concat([roll_30s_test, roll_30[train_len:]], ignore_index=True)
    roll_90s_train = pd.concat([roll_90s_train, roll_90[:train_len]], ignore_index=True)
    roll_90s_test = pd.concat([roll_90s_test, roll_90[train_len:]], ignore_index=True)
    roll_180s_train = pd.concat([roll_180s_train, roll_180[:train_len]], ignore_index=True)
    roll_180s_test = pd.concat([roll_180s_test, roll_180[train_len:]], ignore_index=True)
    roll_360s_train = pd.concat([roll_360s_train, roll_360[:train_len]], ignore_index=True)
    roll_360s_test = pd.concat([roll_360s_test, roll_360[train_len:]], ignore_index=True)
    
train_all.sort_values(by=['Store'], inplace=True)
train_all.reset_index(inplace=True, drop=True)
train_all['Sales7']=roll_7s_train
train_all['Sales14']=roll_14s_train
train_all['Sales30']=roll_30s_train
train_all['Sales90']=roll_90s_train
train_all['Sales180']=roll_180s_train
train_all['Sales360']=roll_360s_train

test_all.sort_values(by=['Store'], inplace=True)
test_all.reset_index(inplace=True, drop=True)
test_all['Sales7']=roll_7s_test
test_all['Sales14']=roll_14s_test
test_all['Sales30']=roll_30s_test
test_all['Sales90']=roll_90s_test
test_all['Sales180']=roll_180s_test
test_all['Sales360']=roll_360s_test

train_all.sort_values(by=['Date'], inplace=True)
test_all.sort_values(by=['Date'], inplace=True)




