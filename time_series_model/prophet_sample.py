# Python
import pandas as pd
from fbprophet import Prophet
# 读入数据集
df = pd.read_csv('examples/example_wp_log_peyton_manning.csv')
df.head()

# 拟合模型
m = Prophet()
m.fit(df)

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=365)
future.tail()

# 预测数据集
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# 展示预测结果
m.plot(forecast)

# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
m.plot_components(forecast)

print(forecast.columns)
Index(['ds', 'trend', 'trend_lower', 'trend_upper', 'yhat_lower', 'yhat_upper',
       'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
       'multiplicative_terms', 'multiplicative_terms_lower',
       'multiplicative_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper',
       'yearly', 'yearly_lower', 'yearly_upper', 'yhat'],
      dtype='object')

#4.4 对节假日和季节性设定先验规模
#如果发现节假日效应被过度拟合了，通过设置参数 holidays_prior_scale 可以调整它们的先验规模来使之平滑，默认下该值取 10 。

#减少这个参数会降低假期效果


#加入周期 
m = Prophet()

m = Prophet(holidays=holidays, holidays_prior_scale=0.05).fit(df)
forecast = m.predict(future)
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
['ds', 'playoff', 'superbowl']][-10:]

m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)

