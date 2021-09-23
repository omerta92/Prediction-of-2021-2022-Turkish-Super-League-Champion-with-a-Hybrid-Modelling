
##############################################################
# Importing Libraries and Setting Options
##############################################################

import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

##############################################################
# Data Reading
##############################################################

df_ = pd.read_excel("C:/Users/omery/Desktop/meta_tsl.xlsx")
df = df_.copy()

##############################################################
# Data Pre-Processing
##############################################################

df.dropna(inplace=True)

today_date = dt.datetime(2021, 9, 23)
df.head()


#########################
# Data Processing % Analysing
#########################

# last_m : number of days from last match until today
# first_m : number of days from first match until today
# m_series : count of match between two teams
# points : points according to result


res_df = df.groupby(['Index']).agg({'SPE_DATE': [lambda date: (date.max() - date.min()).days,
                                                      lambda date: (today_date - date.min()).days],
                                         'ID': lambda num: num.count(),
                                         'Points': lambda points: points.sum()})

res_df.columns = res_df.columns.droplevel(0)
res_df.columns = ['last_m', 'first_m', 'm_series', 'points']

res_df = res_df.merge(df,on='Index', how='left')

res_df = res_df[['ID','Index','last_m','first_m','m_series','points']]

res_df.drop_duplicates()

res_df["avg_points"] = res_df["points"] / res_df["m_series"]

res_df["last_m"] = res_df["last_m"] / 7

res_df["first_m"] = res_df["first_m"] / 7

res_df.head()

##############################################################
# BG-NBD Modelling
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(res_df['m_series'],
        res_df['last_m'],
        res_df['first_m'])

##############################################################
# GAMMA-GAMMA Modelling
##############################################################


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(res_df['m_series'], res_df['avg_points'])


ggf.conditional_expected_average_profit(res_df['m_series'],
                                        res_df['avg_points']).head(10)


ggf.conditional_expected_average_profit(res_df['m_series'],
                                        res_df['avg_points']).sort_values(ascending=False).head(10)

res_df["expected_average_rate"] = ggf.conditional_expected_average_profit(res_df['m_series'],
                                                                             res_df['avg_points'])

res_df=res_df.drop_duplicates()
res_df.head()
##############################################################
# Hybrid Modelling with BGNBD & GGM
##############################################################

rate = ggf.customer_lifetime_value(bgf,
                                   res_df['m_series'],
                                   res_df['last_m'],
                                   res_df['first_m'],
                                   res_df['avg_points'],
                                   time=8,  # for 8 months
                                   freq="W",  # frequency
                                   discount_rate=0.00)

rate.head()
res_df = res_df.reset_index()
rate = rate.reset_index()
rate['prob'] = rate['clv']

res_df.sort_values(by="expected_average_rate", ascending=False).head(50)

res_final = res_df.merge(rate, on="index", how="left")

res_final = res_final[['ID', 'Index', 'last_m', 'first_m', 'm_series', 'points', 'avg_points', 'expected_average_rate', 'prob']]

res_final = res_final.drop_duplicates()

res_final.sort_values(by="prob", ascending=False).head(10)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(res_final[["prob"]])
res_final["prob"] = scaler.transform(res_final[["prob"]])


res_final.sort_values(by="prob", ascending=False).head()



##############################################################
# Segmentation
##############################################################


res_final["bags"] = pd.qcut(res_final["prob"], 6, labels=["very_weak", "weak", "low_med", "high_med","strong","too_strong"])

res_final.head()


res_final.to_csv("championss.csv")