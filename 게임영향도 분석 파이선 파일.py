#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # 산프 분석시작

# In[154]:


import scipy.stats as stats
import pandas as pd
import urllib
import warnings
warnings.filterwarnings('ignore')
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import re


# In[155]:


데이터=pd.read_excel("C:/Users/user/Desktop/산프 우리소프트/분석자료/수정 데이터111.xlsx")
데이터.to_csv("C:/Users/user/Desktop/산프 우리소프트/분석자료/수정 데이터123.csv")
a000=pd.read_csv("C:/Users/user/Desktop/산프 우리소프트/분석자료/수정 데이터123.csv",index_col=0)
a000.isna().sum()
a000
a111=a000.fillna(pd.NA)  
a1111= a111.applymap(lambda x: pd.NA if re.match(r'^\.$', str(x)) else x)
a1111=a1111.dropna()
a1111.isna().sum()


# In[156]:


df=pd.DataFrame(a1111,columns=['기분','A4정답율'])

df['A4정답율'] = pd.to_numeric(df['A4정답율'], errors='coerce')

df_1=df[df.기분==1]['A4정답율'].tolist()
df_2=df[df.기분==2]['A4정답율'].tolist()
df_3=df[df.기분==3]['A4정답율'].tolist()
df_4=df[df.기분==4]['A4정답율'].tolist()
df_5=df[df.기분==5]['A4정답율'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df_1,df_2,df_3,df_4,df_5))


# In[157]:


df1=pd.DataFrame(a1111,columns=['기분','A4난이도_평균'])

df1['A4난이도_평균'] = pd.to_numeric(df1['A4난이도_평균'], errors='coerce')
df1_1=df1[df1.기분==1]['A4난이도_평균'].tolist()
df1_2=df1[df1.기분==2]['A4난이도_평균'].tolist()
df1_3=df1[df1.기분==3]['A4난이도_평균'].tolist()
df1_4=df1[df1.기분==4]['A4난이도_평균'].tolist()
df1_5=df1[df1.기분==5]['A4난이도_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df1_1,df1_2,df1_3,df1_4,df1_5))


# In[158]:


df2=pd.DataFrame(a1111,columns=['기분','A4난이도_표준편차'])
df2['A4난이도_표준편차'] = pd.to_numeric(df2['A4난이도_표준편차'], errors='coerce')
df2_1=df2[df2.기분==1]['A4난이도_표준편차'].tolist()
df2_2=df2[df2.기분==2]['A4난이도_표준편차'].tolist()
df2_3=df2[df2.기분==3]['A4난이도_표준편차'].tolist()
df2_4=df2[df2.기분==4]['A4난이도_표준편차'].tolist()
df2_5=df2[df2.기분==5]['A4난이도_표준편차'].tolist()

#등분산성
from scipy.stats import levene
print(stats.levene(df2_1,df2_2,df2_3,df2_4,df2_5))


# In[159]:


df3=pd.DataFrame(a1111,columns=['기분','A4난이도_최대'])

df3['A4난이도_최대'] = pd.to_numeric(df3['A4난이도_최대'], errors='coerce')

df3_1=df3[df3.기분==1]['A4난이도_최대'].tolist()
df3_2=df3[df3.기분==2]['A4난이도_최대'].tolist()
df3_3=df3[df3.기분==3]['A4난이도_최대'].tolist()
df3_4=df3[df3.기분==4]['A4난이도_최대'].tolist()
df3_5=df3[df3.기분==5]['A4난이도_최대'].tolist()

#등분산성
from scipy.stats import levene
print(stats.levene(df3_1,df3_2,df3_3,df3_4,df3_5))


# In[160]:


df4=pd.DataFrame(a1111,columns=['기분','A4결정시간_평균'])
df4['A4결정시간_평균'] = pd.to_numeric(df4['A4결정시간_평균'], errors='coerce')

df4_1=df4[df4.기분==1]['A4결정시간_평균'].tolist()
df4_2=df4[df4.기분==2]['A4결정시간_평균'].tolist()
df4_3=df4[df4.기분==3]['A4결정시간_평균'].tolist()
df4_4=df4[df4.기분==4]['A4결정시간_평균'].tolist()
df4_5=df4[df4.기분==5]['A4결정시간_평균'].tolist()


#등분산성
from scipy.stats import levene
print(stats.levene(df4_1,df4_2,df4_3,df4_4,df4_5))


# In[161]:


df5=pd.DataFrame(a1111,columns=['기분','A4결정시간_표준편차'])
df5['A4결정시간_표준편차'] = pd.to_numeric(df5['A4결정시간_표준편차'], errors='coerce')

df5_1=df5[df5.기분==1]['A4결정시간_표준편차'].tolist()
df5_2=df5[df5.기분==2]['A4결정시간_표준편차'].tolist()
df5_3=df5[df5.기분==3]['A4결정시간_표준편차'].tolist()
df5_4=df5[df5.기분==4]['A4결정시간_표준편차'].tolist()
df5_5=df5[df5.기분==5]['A4결정시간_표준편차'].tolist()


#등분산성
from scipy.stats import levene
print(stats.levene(df5_1,df5_2,df5_3,df5_4,df5_5))


# In[162]:


df6=pd.DataFrame(a1111,columns=['기분','A4터치_횟수'])
df6['A4터치_횟수'] = pd.to_numeric(df6['A4터치_횟수'], errors='coerce')

df6_1=df6[df6.기분==1]['A4터치_횟수'].tolist()
df6_2=df6[df6.기분==2]['A4터치_횟수'].tolist()
df6_3=df6[df6.기분==3]['A4터치_횟수'].tolist()
df6_4=df6[df6.기분==4]['A4터치_횟수'].tolist()
df6_5=df6[df6.기분==5]['A4터치_횟수'].tolist()

#등분산성
from scipy.stats import levene
print(stats.levene(df6_1,df6_2,df6_3,df6_4,df6_5))


# In[163]:


df7=pd.DataFrame(a1111,columns=['기분','A5정답율'])
df7['A5정답율'] = pd.to_numeric(df7['A5정답율'], errors='coerce')

df7_1=df7[df7.기분==1]['A5정답율'].tolist()
df7_2=df7[df7.기분==2]['A5정답율'].tolist()
df7_3=df7[df7.기분==3]['A5정답율'].tolist()
df7_4=df7[df7.기분==4]['A5정답율'].tolist()
df7_5=df7[df7.기분==5]['A5정답율'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df7_1,df7_2,df7_3,df7_4,df7_5))


# In[164]:


df8=pd.DataFrame(a1111,columns=['기분','A5난이도_평균'])
df8['A5난이도_평균'] = pd.to_numeric(df8['A5난이도_평균'], errors='coerce')

df8_1=df8[df8.기분==1]['A5난이도_평균'].tolist()
df8_2=df8[df8.기분==2]['A5난이도_평균'].tolist()
df8_3=df8[df8.기분==3]['A5난이도_평균'].tolist()
df8_4=df8[df8.기분==4]['A5난이도_평균'].tolist()
df8_5=df8[df8.기분==5]['A5난이도_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df8_1,df8_2,df8_3,df8_4,df8_5))


# In[165]:


df9=pd.DataFrame(a1111,columns=['기분','A5난이도_표준편차'])
df9['A5난이도_표준편차'] = pd.to_numeric(df9['A5난이도_표준편차'], errors='coerce')

df9_1=df9[df9.기분==1]['A5난이도_표준편차'].tolist()
df9_2=df9[df9.기분==2]['A5난이도_표준편차'].tolist()
df9_3=df9[df9.기분==3]['A5난이도_표준편차'].tolist()
df9_4=df9[df9.기분==4]['A5난이도_표준편차'].tolist()
df9_5=df9[df9.기분==5]['A5난이도_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df9_1,df9_2,df9_3,df9_4,df9_5))


# In[166]:


df10=pd.DataFrame(a1111,columns=['기분','A5난이도_최대'])
df10['A5난이도_최대'] = pd.to_numeric(df10['A5난이도_최대'], errors='coerce')

df10_1=df10[df10.기분==1]['A5난이도_최대'].tolist()
df10_2=df10[df10.기분==2]['A5난이도_최대'].tolist()
df10_3=df10[df10.기분==3]['A5난이도_최대'].tolist()
df10_4=df10[df10.기분==4]['A5난이도_최대'].tolist()
df10_5=df10[df10.기분==5]['A5난이도_최대'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df10_1,df10_2,df10_3,df10_4,df10_5))


# In[167]:


df11=pd.DataFrame(a1111,columns=['기분','A5결정시간_평균'])
df11['A5결정시간_평균'] = pd.to_numeric(df11['A5결정시간_평균'], errors='coerce')

df11_1=df11[df11.기분==1]['A5결정시간_평균'].tolist()
df11_2=df11[df11.기분==2]['A5결정시간_평균'].tolist()
df11_3=df11[df11.기분==3]['A5결정시간_평균'].tolist()
df11_4=df11[df11.기분==4]['A5결정시간_평균'].tolist()
df11_5=df11[df11.기분==5]['A5결정시간_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df11_1,df11_2,df11_3,df11_4,df11_5))


# In[168]:


df12=pd.DataFrame(a1111,columns=['기분','A5결정시간_표준편차'])
df12['A5결정시간_표준편차'] = pd.to_numeric(df12['A5결정시간_표준편차'], errors='coerce')

df12_1=df12[df12.기분==1]['A5결정시간_표준편차'].tolist()
df12_2=df12[df12.기분==2]['A5결정시간_표준편차'].tolist()
df12_3=df12[df12.기분==3]['A5결정시간_표준편차'].tolist()
df12_4=df12[df12.기분==4]['A5결정시간_표준편차'].tolist()
df12_5=df12[df12.기분==5]['A5결정시간_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df12_1,df12_2,df12_3,df12_4,df12_5))


# In[169]:


df13=pd.DataFrame(a1111,columns=['기분','A5터치_횟수'])
df13['A5터치_횟수'] = pd.to_numeric(df13['A5터치_횟수'], errors='coerce')

df13_1=df13[df13.기분==1]['A5터치_횟수'].tolist()
df13_2=df13[df13.기분==2]['A5터치_횟수'].tolist()
df13_3=df13[df13.기분==3]['A5터치_횟수'].tolist()
df13_4=df13[df13.기분==4]['A5터치_횟수'].tolist()
df13_5=df13[df13.기분==5]['A5터치_횟수'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df13_1,df13_2,df13_3,df13_4,df13_5))


# In[170]:


df14=pd.DataFrame(a1111,columns=['기분','M5정답율'])
df14['M5정답율'] = pd.to_numeric(df14['M5정답율'], errors='coerce')

df14_1=df14[df14.기분==1]['M5정답율'].tolist()
df14_2=df14[df14.기분==2]['M5정답율'].tolist()
df14_3=df14[df14.기분==3]['M5정답율'].tolist()
df14_4=df14[df14.기분==4]['M5정답율'].tolist()
df14_5=df14[df14.기분==5]['M5정답율'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df14_1,df14_2,df14_3,df14_4,df14_5))


# In[171]:


df15=pd.DataFrame(a1111,columns=['기분','M5난이도_평균'])
df15['M5난이도_평균'] = pd.to_numeric(df15['M5난이도_평균'], errors='coerce')

df15_1=df15[df15.기분==1]['M5난이도_평균'].tolist()
df15_2=df15[df15.기분==2]['M5난이도_평균'].tolist()
df15_3=df15[df15.기분==3]['M5난이도_평균'].tolist()
df15_4=df15[df15.기분==4]['M5난이도_평균'].tolist()
df15_5=df15[df15.기분==5]['M5난이도_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df15_1,df15_2,df15_3,df15_4,df15_5))


# In[172]:


df16=pd.DataFrame(a1111,columns=['기분','M5난이도_표준편차'])
df16['M5난이도_표준편차'] = pd.to_numeric(df16['M5난이도_표준편차'], errors='coerce')

df16_1=df16[df16.기분==1]['M5난이도_표준편차'].tolist()
df16_2=df16[df16.기분==2]['M5난이도_표준편차'].tolist()
df16_3=df16[df16.기분==3]['M5난이도_표준편차'].tolist()
df16_4=df16[df16.기분==4]['M5난이도_표준편차'].tolist()
df16_5=df16[df16.기분==5]['M5난이도_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df16_1,df16_2,df16_3,df16_4,df16_5))


# In[173]:


df17=pd.DataFrame(a1111,columns=['기분','M5난이도_최대'])
df17['M5난이도_최대'] = pd.to_numeric(df17['M5난이도_최대'], errors='coerce')

df17_1=df17[df17.기분==1]['M5난이도_최대'].tolist()
df17_2=df17[df17.기분==2]['M5난이도_최대'].tolist()
df17_3=df17[df17.기분==3]['M5난이도_최대'].tolist()
df17_4=df17[df17.기분==4]['M5난이도_최대'].tolist()
df17_5=df17[df17.기분==5]['M5난이도_최대'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df17_1,df17_2,df17_3,df17_4,df17_5))


# In[174]:


df18=pd.DataFrame(a1111,columns=['기분','M5남은_제한시간_평균'])
df18['M5남은_제한시간_평균'] = pd.to_numeric(df18['M5남은_제한시간_평균'], errors='coerce')

df18_1=df18[df18.기분==1]['M5남은_제한시간_평균'].tolist()
df18_2=df18[df18.기분==2]['M5남은_제한시간_평균'].tolist()
df18_3=df18[df18.기분==3]['M5남은_제한시간_평균'].tolist()
df18_4=df18[df18.기분==4]['M5남은_제한시간_평균'].tolist()
df18_5=df18[df18.기분==5]['M5남은_제한시간_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df18_1,df18_2,df18_3,df18_4,df18_5))


# In[175]:


df19=pd.DataFrame(a1111,columns=['기분','M5남은_제한시간_표준편차'])
df19['M5남은_제한시간_표준편차'] = pd.to_numeric(df19['M5남은_제한시간_표준편차'], errors='coerce')

df19_1=df19[df19.기분==1]['M5남은_제한시간_표준편차'].tolist()
df19_2=df19[df19.기분==2]['M5남은_제한시간_표준편차'].tolist()
df19_3=df19[df19.기분==3]['M5남은_제한시간_표준편차'].tolist()
df19_4=df19[df19.기분==4]['M5남은_제한시간_표준편차'].tolist()
df19_5=df19[df19.기분==5]['M5남은_제한시간_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df19_1,df19_2,df19_3,df19_4,df19_5))


# In[176]:


df20=pd.DataFrame(a1111,columns=['기분','M5남은_제한시간_최대'])
df20['M5남은_제한시간_최대'] = pd.to_numeric(df20['M5남은_제한시간_최대'], errors='coerce')

df20_1=df20[df20.기분==1]['M5남은_제한시간_최대'].tolist()
df20_2=df20[df20.기분==2]['M5남은_제한시간_최대'].tolist()
df20_3=df20[df20.기분==3]['M5남은_제한시간_최대'].tolist()
df20_4=df20[df20.기분==4]['M5남은_제한시간_최대'].tolist()
df20_5=df20[df20.기분==5]['M5남은_제한시간_최대'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df20_1,df20_2,df20_3,df20_4,df20_5))


# In[177]:


df21=pd.DataFrame(a1111,columns=['기분','M5터치_횟수'])
df21['M5터치_횟수'] = pd.to_numeric(df21['M5터치_횟수'], errors='coerce')

df21_1=df21[df21.기분==1]['M5터치_횟수'].tolist()
df21_2=df21[df21.기분==2]['M5터치_횟수'].tolist()
df21_3=df21[df21.기분==3]['M5터치_횟수'].tolist()
df21_4=df21[df21.기분==4]['M5터치_횟수'].tolist()
df21_5=df21[df21.기분==5]['M5터치_횟수'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df21_1,df21_2,df21_3,df21_4,df21_5))


# In[178]:


df22=pd.DataFrame(a1111,columns=['기분','V4전체_정답율'])
df22['V4전체_정답율'] = pd.to_numeric(df22['V4전체_정답율'], errors='coerce')

df22_1=df22[df22.기분==1]['V4전체_정답율'].tolist()
df22_2=df22[df22.기분==2]['V4전체_정답율'].tolist()
df22_3=df22[df22.기분==3]['V4전체_정답율'].tolist()
df22_4=df22[df22.기분==4]['V4전체_정답율'].tolist()
df22_5=df22[df22.기분==5]['V4전체_정답율'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df22_1,df22_2,df22_3,df22_4,df22_5))


# In[179]:


df23=pd.DataFrame(a1111,columns=['기분','V4왼쪽_정답율'])
df23['V4왼쪽_정답율'] = pd.to_numeric(df23['V4왼쪽_정답율'], errors='coerce')

df23_1=df23[df23.기분==1]['V4왼쪽_정답율'].tolist()
df23_2=df23[df23.기분==2]['V4왼쪽_정답율'].tolist()
df23_3=df23[df23.기분==3]['V4왼쪽_정답율'].tolist()
df23_4=df23[df23.기분==4]['V4왼쪽_정답율'].tolist()
df23_5=df23[df23.기분==5]['V4왼쪽_정답율'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df23_1,df23_2,df23_3,df23_4,df23_5))


# In[180]:


df24=pd.DataFrame(a1111,columns=['기분','V4오른쪽_정답율'])
df24['V4오른쪽_정답율'] = pd.to_numeric(df24['V4오른쪽_정답율'], errors='coerce')

df24_1=df24[df24.기분==1]['V4오른쪽_정답율'].tolist()
df24_2=df24[df24.기분==2]['V4오른쪽_정답율'].tolist()
df24_3=df24[df24.기분==3]['V4오른쪽_정답율'].tolist()
df24_4=df24[df24.기분==4]['V4오른쪽_정답율'].tolist()
df24_5=df24[df24.기분==5]['V4오른쪽_정답율'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df24_1,df24_2,df24_3,df24_4,df24_5))


# In[181]:


df25=pd.DataFrame(a1111,columns=['기분','V4난이도_평균'])
df25['V4난이도_평균'] = pd.to_numeric(df25['V4난이도_평균'], errors='coerce')

df25_1=df25[df25.기분==1]['V4난이도_평균'].tolist()
df25_2=df25[df25.기분==2]['V4난이도_평균'].tolist()
df25_3=df25[df25.기분==3]['V4난이도_평균'].tolist()
df25_4=df25[df25.기분==4]['V4난이도_평균'].tolist()
df25_5=df25[df25.기분==5]['V4난이도_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df25_1,df25_2,df25_3,df25_4,df25_5))


# In[182]:


df26=pd.DataFrame(a1111,columns=['기분','V4난이도_표준편차'])
df26['V4난이도_표준편차'] = pd.to_numeric(df26['V4난이도_표준편차'], errors='coerce')

df26_1=df26[df26.기분==1]['V4난이도_표준편차'].tolist()
df26_2=df26[df26.기분==2]['V4난이도_표준편차'].tolist()
df26_3=df26[df26.기분==3]['V4난이도_표준편차'].tolist()
df26_4=df26[df26.기분==4]['V4난이도_표준편차'].tolist()
df26_5=df26[df26.기분==5]['V4난이도_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df26_1,df26_2,df26_3,df26_4,df26_5))


# In[183]:


df27=pd.DataFrame(a1111,columns=['기분','V4난이도_최대'])
df27['V4난이도_최대'] = pd.to_numeric(df27['V4난이도_최대'], errors='coerce')

df27_1=df27[df27.기분==1]['V4난이도_최대'].tolist()
df27_2=df27[df27.기분==2]['V4난이도_최대'].tolist()
df27_3=df27[df27.기분==3]['V4난이도_최대'].tolist()
df27_4=df27[df27.기분==4]['V4난이도_최대'].tolist()
df27_5=df27[df27.기분==5]['V4난이도_최대'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df27_1,df27_2,df27_3,df27_4,df27_5))


# In[184]:


df28=pd.DataFrame(a1111,columns=['기분','V4마지막_터치시간_평균'])
df28['V4마지막_터치시간_평균'] = pd.to_numeric(df28['V4마지막_터치시간_평균'], errors='coerce')

df28_1=df28[df28.기분==1]['V4마지막_터치시간_평균'].tolist()
df28_2=df28[df28.기분==2]['V4마지막_터치시간_평균'].tolist()
df28_3=df28[df28.기분==3]['V4마지막_터치시간_평균'].tolist()
df28_4=df28[df28.기분==4]['V4마지막_터치시간_평균'].tolist()
df28_5=df28[df28.기분==5]['V4마지막_터치시간_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df28_1,df28_2,df28_3,df28_4,df28_5))


# In[185]:


df29=pd.DataFrame(a1111,columns=['기분','V4마지막_터치시간_표준편차'])
df29['V4마지막_터치시간_표준편차'] = pd.to_numeric(df29['V4마지막_터치시간_표준편차'], errors='coerce')

df29_1=df29[df29.기분==1]['V4마지막_터치시간_표준편차'].tolist()
df29_2=df29[df29.기분==2]['V4마지막_터치시간_표준편차'].tolist()
df29_3=df29[df29.기분==3]['V4마지막_터치시간_표준편차'].tolist()
df29_4=df29[df29.기분==4]['V4마지막_터치시간_표준편차'].tolist()
df29_5=df29[df29.기분==5]['V4마지막_터치시간_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df29_1,df29_2,df29_3,df29_4,df29_5))


# In[186]:


df30=pd.DataFrame(a1111,columns=['기분','V4왼쪽_마지막_터치시간_평균'])
df30['V4왼쪽_마지막_터치시간_평균'] = pd.to_numeric(df30['V4왼쪽_마지막_터치시간_평균'], errors='coerce')

df30_1=df30[df30.기분==1]['V4왼쪽_마지막_터치시간_평균'].tolist()
df30_2=df30[df30.기분==2]['V4왼쪽_마지막_터치시간_평균'].tolist()
df30_3=df30[df30.기분==3]['V4왼쪽_마지막_터치시간_평균'].tolist()
df30_4=df30[df30.기분==4]['V4왼쪽_마지막_터치시간_평균'].tolist()
df30_5=df30[df30.기분==5]['V4왼쪽_마지막_터치시간_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df30_1,df30_2,df30_3,df30_4,df30_5))


# In[187]:


df31=pd.DataFrame(a1111,columns=['기분','V4왼쪽_마지막_터치시간_표준편차'])
df31['V4왼쪽_마지막_터치시간_표준편차'] = pd.to_numeric(df31['V4왼쪽_마지막_터치시간_표준편차'], errors='coerce')

df31_1=df31[df31.기분==1]['V4왼쪽_마지막_터치시간_표준편차'].tolist()
df31_2=df31[df31.기분==2]['V4왼쪽_마지막_터치시간_표준편차'].tolist()
df31_3=df31[df31.기분==3]['V4왼쪽_마지막_터치시간_표준편차'].tolist()
df31_4=df31[df31.기분==4]['V4왼쪽_마지막_터치시간_표준편차'].tolist()
df31_5=df31[df31.기분==5]['V4왼쪽_마지막_터치시간_표준편차'].tolist()
# 등분산성
from scipy.stats import levene
print(stats.levene(df31_1,df31_2,df31_3,df31_4,df31_5))


# In[188]:


df32=pd.DataFrame(a1111,columns=['기분','V4오른쪽_마지막_터치시간_평균'])
df32['V4오른쪽_마지막_터치시간_평균'] = pd.to_numeric(df32['V4오른쪽_마지막_터치시간_평균'], errors='coerce')

df32_1=df32[df32.기분==1]['V4오른쪽_마지막_터치시간_평균'].tolist()
df32_2=df32[df32.기분==2]['V4오른쪽_마지막_터치시간_평균'].tolist()
df32_3=df32[df32.기분==3]['V4오른쪽_마지막_터치시간_평균'].tolist()
df32_4=df32[df32.기분==4]['V4오른쪽_마지막_터치시간_평균'].tolist()
df32_5=df32[df32.기분==5]['V4오른쪽_마지막_터치시간_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df32_1,df32_2,df32_3,df32_4,df32_5))


# In[189]:


df33=pd.DataFrame(a1111,columns=['기분','V4오른쪽_마지막_터치시간_표준편차'])
df33['V4오른쪽_마지막_터치시간_표준편차'] = pd.to_numeric(df33['V4오른쪽_마지막_터치시간_표준편차'], errors='coerce')

df33_1=df33[df33.기분==1]['V4오른쪽_마지막_터치시간_표준편차'].tolist()
df33_2=df33[df33.기분==2]['V4오른쪽_마지막_터치시간_표준편차'].tolist()
df33_3=df33[df33.기분==3]['V4오른쪽_마지막_터치시간_표준편차'].tolist()
df33_4=df33[df33.기분==4]['V4오른쪽_마지막_터치시간_표준편차'].tolist()
df33_5=df33[df33.기분==5]['V4오른쪽_마지막_터치시간_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df33_1,df33_2,df33_3,df33_4,df33_5))


# In[190]:


df34=pd.DataFrame(a1111,columns=['기분','V4터치_횟수'])
df34['V4터치_횟수'] = pd.to_numeric(df34['V4터치_횟수'], errors='coerce')

df34_1=df34[df34.기분==1]['V4터치_횟수'].tolist()
df34_2=df34[df34.기분==2]['V4터치_횟수'].tolist()
df34_3=df34[df34.기분==3]['V4터치_횟수'].tolist()
df34_4=df34[df34.기분==4]['V4터치_횟수'].tolist()
df34_5=df34[df34.기분==5]['V4터치_횟수'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df34_1,df34_2,df34_3,df34_4,df34_5))


# In[191]:


df35=pd.DataFrame(a1111,columns=['기분','V4왼쪽_터치_횟수'])
df35['V4왼쪽_터치_횟수'] = pd.to_numeric(df35['V4왼쪽_터치_횟수'], errors='coerce')

df35_1=df35[df35.기분==1]['V4왼쪽_터치_횟수'].tolist()
df35_2=df35[df35.기분==2]['V4왼쪽_터치_횟수'].tolist()
df35_3=df35[df35.기분==3]['V4왼쪽_터치_횟수'].tolist()
df35_4=df35[df35.기분==4]['V4왼쪽_터치_횟수'].tolist()
df35_5=df35[df35.기분==5]['V4왼쪽_터치_횟수'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df35_1,df35_2,df35_3,df35_4,df35_5))


# In[192]:


df36=pd.DataFrame(a1111,columns=['기분','V4오른쪽_터치_횟수'])
df36['V4오른쪽_터치_횟수'] = pd.to_numeric(df36['V4오른쪽_터치_횟수'], errors='coerce')

df36_1=df36[df36.기분==1]['V4오른쪽_터치_횟수'].tolist()
df36_2=df36[df36.기분==2]['V4오른쪽_터치_횟수'].tolist()
df36_3=df36[df36.기분==3]['V4오른쪽_터치_횟수'].tolist()
df36_4=df36[df36.기분==4]['V4오른쪽_터치_횟수'].tolist()
df36_5=df36[df36.기분==5]['V4오른쪽_터치_횟수'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df36_1,df36_2,df36_3,df36_4,df36_5))


# In[193]:


df37=pd.DataFrame(a1111,columns=['기분','V5정답율'])
df37['V5정답율'] = pd.to_numeric(df37['V5정답율'], errors='coerce')

df37_1=df37[df37.기분==1]['V5정답율'].tolist()
df37_2=df37[df37.기분==2]['V5정답율'].tolist()
df37_3=df37[df37.기분==3]['V5정답율'].tolist()
df37_4=df37[df37.기분==4]['V5정답율'].tolist()
df37_5=df37[df37.기분==5]['V5정답율'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df37_1,df37_2,df37_3,df37_4,df37_5))


# In[194]:


df38=pd.DataFrame(a1111,columns=['기분','V5난이도_평균'])
df38['V5난이도_평균'] = pd.to_numeric(df38['V5난이도_평균'], errors='coerce')

df38_1=df38[df38.기분==1]['V5난이도_평균'].tolist()
df38_2=df38[df38.기분==2]['V5난이도_평균'].tolist()
df38_3=df38[df38.기분==3]['V5난이도_평균'].tolist()
df38_4=df38[df38.기분==4]['V5난이도_평균'].tolist()
df38_5=df38[df38.기분==5]['V5난이도_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df38_1,df38_2,df38_3,df38_4,df38_5))


# In[195]:


df39=pd.DataFrame(a1111,columns=['기분','V5난이도_표준편차'])
df39['V5난이도_표준편차'] = pd.to_numeric(df39['V5난이도_표준편차'], errors='coerce')

df39_1=df39[df39.기분==1]['V5난이도_표준편차'].tolist()
df39_2=df39[df39.기분==2]['V5난이도_표준편차'].tolist()
df39_3=df39[df39.기분==3]['V5난이도_표준편차'].tolist()
df39_4=df39[df39.기분==4]['V5난이도_표준편차'].tolist()
df39_5=df39[df39.기분==5]['V5난이도_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df39_1,df39_2,df39_3,df39_4,df39_5))


# In[196]:


df40=pd.DataFrame(a1111,columns=['기분','V5난이도_최대'])
df40['V5난이도_최대'] = pd.to_numeric(df40['V5난이도_최대'], errors='coerce')

df40_1=df40[df40.기분==1]['V5난이도_최대'].tolist()
df40_2=df40[df40.기분==2]['V5난이도_최대'].tolist()
df40_3=df40[df40.기분==3]['V5난이도_최대'].tolist()
df40_4=df40[df40.기분==4]['V5난이도_최대'].tolist()
df40_5=df40[df40.기분==5]['V5난이도_최대'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df40_1,df40_2,df40_3,df40_4,df40_5))


# In[197]:


df41=pd.DataFrame(a1111,columns=['기분','V5결정시간_평균'])
df41['V5결정시간_평균'] = pd.to_numeric(df41['V5결정시간_평균'], errors='coerce')

df41_1=df41[df41.기분==1]['V5결정시간_평균'].tolist()
df41_2=df41[df41.기분==2]['V5결정시간_평균'].tolist()
df41_3=df41[df41.기분==3]['V5결정시간_평균'].tolist()
df41_4=df41[df41.기분==4]['V5결정시간_평균'].tolist()
df41_5=df41[df41.기분==5]['V5결정시간_평균'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df41_1,df41_2,df41_3,df41_4,df41_5))


# In[198]:


df42=pd.DataFrame(a1111,columns=['기분','V5결정시간_표준편차'])

df42['V5결정시간_표준편차'] = pd.to_numeric(df42['V5결정시간_표준편차'], errors='coerce')

df42_1=df42[df42.기분==1]['V5결정시간_표준편차'].tolist()
df42_2=df42[df42.기분==2]['V5결정시간_표준편차'].tolist()
df42_3=df42[df42.기분==3]['V5결정시간_표준편차'].tolist()
df42_4=df42[df42.기분==4]['V5결정시간_표준편차'].tolist()
df42_5=df42[df42.기분==5]['V5결정시간_표준편차'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df42_1,df42_2,df42_3,df42_4,df42_5))


# In[199]:


df43=pd.DataFrame(a1111,columns=['기분','V5터치_횟수'])
df43['V5터치_횟수'] = pd.to_numeric(df43['V5터치_횟수'], errors='coerce')

df43_1=df43[df43.기분==1]['V5터치_횟수'].tolist()
df43_2=df43[df43.기분==2]['V5터치_횟수'].tolist()
df43_3=df43[df43.기분==3]['V5터치_횟수'].tolist()
df43_4=df43[df43.기분==4]['V5터치_횟수'].tolist()
df43_5=df43[df43.기분==5]['V5터치_횟수'].tolist()

# 등분산성
from scipy.stats import levene
print(stats.levene(df43_1,df43_2,df43_3,df43_4,df43_5))


# In[ ]:





# In[245]:


model=ols('A4정답율 ~ C(기분)',df).fit()
print(anova_lm(model))

model1=ols('A4난이도_평균 ~ C(기분)',df1).fit()
print(anova_lm(model1))


model2=ols('A4난이도_표준편차 ~ C(기분)',df2).fit()
print(anova_lm(model2))


model3=ols('A4난이도_최대 ~ C(기분)',df3).fit()
print(anova_lm(model3))

model4=ols('A4결정시간_평균 ~ C(기분)',df4).fit()
print(anova_lm(model4))

model5=ols('A4결정시간_표준편차 ~ C(기분)',df5).fit()
print(anova_lm(model5))


model6=ols('A4터치_횟수 ~ C(기분)',df6).fit()
print(anova_lm(model6))


model7=ols('A5정답율 ~ C(기분)',df7).fit()
print(anova_lm(model7))

model8=ols('A5난이도_평균 ~ C(기분)',df8).fit()
print(anova_lm(model8))


model9=ols('A5난이도_표준편차 ~ C(기분)',df9).fit()
print(anova_lm(model9))


model10=ols('A5난이도_최대 ~ C(기분)',df10).fit()
print(anova_lm(model10))

model11=ols('A5결정시간_평균 ~ C(기분)',df11).fit()
print(anova_lm(model11))

model12=ols('A5결정시간_표준편차 ~ C(기분)',df12).fit()
print(anova_lm(model12))

model13=ols('A5터치_횟수 ~ C(기분)',df13).fit()
print(anova_lm(model13))

model14=ols('M5정답율 ~ C(기분)',df14).fit()
print(anova_lm(model14))

model15=ols('M5난이도_평균 ~ C(기분)',df15).fit()
print(anova_lm(model15))

model16=ols('M5난이도_표준편차 ~ C(기분)',df16).fit()
print(anova_lm(model16))

model17=ols('M5난이도_최대 ~ C(기분)',df17).fit()
print(anova_lm(model17))

model18=ols('M5남은_제한시간_평균 ~ C(기분)',df18).fit()
print(anova_lm(model18))


model19=ols('M5남은_제한시간_표준편차 ~ C(기분)',df19).fit()
print(anova_lm(model19))

model20=ols('M5남은_제한시간_최대 ~ C(기분)',df20).fit()
print(anova_lm(model20))

model21=ols('M5터치_횟수 ~ C(기분)',df21).fit()
print(anova_lm(model21))

model22=ols('V4전체_정답율 ~ C(기분)',df22).fit()
print(anova_lm(model22))

model23=ols('V4왼쪽_정답율 ~ C(기분)',df23).fit()
print(anova_lm(model23))

model24=ols('V4오른쪽_정답율 ~ C(기분)',df24).fit()
print(anova_lm(model24))

model25=ols('V4난이도_평균 ~ C(기분)',df25).fit()
print(anova_lm(model25))

model26=ols('V4난이도_표준편차 ~ C(기분)',df26).fit()
print(anova_lm(model26))

model27=ols('V4난이도_최대 ~ C(기분)',df27).fit()
print(anova_lm(model27))

model28=ols('V4마지막_터치시간_평균 ~ C(기분)',df28).fit()
print(anova_lm(model28))

model29=ols('V4마지막_터치시간_표준편차 ~ C(기분)',df29).fit()
print(anova_lm(model29))

model30=ols('V4왼쪽_마지막_터치시간_평균 ~ C(기분)',df30).fit()
print(anova_lm(model30))

model31=ols('V4왼쪽_마지막_터치시간_표준편차 ~ C(기분)',df31).fit()
print(anova_lm(model31))
model32=ols('V4오른쪽_마지막_터치시간_평균 ~ C(기분)',df32).fit()
print(anova_lm(model32))

model33=ols('V4오른쪽_마지막_터치시간_표준편차 ~ C(기분)',df33).fit()
print(anova_lm(model33))

model34=ols('V4터치_횟수 ~ C(기분)',df34).fit()
print(anova_lm(model34))

model35=ols('V4왼쪽_터치_횟수 ~ C(기분)',df35).fit()
print(anova_lm(model35))

model36=ols('V4오른쪽_터치_횟수 ~ C(기분)',df36).fit()
print(anova_lm(model36))

model37=ols('V5정답율 ~ C(기분)',df37).fit()
print(anova_lm(model37))

model38=ols('V5난이도_평균 ~ C(기분)',df38).fit()
print(anova_lm(model38))

model39=ols('V5난이도_표준편차 ~ C(기분)',df39).fit()
print(anova_lm(model39))

model40=ols('V5난이도_최대 ~ C(기분)',df40).fit()
print(anova_lm(model40))

model41=ols('V5결정시간_평균 ~ C(기분)',df41).fit()
print(anova_lm(model41))

model42=ols('V5결정시간_표준편차 ~ C(기분)',df42).fit()
print(anova_lm(model42))

model43=ols('V5터치_횟수 ~ C(기분)',df43).fit()
print(anova_lm(model43))


# In[246]:


import pandas as pd
from scipy import stats

# 데이터프레임에서 필요한 열 선택
response = df18['M5남은_제한시간_평균']
independent_var = df18['기분']
# 독립변수의 범주 확인
categories = independent_var.unique()
# 그룹별 데이터 생성
groups = [response[independent_var == category] for category in categories]
# Welch's 분산분석 수행
f_value, p_value = stats.f_oneway(*groups)

# 결과 출력
print("Welch's 분산분석 결과:")
print("F-value:", f_value)
print("p-value:", p_value)


# In[247]:


response = df22['V4전체_정답율']
independent_var = df22['기분']
categories = independent_var.unique()
groups = [response[independent_var == category] for category in categories]
# Welch's 분산분석 수행
f_value, p_value = stats.f_oneway(*groups)
# 결과 출력
print("Welch's 분산분석 결과:")
print("F-value:", f_value)
print("p-value:", p_value)


# In[248]:


response = df24['V4오른쪽_정답율']
independent_var = df24['기분']
categories = independent_var.unique()
groups = [response[independent_var == category] for category in categories]
# Welch's 분산분석 수행
f_value, p_value = stats.f_oneway(*groups)
# 결과 출력
print("Welch's 분산분석 결과:")
print("F-value:", f_value)
print("p-value:", p_value)


# In[249]:


response = df25['V4난이도_평균']
independent_var = df25['기분']
categories = independent_var.unique()
groups = [response[independent_var == category] for category in categories]
# Welch's 분산분석 수행
f_value, p_value = stats.f_oneway(*groups)
# 결과 출력
print("Welch's 분산분석 결과:")
print("F-value:", f_value)
print("p-value:", p_value)


# In[250]:


response = df26['V4난이도_표준편차']
independent_var = df26['기분']
categories = independent_var.unique()
groups = [response[independent_var == category] for category in categories]
# Welch's 분산분석 수행
f_value, p_value = stats.f_oneway(*groups)
# 결과 출력
print("Welch's 분산분석 결과:")
print("F-value:", f_value)
print("p-value:", p_value)


# # welch's 분산분석 -> 사후검정

# In[251]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


g18 = sm.stats.multicomp.pairwise_tukeyhsd(model18.resid, df18['기분'])
g22 = sm.stats.multicomp.pairwise_tukeyhsd(model22.resid, df22['기분'])
g24 = sm.stats.multicomp.pairwise_tukeyhsd(model24.resid, df24['기분'])
g25 = sm.stats.multicomp.pairwise_tukeyhsd(model25.resid, df25['기분'])
g26 = sm.stats.multicomp.pairwise_tukeyhsd(model26.resid, df26['기분'])



# In[252]:


print(g18.summary())


# In[253]:


sns.boxplot(x='기분',y='M5남은_제한시간_평균',data=df18)


# In[254]:


print(g22.summary())


# In[255]:


sns.boxplot(x='기분',y='V4전체_정답율',data=df22)


# In[256]:


print(g24.summary())


# In[257]:


sns.boxplot(x='기분',y='V4오른쪽_정답율',data=df24)


# In[258]:


print(g25.summary())


# In[259]:


sns.boxplot(x='기분',y='V4난이도_평균',data=df25)


# In[260]:


print(g26.summary())


# In[261]:


sns.boxplot(x='기분',y='V4난이도_표준편차',data=df26)


# # 아노바 후 사후검정

# In[262]:


from statsmodels.sandbox.stats.multicomp import MultiComparison
import scipy.stats

f1=MultiComparison(df1.A4난이도_평균,df1.기분)
f2=MultiComparison(df2.A4난이도_표준편차,df2.기분)
f4=MultiComparison(df4.A4결정시간_평균,df4.기분)
f5=MultiComparison(df5.A4결정시간_표준편차,df5.기분)
f6=MultiComparison(df6.A4터치_횟수,df6.기분)
f23=MultiComparison(df23.V4왼쪽_정답율,df23.기분)
f27=MultiComparison(df27.V4난이도_최대,df27.기분)
f28=MultiComparison(df28.V4마지막_터치시간_평균,df28.기분)
f29=MultiComparison(df29.V4마지막_터치시간_표준편차,df29.기분)
f30=MultiComparison(df30.V4왼쪽_마지막_터치시간_평균,df30.기분)
f31=MultiComparison(df31.V4왼쪽_마지막_터치시간_표준편차,df31.기분)
f32=MultiComparison(df32.V4오른쪽_마지막_터치시간_평균,df32.기분)
f33=MultiComparison(df33.V4오른쪽_마지막_터치시간_표준편차,df33.기분)
                    


# In[263]:


r1=f1.allpairtest(scipy.stats.ttest_ind, method='bonf')
r1[0]
#기분 1과4, 기분1와5 간의 평균차이가 유의미함


# In[264]:


import seaborn as sns
sns.boxplot(x='기분',y='A4난이도_평균',data=df1)


# In[265]:


r2=f1.allpairtest(scipy.stats.ttest_ind, method='bonf')
r2[0]
#기분 1과4, 기분1와5 간의 평균차이가 유의미함


# In[266]:


sns.boxplot(x='기분',y='A4난이도_표준편차',data=df2)


# In[267]:


r4=f4.allpairtest(scipy.stats.ttest_ind, method='bonf')
r4[0]
#기분 1과4, 기분3와4 간의 평균차이가 유의미함


# In[268]:


sns.boxplot(x='기분',y='A4결정시간_평균',data=df4)


# In[269]:


r5=f5.allpairtest(scipy.stats.ttest_ind, method='bonf')
r5[0]
#기분 1과4, 기분2와4, 기분3과4 간의 평균차이가 유의미함


# In[270]:


sns.boxplot(x='기분',y='A4결정시간_표준편차',data=df5)


# In[271]:


r6=f6.allpairtest(scipy.stats.ttest_ind, method='bonf')
r6[0]
#기분 1과4간의 평균차이가 유의미함


# In[272]:


sns.boxplot(x='기분',y='A4터치_횟수',data=df6)


# In[273]:


r23=f23.allpairtest(scipy.stats.ttest_ind, method='bonf')
r23[0]
#기분 2와5 , 기분 4와5간의 평균차이가 유의미함


# In[274]:


sns.boxplot(x='기분',y='V4왼쪽_정답율',data=df23)


# In[275]:


r27=f27.allpairtest(scipy.stats.ttest_ind, method='bonf')
r27[0]
#기분 1와5 ,기분2와5 ,기분 4와5간의 평균차이가 유의미함


# In[276]:


sns.boxplot(x='기분',y='V4난이도_최대',data=df27)


# In[277]:


r28=f28.allpairtest(scipy.stats.ttest_ind, method='bonf')
r28[0]
#기분 1과5, 2와5 , 3과5, 4와5 간의 평균차이가 유의미함


# In[278]:


sns.boxplot(x='기분',y='V4마지막_터치시간_평균',data=df28)


# In[279]:


r29=f29.allpairtest(scipy.stats.ttest_ind, method='bonf')
r29[0]
#기분 1과5, 2와5 , 3과5, 4와5 간의 평균차이가 유의미함


# In[280]:


sns.boxplot(x='기분',y='V4마지막_터치시간_표준편차',data=df29)


# In[281]:


r30=f30.allpairtest(scipy.stats.ttest_ind, method='bonf')
r30[0]
#기분 1과5, 2와5 , 3과5 간의 평균차이가 유의미함


# In[282]:


sns.boxplot(x='기분',y='V4왼쪽_마지막_터치시간_평균',data=df30)


# In[283]:


r31=f31.allpairtest(scipy.stats.ttest_ind, method='bonf')
r31[0]
#기분 2와 5간의 평균차이가 유의미함


# In[284]:


sns.boxplot(x='기분',y='V4왼쪽_마지막_터치시간_표준편차',data=df31)


# In[285]:


r32=f32.allpairtest(scipy.stats.ttest_ind, method='bonf')
r32[0]
#기분 1과5, 2와5 , 3과5, 4와5 간의 평균차이가 유의미함


# In[286]:


sns.boxplot(x='기분',y='V4오른쪽_마지막_터치시간_평균',data=df32)


# In[287]:


r33=f33.allpairtest(scipy.stats.ttest_ind, method='bonf')
r33[0]


# In[288]:


sns.boxplot(x='기분',y='V4오른쪽_마지막_터치시간_표준편차',data=df33)


# In[ ]:





# In[ ]:




