import pandas as pd

df_train = pd.read_csv('train.csv')

df_pax0 = df_train.loc[df_train['PAX'] == 0]
df_pax1 = df_train.loc[df_train['PAX'] == 1]
df_pax2 = df_train.loc[df_train['PAX'] == 2]
df_pax3 = df_train.loc[df_train['PAX'] == 3]
df_pax4 = df_train.loc[df_train['PAX'] == 4]
df_pax5 = df_train.loc[df_train['PAX'] == 5]
df_pax6 = df_train.loc[df_train['PAX'] == 6]
df_pax7 = df_train.loc[df_train['PAX'] == 7]


df_pax0.to_csv('Pax0.csv', sep=',', encoding='utf-8', index=False)
df_pax1.to_csv('Pax1.csv', sep=',', encoding='utf-8', index=False)
df_pax2.to_csv('Pax2.csv', sep=',', encoding='utf-8', index=False)
df_pax3.to_csv('Pax3.csv', sep=',', encoding='utf-8', index=False)
df_pax4.to_csv('Pax4.csv', sep=',', encoding='utf-8', index=False)
df_pax5.to_csv('Pax5.csv', sep=',', encoding='utf-8', index=False)
df_pax6.to_csv('Pax6.csv', sep=',', encoding='utf-8', index=False)
df_pax7.to_csv('Pax7.csv', sep=',', encoding='utf-8', index=False)