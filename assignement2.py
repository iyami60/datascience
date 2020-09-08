import os   
import csv
import pandas as pd
import pandas as pd
df = pd.read_csv(r'C:\Users\ISSAM\Desktop\py_doc\tkharbii9\olympics.csv', index_col=0, skiprows=1)
census_df = pd.read_csv(r'C:\Users\ISSAM\Desktop\py_doc\tkharbii9\census.csv')

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
#print(df.tail(10))


#part1
def max_gold():
    return df[df['Gold'] == max(df['Gold'])].T.keys()[0]

def max_diff():
    tran = df.T
    C = []
    for i in tran.columns:
        x = df['Gold'][i] - df['Gold.1'][i]
        C.append(abs(x))
    maximum = max(C)
    a = C.index(maximum)
    return df.T.keys()[a]

def biggest_diff():
    tran = df.T
    C = []
    for i in tran.columns:
        x = (df['Gold'][i] - df['Gold.1'][i])/(df['Gold'][i] + df['Gold.1'][i] + df['Gold.2'][i])
        C.append(abs(x))
    maximum = max(C)
    result = C.index(maximum)
    return df.T.keys()[result]


def score():
    slice2 = df.iloc[:,11:14]
    data = []
    for index,row in slice2.iterrows():
        somme  = row['Gold.2']*3 + row['Silver.2']*2 + row['Bronze.2']*1
        data.append(somme)
    result = pd.Series(data = data, index = df.index, name='Points')
    return result


#PART2
def occ():
    C = []
    x=0
    resultat = census_df.groupby('STATE').nunique().index
    for i in range(len(resultat)):
        for index,row in census_df.iterrows():
            if (row['STATE'] == resultat[i]):
                x+=1
        C.append(x)
        x = 0
    result = max(C)
    return C

def occ_consec():
    C = []
    x=0
    resultat = census_df.groupby('STATE').nunique().index
    for i in range(len(resultat)):
        for index,row in census_df.iterrows():
            if (row['STATE'] == resultat[i]):
                x+=1
        C.append(x)
    return C


def list_pop():
    resultat = occ_consec()
    extract = census_df.iloc[0:resultat[0],:]
    C = []
    names =[]
    for j in range(1,len(resultat)):
        a = extract['CENSUS2010POP'].nlargest(3)
        C.append(sum(a))
        extract = census_df.iloc[resultat[j-1]:resultat[j],:]
    res = pd.Series(data=C)
    tool_1 = res.sort_values(ascending=True).index
    tool_2 = census_df.groupby('STNAME').nunique().index
    for t in range(len(tool_1)):
        names.append(tool_2[tool_1[t]])
    return names
    

def answer_seven():
    popul = census_df.iloc[:,9:15].T
    list1 = []
    for i in range(len(census_df)):
        mn = min(popul[i])
        mx = max(popul[i])
        diff = mx-mn
        list1.append(diff)
    maxi = max(list1)
    indx = list1.index(maxi)
    return census_df.iloc[indx]['CTYNAME']

def answer_eight():
    data = pd.DataFrame()
    for index,row in census_df.iterrows():
        if (row['REGION']==1 or row['REGION']==2) :
            data = data.append(census_df.iloc[index])
    dataFinale = data.loc[:,['STNAME','CTYNAME']]
    bool_series = dataFinale['CTYNAME'].str.startswith('Washington', na = False) 
    return dataFinale[bool_series]

def answer_five():
    df=census_df[census_df['SUMLEV'] == 50]
    df = df.groupby( [ "SUMLEV", "STNAME"] ).size().to_frame(name = 'count').reset_index()
    return df


dfx = pd.DataFrame({
    'value':[20.45,22.89,32.12,111.22,33.22,100.00,99.99],
    'product':['table','chair','chair','mobile phone','table','mobile phone','table']
})

df1 = dfx.groupby('product')['value'].sum().to_frame(name= "Somme").reset_index()

df2 = dfx.groupby('product')['value'].sum().to_frame().reset_index().sort_values(by='value')

dfx1 = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
dfx1.rename(columns={0: 'Grades'}, inplace=True)


print("\n")
print(dfx1['Grades'].astype('category'))