import pandas as pd
import numpy as np

def answer_one():
        #read data
        energy = pd.read_excel(r'C:\Users\ISSAM\Desktop\py_doc\tkharbii9\Energy Indicators.xls', sheet_name= 'Energy')  
        energy = energy[17:244]
        energy.drop("Unnamed: 0", axis = 1,inplace =True)
        energy.drop("Unnamed: 2", axis = 1,inplace =True)

        #just make date more readable
        energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
        energy = energy.replace(r'...',np.NaN )
        energy['Energy Supply'] = energy['Energy Supply']*1000000
        energy['Country'] = energy['Country'].replace({'China, Hong Kong Special Administrative Region':'Hong Kong',
                                    'United Kingdom of Great Britain and Northern Ireland':'United Kingdom',
                                    'Republic of Korea':'South Korea',
                                    'United States of America':'United States',
                                    'Iran (Islamic Republic of)':'Iran'}
                                            )
        energy.reset_index(0, inplace=True)
        energy.drop('index',axis=1, inplace=True)

        GDP    = pd.read_csv(r"C:\Users\ISSAM\Desktop\py_doc\tkharbii9\world_bank.csv",skiprows = 4)
        GDP['Country Name'] = GDP['Country Name'].replace({"Korea, Rep.": "South Korea", 
                                                  "Iran, Islamic Rep.": "Iran",
                                                  "Hong Kong SAR, China": "Hong Kong"}
                                                  )
        GDP    = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
        GDP.rename(columns = {'Country Name' : 'Country'}, inplace = True)

        ScimEn = pd.read_excel(io = r'C:\Users\ISSAM\Desktop\py_doc\tkharbii9\scimagojr-3.xlsx')
        ScimEn = ScimEn[0:15]

        df     = pd.merge(ScimEn, energy, how='inner', left_on='Country', right_on='Country')
        new_df = pd.merge(df, GDP, how='inner', left_on='Country', right_on='Country')
        new_df = new_df.set_index('Country')
        return new_df

#print(answer_one())

def answer_two():
        #read data
        energy = pd.read_excel(r'C:\Users\ISSAM\Desktop\py_doc\tkharbii9\Energy Indicators.xls', sheet_name= 'Energy')  
        energy = energy[17:244]
        energy.drop("Unnamed: 0", axis = 1,inplace =True)
        energy.drop("Unnamed: 2", axis = 1,inplace =True)

        #just make date more readable
        energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
        energy = energy.replace(r'...',np.NaN )
        energy['Energy Supply'] = energy['Energy Supply']*1000000
        energy['Country'] = energy['Country'].replace({'China, Hong Kong Special Administrative Region':'Hong Kong',
                                    'United Kingdom of Great Britain and Northern Ireland':'United Kingdom',
                                    'Republic of Korea':'South Korea',
                                    'United States of America':'United States',
                                    'Iran (Islamic Republic of)':'Iran'}
                                            )
        energy.reset_index(0, inplace=True)
        energy.drop('index',axis=1, inplace=True)

        GDP    = pd.read_csv(r"C:\Users\ISSAM\Desktop\py_doc\tkharbii9\world_bank.csv",skiprows = 4)
        GDP['Country Name'] = GDP['Country Name'].replace({"Korea, Rep.": "South Korea", 
                                                  "Iran, Islamic Rep.": "Iran",
                                                  "Hong Kong SAR, China": "Hong Kong"}
                                                  )
        GDP    = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
        GDP.rename(columns = {'Country Name' : 'Country'}, inplace = True)

        ScimEn = pd.read_excel(io = r'C:\Users\ISSAM\Desktop\py_doc\tkharbii9\scimagojr-3.xlsx')
        ScimEn = ScimEn[0:15]

        df     = pd.merge(ScimEn, energy, how='outer', left_on='Country', right_on='Country')
        new_df = pd.merge(df, GDP, how='outer', left_on='Country', right_on='Country')
        new_df = new_df.set_index('Country')
        new_df = new_df.shape[0]-15
        return new_df


def answer_three():
    Top15 = answer_one()
    years_to_keep = np.arange(2006, 2016).astype(str)
    Top15['avgGDP'] = Top15[years_to_keep].mean(axis=1)
    return Top15['avgGDP'].sort_values(ascending=False)


def answer_eleven():
    import numpy as np
    import pandas as pd
    Top15 = answer_one()
    ContinentDict  = {'China':'Asia',
                      'United States':'North America', 
                      'Japan':'Asia',
                      'United Kingdom':'Europe',
                      'Russian Federation':'Europe',
                      'Canada':'North America',
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}
    Top15 = Top15.reset_index()
    Top15['Continent'] = Top15['Country'].map(ContinentDict)
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    result = Top15.copy()
    result = result[['Continent', 'PopEst']]
    result = result.groupby('Continent')['PopEst'].agg({'size': np.size,'sum': np.sum,'mean': np.mean,'std': np.std})
    #result = grouped.agg(['np.size', 'sum', 'mean', 'std'])
    idx = pd.IndexSlice
    #result = result.loc[:, idx['PopEst']]
    #result = result.reset_index()
    #result = result.set_index('Continent')
    return result

Top15 = answer_one()
ContinentDict  = {'China':'Asia',
                      'United States':'North America', 
                      'Japan':'Asia',
                      'United Kingdom':'Europe',
                      'Russian Federation':'Europe',
                      'Canada':'North America',
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}
Top15 = Top15.reset_index()
Top15['Continent'] = Top15['Country'].map(ContinentDict)
Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
result = Top15.copy()
result = result[['Continent', 'PopEst']]
#result = result.groupby('Continent')['PopEst'].agg({'size': np.size,'sum': np.sum,'mean': np.mean,'std': np.std})


x = np.random.binomial(1,0.1,150)
print(x)