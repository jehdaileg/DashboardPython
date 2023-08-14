#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 



#chargement du dataframe 

covid_df = pd.read_excel('cov19.xlsx')

#covid_df.head()

#covid_df.info()

#covid_df.describe()

#covid_df.shape

#check des colonnes et lignes vides, leur pourcentage 
#covid_df.isnull().sum()
#les supprimer 
#covid_df.dropna(inplace = True)

#covid_df.shape

#covid_df.groupby('countriesAndTerritories')['cases', 'deaths'].sum()
#les pays avec le plus grand nombre des cas avec group by Pays, les cas et decedes // countriesAndTerritories
#cases_deaths_by_countries = covid_df.groupby('countriesAndTerritories')['cases', 'deaths'].sum().sort_values('cases',ascending = False)

#cases_deaths_by_countries

#grouper les donnees du dataframe selon kle pays, cas et deces 


#evaluer la somme de cette situation pour rirer conclusion 


#le plays avec le plus grand taux de mortalite 
#nbre de mortalite dufferent de nombre de deces car le taux c'est cas referes / cas deces 
#en sachant que nous avons dja un groupe de cas et deces par pays 

#faire le rapport entre ces cas et deces [pour] avoir le taux de mortalite 


#et stocker les resultats dans une colonne taux_mortalite 

#covid_df['taux_mortalite'] = covid_df['deaths'] / covid_df['cases']

#covid_df.head()


#faire un graphique qui donne une idee bar 

#plt.figure(figsize = (15, 10))
#ax = covid_df['taux_mortalite'].sort_values(ascending = False).head(20).plot.bar()
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha ='right')
#ax.set_xlabel('Country')
#ax.set_ylabel('taux_mortalite')
#ax.set_title('Taux de Mortalite Globale')

#effets du confinement sur le nombre des cas 
##covid_df_month = covid_df.groupby('month')['cases', 'deaths'].sum()

#covid_df_month.head()

#faire un graphique pour avoir un bref sur le nombre de total des cas par mois et le nombre total des morts par mois  avec un plot de kind 'line'
#fig = plt.figure(figsize = (15, 10))
#ax1 = fig.add_subplot(1,2,1)
#ax2 = fig.add_subplot(1,2,2)

#covid_df_month['cases'].plot(kind='line', ax = ax1)
#ax1.set_title('cas par mois ')
#ax1.set_xlabel('Mois')
#ax1.set_ylabel('Nombre de cas')

#covid_df_month['deaths'].plot(kind='line', ax = ax2)
#ax2.set_title('morts par mois')
#ax2.set_xlabel('Mois')
#ax2.set_ylabel('Nombre morts')
#plt.savefig('casdeces.jpg')

#NB cela apres avoir grouper les datas par mois selon les cas et les décès

#analyser les differentes courbes de confinement pour l'allemagen, france, usa, belgique car ils respectent le confinement  en faisant les groupby country en ajoutant les casres et deces 
#germany = covid_df[covid_df['countriesAndTerritories'] == 'Germany']
#month_germany = germany.groupby('month')['cases', 'deaths'].sum()
#df_germany_grouped = month_germany.reset_index()
#df_germany_grouped


#uk = covid_df[covid_df['countriesAndTerritories'] == 'United_Kingdom']
#month_uk = uk.groupby('month')['cases', 'deaths'].sum()
#df_uk_grouped = month_uk.reset_index()
#df_uk_grouped


#france = covid_df[covid_df['countriesAndTerritories'] == 'France']
#month_france = france.groupby('month')['cases', 'deaths'].sum()
#df_france_grouped = month_france.reset_index()
#df_france_grouped

#italy = covid_df[covid_df['countriesAndTerritories'] == 'Italy']
#month_italy = italy.groupby('month')['cases', 'deaths'].sum()
#df_italy_grouped = month_italy.reset_index()
#df_italy_grouped



#on affiche ces datas sur un figure  

#fig = plt.figure(figsize = (20,15))

#ax1 = fig.add_subplot(2,2,1)
#df_germany_grouped.plot(kind='line', x ='month', y ='cases', ax=ax1)
#ax1.set_title('Evolution cases covid en Allemagne')

#ax2 = fig.add_subplot(2,2,2)
#df_france_grouped.plot(kind='line', x ='month', y ='cases', ax = ax2)
#ax2.set_title('Evolution cases covid en France')


#ax3 = fig.add_subplot(2,2,3)
#df_italy_grouped.plot(kind='line', x ='month', y ='cases', ax = ax3)
#ax3.set_title('Evolution cases covid en Italy')


#ax4 = fig.add_subplot(2,2,4)
#df_uk_grouped.plot(kind='line', x ='month', y ='cases', ax = ax4)
#ax4.set_title('Evolution cases covid en United Kingdom')

#plt.savefig('casesByMonthByCountry.jpg')

#situation par continents 
#create a group with cases and deaths in order to do apart analyzis 

continent_df = covid_df.groupby('continentExp')['cases', 'deaths'].sum().sort_values('cases',ascending = False)

continent_df

continent_df['mortality_rate'] = continent_df['deaths'] / continent_df['cases']

continent_df.head(20)



# In[13]:


continent_df.groupby('continentExp')['mortality_rate'].sum().sort_values(ascending = False)


# In[17]:


continent_df.groupby('continentExp')['mortality_rate'].sum().sort_values(ascending = False).plot(kind = 'barh')


# In[ ]:




