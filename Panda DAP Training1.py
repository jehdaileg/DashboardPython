#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


movies = pd.read_csv('movie.csv')

movies.head()  #voir les 5 premieres lignes 
movies.head(10) #voir les 10 premieres lignes 
movies.shape #nombre des lignes et colonnes 
movies.columns #affiche les colonnes 
movies.values  #voir les valeurs des colonnes
movies.index #voir les index

type(movies.values) #constater que les donnees sont en tab numpy 

movies.tail() #voir les 5 dernieres lignes 
movies.tail(10) #voir les 10 dernieres lignes 
movies.sample(5) #voir 5 lignes aleatoires

movies.dtypes #type de chaque colonne(int,string,...)
movies.dtypes.value_counts() #avoir une idee sur le nombre des types de variables 

movies.info() #avoir un appercu global sur le type des donnees 

movies.language


# In[ ]:





# In[17]:


movies.dtypes.value_counts().plot.pie()


# In[3]:


movies.language #voir les differentes valeurs dispos dans la colonne language 
movies['language'] #idem


movies['language'].unique() #voir chaque valeur unique de la colonne 
#idem que 
movies.language.unique()

#compter toutes ces valeurs uniques dispo de la colonne 
len(movies.language.unique()) #inclut le nan 

#compter valeurs par variables 
movies.language.value_counts(ascending=True) #ordre croissant 


#pourcentage 
movies.language.value_counts(normalize=True) *100

movies.head()
#rename a column 
movies.rename(columns = {'director_facebook_likes': 'director_fb_likes'}, inplace = True)

movies.head()

movies_string = movies.select_dtypes('object') #recuperer seulement les colonnes de type object 
movies_string.head()
movies_string.describe()

movies_int_float = movies.select_dtypes(include = 'number')
movies_int_float.describe()

#garder seulement les colonnes listees dans le tableau 
movies_reduce = movies[['director_name','duration','gross', 'genres', 'movie_title', 'plot_keywords', 'budget','title_year', 'language', 'imdb_score']]
movies_reduce.head()




#supprimer les colonnes 
movies_reduce = movies_reduce.drop(['plot_keywords'], axis = 1)

movies.drop([])
movies_reduce.head()

movies.columns

#creer une colonne 

#renommer 
movies_reduce.rename(columns = {'gross':'revenue'}, inplace=True)
movies_reduce.head()
movies_reduce.sort_values('revenue')[:10] #trier selon le revenu des films en croissant 
movies_reduce.sort_values('revenue', ascending=False) #tri decroissant 


#tri sur base de plusieurs colonnes 
movies_reduce.sort_values(['title_year', 'imdb_score'], ascending=False)[:10]

#acces a des colonnes particulieres 
movies['movie_title']   #or movies.movie_title

movies_reduce.iloc[4]#affiche toutes les informations coreespondant a l'index 4 

#filter les lignes d'un dataframe 

#filtrer sur une seule variable 
movies['director_name']

movies['director_name'] == 'Steven Spielberg'




isSteven = movies[movies['director_name'] == 'Steven Spielberg' ]
isSteven.head()
#moyenne score des films de Stevene Sp.

isSteven['imdb_score'].mean()

#filtrer sur deux ou plusieurs variables
#les films francais et dont la duree est inf a 60 

#movies_F = movies[movies['language'] == 'French' & movies['duration'] < 60]
cond1 = movies['language'] == 'French'
cond2 = movies['duration'] < 60 



movies_F = movies[cond1 & cond2]
movies_F.head()

#concatener deux dataframes  

#pd.concat([df1,df2], ignore_index = True)

#merger deux dataframes 

#merge_datas = pd.merge(df1,df2, how="inner", on="Date", suffixes =('_df1', '_df2'))

#suppression des doublons

movies = pd.read_csv('movie.csv')



# In[11]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


movies = pd.read_csv('movie.csv')
movies.head()

#suppression des doublons 

movies.duplicated() #identifie les lignes doublons en false et true

movies.drop_duplicates() #supprimme les doublons sans reset les index 

movies.drop_duplicates().reset_index(drop= True) #supprime tout en reformant les index 

#supprimer les douvblons sur une colonne precise 
#movies.drop_duplicates(['colonne_concernee']).reset_index(drop = True)

#exemple de suppression 

#dataframe.drop_duplicates(['col']).reset_index(drop = True)#supprime les doublons en gardant la premiere occurrence

#dataframe.drop_duplicates(['col'], keep = 'last').reset_index(drop = True) #supprime les doublons grace a la colonne specifiee mais en gardant la derniere occurrence

#remplacer des valeurs dans un dataframe 

#remplacer tous les 20 par 15  //toujours stocker dans une nouvelle variable pour signifier les nouvelles datas 

#--dataframe.replace(to_replace = 20, value=np.nan) #remplacer avec les nan puis 

#--dataframe.replace(to_replace = np.nan, value= 15) #ainsi remplacer les nan(20) par 15 

#remplacer dans une colonne precise 

#--dF['col_to_r'].replace(to_replace = 20, value = np.nan)
#--dF['col_to_r'].replace(to_replace = np.nan, value = 15)

#ou encore remplacer les 10 et 15 par 5 et 7 
#--df.replace(to_replace = [10, 15], value = [5,7])

#dans une colonne precise 
#--df['col_to_r'].replace(to_replace = [10, 15], value = [5, 7])


#remplacer une valeur precise 
#remplacer 4 de la 3 eme ligne(index 2) et de la troisieme colonne 

#--df.loc[2, "col3"] #avec loc 
#--df.iloc[2, 2] #avec iloc 

#--df.iat[2,2] = 5000000 # ou 

#--df.at[2, "col3"] = 5000000

#remplacer les lignes vides (NaN)

#evaluer les lignes vides dans un dataFrame 
#--df.isnull().sum #la somme de toutes les lignes avec des valeurs manquantes 
#--df.isnull().mean() # moyenne de toutes les lignes avec des NaN ou des valeurs manquantes 
        #en pourcentage 
#--(df.isnull().mean()) * 100
#suppression des lignes avec au moins une valeur manquante 
#--df.dropna() #supprime toutes les lignes avec au moins une valeur manquante 

#--df.dropna(how = 'all') #supprime seulement les lignes avec les 100/100 des valeurs manquantes
#suppression des colonnes avec au moins une valeur manquante 

#--df.dropna(axis = 1) 

#suppression des colonnes avec les 100/100 des valeurs manquantes
#--df.dropna(how = 'all', axis = 1)

#remplacer les valeurs manquantes 
#--df.fillna(0) #remplacer toutes les valeurs manquantes par 0 
#--df.mean() #moyenne des valeurs numeriques 
#--df.fillna(df.mean()) #remplacer les valeurs manquantes par differentes moyennes


#--df.col.fillna(method = 'ffill') #remplace dans la colonne les NaN par la valeur qui suit et qui est non nulle 
#-- df.col.fillna(method = 'bfill') #remplace dans la colonne les NaN par la valeur qui vient avant et qui est non nulle 

#transformer les variables numeriques en variables categoriques pour categoriser les donnees  

notes = np.random.randint(20, size=(15,))

notes

#transformer cette liste en dataframe de panda 
notes_df = pd.DataFrame(notes, columns = ['notes'])
notes_df

notes 
#creation de la liste des valeurs qui conditionnent les categories selon le contexte 
#bins = [-1, 10, 12, 15, 18, 20] #ce sont des intervalles , de -1 a 10, de 10 a 12, de 12 a 15, ... 

#creation des categories sur base des valeurs bins 
#bins_names = ['echec', 'passable', 'assez bon', 'Bien', 'tres bien']

#creation de category 
#category = pd.cut(notes, bins, right = False, labels = bins_names)
#category 

#exemple de categorisation deux 
participants_age = [12, 14, 19, 24, 27, 41]

#valeurs intervalles de categorisation 
bins = [1, 10, 17, 25, 39]

bins_names = ['enfant', 'mineur', 'jeune celibataire', 'jeune responsable']

category_age = pd.cut(participants_age, bins, right= False, labels = bins_names)

category_age 

#grouper des donnees








# In[41]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#grouper des donnees 

automobiles = pd.read_csv('Automobile_data.csv')

automobiles.head()

#grouper les datas selon la colonne body-style 
#vision globale de la proportion selon le body-style 
automobiles['body-style'].value_counts()
#voir les analyses preliminaires selon un type specifique 

automobiles.groupby('body-style').describe().T

#analyse de la colonne en global 
automobiles['body-style'].describe().T

#faire un groupe de deux colonnes 
two_groups = automobiles.groupby(['body-style', 'drive-wheels'])
two_groups.groups  #en faire un group 
two_groups.first()

#prendre un groupe de cas specifique 
#creer un groupe sur base de la colonne puis 
auto_group_by_bodyStyle = automobiles.groupby('body-style')
#prendre les convertible de body-style 
auto_convert = auto_group_by_bodyStyle.get_group('convertible')

auto_convert.mean()

#aggreagations 

auto_group_by_bodyStyle.agg(['mean', 'sum', 'max', 'count'])








# In[50]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#analyse des donnees temporielles 

#Objet: Analyse de la consommation d'energie en Allemagne de 2006 à 2017 de manière journalière; analyse des tendances daily 

#import des datas 
consommation = pd.read_csv('opsd_germany_daily.csv')
#verifier les donnees 
consommation.head()
#voir la colonne Date 
consommation.Date
#converir cette colonne Date en datetime 
consommation.Date = pd.to_datetime(consommation.Date)
#ou consommation['Date'] = pd.to_datetime(consommation['Date'])
consommation.Date 
#verifier si des lignes en colonne Date sont nulles 
consommation['Date'].isnull().any()

#Faire de la colonne Date l'index du dataframe
consommation = consommation.set_index('Date')
consommation.head()

#voir les lignes vides 
consommation.isnull().mean()

#remplacer les lignes vides de la colonne Consumption ou consommation avec des valeurs approximatives
consommation['Consumption'].isnull().any()
#consommation['Consumption'].ffill()

#check de l'index 
consommation.index
#creer une colonne Year 
consommation['Year'] = consommation.index.year

#creer une colonne Mois 
consommation["Month"] = consommation.index.month 

#creer une colonne jour avec day_name 
consommation["day_name"] = consommation.index.day_name()
#voir les donnees avec les nouvelles colonnes
consommation.head()

#compter le nombre d'annees pour avoir idee sur la quantiy des datas for analyzing 
consommation.Year.value_counts()

#creer une colonne is_weekend pour check true or false si c'est un weekend ou pas (saturday or sunday ou isin(saturday, sunday))
#consommation['is_weekend'] = consommation.day_name == 'Saturday' | consommation.day_name == 'Sunday'
consommation['is_weekend']  = consommation.day_name.isin(['Saturday', 'Sunday'])
#checker la data avec la nouvelle colonne 
consommation.head()

#checker l'annee Max 
consommation.Year.max()
#selectionner les lignes du mois de Mai 2010 a Aout 2013 
consommation.loc["2010-05-01": "2013-08-01"]

#analyse proprement dites avec des graphiques seaborn et matplotlib 
#consommation en energie de 2006 et 2017 
#consommation.loc["2006":"2017", "Consumption"].plot().plt
#ou en ameliorant 
#--sns.set_style('whitegrid')
#--consommation.loc["2006": "2017", "Consumption"].plot(linewidth = 5, figsize= (12, 8))
#plt.show()
#afficher le jour dela semaine  avec beaucoup plus de consommation globale 
consommation.groupby('day_name').sum()['Consumption'].sort_values(ascending = False)
#afficher la moyenne de consommation par jour globalement 
#((consommation.groupby('day_name').mean()['Consumption'])*100).sort_values(ascending = False)

#explications: Je groupe par jour et je fais la somme des consommations et je les classe du plus grand au plus petit 
#afficher le total des valeurs par jour 
consommation.groupby('day_name').sum()

#afficher maintenant le total de la consommation globale par jour du plus petit au plus grand 
consommation.groupby('day_name').sum()['Consumption'].sort_values()
#creer un groupday group par jour 
consommation_groupday = consommation.groupby('day_name')
wednesday_data = consommation_groupday.get_group('Wednesday')
wednesday_data

#creer un graphique avec seaborn pour les jours avec barplot 
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#sns.barplot(x = consommation_groupday['day_name'], y = consommation_groupday['Consumption'], order = order)
#plt.show()

#consommation.head()


#voir les consommation par annee avec Facetgrid avec Grid
#grid = sns.FacetGrid(consommation, row="Year")
#grid.map(plt.plot, 'Consumption')
#zoomer sur 2016 en faisant une copie des donnees 
data_2016 = consommation.loc['2016'].copy()
data_2016.head()

#visulation des donnees 2016 avec scatterplot en fixant des limites 
#sns.scatterplot(x = data_2016.index, y = data_2016.Consumption)
#plt.xlim(data_2016.index.min(), data_2016.index.max())


#faire un graphique de comparaison de consommation en energie en 2016, de la prodution d'energie solaire en 2016 et d'energie eolienne sur un mm graphique 
#le faire avec scatterplot de seaborn

#fig, axes = plt.subplots(figsize=(16,10), nrows = 3, ncols = 1)

#sns.scatterplot(data = data_2016, x = data_2016.index, y = data_2016.Consumption, ax = axes[0])
#axes[0].set_title('Consommation en energie en 2016')
#axes[0].set_xlim(data_2016.index.min(), data_2016.index.max())

#sns.scatterplot(data = data_2016, x = data_2016.index, y = 'Solar', ax = axes[1])
#axes[1].set_title('Production Energie solaire en 2016')
#axes[1].set_xlim(data_2016.index.min(), data_2016.index.max())

#sns.scatterplot(data = data_2016, x = data_2016.index, y = 'Wind', ax = axes[2])
#axes[2].set_title('Production Energie Eolienne en 2016')
#axes[2].set_xlim(data_2016.index.min(), data_2016.index.max())
#fig.tight_layout(pad = 2)
#plt.savefig('scatterplot.jpg')


#comparaison de la consommation et de la prodution solaire avec lineplot 
#fig, ax = plt.subplots(figsize=(16,10), nrows = 1, ncols = 1)

#sns.lineplot(data = data_2016, x = data_2016.index, y = 'Consumption', ax = ax, legend = 'brief', label = 'Consommation')

#sns.lineplot(data = data_2016, x = data_2016.index, y = 'Solar', ax = ax, legend = 'brief', label = 'Energie solaire')





#consommation par mois avec un boxplot 
#-sns.boxplot(data = data_2016, x = 'Month', y = 'Consumption')



#consommation pour voir le jour qu'on consomme le plus avec boxplot 

#g = sns.boxplot(data = consommation, x = 'day_name', y = 'Consumption')
#g.set_xticklabels(rotation = 30, labels=consommation["day_name"].unique())
#plt.show()


#consommation en semaine vs le weekend avec boxplot 

#g = sns.boxplot(data = consommation, x = 'is_weekend', y = 'Consumption')
#g.set_xticklabels(rotation = 30, labels=consommation["is_weekend"].unique())
#plt.show()


#le resampling pour grouper ,,,, ex, consommaion par semaine 
#supprimer les colonnes Year Month et Weekend 


#consommation par semaine

consommation.loc["2015"].resample('W').mean()

#consommation en moyenne par mois 

consommation.loc["2015"].resample('M').mean()



#consommation par trimestre avec resample 

consommation.loc["2015"].resample('Q').mean()









# In[83]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

automobiles = pd.read_csv('Automobile_data.csv')

automobiles.head()
#visualisation des donnees avec MatplotLib 

#cfr partie up 
#un graphique avec methode de Matlab 

#cfr partie up 

#un graphique en POO 

#cfr partie up

# graphique avec panda avec kind de plot 
#utomobiles.plot(kind = 'scatter', x = 'body-style', y = 'fuel-type')


#utilisation de plot.bar() ou plot(kind ='bar')  //ou barh
#automobiles['body-style'].value_counts().plot(kind='pie')

#automobiles['body-style'].plot(kind='hist')
#visualisation des donnees avec Seaborn avec les datas du titanic

titanic = pd.read_excel('titanic_dataset_3.xls')

titanic.head()

#voir la distribution des sexes 
#titanic.sex.value_counts().plot(kind='barh') #OU AVEC bar 
#utilisation de boxplot de la Moyenne, Mediane ou distribution des datas selon une variable definie 
#sns.boxplot(data = titanic, x='age') #sur axe x 
#sns.boxplot(data = titanic, y = 'age')
#DISTRIBUTION
#distribution avec kdeplot(Age)
#sns.kdeplot(titanic.age)


#distribution avec distplot  

#sns.distplot(titanic.age)

#analyse bivarielle pour analyser une variable categorielle et numerique  
#avec violinplot
#sns.violinplot(x = 'survived', y = 'age', data = titanic)

#countplot pour analyser une varaible categorielle 
#sns.countplot(x = 'sex', data = titanic)


#analyser plusieurs variables numeriques pour la correlation avec heatmap 
#voir d'abord la correlation de toutes les valeurs des variables numeriques 
titanic.corr()
#
#--sns.heatmap(titanic.corr(), annot= True, cmap='YlGnBu')

#utiliser heatmap pour la correlation entre plusieurs variables 
#sns.heatmap(titanic.corr()[['survived', 'age']], annot= True, cmap='YlGnBu')



#analyse multivariee 

#Histogramme des hommes qui ont survecu 
#creer groupe des hommes et utilisr hist()
#cond_h_s = (titanic.sex == 'male') & (titanic.survived == 1)
#titanic[cond_h_s]['age'].hist()


#utiliser FacetGrid pour la distribution de l'age des hommes qui nn'ont pas survecu et qui ont survecu et pour les femmes selon Age
#grid = sns.FacetGrid(titanic, col='sex', row= 'survived')
#grid.map(plt.hist, 'age')

#utiliser regplot pour la Regression 

#sns.regplot(titanic, x = 'age', y='fare')

#deux variables categoriques et deux variables numerques, le lmplot (Exe: age, fare, survived, sex )
#sns.lmplot(data = titanic, x='age', y = 'fare', col='sex', row='survived')

#utilisation de stripplot pour la distribution d'une variable categorique et une variable numerique 
#sns.stripplot(data = titanic, x='survived', y ='age')



#idem mais avec swarmplot pour constater la largeur 

#sns.swarmplot(data = titanic, x='survived', y ='age')


#utilisation de catplot 

sns.catplot(data = titanic, x='survived', y ='age', hue='sex')


#utilisation de pairplot pour correspondre deux variables deux a deux sur le numerique avec l'option hue pour une categorisation 



 






# In[ ]:




