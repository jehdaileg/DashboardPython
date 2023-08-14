#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 



#chargement des datas 

sales_datas_2019 = pd.read_csv('Sales_2019_datas.csv')

sales_datas_2019.head(10)

#Creation de la copie des donnees pour en garder toujours l'original avant le traitement 

sales_df = sales_datas_2019.copy()


#check les datas 
#sales_df.head(15)

#sales_df.columns #colonnes 

#sales_df.shape #nombre de lignes et des colonnes #verifier le nombre des lignes et des colonnes 

#sales_df.dtypes.value_counts().plot.pie() #avoir une vue globale sur les variables dont il est question

#sales_df.info() #vue globale sur notre dataframe 

#checker les valeurs manquantes par lignes et par colonnes 
#sales_df.isnull().any() #au moins une valeur manquante dans des colonnes 

#sales_df.isnull().sum(axis = 0) #valeurs manquantes par ligne 


#extraire les valeurs manquantes et les stocker dans une variable pour les colonnes 

#valeurs_manquantes = sales_df[sales_df.isnull().any(axis = 1)]

#valeurs_manquantes.isnull().all()


#Ssupprimer les valeurs manquantes de notre dataset et verifier a nouveau le nombre des lignes et colonnes 
#sales_df.dropna(inplace = True)




#reverifier la presence des valeurs manquantes apres suppression 
#sales_df.isnull().any()

#appercu d'une analyse globale 
#sales_df.describe()

#afficher toutes les lignes dans lesquelles nous n'avons pas de valeur correct
#Order ID et Order Date, enlever ces textes qui n'ont pas de place en cet endroit 
#supprimer les valeurs des donnees qui ne sont pas vraies, exemple supprimer les textes d'une colonne ProductId et les stocker dans une nouvelle variable 
#sales_df['Order ID']
#récupère toutes les lignes qui n'ont pas de valeur nombre selon la colonne Date ou Order ID car ce sont deux variables importantes pour l'analyse sans quoi ça ne marche pas 
#on supprime grace a l'index de la ligne 
#avant regardons d'abord ce tableau qui garde toutes ces valeurs abberantes 
#sales_df.loc[~sales_df['Order ID'].str.isdigit(), :]
#suppression de toutes ces donnees de ce tableau abberant
#sales_df_clean = sales_df.drop(sales_df.loc[sales_df['Order Date'] == 'Order Date', :].index)
#test encore si ce tableau a encore des datas 
#sales_df_clean.loc[~sales_df['Order ID'].str.isdigit(), :]


#changer le type des colonnes ou variables surtout les colonnes date en datetime
#convertir la Quantite commande en int 
#sales_df_clean.columns 

#sales_df_clean['Quantity Ordered'] = sales_df_clean['Quantity Ordered'].astype('int')

#sales_df_clean.info()  #checking 

#changer en colonnes numerique le prix
#sales_df_clean['Price Each'] = pd.to_numeric(sales_df_clean['Price Each'])

#check 
#sales_df_clean.info()

#date 
#sales_df_clean['Order Date'] = pd.to_datetime(sales_df_clean['Order Date'])

#check 
#sales_df_clean.head()
#sales_df_clean['Order Date']

#verifier le type des variables qui sont disponibles 


#commencer les analyses proprement dites 
#savoir indexer la colonne date de notre dataframe 
#sales_df_clean.index
#sales_df_clean = sales_df_clean.set_index('Order Date')
#check 
sales_df_clean.head()

#trouver le meilleur mois de vente 

#creer une colonne mois 
#sales_df_clean['Mois'] = sales_df_clean.index.month_name() #mois 
#sales_df_clean['Annee'] = sales_df_clean.index.year#annee 
#sales_df_clean['Jour'] = sales_df_clean.index.day_name() #jour 

#check 
#sales_df_clean.head()

#Pour avoir le meilleur mois, on doit avoir le chiffre d'affaire 
#---sales_df_clean['Chiffre Affaire'] = sales_df_clean['Quantity Ordered'] * sales_df_clean['Price Each']

sales_df_clean.head()

#une fois avoir la colonne du chiffre d'affaire, la comparer pour chaque mois (le total ou la somme qu'on prend en compte)
#--sales_df_clean.groupby('Mois').sum()['Chiffre Affaire'].sort_values(ascending = False).plot().bar #voir et avoir un appercu
#----methode 2 
#sales_df_clean.groupby('Mois')['Chiffre Affaire'].sum().sort_values(ascending = False).plot().bar


#ordonner les mois selon une liste mois 

order = ['January','February','March','April','May','June','July','August','September','October','November','December']


#faire un plot bar pour afficher le montant par mois et voir le plus importe
#utilisation de loc pour matcher les mois en ordre 
#sales_df_clean.groupby('Mois')['Chiffre Affaire'].sum().loc[order].plot().bar(figsize=(10, 8))

#confirmer et afficher la valeur de vente du mois en question 
#sales_df_clean.groupby('Mois')['Chiffre Affaire'].sum().loc[order].plot.bar(figsize = (10,8))
##plt.title('Chiffre Affaire Mensuel pour 2019')
#plt.savefig('venteMensuel.jpg')

#Dans quelle ville on a le plus vendus ? 

#Voir un peu toutes les adresses en mode unique
sales_df_clean['Purchase Address'].unique() #adresses uniques dans la data

#Extraire la ville du Purchase Adresse 
#tester et trouver l'index de la ville dans le Purchase Adresse et surppimer les espaces 
#creer une fonction get_ville qui prend en parametre une adresse et extrait l'index de la ville decouverte 

#def get_ville(adresse):
    #return adresse.split(',')[1].strip()

#check  si ca recupere l'index 1 qui est la ville 
#get_ville('225 5th St, Los Angeles, CA 90001')


#creer la colonne ville tout en appliquant la fonction get_ville sur notre Purchase Adresse du dataset 
#sales_df_clean['Ville'] = sales_df_clean['Purchase Address'].apply(get_ville)

#check
#sales_df_clean.head()
#cheker les valeurs de la colonne ville en mode unique 
#sales_df_clean['Ville'].unique()

#voir maintenant la vente recorde des villes 

#sales_df_clean.groupby('Ville')['Chiffre Affaire'].sum().sort_values(ascending = False)

#Faire un plot barh pour voir un peu le record des ventes par ville apres avoir grouper selon les 

#ca_ville = sales_df_clean.groupby('Ville')['Chiffre Affaire'].sum().sort_values(ascending = False)
#creer un groupe 
#ca_ville = sales_df_clean.groupby('Ville').sum()['Chiffre Affaire'].sort_values(ascending = False)

#ca_ville.plot(kind = 'barh', figsize=(10,8))
#plt.xlabel('Chiffre Affaire in Millions')
#plt.title('Chiffre Affaire Par ville')
#A quelle heure faire la publicit pour augmenter les ventes ? 
#demarches 
#analyser les ventes par heures pour faire le lien entre heure et ventes 
#creer ainsi les colonnes heures et time 
#sales_df_clean['Heure'] = sales_df_clean.index.hour
#sales_df_clean['Time'] = sales_df_clean.index.time

#sales_df_clean.head()

#regrouper les ventes par heures 
#sales_df_clean.groupby('Heure')['Chiffre Affaire'].sum().sort_values(ascending = False)

#les stocker et les afficher en les creant dans un dataframe Panda 

ca_heures = pd.DataFrame(sales_df_clean.groupby('Heure')['Chiffre Affaire'].sum())

ca_heures

#afficher une lineplot avec seaborn avec xticks ranger entre 0 et 24 pour signifier 0 et 24 heures 
#sns.lineplot(data = ca_heures['Chiffre Affaire'])
#plt.xticks(ticks = range(0,24))
#plt.grid()
#plt.show()
#plt.savefig('VenteparheureGrid.jpg')


#Quels sont les produits qui sont les plus achetés ensemble 
#les produits qui sont vendus ensemble ont le meme product id 
#checker les valeurs duplicated de la colonne Product ID avec l'option keep a False

sales_df_clean[sales_df_clean['Order ID'].duplicated(keep= False)]

#concatener le Product ID qui s'est repeté, le concater et l'associer avec chaque différent produit y référent 
#(exemple, pour un id 1000 on a un smartphone, pc, casques...)



#et stocker ces donnees dans une nouvelle variable pour ne pas alterer notre dataframe 


#faire une suite d'operations de groupage, de mise ensemble pour trouver les produits qui sont le plues vendus ensembe 



#trouver le produit le plus vendu
#grouper juste le produit selon la somme de la quantite ordonnee 







# In[ ]:




