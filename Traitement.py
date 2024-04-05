from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chemin de la BDD
chemin = "C:/Users/MOUSTAPHA BOYE/PROJETS AVEC VSCODE/Desgagnes/BDD 2024"
# Nom du fichier excel
fichier = "BDD_2024.xlsx"
# Chemin complet
chemin_complet = f"{chemin}/{fichier}" 
# Chargement des données depuis un fichier Excel
df = pd.read_excel(chemin_complet, sheet_name='Feuil1')

df = df.dropna(how='all')
df = df.drop_duplicates()
df = df.dropna(subset = ['Ident_Type'])
df[df['Ident_Type'].isnull()]
#supprimer les données avec les entêtes
indexPavillon = df[df['Ident_Pavillon']=='Pavillon'].index
df.drop(indexPavillon, inplace=True)

#Client Desgagnes / Unité d'affaires
df['Ident_CustomerReference'] = df['Ident_CustomerReference'].fillna('Client_Inconnu')
#Position des Navire
df['Position_Port_Latitude'] = df['Position_Port_Latitude'].fillna('En Opération')
df[['Position_Section_Longitude','Position_Pays','Position_Code','Position_Province']] = df[[
    'Position_Section_Longitude','Position_Pays','Position_Code','Position_Province']].fillna('Missing')
#Les valeurs manquantes sous condition d'une latitude non nulle
df.loc[df['Position_Port_Latitude'].notna(),['Position_Pays','Position_Code','Position_Province'
       ]] = df.loc[df['Position_Port_Latitude'].notna(),['Position_Pays','Position_Code',
                  'Position_Province']].fillna('Missing')


#Imputations des valeurs manquantes
df.loc[df['Position_Port_Latitude'] == 'Cap Aux Meules', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','CMS','']
df.loc[df['Position_Port_Latitude'] == 'Cap Aux Meules', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','CMS','']
df.loc[df['Position_Port_Latitude'] == 'Cap Aux Meules', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','CMS','']
df.loc[df['Position_Port_Latitude'] == 'Montreal ', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','MTR','QC']
df.loc[df['Position_Port_Latitude'] == 'Montreal ', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','MTR','QC']
df.loc[df['Position_Port_Latitude'] == 'Montreal ', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','MTR','QC']
df.loc[df['Position_Port_Latitude'] == 'Montreal ', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','MTR','QC']
df.loc[df['Position_Port_Latitude'] == 'Montreal 97', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','MTR','QC']
df.loc[df['Position_Port_Latitude'] == 'Montreal Norcan 74', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','MTR','QC']
df.loc[df['Position_Port_Latitude'] == 'Montreal Suncor 109', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','MTR','QC']
df.loc[df['Position_Port_Latitude'] == 'Tracy anchorage ', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','ST6','QC']
df.loc[df['Position_Port_Latitude'] == 'Tracy Kildair Dock', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','ST6','QC']
df.loc[df['Position_Port_Latitude'] == 'Trios Rivieres', ['Position_Pays', 'Position_Code','Position_Province']] = ['CA','TRR','QC']
#Traitement de la distance priorisée
df['Distance_Priorisee'] = df['Distance_Priorisee'].fillna(0)
#Traitement des consommations
df['ConsommationHFO_Bouilloire'] = df['ConsommationHFO_Bouilloire'].fillna(0)
df['ConsommationHFO_Chauffagecargaison'] = df['ConsommationHFO_Chauffagecargaison'].fillna(0)
df['ConsommationHFO_EnginPrincipal'] = df['ConsommationHFO_EnginPrincipal'].fillna(0)
df['ConsommationHFO_EnginAuxiliaire'] = df['ConsommationHFO_EnginAuxiliaire'].fillna(0)
df['ConsommationHFO_Cargopumps'] = df['ConsommationHFO_Cargopumps'].fillna(0)
df['ConsommationHFO_Systemegazinterne'] = df['ConsommationHFO_Systemegazinterne'].fillna(0)
df['ConsommationMDO_Bouilloire'] = df['ConsommationMDO_Bouilloire'].fillna(0)
df['ConsommationMDO_Chauffagecargaison'] = df['ConsommationMDO_Chauffagecargaison'].fillna(0)
df['ConsommationMDO_EnginPrincipal'] = df['ConsommationMDO_EnginPrincipal'].fillna(0)
df['ConsommationMDO_Cargopumps'] = df['ConsommationMDO_Cargopumps'].fillna(0)
df['ConsommationMDO_Systemegazinterne'] = df['ConsommationMDO_Systemegazinterne'].fillna(0)
df['ConsommationLNG_Bouilloire'] = df['ConsommationLNG_Bouilloire'].fillna(0)
df['ConsommationLNG_Chauffagecargaison'] = df['ConsommationLNG_Chauffagecargaison'].fillna(0)
df['ConsommationLNG_EnginPrincipal'] = df['ConsommationLNG_EnginPrincipal'].fillna(0)
df['ConsommationLNG_EnginAuxiliaire'] = df['ConsommationLNG_EnginAuxiliaire'].fillna(0)
df['ConsommationLNG_Cargopumps'] = df['ConsommationLNG_Cargopumps'].fillna(0)
df['ConsommationLNG_Systemegazinterne'] = df['ConsommationLNG_Systemegazinterne'].fillna(0)
df['Bunkering_Quantite'] = df['Bunkering_Quantite'].fillna(0)


############# TRAITEMENT DE LA DATE ET HEURE ###############
#Filtre des Evenements
df = df[df['Ident_Evenement'].isin(['Arrival at port or position','Departure from port or position'])]

df['Distance_DateheureQuebec'] = pd.to_datetime(df['Distance_DateheureQuebec'])
#Trier les données
df.sort_values(by=(['Ident_Navire','Distance_DateheureQuebec']))
#Création d'une colonne temporaire pour les Evenements Arrivée et Départ
#df['Temp_Event'] = df['Ident_Evenement'].where(df['Ident_Evenement'].isin(['Arrival at port or position','Departure from port or position']))
#Calculer la différence de temps
df['DiffTemps'] =  df['Distance_DateheureQuebec'] - df.groupby('Ident_Navire')['Distance_DateheureQuebec'].shift(1)
# Convertir la différence de temps en une unité de mesure souhaitée, par exemple en minutes
df['DiffTempsHeure'] = df['DiffTemps'].dt.total_seconds() / 3600
# Créer une colonne pour vérifier si la ligne actuelle et la précédente appartiennent au même navire
df['MemeNavire'] = df['Ident_Navire'] == df['Ident_Navire'].shift(1)
# Mettre à NaN la différence de temps lorsqu'il ne s'agit pas du même navire
df.loc[~df['MemeNavire'], 'DiffTempsHeure'] = np.nan
df.to_excel('C:/Users/MOUSTAPHA BOYE/PROJETS AVEC VSCODE/Git_Desgagnes/Sorties/DiffTemps.xlsx', index=False, engine='openpyxl')

colonnes = ['Ident_Pavillon','Ident_Type','Ident_Navire','Ident_Evenement','Distance_Duree_h',
            'Distance_Priorisee','Vitessecalcule','ConsommationHFO_Bouilloire',
            'ConsommationHFO_Chauffagecargaison','ConsommationHFO_EnginPrincipal','Bunkering_Quantite',
            'ConsommationHFO_EnginAuxiliaire','ConsommationHFO_Cargopumps','ConsommationHFO_Systemegazinterne',
            'ConsommationMDO_Bouilloire','ConsommationMDO_Chauffagecargaison','ConsommationMDO_EnginPrincipal',
            'ConsommationMDO_EnginAuxiliaire','ConsommationMDO_Cargopumps','ConsommationMDO_Systemegazinterne',
            'ConsommationLNG_Bouilloire','ConsommationLNG_Chauffagecargaison','ConsommationLNG_EnginPrincipal',
            'ConsommationLNG_EnginAuxiliaire','ConsommationLNG_Cargopumps','ConsommationLNG_Systemegazinterne'
]
df_reg = df[colonnes]
df_reg['Consommation_tot'] = df_reg[['ConsommationHFO_Bouilloire','ConsommationHFO_Chauffagecargaison',
            'ConsommationHFO_EnginPrincipal','ConsommationLNG_Systemegazinterne','ConsommationHFO_EnginAuxiliaire',
            'ConsommationHFO_Cargopumps','ConsommationHFO_Systemegazinterne','ConsommationMDO_Bouilloire',
            'ConsommationMDO_Chauffagecargaison','ConsommationMDO_EnginPrincipal','ConsommationMDO_EnginAuxiliaire','ConsommationMDO_Cargopumps','ConsommationMDO_Systemegazinterne',
            'ConsommationLNG_Bouilloire','ConsommationLNG_Chauffagecargaison','ConsommationLNG_EnginPrincipal',
            'ConsommationLNG_EnginAuxiliaire','ConsommationLNG_Cargopumps']].sum(axis=1)


# Initialisation des variables nécessaires
target_variable = 'Consommation_tot'

# Suppression sécurisée des colonnes : l'argument errors='ignore' permet d'ignorer les colonnes non trouvées
columns_to_drop = [target_variable,'ConsommationHFO_Bouilloire','ConsommationHFO_Chauffagecargaison',
            'ConsommationHFO_EnginPrincipal','ConsommationLNG_Systemegazinterne','ConsommationHFO_EnginAuxiliaire',
            'ConsommationHFO_Cargopumps','ConsommationHFO_Systemegazinterne','ConsommationMDO_Bouilloire',
            'ConsommationMDO_Chauffagecargaison','ConsommationMDO_EnginPrincipal','ConsommationMDO_EnginAuxiliaire','ConsommationMDO_Cargopumps','ConsommationMDO_Systemegazinterne',
            'ConsommationLNG_Bouilloire','ConsommationLNG_Chauffagecargaison','ConsommationLNG_EnginPrincipal',
            'ConsommationLNG_EnginAuxiliaire','ConsommationLNG_Cargopumps']
X = df_reg.drop(columns=columns_to_drop, errors='ignore')
y = df_reg[target_variable]

# Sélection des types de colonnes pour la transformation, en excluant les datetime
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Séparation des données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuration du pipeline de prétraitement et de modélisation
# Mise à jour du préprocesseur pour inclure SimpleImputer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())]), categorical_features)
    ]
)

models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), SVR()]

best_model = None
best_r2_adjusted = -np.inf
best_model_type = None

# Boucle sur les différents modèles
for model in models:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    r2_adjusted = 1 - ((1-r2) * (n-1) / (n-p-1))
    
    if r2_adjusted > best_r2_adjusted:
        best_model = pipeline
        best_r2_adjusted = r2_adjusted
        best_model_type = type(model).__name__

if best_model:
    print(f"Meilleur modèle trouvé: {best_model_type} avec un R² ajusté de {best_r2_adjusted:.4f}")
        # Vérification pour les modèles linéaires avec 'coef_'
    if hasattr(best_model.named_steps['model'], 'coef_'):
        coefficients = best_model.named_steps['model'].coef_
        
        # Récupération des noms de fonctionnalités après l'encodage OneHot
        # Nous appelons get_feature_names_out directement sur le ColumnTransformer après le pipeline a été ajusté
        feature_names_out = best_model.named_steps['preprocessor'].get_feature_names_out()
        
        # Préparation des noms de fonctionnalités finales en incluant les variables numériques et catégorielles
        feature_names = feature_names_out
        
        # Création du DataFrame pour les coefficients avec les noms de fonctionnalités corrects
        coeff_df = pd.DataFrame(coefficients.flatten(), index=feature_names, columns=['Coefficient'])
        
        # Exportation du DataFrame dans un fichier Excel
        coeff_df.to_excel('C:/Users/MOUSTAPHA BOYE/PROJETS AVEC VSCODE/Coefficient_Regression_Best_Model.xlsx', sheet_name='Coefficients')
        #df_final.to_excel('C:/Users/MOUSTAPHA BOYE/PROJETS AVEC VSCODE/BaseRegression.xlsx', sheet_name='Base')
else:
    print("Aucun modèle trouvé avec un R² ajusté supérieur à 95%")


