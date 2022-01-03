
#-------------------------------------------------------------------------------------------------------------------------
# One hot encoder (avec récupération des labels)

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
import pandas as pd
import numpy as np

class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_<{self.categories_[i][j]}>')
                j += 1
        return new_columns
    
#-------------------------------------------------------------------------------------------------------------------------

# Targer Encoding ou One Hot Encoding (1 nouvelle colonne crée)
def encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]], columns=[column,fix_column])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = trained_model.transform(dataframe_work[column])
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe


# Label Encoding ou One Hot Encoding (1 nouvelle colonne crée)

def label_encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]], columns=[column,fix_column])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = dataframe_work[column].apply(lambda x: trained_model.transform([x])[0] if pd.notna(x) else np.NaN)
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe

# Targer Encoding ou One Hot Encoding (1 nouvelle colonne crée)
def target_encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]], columns=[column,fix_column])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = trained_model.transform(dataframe_work[column])
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe


# ONE-HOT-ENCODING (plusieurs nouvelles colonnes crées)
def vector_encoding_transform_with_merge(dataframe, column, fix_column, trained_model):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work_transformed = pd.DataFrame(trained_model.transform(dataframe_work))

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work_transformed.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work_transformed, on=fix_column)
    
    return dataframe













#----------------------------------------------------------------------------------------------

def SAVE_encoding_transform_with_merge(dataframe, column, fix_column, trained_model, column_new_name):
    """Fonction transfroam,nt une colonne de dataframe à partir d'un modèle fitté"""
    """renseigner le modèle fitté"""
    """Indiquer le nouveau nom de colonne"""
    """Indiquer une colonne fixe pour le merge"""
    # Création d'un dataframe avec la colonne souhaitée et une colonne repère en index pour éviter de perdre l'ordre après .transform (re-indexage possible de la fonction):
    dataframe_work = pd.DataFrame(dataframe[[column,fix_column]])
    dataframe_work.set_index([fix_column], inplace = True)

    # Transform
    dataframe_work[column_new_name] = trained_model.transform(dataframe_work[column])
    dataframe_work.drop(column, axis=1, inplace=True)

    # La colonne repère a été passée en index puis réapparaît après un reset index:
    dataframe_work.reset_index(inplace=True)

    # Merge avec colonne commune fix_column:
    dataframe = pd.merge(dataframe, dataframe_work, on=fix_column)
    
    return dataframe




