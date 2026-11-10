import pandas as pd
from sklearn.impute import SimpleImputer

def split_features_target(df: pd.DataFrame, target_col: str = 'MEDV'):
    #x - dane wejściowe
    X = df.drop(columns=[target_col])
    #y - zmienna docelowa
    y = df[target_col]
    return X, y


def impute_missing_values(X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    #uzupelnienie braków medianą
    imputer = SimpleImputer(strategy=strategy)
    X_imputed_array = imputer.fit_transform(X)
    
    #przywrócenie oryginalnych indeksów
    X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns, index=X.index)
    return X_imputed