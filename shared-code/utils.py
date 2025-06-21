import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error,make_scorer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ID_COL='Id'
TARGET_COL='SalePrice'
INDEX_COL='Index'
Kfolds=6 # Za da imat 20% vo validation datasetot 


cols = np.array(
        [
            ("HouseStyle", "ForFamilyType"),
            ("BldgType", "Stories"),
            ("BsmtFinType1", "Bsmt1Type"),
            ("BsmtFinSF1", "Bsmt1Sf"),
            ("BsmtFinType2", "Bsmt2Type"),
            ("BsmtFinSF2", "Bsmt2Sf"),
        ]
    )


def change_cols_names_return_new(df:pd.DataFrame,old_new_pairs:np.ndarray[tuple[str,str]]):
    tmp=df.columns.values 
    indx_oldcols=np.where(np.isin(tmp,old_new_pairs[:,0]))[0]
    for i,indx in enumerate(indx_oldcols):
        tmp[indx]=old_new_pairs[i][1]
    return tmp


def modify_features(df: pd.DataFrame):
    df1 = df.copy(deep=True)
    df1.loc[:, "AgeFromLastRemodelingPriSale"] = df1["YrSold"] - df1["YearRemodAdd"]
    df1.drop(["YearBuilt", "YearRemodAdd"], axis=1, inplace=True)
    
    NEW_COL = "GarageAgePriSale"

    df1.loc[:, NEW_COL] = df1["YrSold"] - df1["GarageYrBlt"]
    df1.drop(["GarageYrBlt", "YrSold"], axis=1, inplace=True)

    new_cols = change_cols_names_return_new(df1, cols)
    df1.columns = new_cols
    
    df1 = df1.rename(str, axis="columns")
    
    return df1

class PdWrapper:
    def __init__(self, imp):
        self.imp = imp

    def __call__(self, X: pd.DataFrame):
        return make_pd_from_col_trans(X, self.imp)

def make_pd_from_col_trans(data, trans):
    features = None
    if isinstance(trans, ColumnTransformer): 
        features = [s.split('__')[1] for s in trans.get_feature_names_out()]
    else:
        raise NotImplementedError("Zaborev sho beshe tuka")
    return pd.DataFrame(data=data, columns=features)

def conv_npfloat64(df:pd.DataFrame): 
    return df.astype(np.float64)

def apply_log(x):
    return 20 if x <= 0 else np.log(x)


np_apply_log = np.vectorize(apply_log)


def rmse_logs(y_true, y_pred):
    y_true_tmp = np_apply_log(y_true)
    y_pred_tmp = np_apply_log(y_pred)

    return root_mean_squared_error(y_true_tmp, y_pred_tmp)


house_pricing_metric = make_scorer(rmse_logs)