"""Fold-safe preprocessing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .schema import validate_feature_list

@dataclass(frozen=True)
class FeatureSpecification:
    numeric:tuple[str,...]; categorical:tuple[str,...]
    @property
    def all(self)->tuple[str,...]: return self.numeric+self.categorical
    def validate(self,frame:pd.DataFrame)->None:
        validate_feature_list(self.all); missing=[c for c in self.all if c not in frame.columns]
        if missing: raise ValueError(f"feature columns missing from data: {missing}")
        overlap=set(self.numeric).intersection(self.categorical)
        if overlap: raise ValueError(f"columns listed as numeric and categorical: {sorted(overlap)}")

def make_sklearn_preprocessor(features:FeatureSpecification,*,scale_numeric:bool,one_hot_min_frequency:int|float|None=10,sparse_output:bool=True)->ColumnTransformer:
    numeric_steps=[("impute",SimpleImputer(strategy="median",add_indicator=True))]
    if scale_numeric: numeric_steps.append(("scale",StandardScaler(with_mean=True)))
    numeric=Pipeline(numeric_steps)
    categorical=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=one_hot_min_frequency,sparse_output=sparse_output))])
    return ColumnTransformer([("numeric",numeric,list(features.numeric)),("categorical",categorical,list(features.categorical))],remainder="drop",verbose_feature_names_out=False)

def prepare_native_categoricals(train:pd.DataFrame,other:Iterable[pd.DataFrame],features:FeatureSpecification)->tuple[pd.DataFrame,list[pd.DataFrame]]:
    features.validate(train); train_out=train.loc[:,features.all].copy(); others=[f.loc[:,features.all].copy() for f in other]
    for c in features.categorical:
        cats=pd.Index(train_out[c].dropna().astype(str).unique()); train_out[c]=pd.Categorical(train_out[c].astype("string"),categories=cats)
        for f in others: f[c]=pd.Categorical(f[c].astype("string"),categories=cats)
    for c in features.numeric:
        train_out[c]=pd.to_numeric(train_out[c],errors="coerce")
        for f in others: f[c]=pd.to_numeric(f[c],errors="coerce")
    return train_out,others
