import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import KroghInterpolator, Akima1DInterpolator, BarycentricInterpolator
from sklearn import impute
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_iterative_imputer

assert 'DataFrame' in dir(pd)
assert len(dir(enable_iterative_imputer))
assert 'IterativeImputer' in dir(impute)
from sklearn.impute import IterativeImputer


def minmaxscale(x):
    x2 = x + np.abs(np.min(x))
    x2 /= np.max(x2)
    return x2


class BarycentricImputer(TransformerMixin):
    def fit(self, df2, basis, target):
        self.basis = basis
        if isinstance(target, str):
            target = [target]
        self.target = target
        self.aki = {}
        for col in self.target:
            df2sorted = df2.sort_values(basis).dropna(subset=[basis, col])
            self.aki[col] = BarycentricInterpolator(
                xi=df2sorted[basis].values,
                yi=df2sorted[col].values
            )

    def transform(self, df2):
        for col in self.target:
            df2[f'{col}_bary'] = self.aki[col](df2[self.basis].values)
        return df2


class AkimaImputer(TransformerMixin):
    def fit(self, df2, basis, target):
        self.basis = basis
        if isinstance(target, str):
            target = [target]
        self.target = target
        self.aki = {}
        for col in self.target:
            df2sorted = df2.sort_values(basis).dropna(subset=[basis, col])
            self.aki[col] = Akima1DInterpolator(
                x=df2sorted[basis].values,
                y=df2sorted[col].values
            )

    def transform(self, df2):
        for col in self.target:
            df2[f'{col}_akima'] = self.aki[col](df2[self.basis].values)
        return df2


class KroghImputer(TransformerMixin):
    def __init__(self):
        super().__init__()
        self.basis = None
        self.target = None
        self.ki = None

    def fit(self, df2, basis, target):
        self.basis = basis
        if isinstance(target, str):
            target = [target]
        self.target = target
        self.ki = {}
        for col in self.target:
            df2sorted = df2.sort_values(basis).dropna(subset=[basis, col])
            self.ki[col] = KroghInterpolator(
                xi=df2sorted[basis].values,
                yi=df2sorted[col].values
            )

    def transform(self, df2):
        for col in self.target:
            df2[f'{col}_krogh'] = self.ki[col](df2[self.basis].values)
        return df2


@pytest.fixture(name="di_df")
def di_fixture():
    from scipy.interpolate import Rbf
    x, y, z, d = np.random.rand(4, 50)
    rbfi = Rbf(x, y, z, d, function='gaussian')  # radial basis function interpolator instance
    xi = yi = zi = np.linspace(0, 1, 20)
    di = rbfi(xi, yi, zi)  # interpolated values
    di2 = di.copy()
    di2[3:12] = np.nan
    df2 = pd.DataFrame(dict(xi=xi, yi=yi, zi=zi, di=di, di2=di2))
    return df2


def test_krogh_imputer(di_df):
    """
    test for KroghImputer
    Parameters
    ----------
    di_df

    Returns
    -------

    """
    kit = KroghImputer()
    kit.fit(di_df, basis='di', target=['di2'])
    result = kit.transform(di_df)
    assert result.shape == (20, 6)


def test_akima_imputer(di_df):
    """
    test for AkimaImputer
    Parameters
    ----------
    di_df

    Returns
    -------

    """
    akit = AkimaImputer()
    akit.fit(di_df, basis='di', target=['di2'])
    result = akit.transform(di_df)
    assert result.shape == (20, 6)


def test_barycentric_imputer(di_df):
    """
    test for BarycentricImputer
    Parameters
    ----------
    di_df

    Returns
    -------

    """
    byit = BarycentricImputer()
    byit.fit(di_df, basis='di', target=['di2'])
    result = byit.transform(di_df)
    assert result.shape == (20, 6)


def test_iterative_imputer(di_df):
    """
    test for IterativeImputer
    Parameters
    ----------
    di_df

    Returns
    -------

    """
    imputer = IterativeImputer()
    imputer.fit(di_df[['di', 'di2']])
    result = di_df.copy()
    result['di2'] = imputer.transform(di_df[['di', 'di2']])[:, 1]
    assert result.shape == (20, 5)
