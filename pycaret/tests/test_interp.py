import pytest
from scipy.interpolate import KroghInterpolator
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from scipy.interpolate import KroghInterpolator, Akima1DInterpolator


def minmaxscale(x):
    x2 = x + np.abs(np.min(x))
    x2 /= np.max(x2)
    return x2


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
            df2[f'{col}_interp'] = self.aki[col](df2[self.basis].values)
        return df2


class KroghImputer(TransformerMixin):
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
            df2[f'{col}_interp'] = self.ki[col](df2[self.basis].values)
        return df2


@pytest.fixture(name="di_df")
def di_fixture():
    import numpy as np
    import pandas as pd
    from scipy.interpolate import Rbf
    x, y, z, d = np.random.rand(4, 50)
    rbfi = Rbf(x, y, z, d, function='gaussian')  # radial basis function interpolator instance
    xi = yi = zi = np.linspace(0, 1, 20)
    di = rbfi(xi, yi, zi)  # interpolated values
    di2 = di.copy()
    di2[3:12] = np.nan
    df2 = pd.DataFrame(dict(xi=xi, yi=yi, zi=zi, di=di, di2=di2))
    return df2


def test_KroghImputer(di_df):
    kit = KroghImputer()
    kit.fit(di_df, basis='di', target=['di2'])
    result = kit.transform(di_df)
    assert result.shape == (20, 6)


def test_AkimaImputer(di_df):
    akit = AkimaImputer()
    akit.fit(di_df, basis='di', target=['di2'])
    result = akit.transform(di_df)
    assert result.shape == (20, 6)
