import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numpy.random import default_rng
from numpy.random import SeedSequence
from fitter import Fitter
DOI = ["loglaplace", "mielke", "kappa3", "burr", "burr12", "lognorm", "gamma", "expon", "cauchy", "exponpow"]
data = pd.read_excel("metal_tests.xlsx", na_values = "ND")
# data.info()
data.to_csv("data.csv")
flower = data.loc[(data.accessioning_type == "Flower") | (data.accessioning_type == "Leaf/Mixed Plant Material"),:].copy()
# flower.info()
floc = flower.replace("ND", np.nan)
# floc.info()
arse = floc.arsenic_ug_g.dropna().values
sarse = np.sort(arse)
fa = Fitter(sarse[5:-5], distributions=DOI, timeout=60)
fa.fit()
print(fa.summary())