import numpy as np
import seaborn as sb
from scipy import stats
import matplotlib.pyplot as plt



DIST_CONTINU = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]


for i in DIST_CONTINU:
    print(i)
    class_method = getattr(stats, i)
    a = class_method.rvs(a =1, size = 1000, random_state = 42)
    sb.histplot(a, bins = 30, kde= True)
    plt.show()