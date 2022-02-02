import pandas as pd
import numpy as np
from numpy.random import default_rng
import scipy.stats as stats
import seaborn as sb
from scipy.stats import chi2
from sklearn.model_selection import GroupShuffleSplit
from numpy.random import SeedSequence
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

def test_p_eff(k = 5, chi_param = 1, tests =2000, pass_frac = 0.95, fail = 1.2, rng = default_rng(42)):
    a = rng.chisquare(chi_param, tests)
    crit = chi2.ppf(pass_frac, chi_param)
    scaling = crit/fail
    a_scaled = a/scaling
    temp_pools = []
    gaps = np.arange(0,len(a_scaled),k)
    for i in gaps:
        b = np.mean(a_scaled[i:i+k])
        temp_pools.append(b)

    count = 0
    for i in temp_pools:
        if i >= fail/k:
            count += 1
    total_tests = count * k + len(temp_pools)
    # print("Pools failing the pooled test criteria: {}".format(count))
    # print("Tests needed in the pool stage: {}".format(len(temp_pools)))
    # print("Total tests needed: {}".format(total_tests))
    # print("Total tests needed for individual tests: {}".format(len(a_scaled)))
    # print("Pooled testing requirement compared to individual tests: {:.2%}".format(total_tests/len(a_scaled)))
    # print("ran 1")
    return total_tests/len(a_scaled)

def wrapper_calc(arg_dict):
    kwargs = arg_dict
    # print(type(kwargs))
    # print(kwargs)
    return test_p_eff(**kwargs)

if __name__ == "__main__":
    kk_val = [2,3,4,5,6,7,8,9,10]
    per_fail=[.75, .8, .85, .9, .95, .975, .99, .999]
    test_matrix = np.zeros((len(kk_val),len(per_fail)))
    # print(test_matrix)
    # print(test_matrix.shape)
    tt= []
    begin = time.perf_counter()
    with Pool(processes=8) as pool:
        for c,v in enumerate(per_fail):
            for cc,vv in enumerate(kk_val):
                no_tests = 4000
                tests = np.empty((no_tests, 1), float)
    
                tests[:] = np.nan
                # print(tests)
                alpha = SeedSequence().entropy
                args = np.arange(no_tests) + alpha
                dic_list = []
                base_dic = {"k":vv, "pass_frac":v}
                for i in args:
                    # print(i)
                    base_dic["rng"] = default_rng(i)
                    dic_list.append(base_dic.copy())
    
                tests = pool.map(wrapper_calc, dic_list)
                # for i in np.arange(no_tests):
                #     # print(i)
                #     b = i + 50
                #     # print(b)
                #     blank = pool.apply_async(test_p_eff, kwds={"k":vv, "pass_frac":v, "rng":default_rng(alpha + b)})
                    
                t_mean =np.mean(tests)
                print("Average pooled tests required: {:.2%} for a pool size of {}, pass rate of: {}".format(t_mean, vv, v))
                test_matrix[c-1,cc-1]=t_mean
                tt.append([v,vv,t_mean])
    end = time.perf_counter()
    print("That run took {} seconds".format(end-begin))