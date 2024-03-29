
import numpy as np
from numpy.random import default_rng
import matrix3 as gen
import pandas as pd

from scipy.stats import chi2
from numpy.random import SeedSequence

from multiprocessing import Pool
import time

def de_conv_pool(tests, fail, batch):
    test_count = 0
    for i in range(len(tests)):
        # print("val of i:", i)
        # print("val of batch-i:", batch - i)
        # print("remaining samples:", tests[i:])
        # print("mean of remaining samples:", np.mean(tests[i:]))
        # print("sub group fail level: ", fail/(batch-i), "\n")
        if np.mean(tests[i:]) >= fail/(batch - i):
            test_count = test_count + 1
    return test_count


def booter(sample_count: int, dist: np.array) -> np.array:
    boot_gen = np.random.default_rng()
    sample = np.empty(sample_count)
    for i in np.arange(0, sample_count):
        ind = boot_gen.integers(0, len(dist)-1,size=1)
        sample[i] = dist[ind].copy()
    return sample


def test_p_conv_eff(k = 5, dist_parm = (1,1,1), chi_param = 1, tests =2000, pass_frac = 0.95, fail = 1.2, loq=0.3, loq_scale = 50, rng = default_rng(42)):
    if fail/k < loq/loq_scale: #  evaluate if batch size is too big for limit of detection
        print("inapropriate batchsize for fail conc : LoQ")
        return np.nan
    # a = rng.chisquare(chi_param, tests) # build chisquare probability distribution
    # crit = chi2.ppf(pass_frac, chi_param) # identify the x value corresponding to the desired pass rate
    # scaling = crit/fail
    # a_scaled = a/scaling # rescale the sample distribution so that the critical value == fail value

    # a_scaled = gen.gen_sample(sample_count=tests, pool=k, seed=rng, dist_parm = dist_parm)
    cali = pd.read_csv("cali_data.csv")
    a_scaled = booter(tests, cali.cadmium)
 
    gaps = np.arange(0,len(a_scaled),k) # build the pools
    # sub_tests = 0 #counter for individual tests
    # fails = [x for x in a_scaled if x >=1.2]
    # print("number of fails: ", len(fails))
    # print("the fails: ", fails)
    

    test_count = len(gaps)
    for i in gaps:
        b = np.mean(a_scaled[i:i+k])
        # print(k==len(a_scaled[i:i+k]))
             
        if b >= fail/k:
            test_count = test_count + de_conv_pool(a_scaled[i:i+k], fail, len(a_scaled[i:i+k]))
             
    


    # count = 0
    # for i in temp_pools: #count the number of pools that fail
    #     if i >= fail/k:
    #         count += 1
    # total_tests = count * k + len(temp_pools) # count the total number of tests
    # print("Pools failing the pooled test criteria: {}".format(count))
    # print("Tests needed in the pool stage: {}".format(len(temp_pools)))
    # print("Total tests needed: {}".format(total_tests))
    # print("Total tests needed for individual tests: {}".format(len(a_scaled)))
    # print("Pooled testing requirement compared to individual tests: {:.2%}".format(total_tests/len(a_scaled)))
    return test_count/len(a_scaled)


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
    # return test_p_eff(**kwargs)
    return test_p_conv_eff(**kwargs)

if __name__ == "__main__":
    kk_val = [2,3,4,5,6,7,8,9,10,11,12,13,14]
    # per_fail=[.5, .75, .8, .85, .9, .95, .975, .99, .999]
    # kk_val = [3,4]
    per_fail=[.9]
    # test_matrix = np.zeros((len(kk_val),len(per_fail)))
    # print(test_matrix)
    # print(test_matrix.shape)

    begin = time.perf_counter()
    runs = [({"dist_parm":(1.5233660418119728, 0.0027288920577292503, 0.06939280546211554), "fail": 0.5}, "pb_single.csv"),
            ({"dist_parm":(1.1504315889666699, -0.0020480895355545316, 0.06710884489857513), "fail": 0.2}, "as_single.csv"),
            ({"dist_parm":(1.0858164361757576, 0.00019808453407679164, 0.036053458447836084), "fail": 0.2}, "cd_single.csv"),
            ({"dist_parm":(0.7502321546956426, -0.000961540716204136, 0.007837295888340656), "fail": 0.1}, "hg_single.csv")]
    for r  in runs:
        tt =[]
        with Pool(processes=8) as pool:
            for c,v in enumerate(per_fail):
                for cc,vv in enumerate(kk_val):
                    no_tests = 100
                    tests = np.empty((no_tests, 1), float)
        
                    tests[:] = np.nan
                    # print(tests)
                    alpha = SeedSequence().entropy
                    args = np.arange(no_tests) + alpha
                    dic_list = []
                    base_dic = {"k":vv, "pass_frac":v, "tests":2000}
                    base_dic.update(r[0])
                    for i in args:
                        # print(i)
                        base_dic["rng"] = default_rng(i) # production version
                        # base_dic["rng"] = default_rng(42) # testing version
                        dic_list.append(base_dic.copy())
        
                    tests = pool.map(wrapper_calc, dic_list)
                    # for i in np.arange(no_tests):
                    #     # print(i)
                    #     b = i + 50
                    #     # print(b)
                    #     blank = pool.apply_async(test_p_eff, kwds={"k":vv, "pass_frac":v, "rng":default_rng(alpha + b)})
                        
                    t_mean =np.mean(tests)
                    print("Average pooled tests required: {:.2%} for a pool size of {}, pass rate of: {}".format(t_mean, vv, v))
                    # test_matrix[c-1,cc-1]=t_mean
                    tt.append([v,vv,t_mean])
        end = time.perf_counter()
        print("That run took {} seconds".format(end-begin))
        print(tt)
        out = pd.DataFrame(tt, columns=["ignore", "pool", "tests"])
        out.to_csv(r[1])