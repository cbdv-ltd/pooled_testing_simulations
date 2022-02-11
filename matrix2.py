from operator import not_
import numpy as np
from numpy.random import default_rng
from scipy.stats import chi2
from numpy.random import SeedSequence
from multiprocessing import Pool
import time
from scipy import stats
import pandas as pd
import check_matrix


DIST_CONTINU = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]


def get_aves(matrix, fail):
    x, y = matrix.shape
    xa = {}
    ya = {}

    for i in range(x):
        if (x-np.sum(np.isnan(matrix[i,:]))) == 0:
            xa[i] = 0        
        elif np.nanmean(matrix[i,:]) < fail/(x-np.sum(np.isnan(matrix[i,:]))):
            xa[i] = 0
        else:
            xa[i] = np.nanmean(matrix[i,:])

        if (x-np.sum(np.isnan(matrix[:,i]))) == 0:
            ya[i] = 0        
        elif np.nanmean(matrix[:,i]) < fail/(x-np.sum(np.isnan(matrix[:,i]))):
            ya[i] = 0
        else:
            ya[i] = np.nanmean(matrix[:,i])
    
    return xa, ya


def fail_test(matrix, fail, debug = 0):
    # print("debug status: ", debug)
    failure = 0
    x, y = get_aves(matrix, fail)
    xc = np.sum(np.isfinite(matrix), axis=1) # may have a bug here needs to be checked
    yc = np.sum(np.isfinite(matrix), axis=0)
    if debug == 1:
        print(matrix)
        print(xc)
        print(yc)
        print(x)
        print(y)
    # matrix[1,2:] = np.nan
    # xc = np.sum(np.isfinite(matrix), axis=0)
    # yc = np.sum(np.isfinite(matrix), axis=1)
    # print(matrix)
    # print(xc)
    # print(yc)
    if debug == 1:
        print("debug on")
        print(fail)
        print("fail y average adjusted: ",fail/yc)
        print("y averages", np.array([*y.values()]))
        for i in range(matrix.shape[0]):
            print("{}th, matrix in 0 dim {}".format(i, matrix[i,:]))
        print("failure matrix",np.array([*y.values()]) >= fail/yc)
        print("sum: ", np.sum(np.array([*y.values()]) >= fail/yc))
    y_fail = False
    x_fail = False
    for i in range(matrix.shape[0]):
        if (y[i] > 0) and (y[i] >= fail/np.sum(np.isfinite(matrix[:,i]))):
            y_fail += True
        if (x[i] > 0) and (x[i] >= fail/np.sum(np.isfinite(matrix[i,:]))):
            x_fail += True

    if y_fail > 0:
        # print("fail on y")
        failure += 1
    if x_fail > 0:
        # print("fail on x")
        failure += 1
    # print("failure state: ", failure)
    return failure




def guess_sample(matrix, fail, tests = 0, ool = 0):
    # print("guessing loop test value: ", tests)
    avx, avy = get_aves(matrix, fail)
    
    if fail_test(matrix, fail) < 2:
        # print("guessing loop returning test value: ", tests)
        return tests, ool
    avxi = max(avx, key = avx.get)
    avyi = max(avy, key = avy.get)
    not_null = np.isnan(matrix[avxi, avyi])
    if avx[avxi] > avy[avyi]:
        count = matrix.shape[0]
        while not_null == True:
            # print("initial guess y: ", avyi)
            avy[avyi] = 0
            avyi = max(avy, key = avy.get)
            not_null = np.isnan(matrix[avxi, avyi])
            # print("reguessed y", avyi)
            # print("y dictionary: ", avy)
            count = count - 1
            if count == -1:
                return tests, ool
                # raise ValueError("cannot find a Y value for biggest X ave \n{}".format(matrix))
        tests += 1
        if matrix[avxi, avyi] >= fail:
            ool += 1
        matrix[avxi, avyi] = np.nan
        
        
    else:
        count  = matrix.shape[0]
        while not_null == True:
            # print("initial guess x", avxi)
            avx[avxi] = 0
            avxi = max(avx, key = avx.get)
            not_null = np.isnan(matrix[avxi, avyi])
            # print("reguessed x", avxi)
            # print("x dictionary: ", avx)
            count = count - 1
            if count == -1:
                return tests, ool
                # raise ValueError("cannot find a x value for biggest Y ave \n{}".format(matrix))
        tests += 1
        if matrix[avxi, avyi] >= fail:
            ool += 1
        matrix[avxi, avyi] = np.nan
        

    # print("bottom return test count: ", tests)
    # print(matrix)
    tests, ool = guess_sample(matrix, fail, tests, ool)
    return tests, ool
    
    # print("Diagnostics:\n")
    # print("failure test: ", fail_test(matrix, fail, 1))
    # print(matrix)
    # print("x averages: ", avx)
    # print("y averages: ", avy)
    # print("test count: ", tests)
    # print("fail value: ", fail)
    # raise ValueError("Not sure how we got here")
    
    

def matrix2(k = 5, chi_param = 1, tests =2000, pass_frac = 0.95, fail = 1.2, loq=0.3, loq_scale = 2, rng = 42):
    sample = gen_sample(sample_count=tests, fail=fail, pass_rate=pass_frac, pool=k*k, seed=rng)
    matrix = np.resize(sample, (len(sample)//(k*k),k,k))
    test_count = 0
    total_fail = 0
    z, x, y = matrix.shape
    for i in range(z):
        t , f = check_matrix.guess(matrix[i,:,:], fail)
        test_count += t
        total_fail += f
            

    # print("Test count for this sample: ", test_count)
    # print("Failed samples found: ", total_fail)
    # print("Actual number of failed samples: ", len([x for x in sample if x>=fail])
    return test_count/len(sample)


def gen_sample(sample_count: int=2000, fail: float=1.2, distribution: str="chi2", pass_rate: float=0.95,  pool: int=8, seed: int=42, freedom: float=1):
    # sample count is the minimum sample count, sample_count < returned sample size <= sample_size + pool    
    if distribution not in DIST_CONTINU:
        raise ValueError("{} is not a continuous distribution".format(distribution))
    rng = default_rng(seed)
    class_method = getattr(stats, distribution)    
    a = class_method.rvs(df = freedom, size = sample_count + pool -1, random_state = rng)
    a_scaled = a/(class_method.ppf(q = pass_rate, df =freedom)/fail)
    c = (len(a_scaled) // pool) * pool # find ideal sample count (all pool sizes equal, no residual)
    # print("Sample size stats, count: {}, pool: {}, ideal pool: {}".format(len(a_scaled), pool, c))
    # print(b == len(a_scaled[:b]))
    return a_scaled[:c]

def wrapper_calc(arg_dict):
    kwargs = arg_dict
    # print(type(kwargs))
    # print(kwargs)
    # return test_p_eff(**kwargs)
    return matrix2(**kwargs)

if __name__ == "__main__":
    kk_val = [7,8,9,10,11,12]
    # per_fail=[.5, .75, .8, .85, .9, .95, .975, .99, .996, .999]
    # kk_val = [5]
    per_fail=[.996]
    tt= []
    begin = time.perf_counter()
    sample_count = 2000
    no_tests = 400
    for c,v in enumerate(per_fail):
        for cc,vv in enumerate(kk_val):
            tests = np.empty((no_tests, 1), float)
            tests[:] = np.nan
            alpha = SeedSequence().entropy
            args = np.arange(no_tests) + alpha
            dic_list = []
            base_dic = {"k":vv, "pass_frac":v, "tests":sample_count}
            for i in args:
                # print(i)
                base_dic["rng"] = i # production version
                # base_dic["rng"] = 42 # testing version
                dic_list.append(base_dic.copy())
            for ccc,vvv in enumerate(dic_list):
                tests[ccc] = wrapper_calc(vvv)
            t_mean = np.mean(tests)
            print("Average pooled tests required: {:.2%} for a pool size of {}, sample size of {} pass rate of: {}, over {} test runs.\n----------------".format(t_mean, vv, sample_count, v, no_tests))
            tt.append([v,vv,t_mean])

    # with Pool(processes=8) as pool:
    #     for c,v in enumerate(per_fail):
    #         for cc,vv in enumerate(kk_val):
                
    #             tests = np.empty((no_tests, 1), float)
    #             tests[:] = np.nan
    #             alpha = SeedSequence().entropy
    #             args = np.arange(no_tests) + alpha
    #             dic_list = []
    #             base_dic = {"k":vv, "pass_frac":v, "tests":sample_count}
    #             for i in args:
    #                 # print(i)
    #                 # base_dic["rng"] = i # production version
    #                 base_dic["rng"] = 42 # testing version
    #                 dic_list.append(base_dic.copy())
    #             tests = pool.map(wrapper_calc, dic_list)
    #             t_mean = np.mean(tests)
    #             print("Average pooled tests required: {:.2%} for a pool size of {}, pass rate of: {}\n----------------".format(t_mean, vv, v))
    #             tt.append([v,vv,t_mean])
    end = time.perf_counter()
    print("That run took {} seconds".format(end-begin))
    print(tt)




# m = np.array([[0,0,np.nan,3.5], [4,np.nan,0,0], [1,1,1.1,1], [0,np.nan,12,0]])

# a = fail_test(m,4,1)
# fail_lvl = 5
# m = np.array([[0,1,5,10,3],[3,2,7,4,1],[5,2,9,1,2],[3,1,2,0,3],[1,1,0,6,0]], float)
# m11 = np.array([[0,1,np.nan,np.nan,3],[np.nan,2,np.nan,np.nan,1],[np.nan,2,np.nan,1,2],[3,np.nan,2,np.nan,np.nan],[1,1,0,np.nan,0]], float)
# m14 = np.array([[np.nan,1,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,4,1],[np.nan,2,np.nan,np.nan,2],[3,np.nan,2,np.nan,np.nan],[1,1,0,np.nan,0]], float)
# print(m14)
# a,b  = guess_sample(m14, fail_lvl)
# print(a, b)
# print(m11)
# a,b  = guess_sample(m11, fail_lvl)
# print(a, b)
# print(m)
# a,b  = guess_sample(m, fail_lvl)
# print(a, b)



# Mprime = np.array([[           np.nan,    8.58349006e-02, 4.91154148e-01, 6.22954288e-01,  2.53602065e-01],
#                       [        np.nan,    8.78314721e-02, 8.46660451e-01, 3.64376599e-05,  3.65600298e-01],
#                       [7.93157972e-01,            np.nan, 2.39595683e-03, 3.38072837e-02,  1.63071299e-01],
#                       [4.81439648e-01,            np.nan, 1.60141019e-01, 1.53667450e-01,  3.97597566e-04],
#                       [        np.nan,            np.nan,         np.nan, 5.82260980e-01,  1.02998046e-02]])

# c = fail_test(Mprime, 1.2)
# print("C: ", c)

# dx, dy = get_aves(Mprime, 1.2)
# print("dx: ", dx, "dy: ", dy)
# print("dy: ", dy)
# a, b = guess_sample(Mprime, 1.2)
# print(a, b)