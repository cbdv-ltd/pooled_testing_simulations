import numpy as np
from numpy.random import default_rng
from numpy.random import SeedSequence
from multiprocessing import Pool
import time
from scipy import stats
import pandas as pd
import check_matrix
import seaborn as sb


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
    failure = 0
    x, y = get_aves(matrix, fail)
    xc = np.sum(np.isfinite(matrix), axis=1) 
    yc = np.sum(np.isfinite(matrix), axis=0)
    if debug == 1:
        print(matrix)
        print(xc)
        print(yc)
        print(x)
        print(y)
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
        failure += 1
    if x_fail > 0:
        failure += 1
    return failure



def guess_sample(matrix, fail, tests = 0, ool = 0):
    avx, avy = get_aves(matrix, fail)
    if fail_test(matrix, fail) < 2:
        return tests, ool
    avxi = max(avx, key = avx.get)
    avyi = max(avy, key = avy.get)
    not_null = np.isnan(matrix[avxi, avyi])
    if avx[avxi] > avy[avyi]:
        count = matrix.shape[0]
        while not_null == True:
            avy[avyi] = 0
            avyi = max(avy, key = avy.get)
            not_null = np.isnan(matrix[avxi, avyi])
            count = count - 1
            if count == -1:
                return tests, ool
        tests += 1
        if matrix[avxi, avyi] >= fail:
            ool += 1
        matrix[avxi, avyi] = np.nan
        
        
    else:
        count  = matrix.shape[0]
        while not_null == True:
            avx[avxi] = 0
            avxi = max(avx, key = avx.get)
            not_null = np.isnan(matrix[avxi, avyi])
            count = count - 1
            if count == -1:
                return tests, ool
        tests += 1
        if matrix[avxi, avyi] >= fail:
            ool += 1
        matrix[avxi, avyi] = np.nan
    tests, ool = guess_sample(matrix, fail, tests, ool)
    return tests, ool
    


def matrix2(k = 5, dist_parm = (1, 0, 1), tests =2000, pass_frac = 0.95, fail = 1.2, loq=0.3, loq_scale = 2, rng = 42):
    sample = gen_sample(sample_count=tests, fail=fail, pass_rate=pass_frac, pool=k*k, seed=rng, dist_parm=dist_parm)
    matrix = np.resize(sample, (len(sample)//(k*k),k,k))
    test_count = 0
    total_fail = 0
    z, x, y = matrix.shape
    for i in range(z):
        t , f = check_matrix.guess(matrix[i,:,:], fail)
        test_count += t
        total_fail += f
    return test_count/len(sample)


def gen_sample(sample_count: int=2000, fail: float=1.2, distribution: str="lognorm", pass_rate: float=0.95,  pool: int=8, seed: int=42, dist_parm: tuple=(1,0,1)):
    # sample count is the minimum sample count, sample_count < returned sample size <= sample_size + pool    
    if distribution not in DIST_CONTINU:
        raise ValueError("{} is not a continuous distribution".format(distribution))
    rng = default_rng(seed)
    class_method = getattr(stats, distribution)    
    a = class_method.rvs(s=dist_parm[0], loc=dist_parm[1], scale=dist_parm[2], size=sample_count + pool -1, random_state = rng)
    c = (len(a) // pool) * pool # find ideal sample count (all pool sizes equal, no residual)
    return a[:c]

def wrapper_calc(arg_dict):  #wrapper function for passing multiple keywords into multiprocessing map pool
    kwargs = arg_dict
    return matrix2(**kwargs)

if __name__ == "__main__":
    kk_val = np.arange(3,17,1)
    # kk_val = [8,9,10,11,12]
    # per_fail=[.5, .75, .8, .85, .9, .95, .975, .99, .996, .999]
    # kk_val = [5]
    per_fail=[.996]
    
    begin = time.perf_counter()
    sample_count = 2000
    no_tests = 4000
    runs = [({"dist_parm":(1.5233660418119728, 0.0027288920577292503, 0.06939280546211554), "fail": 0.5}, "pb_matrix.csv"),
            ({"dist_parm":(1.1504315889666699, -0.0020480895355545316, 0.06710884489857513), "fail": 0.2}, "as_matrix.csv"),
            ({"dist_parm":(1.0858164361757576, 0.00019808453407679164, 0.036053458447836084), "fail": 0.2}, "cd_matrix.csv"),
            ({"dist_parm":(0.7502321546956426, -0.000961540716204136, 0.007837295888340656), "fail": 0.1}, "hg_matrix.csv")]
    for r in runs:
        tt= []
        with Pool(processes=16) as pool:
            for c,v in enumerate(per_fail):
                for cc,vv in enumerate(kk_val):
                    tests = np.empty((no_tests, 1), float)
                    tests[:] = np.nan
                    alpha = SeedSequence().entropy
                    args = np.arange(no_tests) + alpha
                    dic_list = []
                    base_dic = {"k":vv, "pass_frac":v, "tests":sample_count}
                    base_dic.update(r[0])
                    # base_dic = {"k":vv, "pass_frac":v, "tests":sample_count, "dist_parm":(1.4525204178329565, 0.0001955834220820136, 0.07417152849481998)} #hg
                    # base_dic = {"k":vv, "pass_frac":v, "tests":sample_count, "dist_parm":(1.4525204178329565, 0.0001955834220820136, 0.07417152849481998)} #cd
                    # base_dic = {"k":vv, "pass_frac":v, "tests":sample_count, "dist_parm":(1.4525204178329565, 0.0001955834220820136, 0.07417152849481998)} #as
                    for i in args:
                        base_dic["rng"] = i # production version
                        # base_dic["rng"] = 42 # testing version
                        dic_list.append(base_dic.copy())
                    tests = pool.map(wrapper_calc, dic_list)
                    t_mean = np.mean(tests)
                    print("Average pooled tests required: {:.2%} for a pool size of {}, pass rate of: {}\n----------------".format(t_mean, vv, v))
                    tt.append([v,vv,t_mean])
        end = time.perf_counter()
        print("That run took {} seconds".format(end-begin))
        print(tt)
        out = pd.DataFrame(tt, columns=["pass", "pool", "percentage"])
        out.to_csv(r[1])
