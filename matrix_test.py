from email.policy import default
import numpy as np
from numpy.random import default_rng
from scipy.stats import chi2
from numpy.random import SeedSequence
from multiprocessing import Pool
import time
from scipy import stats
import pandas as pd
# from exceptions import Exception

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

dist_continu = [d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]

def gen_sample(sample_count: int=2000, fail: float=1.2, distribution: str="chi2", pass_rate: float=0.95,  pool: int=8, seed: int=42, freedom: float=1):
    # sample count is the minimum sample count, sample_count < returned sample size <= sample_size + pool    
    if distribution not in dist_continu:
        raise ValueError("{} is not a continuous distribution".format(distribution))
    rng = default_rng(seed)
    class_method = getattr(stats, distribution)
    
    a = class_method.rvs(df = freedom, size = sample_count + pool -1, random_state = rng)
    a_scaled = a/(class_method.ppf(q = pass_rate, df =freedom)/fail)
    c = (len(a_scaled) // pool) * pool # find ideal sample count (all pool sizes equal, no residual)
    # print("Sample size stats, count: {}, pool: {}, ideal pool: {}".format(len(a_scaled), pool, c))
    # print(b == len(a_scaled[:b]))
    return a_scaled[:c]

def test_p_conv_eff(k = 5, chi_param = 1, tests =2000, pass_frac = 0.95, fail = 1.2, loq=0.3, loq_scale = 2, rng = 42):
    if fail/k < loq/loq_scale: #  evaluate if batch size is too big for limit of detection
        return np.nan
    sample = gen_sample(sample_count=tests, fail=fail, pass_rate=pass_frac, pool=k, seed=rng)
    print("sample size: {}, pool: {}, residual: {}".format(len(sample), k, len(sample)%k))
    gaps = np.arange(0,len(sample),k) # build the pools
    test_count = len(gaps)
    for i in gaps:
        b = np.mean(sample[i:i+k])
        if b >= fail/k:
            test_count = test_count + de_conv_pool(sample[i:i+k], fail, len(sample[i:i+k]))
        # if len(a_scaled[i:i+k]) != k:
            # print("shorty")
    return test_count/len(sample)



def matrix_pool(k = 5, chi_param = 1, tests =2000, pass_frac = 0.95, fail = 1.2, loq=0.3, loq_scale = 2, rng = 42):
    sample = gen_sample(sample_count=tests, fail=fail, pass_rate=pass_frac, pool=k*k, seed=rng)
    matrix = np.resize(sample, ((len(sample)//(k*k)),k,k))
    # print("old matrix shape: {}\nnew matrix shape: {}\nnew matrix: {}".format(sample.shape, matrix.shape, matrix))
    # print("Number of failed samples: ", len([x for x in sample if x>=fail])) #how many failed individual samples in the sample
    # print("the 2, 2 values: {}".format(matrix[:,2,2]))
    test_count = 0
    total_fail = 0 
    z,x,y = matrix.shape
    for i in range(z):
        test_count += 1
        x_tests_pos = []
        y_tests_pos = []
        x_tests_cert = []
        y_tests_cert = []
        for xi in range(x):
            # if np.mean(matrix[i,xi,:]) >= fail:
            #     x_tests_cert.append(xi)
            if np.mean(matrix[i,xi,:]) >= fail/k:
                x_tests_pos.append([xi, np.mean(matrix[i,xi,:])])

        for yi in range(y):
            # if np.mean(matrix[i,:,yi]) >= fail:
            #     y_tests_cert.append(yi)
            if np.mean(matrix[i,:,yi]) >= fail/k:
                y_tests_pos.append([yi, np.mean(matrix[i,:,yi])])
        y_arr = np.array(y_tests_pos)
        x_arr = np.array(x_tests_pos)
        # print(y_arr.shape, x_arr.shape)
        
        if y_arr.shape[0] and x_arr.shape[0] >= 1:
            scratch = np.copy(matrix[i,:,:])
            y_max = np.argmax(y_arr[:, 1])
            x_max = np.argmax(x_arr[:, 1])
            y_max_m = int(y_arr[y_max, 0])
            x_max_m = int(x_arr[x_max, 0])
            # print("ymax {} xmax {}".format(y_max_m, x_max_m))
            # print("guess: ", matrix[i, x_max_m, y_max_m])
            test_count +=1
            # print("scratch:\n",scratch)
            if scratch[x_max_m, y_max_m] >= fail:
                total_fail += 1
            scratch[x_max_m, y_max_m] = np.nan
            # print("scratch:\n",scratch)
            # print(np.count_nonzero(np.isnan(scratch[xi,:])))
            while len(x_tests_pos) and len(y_tests_pos) > 0:
                x_tests_pos = []
                y_tests_pos = []
                for xi in range(x):
                    if np.nanmean(scratch[xi,:]) > fail / (k - np.count_nonzero(np.isnan(scratch[xi,:]))):
                        x_tests_pos.append([xi, np.nanmean(scratch[xi,:])])
                for yi in range(y):
                    if np.nanmean(scratch[:,yi]) > fail / (k - np.count_nonzero(np.isnan(scratch[:,yi]))):
                        x_tests_pos.append([yi, np.nanmean(scratch[:,yi])])
                y_arr = np.array(y_tests_pos)
                x_arr = np.array(x_tests_pos)
                if y_arr.shape[0] and x_arr.shape[0] >= 1:
                    print(y_arr.shape, x_arr.shape)
                    y_arr = np.array(y_tests_pos)
                    x_arr = np.array(x_tests_pos)
                    y_max = np.argmax(y_arr[:, 1])
                    x_max = np.argmax(x_arr[:, 1])
                    y_max_m = int(y_arr[y_max, 0])
                    x_max_m = int(x_arr[x_max, 0])
                    # print("Next guess:", scratch[x_max_m, y_max_m])
                    test_count += 1
                    if scratch[x_max_m, y_max_m] >= fail:
                        total_fail += 1
                    scratch[x_max_m, y_max_m] = np.nan
                    # print("Updated Scratch:\n", scratch)


            # print("final scratch:\n",scratch)
 
        # print("pool defo fails by row", x_tests_cert)
        # print("pool possibly fails by row", x_tests_pos)
        # print("pool defo fails by column", y_tests_cert)
        # print("pool possibly fails by column", y_tests_pos)
        # print("Guess pool 3: ", matrix[1,0,1])
        
        # print("-------------------------------")

        # for a in x_tests:
        #     for b in y_tests:


    print("Test count for this sample: ", test_count)
    print("Failed samples found: ", total_fail)
    print("Actual number of failed samples: ", len([x for x in sample if x>=fail]))
    return test_count/len(sample)

def matrix_pool_t(k = 5, chi_param = 1, tests =2000, pass_frac = 0.95, fail = 1.2, loq=0.3, loq_scale = 2, rng = 42):
    sample = gen_sample(sample_count=tests, fail=fail, pass_rate=pass_frac, pool=k*k, seed=rng)
    matrix = np.resize(sample, ((len(sample)//(k*k)),k,k))
    # print("old matrix shape: {}\nnew matrix shape: {}\nnew matrix: {}".format(sample.shape, matrix.shape, matrix))
    print("Number of failed samples: ", len([x for x in sample if x>=fail])) #how many failed individual samples in the sample
    # print("the 2, 2 values: {}".format(matrix[:,2,2]))
    test_count = 0
    total_fail = 0 
    z,x,y = matrix.shape
    for i in range(z):
        test_count += 1
        x_tests_pos = []
        y_tests_pos = []
        x_tests_cert = []
        y_tests_cert = []
        for xi in range(x):
            # if np.mean(matrix[i,xi,:]) >= fail:
            #     x_tests_cert.append(xi)
            if np.mean(matrix[i,xi,:]) >= fail/k:
                x_tests_pos.append([xi, np.mean(matrix[i,xi,:])])

        for yi in range(y):
            # if np.mean(matrix[i,:,yi]) >= fail:
            #     y_tests_cert.append(yi)
            if np.mean(matrix[i,:,yi]) >= fail/k:
                y_tests_pos.append([yi, np.mean(matrix[i,:,yi])])
        y_arr = np.array(y_tests_pos)
        x_arr = np.array(x_tests_pos)
        # print(y_arr.shape, x_arr.shape)
        
        if y_arr.shape[0] and x_arr.shape[0] >= 1:
            scratch = np.copy(matrix[i,:,:])
            y_max = np.argmax(y_arr[:, 1])
            x_max = np.argmax(x_arr[:, 1])
            y_max_m = int(y_arr[y_max, 0])
            x_max_m = int(x_arr[x_max, 0])
            # print("ymax {} xmax {}".format(y_max_m, x_max_m))
            print("guess: ", matrix[i, x_max_m, y_max_m])
            test_count +=1
            print("scratch:\n",scratch)
            if scratch[x_max_m, y_max_m] >= fail:
                total_fail += 1
            scratch[x_max_m, y_max_m] = np.nan
            print("scratch:\n",scratch)
            # print(np.count_nonzero(np.isnan(scratch[xi,:])))
            while (len(x_tests_pos) and len(y_tests_pos)) > 0:
                x_tests_pos = []
                y_tests_pos = []
                for xi in range(x):
                    if np.nanmean(scratch[xi,:]) > (fail / (k - np.count_nonzero(np.isnan(scratch[xi,:])))):
                        x_tests_pos.append([xi, np.nanmean(scratch[xi,:])])
                for yi in range(y):
                    if np.nanmean(scratch[:,yi]) > (fail / (k - np.count_nonzero(np.isnan(scratch[:,yi])))):
                        y_tests_pos.append([yi, np.nanmean(scratch[:,yi])])
                y_arr = np.array(y_tests_pos)
                x_arr = np.array(x_tests_pos)
                print("failing pool shapes:", y_arr.shape, x_arr.shape)
                print("lengths of failing lists:", len(y_tests_pos), len(x_tests_pos))
                print("failing x pool", x_arr)
                print("failing y pool", y_arr)
                if y_arr.shape[0] and x_arr.shape[0] >= 1:
                    print(y_arr.shape, x_arr.shape)
                    y_arr = np.array(y_tests_pos)
                    x_arr = np.array(x_tests_pos)
                    y_max = np.argmax(y_arr[:, 1])
                    x_max = np.argmax(x_arr[:, 1])
                    y_max_m = int(y_arr[y_max, 0])
                    x_max_m = int(x_arr[x_max, 0])
                    if y_arr[y_max, 1] > x_arr[x_max, 1]:

                    print("Next guess:", scratch[x_max_m, y_max_m])
                    print("guess co-ords", x_max_m, y_max_m)
                    print("co-ord test", scratch[x_max_m, :])

                    # if scratch[x_max_m, y_max_m] <1:
                    #     # raise ValueError("oooph not sure how we got here but that shouldn't be possible")
                    #     break
                    #     raise ValueError("oooph not sure how we got here but that shouldn't be possible")
                    test_count += 1
                    if scratch[x_max_m, y_max_m] >= fail:
                        total_fail += 1
                    scratch[x_max_m, y_max_m] = np.nan
                    print(len(x_tests_pos), len(y_tests_pos))
                    print("Updated Scratch:\n", scratch)


            print("final scratch:\n",scratch)
 
        # print("pool defo fails by row", x_tests_cert)
        # print("\npool possibly fails by row", x_tests_pos)
        # print(x_tests_pos.shape)
        # print("pool defo fails by column", y_tests_cert)
        # print("\npool possibly fails by column", y_tests_pos)
        # print("Guess pool 3: ", matrix[1,0,1])
        
        # print("-------------------------------")

        # for a in x_tests:
        #     for b in y_tests:


    print("Test count for this sample: ", test_count)
    print("Failed samples found: ", total_fail)
    print("Actual number of failed samples: ", len([x for x in sample if x>=fail]))
    return test_count/len(sample)


def wrapper_calc(arg_dict):
    kwargs = arg_dict
    # print(type(kwargs))
    # print(kwargs)
    # return test_p_eff(**kwargs)
    return matrix_pool_t(**kwargs)

if __name__ == "__main__":
    # kk_val = [2,3,4,5,6,7,8]
    # per_fail=[.5, .75, .8, .85, .9, .95, .975, .99, .999]
    kk_val = [3,4]
    per_fail=[.9]
    tt= []
    begin = time.perf_counter()
    sample_count = 67
    no_tests = 1
    with Pool(processes=8) as pool:
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
                    # base_dic["rng"] = i # production version
                    base_dic["rng"] = 42 # testing version
                    dic_list.append(base_dic.copy())
                tests = pool.map(wrapper_calc, dic_list)
                t_mean = np.mean(tests)
                print("Average pooled tests required: {:.2%} for a pool size of {}, pass rate of: {}\n----------------".format(t_mean, vv, v))
                tt.append([v,vv,t_mean])
    end = time.perf_counter()
    print("That run took {} seconds".format(end-begin))
    # print(tt)

