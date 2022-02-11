import numpy as np


def guess2(matrix, fail: float, tests = 0, ool = 0):
    dim = matrix.shape[0]
    m = np.ones((dim, dim, 3))
    m[:,:,0] = matrix.copy()
    # m[1,4,1] = 0
    # m[:,3,2] = 0
    m_tests = dim * dim - np.sum(m[:,:,1])
    print("no of tests run", m_tests)
    m_use = m[:,:,0]*np.minimum(m[:,:,1], m[:,:,2])
    x_sum_m = np.sum(m_use, axis = 1)
    x_av_m = x_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 1)
    y_sum_m = np.sum(m_use, axis = 0)
    y_av_m = y_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 0)

    count = 0
    elim = np.sum(np.minimum(m[:,:,1], m[:,:,2]))

    # soph = (x_av_m.reshape(dim, 1) + y_av_m.reshape(1, dim)) * np.minimum(m[:,:,1], m[:,:,2])
    # print(soph)
    # print("elim: ", elim)
    # print(m_use)
    # print(x_sum_m)
    # print(x_av_m)
    # print(y_sum_m)
    # print(y_av_m)
    # print(np.unravel_index(np.argmax(soph, axis=None), soph.shape))

    while elim > 0 and count < dim*dim + 1:
        # x_a_max = np.argmax(x_av_m)
        # y_a_max = np.argmax(y_av_m)
        soph = (x_av_m.reshape(dim, 1) + y_av_m.reshape(1, dim)) * np.minimum(m[:,:,1], m[:,:,2])
        x_a_max, y_a_max = np.unravel_index(np.argmax(soph, axis=None), soph.shape)
        m[x_a_max, y_a_max, 1] = 0        
        print("x-max: {}, y-max: {}, value: {}".format(x_a_max, y_a_max, m[x_a_max, y_a_max, 0]))

        m_tests = dim * dim - np.sum(m[:,:,1])
        m_use = m[:,:,0]*np.minimum(m[:,:,1], m[:,:,2])
        x_sum_m = np.sum(m_use, axis = 1)
        x_av_m = x_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 1)
        y_sum_m = np.sum(m_use, axis = 0)
        y_av_m = y_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 0)
        elim = np.sum(np.minimum(m[:,:,1], m[:,:,2]))
        for iy in dim:
            if np.sum(m_use[:,iy]) < fail:
                m[:,iy,2] = 0
            if np.sum(m_use[iy,:]) < fail:
                m[:,iy,2] = 0
            

        print(np.unravel_index(np.argmax(soph, axis=None), soph.shape))
        elim = np.sum(np.minimum(m[:,:,1], m[:,:,2]))
        if m[x_a_max, y_a_max, 0] >= fail:
            ool += 1
        
        
        
        count += 1
        print("Loop Count: {}, Eliminated samples: {}, Tested samples: {}, Over limit samples: {}".format(count, dim**2 - elim, m_tests, ool))
    return m_tests + 2*dim, ool


def guess(matrix, fail: float, tests = 0, ool = 0):
    dim = matrix.shape[0]
    m = np.ones((dim, dim, 3))
    m[:,:,0] = matrix.copy() + 0.0000000001
    elim = np.sum(m[:,:,2])
    count = 0
    while count < dim * dim + 2 and elim > 0:        
        # print("loop start\n-------------------------\n")
        average_pos_matrix = m[:,:,0] * m[:,:,1]
        x_av = np.sum(average_pos_matrix, axis =1) / np.maximum(1, np.sum(m[:,:,1], axis =1))
        y_av = np.sum(average_pos_matrix, axis =0) / np.maximum(1, np.sum(m[:,:,1], axis =0))
        x_fail = fail / np.maximum(1, np.sum(m[:,:,1], axis =1))
        y_fail = fail / np.maximum(1, np.sum(m[:,:,1], axis =0))
        # print("posibility matrix:\n", average_pos_matrix)
        # print("testing layer:\n", m[:,:,1])
        # print("x av ",x_av, "\ny av ", y_av)
        # print("x_fail: {}, \ny_fail, {}\n shapes x: {}, y: {}".format(x_fail, y_fail, x_fail.shape, y_fail.shape))
        # print("x fail-av comparison:\n", (x_av > x_fail).reshape(dim,1) * m[:,:,2])
        # print("y fail-av comparison:\n", (y_av > y_fail).reshape(1,dim) * m[:,:,2])
        # print("-----------------------------\n",(x_av > x_fail).reshape(dim,1) * m[:,:,2]*(y_av > y_fail).reshape(1,dim),"\n-------------------------\n")
        m[:,:,2] = (x_av > x_fail).reshape(dim,1) * m[:,:,2]*(y_av > y_fail).reshape(1,dim)
        
        decision_matrix = x_av.reshape(dim,1) * y_av.reshape(1, dim)
        x, y = np.unravel_index(np.argmax(decision_matrix, axis=None), decision_matrix.shape)
        choice_matrix = m[:,:,0] * np.minimum(m[:,:,1], m[:,:,2])

        if choice_matrix[x, y]  == 0:
            count_two = 0
            while count_two < dim **2 and choice_matrix[x, y] == 0:
                decision_matrix[x,y] = 0
                x, y = np.unravel_index(np.argmax(decision_matrix, axis=None), decision_matrix.shape)
                count_two += 1

        m[x,y,1] = 0
        # print("Itteration: {}, remove value: {} at x: {}, y: {}".format(count, m[x,y,0], x, y))
        if m[x,y,0] >= fail:
            ool += 1
        # print("pre combo\n",m[:,:,2])
        m[:,:,2] = m[:,:,2] * m[:,:,1]
        # print("post combo\n",m[:,:,2])
        elim = np.sum(m[:,:,2])
        count += 1

    # print(count)
    # print(m[:,:,1])
    # print(m[:,:,2])
    m_tests = dim * dim - np.sum(m[:,:,1])
    return m_tests + 2*dim, ool

# m = np.array([[0,1,5,10,3],[3,2,7,4,1],[5,2,9,1,2],[3,1,2,0,3],[1,1,0,6,0]], float)
# m11 = np.array([[0,1,np.nan,np.nan,3],[np.nan,2,np.nan,np.nan,1],[np.nan,2,np.nan,1,2],[3,np.nan,2,np.nan,np.nan],[1,1,0,np.nan,0]], float)
# m14 = np.array([[np.nan,1,np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan,4,1],[np.nan,2,np.nan,np.nan,2],[3,np.nan,2,np.nan,np.nan],[1,1,0,np.nan,0]], float)
# Mprime = np.array([[           np.nan,    8.58349006e-02, 4.91154148e-01, 6.22954288e-01,  2.53602065e-01],
#                       [        np.nan,    8.78314721e-02, 8.46660451e-01, 3.64376599e-05,  3.65600298e-01],
#                       [7.93157972e-01,            np.nan, 2.39595683e-03, 3.38072837e-02,  1.63071299e-01],
#                       [4.81439648e-01,            np.nan, 1.60141019e-01, 1.53667450e-01,  3.97597566e-04],
#                       [        np.nan,            np.nan,         np.nan, 5.82260980e-01,  1.02998046e-02]])
# print(m)
# print(guess(m, 5))

# print("order should be 5,9,10,7,4,5,0")
# print(m11)
# print(guess(m11, 5))
# print(m14)
# print(guess(m14, 5))
# print(Mprime)
# print(guess(Mprime, 1.2))