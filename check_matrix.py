import numpy as np


# def guess2(matrix, fail: float, tests = 0, ool = 0):
#     dim = matrix.shape[0]
#     m = np.ones((dim, dim, 3))
#     m[:,:,0] = matrix.copy()
#     # m[1,4,1] = 0
#     # m[:,3,2] = 0
#     m_tests = dim * dim - np.sum(m[:,:,1])
#     print("no of tests run", m_tests)
#     m_use = m[:,:,0]*np.minimum(m[:,:,1], m[:,:,2])
#     x_sum_m = np.sum(m_use, axis = 1)
#     x_av_m = x_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 1)
#     y_sum_m = np.sum(m_use, axis = 0)
#     y_av_m = y_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 0)

#     count = 0
#     elim = np.sum(np.minimum(m[:,:,1], m[:,:,2]))

#     # soph = (x_av_m.reshape(dim, 1) + y_av_m.reshape(1, dim)) * np.minimum(m[:,:,1], m[:,:,2])
#     # print(soph)
#     # print("elim: ", elim)
#     # print(m_use)
#     # print(x_sum_m)
#     # print(x_av_m)
#     # print(y_sum_m)
#     # print(y_av_m)
#     # print(np.unravel_index(np.argmax(soph, axis=None), soph.shape))

#     while elim > 0 and count < dim*dim + 1:
#         # x_a_max = np.argmax(x_av_m)
#         # y_a_max = np.argmax(y_av_m)
#         soph = (x_av_m.reshape(dim, 1) + y_av_m.reshape(1, dim)) * np.minimum(m[:,:,1], m[:,:,2])
#         x_a_max, y_a_max = np.unravel_index(np.argmax(soph, axis=None), soph.shape)
#         m[x_a_max, y_a_max, 1] = 0        
#         print("x-max: {}, y-max: {}, value: {}".format(x_a_max, y_a_max, m[x_a_max, y_a_max, 0]))

#         m_tests = dim * dim - np.sum(m[:,:,1])
#         m_use = m[:,:,0]*np.minimum(m[:,:,1], m[:,:,2])
#         x_sum_m = np.sum(m_use, axis = 1)
#         x_av_m = x_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 1)
#         y_sum_m = np.sum(m_use, axis = 0)
#         y_av_m = y_sum_m / np.sum(np.minimum(m[:,:,1], m[:,:,2]), axis = 0)
#         elim = np.sum(np.minimum(m[:,:,1], m[:,:,2]))
#         for iy in dim:
#             if np.sum(m_use[:,iy]) < fail:
#                 m[:,iy,2] = 0
#             if np.sum(m_use[iy,:]) < fail:
#                 m[:,iy,2] = 0
            

#         print(np.unravel_index(np.argmax(soph, axis=None), soph.shape))
#         elim = np.sum(np.minimum(m[:,:,1], m[:,:,2]))
#         if m[x_a_max, y_a_max, 0] >= fail:
#             ool += 1
        
        
        
#         count += 1
#         print("Loop Count: {}, Eliminated samples: {}, Tested samples: {}, Over limit samples: {}".format(count, dim**2 - elim, m_tests, ool))
#     return m_tests + 2*dim, ool


def guess(matrix, fail: float, tests = 0, ool = 0):
    dim = matrix.shape[0]
    m = np.ones((dim, dim, 3))
    m[:,:,0] = matrix.copy() + 0.0000000001
    elim = np.sum(m[:,:,2])
    count = 0
    while count < dim * dim + 2 and elim > 0:        

        average_pos_matrix = m[:,:,0] * m[:,:,1]
        x_av = np.sum(average_pos_matrix, axis =1) / np.maximum(1, np.sum(m[:,:,1], axis =1))
        y_av = np.sum(average_pos_matrix, axis =0) / np.maximum(1, np.sum(m[:,:,1], axis =0))
        x_fail = fail / np.maximum(1, np.sum(m[:,:,1], axis =1))
        y_fail = fail / np.maximum(1, np.sum(m[:,:,1], axis =0))
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
        if m[x,y,0] >= fail:
            ool += 1
        m[:,:,2] = m[:,:,2] * m[:,:,1]
        elim = np.sum(m[:,:,2])
        count += 1
    m_tests = dim * dim - np.sum(m[:,:,1])
    return m_tests + 2*dim, ool