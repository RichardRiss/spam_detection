from numpy import * 

data_mat = matrix([[1.0, 2.1],  # 1.0
                    [2.0, 1.1],  # 1.0
                    [1.3, 1.0],  # -1.0
                    [1.0, 1.0],  # -1.0
                    [1.5, 1.6]  # 1.0
                ])

classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

def stumpClassify(data_mat, dimen, thresh_val, thresh_ineq):
    ret_array = ones((shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dimen] > thresh_val] = -1.0
    return ret_array


def buildStump(data_arr, labels, D):
    data_mat = mat(data_arr)
    labels_mat = mat(labels).T
    m, n = shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = mat(zeros((m, n)))
    min_error = inf
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequel in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predict_val = stumpClassify(data_mat, i, thresh_val, inequel)
                error_array = mat(ones((m, 1)))
                error_array[predict_val == labels_mat] = 0
                weight_error = D.T * error_array
                print("split: dim %d, thresh %.2f, thresh ineqal：%s, the weighted error is %.3f" % (
                    i, thresh_val, inequel, weight_error))
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = predict_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequel
    return best_stump, min_error, best_class_est

if __name__ == '__main__':
    # Build array with equal weights 0,2 for 5 elements
    D = mat(ones((5,1))/5)
    best_stump, min_error, best_classestimate = buildStump(data_mat,classLabels,D)
    print("Best stump: {}".format(best_stump))
    #print("Minimal error: {}".format(min_error))
    #print("Best Class Estimate: {}".format(best_classestimate))