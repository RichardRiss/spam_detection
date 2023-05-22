#!/usr/bin/env python3
# python 3.10.11


from numpy import *
import matplotlib.pyplot as plt


def plot_data(data_mat, labels):
    # Extract x and y coordinates from the data matrix
    x = data_mat[:, 0].tolist()
    y = data_mat[:, 1].tolist()

    # Create a scatter plot
    fig, ax = plt.subplots()
    for i in range(len(x)):
        if labels[i] == 1.0:
            ax.plot(x[i], y[i], 'ro')  # Red dots for label 1.0
        else:
            ax.plot(x[i], y[i], 'bs')  # Blue squares for label -1.0

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Example Data')

    # Display the plot
    plt.show(block=False)
    return fig, ax


def plot_threshold(fig, ax, data):
    plt.ion()
    for item in data:
        dim = item['dim']
        thresh = item['thresh']

        if dim == 1:
            ax.axhline(y=thresh, color='r', linestyle='--')
        elif dim == 0:
            ax.axvline(x=thresh, color='r', linestyle='--')


def loadSimpleData():
    data_mat = matrix([[1.0, 2.1],  # 1.0
                       [2.0, 1.1],  # 1.0
                       [1.3, 1.0],  # -1.0
                       [1.0, 1.0],  # -1.0
                       [1.5, 1.6]  # 1.0
                       ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, labels


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


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump

        alpha = float(
            0.5 * log((1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        # print "classEst: ",classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return sign(aggClassEst)


def main():
    datas_mat, labels = loadSimpleData()
    fig, ax = plot_data(datas_mat, labels)
    # best_stump, min_error, best_class_est = buildStump(datas_mat,labels,mat(ones((5,1))/5))
    # print best_stump, min_error, best_class_est
    classify_array, aggClassEst = adaBoostTrainDS(datas_mat, labels, 9)
    plot_threshold(fig, ax, classify_array)
    print(adaClassify([[2, 1], [0, 0]], classify_array))


if __name__ == '__main__':
    main()