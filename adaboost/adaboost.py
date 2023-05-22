#!/usr/bin/env python3
# python 3.10.11


from numpy import *
import matplotlib.pyplot as plt


######################
# Plotting
######################

def plot_data(data_mat, labels, title='Example Data', plotnum=0, ax=None):
    # Extract x and y coordinates from the data matrix
    x = data_mat[:, 0].tolist()
    y = data_mat[:, 1].tolist()

    # Create a scatter plot
    if plotnum == 0:
        fig, ax = plt.subplots(2, 1, constrained_layout=True)
    for i in range(len(x)):
        if labels[i] == 1.0:
            ax[plotnum].plot(x[i], y[i], 'ro')  # Red dots for label 1.0
        else:
            ax[plotnum].plot(x[i], y[i], 'bs')  # Blue squares for label -1.0

    # Set labels and title
    ax[plotnum].set_xlabel('X')
    ax[plotnum].set_ylabel('Y')
    # ax[plotnum].set_xlim(left=0)
    # ax[plotnum].set_ylim(bottom=0)
    ax[plotnum].set_title(title)

    # Display the plot
    plt.show(block=False)
    return ax


def plot_threshold(ax, data, plotnum=0):
    plt.ion()
    for item in data:
        dim = item['dim']
        thresh = item['thresh']
        color = 'r' if item['ineq'] == 'gt' else 'b'
        if dim == 1:

            ax[plotnum].axhline(y=thresh, color=color, linestyle='--')
        elif dim == 0:
            ax[plotnum].axvline(x=thresh, color=color, linestyle='--')


def plot_roc(predStrengths, classLabels, title, ax=None, plotnum=0):
    subplots = 4
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    if ax is None:
        fig, ax = plt.subplots(subplots, 1, constrained_layout=True)

    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax[plotnum].plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], color='b')
        cur = (cur[0] - delX, cur[1] - delY)

    ax[plotnum].plot([0, 1], [0, 1], 'b--')
    ax[plotnum].set_xlabel('False positive rate')
    ax[plotnum].set_ylabel('True positive rate')
    ax[plotnum].set_title(title)
    ax[plotnum].set_xlim(left=0)
    ax[plotnum].set_ylim(bottom=0)
    print(f'The Area Under the Curve for "{title}" is: {ySum * xStep}')

    block = plotnum == (subplots - 1)
    plt.show(block=block)
    return ax


######################
# Data Handling
######################

def loadSimpleData():
    data_mat = matrix([[1.0, 2.1],  # 1.0
                       [2.0, 1.1],  # 1.0
                       [1.3, 1.0],  # -1.0
                       [1.0, 1.0],  # -1.0
                       [1.5, 1.6]  # 1.0
                       ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, labels


def loadSimpleData2():
    data_mat = matrix([[1.0, 2.1],  # 1.0
                       [2.0, 1.0],  # 1.0
                       [1.3, 1.0],  # -1.0
                       [1.0, 1.0],  # -1.0
                       [1.5, 1.6]  # 1.0
                       ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, labels


def loadDataSet(fileName):
    with open(fileName, 'r') as fr:
        lines = fr.readlines()

    numFeat = len(lines[0].strip().split('\t'))
    dataMat = []
    labelMat = []

    for line in lines:
        curLine = line.strip().split('\t')
        lineArr = [float(curLine[i]) for i in range(numFeat - 1)]
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat


######################
# Data training
######################

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


######################
# Classification
######################

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


def error_rate(arr_predictions, array_labels):
    err_arr = mat(ones((size(arr_predictions, 0), 1)))
    err_sum = err_arr[arr_predictions != mat(array_labels).T].sum()
    return err_sum / size(arr_predictions, 0)


def main():
    ###############
    # SimpleData
    ###############
    # normal simple data
    datas_mat, labels = loadSimpleData()
    ax = plot_data(datas_mat, labels, "Simple Data")
    classify_array, aggClassEst = adaBoostTrainDS(datas_mat, labels, 10)
    plot_threshold(ax, classify_array)

    # slightly changed simple data
    datas_mat2, labels2 = loadSimpleData2()
    plot_data(datas_mat2, labels2, "Simple Data2", 1, ax)
    classify_array2, aggClassEst2 = adaBoostTrainDS(datas_mat2, labels2, 10)
    plot_threshold(ax, classify_array2, 1)

    #print(adaClassify([[2, 1], [0, 0]], classify_array2))

    ###############
    # Complex data
    ###############
    datas_mat2, labels2 = loadDataSet("horseColicTraining2.txt")
    # Train 10 weak learners
    classify_array10, aggClassEst10 = adaBoostTrainDS(datas_mat2, labels2, 10)
    # Train 50 weak learner
    classify_array50, aggClassEst50 = adaBoostTrainDS(datas_mat2, labels2, 50)
    # Train 100 weak learner
    classify_array100, aggClassEst100 = adaBoostTrainDS(datas_mat2, labels2, 100)
    # Train 500 weak learner
    classify_array500, aggClassEst500 = adaBoostTrainDS(datas_mat2, labels2, 500)

    # Classify with testData
    test_data, test_label = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(test_data, classify_array10)
    prediction50 = adaClassify(test_data, classify_array50)
    prediction100 = adaClassify(test_data, classify_array100)
    prediction500 = adaClassify(test_data, classify_array500)

    # Get the error rate
    error10 = error_rate(prediction10, test_label)
    print(f"Error rate for 10 weak learners is: {error10}.")
    error50 = error_rate(prediction50, test_label)
    print(f"Error rate for 50 weak learners is: {error50}.")
    error100 = error_rate(prediction100, test_label)
    print(f"Error rate for 100 weak learners is: {error100}.")
    error500 = error_rate(prediction500, test_label)
    print(f"Error rate for 500 weak learners is: {error500}.")

    # Plot receiver operating characteristic (ROC)
    ax = plot_roc(aggClassEst10.T, labels2, "ROC for 10 decision stumps")
    plot_roc(aggClassEst50.T, labels2, "ROC for 50 decision stumps", ax, 1)
    plot_roc(aggClassEst100.T, labels2, "ROC for 100 decision stumps", ax, 2)
    plot_roc(aggClassEst500.T, labels2, "ROC for 500 decision stumps", ax, 3)


if __name__ == '__main__':
    main()
