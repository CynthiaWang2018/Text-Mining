def evaluate(true_value_path, result_path):
    f = open(true_value_path, 'r')
    true_value = [i.split()[0] for i in f.readlines()]
    f.close()

    f = open(result_path, 'r')
    predictions = [i.strip() for i in f.readlines()]
    f.close()

    n = len(true_value)
    TP = len([i for i in range(n) if true_value[i] == '+1' and predictions[i] == '+1'])
    FN = len([i for i in range(n) if true_value[i] == '+1' and predictions[i] == '-1'])
    FP = len([i for i in range(n) if true_value[i] == '-1' and predictions[i] == '+1'])
    TN = len([i for i in range(n) if true_value[i] == '-1' and predictions[i] == '-1'])

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / n
    # error_rate = 1 - accuracy
    F1 = 2 * precision * recall / (precision + recall)
    print('result file',result_path)
    print('-'*30)
    print('recall :' + str(recall))
    print('precision :' + str(precision))
    print('accuracy :' + str(accuracy))
    print('F1 score :' + str(F1))
    print('\n')
if __name__ == '__main__':
    true_value_path = './1_raw_dataset/test.txt'
    result_path_1 = './7_result2/BernoulliNB.txt'
    result_path_2 = './7_result2/LinearSVC.txt'
    result_path_3 = './7_result2/LR.txt'
    result_path_4 = './7_result2/naiveBayes.txt'
    result_path_5 = './7_result2/NuSVC.txt'
    result_path_6 = './7_result2/SVC.txt'
    evaluate(true_value_path, result_path_1)
    evaluate(true_value_path, result_path_2)
    evaluate(true_value_path, result_path_3)
    evaluate(true_value_path, result_path_4)
    evaluate(true_value_path, result_path_5)
    evaluate(true_value_path, result_path_6)

