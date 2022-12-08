def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    accuracy = sum(prediction==ground_truth)/len(prediction)
    precision = sum((prediction==True)&(prediction==ground_truth))/sum(prediction==True)
    recall = sum((prediction==True)&(prediction==ground_truth))/(sum((prediction==True)&(prediction==ground_truth))+
                                                    sum((prediction==False)&(prediction!=ground_truth)))
    f1 = 2*precision*recall/(precision+recall)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    return sum(prediction==ground_truth)/len(prediction)
