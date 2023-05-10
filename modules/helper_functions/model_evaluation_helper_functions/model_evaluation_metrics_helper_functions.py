from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score

def calculate_all_evaluation_metrics(
        ground_truths,
        model_predictions
    ):
    """
    Calculates all the evaluation metrics
    """
    # Calculate the weighted f1 score
    weighted_f1_score = calculate_weighted_f1_score(
        ground_truths=ground_truths,
        model_predictions=model_predictions
    )

    # Calculate the accuracy
    accuracy = calculate_accuracy(
        ground_truths=ground_truths,
        model_predictions=model_predictions
    )

    # Calculate the average weighted recall
    weighted_recall = calculate_weighted_recall(
        ground_truths=ground_truths,
        model_predictions=model_predictions
    )

    # Calculate the confusion matrix
    confusion_matrix = calculate_confusion_matrix(
        ground_truths=ground_truths,
        model_predictions=model_predictions
    )

    return weighted_f1_score, accuracy, weighted_recall, confusion_matrix

def calculate_weighted_f1_score(
        ground_truths,
        model_predictions
    ):
    """
    Calculates the weighted f1 score
    """
    return f1_score(
        y_true=ground_truths,
        y_pred=model_predictions,
        average='weighted'
    )

def calculate_accuracy(
        ground_truths,
        model_predictions
    ):
    """
    Calculates the accuracy
    """
    return accuracy_score(
        y_true=ground_truths,
        y_pred=model_predictions
    )

def calculate_weighted_recall(
        ground_truths,
        model_predictions
    ):
    """
    Calculates the weighted recall
    """
    return recall_score(
        y_true=ground_truths,
        y_pred=model_predictions,
        average='weighted'
    )

def calculate_confusion_matrix(
        ground_truths,
        model_predictions
    ):
    """
    Calculates the confusion matrix
    """
    return confusion_matrix(
        y_true=ground_truths,
        y_pred=model_predictions
    )