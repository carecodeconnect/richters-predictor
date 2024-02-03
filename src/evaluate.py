from sklearn.metrics import f1_score

def evaluate_model(pipe, X_train, X_valid, y_train, y_valid):
    """
    Evaluate the model using the F1 score.

    Args:
        pipe: trained model
        X_train: training data
        X_valid: validation data
        y_train: training labels
        y_valid: validation labels

    Returns:
        score_valid: F1 score on validation data
        score_train: F1 score on training data
    """
    pipe.fit(X_train, y_train)
    preds_valid = pipe.predict(X_valid)
    preds_train = pipe.predict(X_train)

    score_valid = f1_score(y_valid, preds_valid, average='micro')
    score_train = f1_score(y_train, preds_train, average='micro')

    return score_valid, score_train