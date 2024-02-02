from sklearn.metrics import f1_score

def evaluate_model(pipe, X_train, X_valid, y_train, y_valid):
    pipe.fit(X_train, y_train)
    preds_valid = pipe.predict(X_valid)
    preds_train = pipe.predict(X_train)

    score_valid = f1_score(y_valid, preds_valid, average='micro')
    score_train = f1_score(y_valid, preds_valid, average='micro')
    
    return score_valid, score_train