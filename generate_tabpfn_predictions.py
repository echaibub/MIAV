from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor


def generate_tabpfn_classifier_prediction(X_train, X_test, y_train):
    
    # Initialize a classifier
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    
    # Predict labels
    predictions = clf.predict(X_test)
    
    return {"predictions": predictions}



def generate_tabpfn_regression_prediction(X_train, X_test, y_train):

    # Initialize the regressor
    reg = TabPFNRegressor(device='auto')
    reg.fit(X_train, y_train)

    # Generate prediction
    preds = reg.predict(X_test)

    return {"predictions": preds}

