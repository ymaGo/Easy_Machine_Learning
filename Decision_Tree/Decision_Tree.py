    from sklearn import tree
    from sklearn.model_selection im
    port train_test_split
    import numpy as np

    # There are 9 girls and 8 cats, there are 7 features, 1 for yes 0 for no
    features = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 1, 0]
    ])

    # 1 ofr girl and 0 for cat  
    labels = np.array([
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ])

    # 20% of data for test, others for train
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=0,
    )

    # trian the model
    clf = tree.DecisionTreeClassifier()
    clf.fit(X=X_train, y=y_train)

    # test the model
    print(clf.predict(X_test))
    
    # compare the test and predicted results.
    print(clf.score(X=X_test, y=y_test))

    # predict HelloKitty
    HelloKitty = np.array([[1,1,1,1,1,1,1]])
    print(clf.predict(HelloKitty))
    
