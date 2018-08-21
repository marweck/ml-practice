from sklearn import tree, neural_network, svm

clf = tree.DecisionTreeClassifier()
nn = neural_network.MLPClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43], [179, 70, 40]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'female']

clf.fit(X, Y)
pred_tree = clf.predict([[175, 65, 41]])
print(pred_tree)

nn.fit(X, Y)
pred_nn = nn.predict([[175, 65, 41]])
print(pred_nn)
