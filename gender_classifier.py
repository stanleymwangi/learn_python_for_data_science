from sklearn import tree
from sklearn import svm


# schema [height, weight, shoe_size]
X = [[165, 77, 7], [170, 80, 6], [165, 65, 5], [180, 90, 10], [150, 50, 4], [165, 70, 5], [173, 60, 6],
     [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]
     ]

# list of labels
Y = ["male", "male", "female", "male", "female", "female", "female",
     'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male'
     ]

# initialise the decision tree model
classifier = tree.DecisionTreeClassifier();

# initialise the linear SVC model
svm_classifier = svm.SVC(kernel="linear", C=1.0);

# initialise the KNeighbours model


# train the decision tree model
classifier.fit(X, Y)

# train the SVC model
svm_classifier.fit(X, Y)

# train the KNeighbours model


# serve up a prediction
npc = [[170, 70, 7]]
print(npc)

prediction = classifier.predict(npc)
svm_prediction = svm_classifier.predict(npc)
print("A height of {} cm, weight of {} kg and shoe size {} most likely means you are a:"
       .format(npc[0][0], npc[0][1], npc[0][2]))

print("{}, (decision tree)\n{}, (linear svm)\n".format(prediction, svm_prediction))