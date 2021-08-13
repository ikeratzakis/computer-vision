import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import Counter

n_jobs = -1
# Load digits dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
print('Number of samples:', len(digits.images))
print('Class (digits) distribution:', Counter(digits.target))
images = digits.images.reshape((n_samples, -1))

# Split data 70/30 train/test
X_train, X_test, y_train, y_test = train_test_split(images, digits.target, test_size=0.3, shuffle=False)

# Prepare SVM classifier and grid search method. Dataset looks balanced, so accuracy metric can be used
svc = svm.SVC()
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5, 10], 'gamma': ('scale', 'auto')}
clf = GridSearchCV(svc, parameters, verbose=3)
print('Performing grid search and fitting SVM on data...')
clf.fit(images, digits.target)
print('Best parameters:', clf.best_params_, 'Best score:', clf.best_score_)
cm = metrics.plot_confusion_matrix(clf, X_test, y_test)
cm.figure_.suptitle('Confusion Matrix')

# Plot confusion matrix
print(f"Confusion matrix:\n{cm.confusion_matrix}")

plt.show()

# Now let's run PCA with three different numbers of kept components to see how the results are affected
scaler = MinMaxScaler()
images = scaler.fit_transform(images)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Use the best estimated parameters from the previous training
scores = []
clf = svm.SVC(kernel='rbf', C=5, gamma='scale')
print('Performing PCA and classifying...')
for n_components in [2, 3, 4]:
    pca = PCA(n_components=n_components)
    clf.fit(pca.fit_transform(X_train), y_train)
    scores.append(clf.score(pca.transform(X_test), y_test))

print('PCA accuracies (n_components = 2, 3, 4):', scores)
