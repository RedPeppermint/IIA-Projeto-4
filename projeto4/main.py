import numpy
from sklearn.model_selection import *
from sklearn.neighbors import *
from sklearn.tree import *

from utilsAA import *


def write_to_file(file_name, first, second, third, forth):
    f = open(file_name, "w")
    for i in range(len(third)):
        for j in range(len(first)):
            f.write(str(first[j]) + ": " + str(second[i][j]) + ("\n" if j == len(first) - 1 else "; "))
        f.write(str(third[i]) + " +/- " + str(forth[i]) + '\n\n')


def my_sort(first, second, third):
    return zip(*sorted(zip(first, second, third)))


def encode_all(data):
    # encode gender
    data[:, 0] = encode_feature(data[:, 0])
    # encode customer type
    data[:, 1] = encode_feature(data[:, 1])
    # encode type of travel
    data[:, 3] = encode_feature((data[:, 3]))
    # encode class
    data[:, 4] = encode_feature(data[:, 4])
    return data


def remove_client_id(data):
    data = data[:, 1:]
    return data


# prediction, original
def percentage_right(pre, ori):
    acc = 0
    for i in range(1, len(pre)):
        if pre[i] == ori[i]:
            acc += 1
    print(acc)
    print(len(pre))
    print(len(ori))
    return 100 * acc / len(pre)


raw_data = load_data("airline.csv", False)

raw_data = (remove_client_id(raw_data[0]),) + raw_data[1:]
features = encode_all(raw_data[0])
klass = raw_data[1]
feature_names = raw_data[2][1:]
klass_names = raw_data[3]

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1)
dtc.fit(features, klass)
plt.figure(figsize=[50, 50])
plot_tree(dtc, feature_names=feature_names, class_names=klass_names, filled=True)
# plt.show()
# plot_frontiers(features, klass, dtc, feature_names, klass_names, nclasses=2)

raw_data = load_data("test.csv", True)
raw_data = (remove_client_id(raw_data[0]),) + raw_data[1:]
new_features = encode_all(raw_data[0])

prediction = dtc.predict(new_features)
# print(prediction[:5])
print(percentage_right(prediction, klass))
prediction = dtc.predict(features)
print(percentage_right(prediction, klass))

print()
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(features, klass)
prediction = clf.predict(new_features)
print(percentage_right(prediction, klass))
prediction = clf.predict(features)
print(percentage_right(prediction, klass))

print("\nSCORES")
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1)
scores1 = cross_val_score(dtc, X=features, y=klass, cv=10, n_jobs=-1)
print(scores1)
print(numpy.mean(scores1), numpy.std(scores1))
print()
clf = KNeighborsClassifier(n_neighbors=1)
scores2 = cross_val_score(clf, X=features, y=klass, cv=10, n_jobs=-1)
print(scores2)
print(numpy.mean(scores2), numpy.std(scores2))

print("\nMELHORES VALORES PARA DECISION TREE")
criteria = ["min_samples_split", "min_samples_leaf", "max_depth_tree"]
values = []
means = []
deviations = []
for min_samples_split in range(2, 20):
    for min_samples_leaf in range(1, 20):
        for max_depth_tree in range(0, 20):
            dtc = DecisionTreeClassifier(criterion='entropy',
                                         max_depth=None if max_depth_tree == 0 else max_depth_tree,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf)
            scores = cross_val_score(dtc, X=features, y=klass, cv=10, n_jobs=-1)
            values += [[min_samples_split, min_samples_leaf, max_depth_tree]]
            means += [numpy.mean(scores)]
            deviations += [numpy.std(scores)]

means, values, deviations = my_sort(means, values, deviations)
write_to_file("tree.txt", criteria, values, means, deviations)

print("MELHORES VALORES PARA K NEIGHBORS")
criteria = ["n_neighbors"]
values = []
means = []
deviations = []

for n_neighbors in range(1, 20):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(clf, X=features, y=klass, cv=10, n_jobs=-1)
    values += [[n_neighbors]]
    means += [numpy.mean(scores)]
    deviations += [numpy.std(scores)]

means, values, deviations = my_sort(means, values, deviations)
write_to_file("knei.txt", criteria, values, means, deviations)

# best: tree, min_sample_split ?? qq, min_sample_leaf 1, max_depth_tree >= 6
dtc = DecisionTreeClassifier(criterion='entropy',
                             max_depth=15,
                             min_samples_split=7,
                             min_samples_leaf=1)
dtc.fit(features, klass)
scores = cross_val_score(dtc, X=features, y=klass, cv=10, n_jobs=-1)
print(numpy.mean(scores))
prediction = dtc.predict(new_features)
save_data("results.csv", prediction)
#10 3 1