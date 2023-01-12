import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def plot_loss_curve(index_list, loss_se, loss_res):
    plt.title("loss curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(index_list, loss_se, marker='s', markersize=3)
    plt.plot(index_list, loss_res, marker='*', markersize=3)

    plt.legend(['SENet', 'ResNet'])

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


def plot_accuracy(index_list, accuracy_list_train, accuracy_list_test, name):
    plt.title("Accuracy of " + name)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(index_list, accuracy_list_train, marker='s', markersize=3)
    plt.plot(index_list, accuracy_list_test, marker='*', markersize=3)

    max_a = 0
    max_b = 0
    for a, b in zip(index_list, accuracy_list_test):
        if b > max_b:
            max_a = a
            max_b = b

    plt.text(max_a, max_b, max_b, ha='center', va='bottom', fontsize=10)

    plt.legend(['Train', 'Test'])

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


def plot_correlation_matrix():
    matrix_se = [[88, 1, 2, 3, 5, 5, 0, 1, 0, 3, ], [10, 62, 8, 1, 3, 0, 0, 0, 0, 9, ],
                 [6, 3, 77, 2, 5, 5, 0, 0, 0, 3, ], [8, 1, 11, 15, 7, 3, 0, 1, 0, 3, ],
                 [1, 4, 6, 2, 23, 0, 0, 0, 1, 1, ], [6, 1, 7, 5, 4, 16, 1, 5, 0, 0, ],
                 [0, 0, 0, 0, 0, 1, 33, 5, 0, 1, ], [0, 0, 1, 1, 0, 8, 5, 18, 0, 0, ],
                 [0, 1, 3, 0, 1, 0, 0, 0, 35, 0, ], [3, 4, 2, 2, 1, 0, 0, 0, 2, 15, ]]
    matrix_res = [[67, 9, 8, 3, 5, 7, 0, 4, 0, 5, ], [4, 67, 13, 0, 2, 0, 0, 1, 2, 4, ],
                  [5, 10, 76, 2, 6, 0, 0, 1, 1, 0, ], [20, 6, 6, 4, 5, 3, 1, 4, 0, 0, ],
                  [2, 5, 9, 2, 18, 1, 0, 0, 1, 0, ], [3, 2, 5, 3, 3, 15, 2, 9, 1, 2, ],
                  [0, 0, 2, 0, 0, 0, 31, 7, 0, 0, ], [3, 0, 0, 0, 1, 2, 5, 22, 0, 0, ],
                  [0, 2, 2, 0, 1, 0, 1, 0, 34, 0, ], [3, 13, 2, 0, 0, 0, 1, 2, 3, 5, ]]
    matrix_se = np.array(matrix_se).astype(np.float32)
    matrix_res = np.array(matrix_res).astype(np.float32)

    for i in range(10):
        # print(sum(matrix_se[i]))
        matrix_se[i] = matrix_se[i] / sum(matrix_se[i])
        matrix_res[i] = matrix_res[i] / sum(matrix_res[i])

    cmap = sns.diverging_palette(220, 10, s=75, l=40, n=9, center="light", as_cmap=True)
    sns.heatmap(matrix_se, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, cbar=True)
    plt.show()

    sns.heatmap(matrix_res, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, cbar=True)
    plt.show()


if __name__ == "__main__":
    path_net2 = "../net2.log"

    loss_list_se = np.zeros(20)
    loss_list_res = np.zeros(20)
    index = list(range(1, 21))

    train_accuracy_se = np.zeros(20)
    train_accuracy_res = np.zeros(20)
    test_accuracy_se = np.zeros(20)
    test_accuracy_res = np.zeros(20)

    record_number = 1
    epoch_number = 0
    cur_loss_list = loss_list_se
    cur_train_accuracy = train_accuracy_se
    cur_test_accuracy = test_accuracy_se
    with open(path_net2) as f:
        for line in f:
            if line.count("loss"):
                line = line.replace(" ", "").replace("[", "").replace("]", "").replace("loss:", ",").replace("\n", "")
                line_list = line.split(",")
                cur_number = int(line_list[0])
                if cur_number != record_number:
                    if cur_number < record_number:
                        cur_loss_list = loss_list_res
                    record_number = cur_number
                cur_loss_list[record_number - 1] += float(line_list[2]) / 5
            elif line.count("Accuracy:"):
                line = line.replace(" ", "").replace("Epoch:", "").replace("Loss:", ""). \
                    replace("Accuracy:", "").replace("Totalitems:", "")
                line_list = line.split(",")
                if len(line_list) == 3:
                    if int(line_list[0]) < epoch_number:
                        cur_train_accuracy = train_accuracy_res
                        cur_test_accuracy = test_accuracy_res
                    epoch_number = int(line_list[0])
                    cur_train_accuracy[epoch_number] = float(line_list[2])
                else:
                    cur_test_accuracy[epoch_number] = float(line_list[0])

    loss_list_se = loss_list_se / 6
    loss_list_res = loss_list_res / 6

    plot_loss_curve(index, loss_list_se, loss_list_res)

    plot_accuracy(index, train_accuracy_se, test_accuracy_se, "SENet")

    plot_accuracy(index, train_accuracy_res, test_accuracy_res, "ResNet")

    plot_correlation_matrix()
