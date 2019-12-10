import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
plt.switch_backend('agg')


def save_plot(iters, loss, acc, title):
    # create two legends
    plt.plot(iters, loss, '-b', label='loss')
    plt.plot(iters, acc, '-r', label='accuracy')

    # set up details
    plt.xlabel('# iter')
    plt.legend(loc='upper left')
    plt.title(title)

    plt.savefig('loss_acc.png')

    plt.show()
