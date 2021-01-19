import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
from sklearn.metrics import confusion_matrix
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
def scatterplot(pos,label = None):
    '''
    simply plots the given data.pos in 3D
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:,0],pos[:,1],pos[:,2])
    if label is not None:
        ax.text2D(0.05, 0.95, label, transform=ax.transAxes)
    plt.show()

def accuracies(acc):
    N = range(len(acc))
    plt.plot(N,acc)
    plt.ylim(0.2,0.95)
    plt.xlabel('epochs',fontdict = font)
    plt.ylabel('validation accuracy',fontdict = font)
    plt.savefig('images/image.png')
def heatplot(preds,correct,epoch):
    f, ax = plt.subplots(figsize=(20, 20))
    conf = confusion_matrix(preds,correct)
    sn.heatmap(conf, annot=True)
    plt.savefig('images/confusion'+str(epoch)+'.png')
    plt.clf()
    plt.xlabel('True',fontdict = font)
    plt.ylabel('Prediction',fontdict = font)
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()