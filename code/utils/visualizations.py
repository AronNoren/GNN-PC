import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatterplot(pos):
    '''
    simply plots the given data.pos in 3D
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:,0],pos[:,1],pos[:,2])
    plt.show()
