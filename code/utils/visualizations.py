import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatterplot(pos):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    #for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    #    xs = randrange(n, 23, 32)
    #    ys = randrange(n, 0, 100)
    #    zs = randrange(n, zlow, zhigh)
    #    ax.scatter(xs, ys, zs, marker=m)
    ax.scatter(pos[:,0],pos[:,1],pos[:,2])
    #ax.axis('equal')
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    plt.axis('off')
    plt.show()
