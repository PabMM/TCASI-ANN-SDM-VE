import random
import numpy as np
import matplotlib.pyplot as plt

def random_factor(value, range_val=0.05):
    alpha = random.uniform(-range_val, range_val)
    return value * (1 + alpha)

random_factor_vec = np.vectorize(random_factor)

points = np.array([[2,1],
                   [2,9],
                   [8,1],
                   [9,8],
                   [5,5]])

x = [0, 10]
y = [0, 10]


plt.plot(x, y,color='red')

for i in range(300):
    points_var = random_factor_vec(points)
    plt.scatter(points_var[:,0],points_var[:,1],color='green',s=1)

plt.scatter(points[:,0],points[:,1],edgecolors='black')
#plt.axis('equal')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel(r'$S_1$')
plt.ylabel(r'$S_2$')
plt.title(r'Point cloud of Specifications. $\mathrm{range} = 5\%$')
plt.savefig('SCHEMES\pc_specifications.pdf')
