import numpy as np
import matplotlib.pyplot as plt

# Generate 2D grid of random values
x = 10*np.random.random(121)
y =  10*np.random.random(121)
# Plot the grid
plt.scatter(x,y,edgecolors='black')


# Set axis labels and title
plt.xlabel(r'$DV_1$')
plt.ylabel(r'$DV_2$')
plt.savefig('SCHEMES/DS_randompoints.pdf')

plt.show()
plt.clf()
# Show the plot

x = np.linspace(0, 10, 11)
y = np.linspace(0, 10, 11)

X, Y = np.meshgrid(x, y)
# Generate 2D grid of random values
x = 10*np.random.random(121)
y =  10*np.random.random(121)
# Plot the grid
plt.scatter(X,Y,c='red',edgecolors='black')


# Set axis labels and title
plt.xlabel(r'$DV_1$')
plt.ylabel(r'$DV_2$')

plt.savefig('SCHEMES/DS_grid.pdf')
plt.show()