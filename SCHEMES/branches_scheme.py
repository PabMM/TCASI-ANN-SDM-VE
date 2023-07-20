import numpy as np
import matplotlib.pyplot as plt
def divide_vectors(x, y):
    divisions = []
    start = 0
    increasing = None

    for i in range(1, len(y)):
        if increasing is None:
            if y[i] > y[i - 1]:
                increasing = True
            elif y[i] < y[i - 1]:
                increasing = False
        elif increasing and y[i] < y[i - 1]:
            divisions.append((x[start:i], y[start:i]))
            start = i
            increasing = False
        elif not increasing and y[i] > y[i - 1]:
            divisions.append((x[start:i], y[start:i]))
            start = i
            increasing = True

    divisions.append((x[start:], y[start:]))  # Append the last division

    return divisions

def find_transitions(x,y):
    
    increasing = None
    x_transition = [x[0],x[np.argmin(y)]]
    y_transition = [y[0],np.min(y)]

    for i in range(1, len(y)):
        if increasing is None:
            if y[i] > y[i - 1]:
                increasing = True
            elif y[i] < y[i - 1]:
                increasing = False
        elif increasing and y[i] < y[i - 1]:
            x_transition.append(x[i - 1])
            y_transition.append(y[i - 1])
            increasing = False
        elif not increasing and y[i] > y[i - 1]:
            x_transition.append(x[i - 1])
            y_transition.append(y[i - 1])
            increasing = True
    
    
    x_transition = np.array(x_transition)
    y_transition = np.array(y_transition)
    
    
    sort_indices = np.argsort(y_transition)
    return x_transition[sort_indices],y_transition[sort_indices]

# Example usage
x = np.linspace(0,11,100)
y = np.sin(x)*x

result = divide_vectors(x, y)


# Print the divided vectors
tex_leg = []
for i, (x_div, y_div) in enumerate(result):
    plt.plot(x_div,y_div)
    tex_leg.append(f'g{i+1}')
    
    
plt.legend(tex_leg,loc='upper right')

x_trans,y_trans = find_transitions(x,y)
x_fin = np.max(x)
aux = y_trans[0]
i = 0
for x_t,y_t  in zip(x_trans,y_trans):
    plt.plot([0,x_fin],[y_t,y_t],'--',c='black')  
    plt.scatter(x_t,y_t,c='black')
    
    x_start = 0
    y_start = y_t
    y_end = aux
    
    text = f'I{i}'
    if i != 0:
    # Dibujar la doble flecha vertical
        plt.annotate('', xy=(x_start, y_start), xytext=(x_start, y_end),
                    arrowprops=dict(arrowstyle='<->'))

        # Agregar el texto a la izquierda de la flecha
        plt.annotate(text, xy=(x_start+0.25, (y_start+y_end)/2),
                    xytext=(-10, 0),
                    textcoords='offset points',
                    ha='right', va='center')
    aux = y_t
    i +=1

plt.savefig('SCHEMES/branches.pdf')
plt.pause(3)