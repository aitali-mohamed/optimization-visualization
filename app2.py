import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Define the functions and their gradients
def f(x, y):
    return -2 * np.exp(-((x - 1) ** 2 + y ** 2) / 0.2) - 3 * np.exp(-((x + 1) ** 2 + y ** 2) / 0.2) + x ** 2 + y ** 2

def grad_f(x, y):
    h = 1e-7
    grad_x = (f(x + h, y) - f(x, y)) / h
    grad_y = (f(x, y + h) - f(x, y)) / h
    return np.array([grad_x, grad_y])

def rosenbrock(x, y):
    return np.power(1 - x, 2) + 2 * np.power(y - x * x, 2)

def grad_rosenbrock(x, y):
    grad_x = -2 * (1 - x) - 4 * (y - x * x) * x
    grad_y = 2 * (y - x * x)
    return np.array([grad_x, grad_y])

def rastrigin(x, y):
    return 0.2 * (np.sin(10 * x - np.pi / 2) + np.sin(10 * y - np.pi / 2)) + np.power(x, 2) + np.power(y, 2)

def grad_rastrigin(x, y):
    h = 1e-7
    grad_x = (rastrigin(x + h, y) - rastrigin(x, y)) / h
    grad_y = (rastrigin(x, y + h) - rastrigin(x, y)) / h
    return np.array([grad_x, grad_y])

# Optimization methods
def sgd(grad_func, x0, y0, learning_rate, num_steps):
    history = [(x0, y0)]
    for _ in range(num_steps):
        gradient = grad_func(x0, y0)
        x0 -= learning_rate * gradient[0]
        y0 -= learning_rate * gradient[1]
        history.append((x0, y0))
    return history

def momentum(grad_func, x0, y0, learning_rate, num_steps, moment):
    v_x, v_y = 0, 0
    history = [(x0, y0)]
    for _ in range(num_steps):
        gradient = grad_func(x0, y0)
        v_x = moment * v_x - learning_rate * gradient[0]
        v_y = moment * v_y - learning_rate * gradient[1]
        x0 += v_x
        y0 += v_y
        history.append((x0, y0))
    return history

def rmsprop(grad_func, x0, y0, learning_rate, num_steps, decay_rate, eps):
    cache_x, cache_y = 0, 0
    history = [(x0, y0)]
    for _ in range(num_steps):
        gradient = grad_func(x0, y0)
        cache_x = decay_rate * cache_x + (1 - decay_rate) * gradient[0] ** 2
        cache_y = decay_rate * cache_y + (1 - decay_rate) * gradient[1] ** 2
        x0 -= learning_rate * gradient[0] / (np.sqrt(cache_x) + eps)
        y0 -= learning_rate * gradient[1] / (np.sqrt(cache_y) + eps)
        history.append((x0, y0))
    return history

def adam(grad_func, x0, y0, learning_rate, num_steps, beta_1, beta_2, eps):
    m_x, m_y = 0, 0
    v_x, v_y = 0, 0
    history = [(x0, y0)]
    for _ in range(num_steps):
        gradient = grad_func(x0, y0)
        m_x = beta_1 * m_x + (1 - beta_1) * gradient[0]
        m_y = beta_1 * m_y + (1 - beta_1) * gradient[1]
        v_x = beta_2 * v_x + (1 - beta_2) * gradient[0] ** 2
        v_y = beta_2 * v_y + (1 - beta_2) * gradient[1] ** 2
        x0 -= learning_rate * m_x / (np.sqrt(v_x) + eps)
        y0 -= learning_rate * m_y / (np.sqrt(v_y) + eps)
        history.append((x0, y0))
    return history

# Streamlit interface
st.title("Optimization Paths Visualization")
x0, y0 = st.slider("Starting Point (x, y)", -2.0, 2.0, (-1.0, -1.0), 0.1)

functions = ['Quadratic-Gaussian', 'Rosenbrock', 'Rastrigin']
selected_function = st.radio("Select Function", functions)

methods = ['SGD', 'Momentum', 'RMSProp', 'Adam']
selected_methods = st.multiselect("Select Optimization Methods", methods, default=methods)

learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
moment = st.slider("Momentum", 0.0, 0.99, 0.8, 0.01)
decay_rate = st.slider("Decay Rate (RMSProp)", 0.0, 0.99, 0.99, 0.01)
num_steps = st.slider("Number of Steps", 10, 1000, 50, 10)

# Select function and gradient based on user choice
if selected_function == 'Quadratic-Gaussian':
    func = f
    grad_func = grad_f
    x_domain = np.linspace(-2, 2, 100)
    y_domain = np.linspace(-2, 2, 100)
elif selected_function == 'Rosenbrock':
    func = rosenbrock
    grad_func = grad_rosenbrock
    x_domain = np.linspace(-1.5, 1.5, 100)
    y_domain = np.linspace(-1, 3, 100)
elif selected_function == 'Rastrigin':
    func = rastrigin
    grad_func = grad_rastrigin
    x_domain = np.linspace(-2, 2, 100)
    y_domain = np.linspace(-2, 2, 100)


X, Y = np.meshgrid(x_domain, y_domain)
Z = func(X, Y)

fig = go.Figure(data=go.Contour(z=Z, x=x_domain, y=y_domain, colorscale='YlGnBu'))

# Plot optimization paths
for method in selected_methods:
    if method == 'SGD':
        path = sgd(grad_func, x0, y0, learning_rate, num_steps)
    elif method == 'Momentum':
        path = momentum(grad_func, x0, y0, learning_rate, num_steps, moment)
    elif method == 'RMSProp':
        path = rmsprop(grad_func, x0, y0, learning_rate, num_steps, decay_rate, 1e-6)
    elif method == 'Adam':
        path = adam(grad_func, x0, y0, learning_rate, num_steps, 0.9, 0.999, 1e-6)

    path = np.array(path)
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines+markers', name=method))

fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers', name='Start Point', marker=dict(color='black', size=10)))

fig.update_layout(
    title="Contour Plot with Optimization Paths",
    xaxis_title="x",
    yaxis_title="y",
    legend=dict(
        orientation="h", # horizontal orientation
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(fig)

# Download button
path_df = pd.DataFrame(path, columns=['x', 'y'])
csv = path_df.to_csv(index=False)
st.download_button("Download Optimization Path", csv, "optimization_path.csv")

# Conditional markdown content
if selected_function == 'Quadratic-Gaussian':
    st.markdown("""
    ### Function Definition
    The function used for the optimization visualization is:
    """)
    latext = r'''
    $$ f(x,y)=x^2+y^2 - 2 e^{-(x-1)^2+y^2/0.2} - 3 e^{-(x+1)^2+y^2/0.2} $$
    '''
    st.write(latext)
    st.markdown("""
    It is a quadratic "bowl" with two gaussians creating minima at (1, 0) and (-1, 0) respectively. The size of these minima is controlled by the a and b parameters.

    ### Observations
    1. **Different Minima**: Starting from the same point, different algorithms will converge to different minima. Often, SGD and SGD with momentum will converge to the poorer minimum (the one on the right) while RMSProp and Adam will converge to the global minimum. For this particular function, Adam is the algorithm that converges to the global minimum from the most initializations.
    """)
    st.image('optim_viz_only_adam.png', caption='Only Adam (in green) converges to the global minimum.')
    st.markdown("""
    2. **The effects of momentum**: Augmenting SGD with momentum has many advantages and often works better than the other standard algorithms for an appropriately chosen learning rate. However, with the wrong learning rate, SGD with momentum can overshoot minima and this often leads to a spiraling pattern around the minimum.
    """)
    st.image('optim_viz_momentum.png', caption='SGD with momentum spiraling towards the minimum.')
    st.markdown("""
    3. **Standard SGD does not get you far**: SGD without momentum consistently performs the worst. The learning rate for SGD on the visualization is set to be artificially high (an order of magnitude higher than the other algorithms) in order for the optimization to converge in a reasonable amount of time.
    """)
elif selected_function == 'Rosenbrock':
    st.markdown("""
    ### Function Definition
    #### Rosenbrock Function:
    """)
    latext = r'''
    $$ f(x,y) = (1 - x)^2 + 2(y - x^2)^2 $$
    '''
    st.write(latext)
    st.markdown("""
    The Rosenbrock function has a single global minimum inside a parabolic shaped valley. Most algorithms rapidly converge to this valley, but it is typically difficult to converge to the global minimum within this valley.
    """)
    st.image('optim_viz_rosenbrock.gif', caption='All algorithms find the global minimum but through very different paths.')
    st.markdown("""
    While all algorithms converge to the optimum, the adaptive and non-adaptive optimization algorithms approach the minimum through different paths. In higher dimensional problems, like in deep learning, different optimization algorithms will likely explore very different areas of parameter space.
    """)
elif selected_function == 'Rastrigin':
    st.markdown("""
    #### Rastrigin Function:
    The function used for the optimization visualization is:
    """)
    latext = r'''
    $$ f(x,y)=0.2 (\sin(10x - \frac{\pi}{2}) + \sin(10y - \frac{\pi}{2})) + x^2 + y^2 $$
    '''
    st.write(latext)
    st.markdown("""
    A Rastrigin function is a quadratic bowl overlayed with a grid of sine bumps creating a large number of local minima.""")

    st.image('optim_viz_rastrigin.gif', caption='SGD with momentum reaches the global optimum while all other algorithms get stuck in the same local minimum.')
    st.markdown("""
    In this example, SGD with momentum outperforms all other algorithms using the default parameter settings. The speed built up from the momentum allows it to power through the sine bumps and converge to the global minimum when other algorithms don’t. Of course, this would not necessarily be the case if the sine bumps had been scaled or spaced differently. Indeed, on the first function in this post, Adam performed the best while SGD with momentum performs the best on the Rastrigin function. This shows that there is no single algorithm that will perform the best on all functions, even in simple 2D cases.
    """)