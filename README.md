# Interactive Visualization of Optimization Algorithms

This repository contains a Streamlit app for visualizing the optimization paths of various algorithms on different test functions. The implemented functions include a custom quadratic function with Gaussian minima, the Rosenbrock function, and the Rastrigin function. The app allows users to interactively modify parameters and observe the impact on the optimization process.

## Features

- Visualization of optimization paths for the following algorithms:
  - Stochastic Gradient Descent (SGD)
  - Momentum
  - RMSProp
  - Adam
- Customizable parameters for learning rate, momentum, decay rate, and number of steps.
- Contour plots of the functions with optimization paths overlaid.
- Support for three test functions:
  - Custom quadratic function with Gaussian minima
  - Rosenbrock function
  - Rastrigin function

## Installation

To run this app, you'll need to have Python installed on your machine. The app was developed using Python 3.8. Follow the steps below to set up the environment and run the app:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/optimization-visualization.git
    cd optimization-visualization
    ```

2. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

4. Open your web browser and go to the URL provided by Streamlit (typically [http://localhost:8501](http://localhost:8501)) to access the app.

## Usage

Once the app is running, you can interact with the following features:

- **Starting Point**: Adjust the starting point (x, y) for the optimization.
- **Select Optimization Methods**: Choose one or more optimization algorithms to visualize.
- **Adjust Parameters**: Modify parameters such as learning rate, momentum, decay rate, and number of steps.
- **Download Optimization Path**: Download the computed optimization path as a CSV file.
