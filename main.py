#Static SOM
# import numpy as np
# from minisom import MiniSom
#
# # Generate sample data
# data = np.random.rand(100, 5)  # Assuming 100 data points with 5 features
#
# # Define SOM parameters
# map_size = (10, 10)  # Size of the SOM grid
# input_len = data.shape[1]  # Length of input vector (number of features)
# sigma = 1.0  # Initial neighborhood radius
# learning_rate = 0.5  # Initial learning rate
# num_epochs = 100  # Number of training epochs
#
# # Initialize and train the SOM
# som = MiniSom(*map_size, input_len, sigma=sigma, learning_rate=learning_rate)
# som.train_random(data, num_epochs)
#
# # Get the SOM grid coordinates for each data point
# grid_coordinates = som.win_map(data)
#
# # Print the grid coordinates and corresponding data points
# for c in grid_coordinates:
#     print(c, grid_coordinates[c])
#
# # Visualize the SOM
# import matplotlib.pyplot as plt
#
# # Get the coordinates of each grid cell in the SOM
# xx, yy = np.meshgrid(range(map_size[0]), range(map_size[1]))
# xx = xx.ravel()
# yy = yy.ravel()
#
# # Compute the mean distance between the weight vectors and their neighbors
# mean_distances = som.distance_map()
#
# # Plot the SOM grid
# plt.figure(figsize=(map_size[0], map_size[1]))
# plt.pcolor(xx.reshape(map_size), yy.reshape(map_size), mean_distances, cmap='bone_r')
# plt.colorbar()
#
# # Mark the grid cells with the data points
# for i, x in enumerate(data):
#     w = som.winner(x)
#     plt.plot(w[0]+0.5, w[1]+0.5, 'ro', marker='o', markersize=5, markeredgecolor='k')
#
# plt.title('Self-Organizing Map')
# plt.show()

#DyNAMIC OUTPUT
# import numpy as np
# from minisom import MiniSom
# import matplotlib.pyplot as plt
#
# # Generate sample data
# data = np.random.rand(100, 5)  # Assuming 100 data points with 5 features
#
# # Function to get a valid float input from the user
# def get_valid_float_input(prompt):
#     while True:
#         try:
#             value = float(input(prompt))
#             return value
#         except ValueError:
#             print("Invalid input. Please enter a numerical value.")
#
# # Sidebar for user inputs
# map_size = tuple(int(x) for x in input("Enter the SOM grid size (e.g., 10 10): ").split())
# sigma = get_valid_float_input("Enter the initial neighborhood radius (sigma): ")
# learning_rate = get_valid_float_input("Enter the initial learning rate: ")
# num_epochs = int(get_valid_float_input("Enter the number of training epochs: "))
#
# # Define SOM parameters
# input_len = data.shape[1]  # Length of input vector (number of features)
#
# # Initialize and train the SOM
# som = MiniSom(*map_size, input_len, sigma=sigma, learning_rate=learning_rate)
# som.train_random(data, num_epochs)
#
# # Get the SOM grid coordinates for each data point
# grid_coordinates = som.win_map(data)
#
# # Print the grid coordinates and corresponding data points
# for c in grid_coordinates:
#     print(c, grid_coordinates[c])
#
# # Get the coordinates of each grid cell in the SOM
# xx, yy = np.meshgrid(range(map_size[0]), range(map_size[1]))
# xx = xx.ravel()
# yy = yy.ravel()
#
# # Compute the mean distance between the weight vectors and their neighbors
# mean_distances = som.distance_map()
#
# # Plot the SOM grid
# plt.figure(figsize=(map_size[0], map_size[1]))
# plt.pcolor(xx.reshape(map_size), yy.reshape(map_size), mean_distances, cmap='bone_r')
# plt.colorbar()
#
# # Mark the grid cells with the data points
# for i, x in enumerate(data):
#     w = som.winner(x)
#     plt.plot(w[0]+0.5, w[1]+0.5, 'ro', marker='o', markersize=5, markeredgecolor='k')
#
# plt.title('Self-Organizing Map')
# plt.show()

#Command line live SOM
# import numpy as np
# from minisom import MiniSom
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# # Generate sample data
# data = np.random.rand(100, 5)  # Assuming 100 data points with 5 features
#
# # Function to get a valid float input from the user
# def get_valid_float_input(prompt):
#     while True:
#         try:
#             value = float(input(prompt))
#             return value
#         except ValueError:
#             print("Invalid input. Please enter a numerical value.")
#
# # Sidebar for user inputs
# map_size = tuple(int(x) for x in input("Enter the SOM grid size (e.g., 10 10): ").split())
# sigma = get_valid_float_input("Enter the initial neighborhood radius (sigma): ")
# learning_rate = get_valid_float_input("Enter the initial learning rate: ")
# num_epochs = int(get_valid_float_input("Enter the number of training epochs: "))
#
# # Define SOM parameters
# input_len = data.shape[1]  # Length of input vector (number of features)
#
# # Initialize the SOM
# som = MiniSom(*map_size, input_len, sigma=sigma, learning_rate=learning_rate)
#
# # Create a figure and axis for the plot
# fig, ax = plt.subplots()
#
# # Update function for the animation
# def update(frame):
#     # Train the SOM for one epoch
#     som.train_random(data, 1)
#
#     # Clear the current plot
#     ax.clear()
#
#     # Get the SOM grid coordinates for each data point
#     grid_coordinates = som.win_map(data)
#
#     # Get the coordinates of each grid cell in the SOM
#     xx, yy = np.meshgrid(range(map_size[0]), range(map_size[1]))
#     xx = xx.ravel()
#     yy = yy.ravel()
#
#     # Compute the mean distance between the weight vectors and their neighbors
#     mean_distances = som.distance_map()
#
#     # Plot the SOM grid
#     ax.pcolor(xx.reshape(map_size), yy.reshape(map_size), mean_distances, cmap='bone_r')
#     ax.set_title(f'Self-Organizing Map - Epoch: {frame + 1}/{num_epochs}')
#
#     # Mark the grid cells with the data points
#     for i, x in enumerate(data):
#         w = som.winner(x)
#         ax.plot(w[0]+0.5, w[1]+0.5, 'ro', marker='o', markersize=5, markeredgecolor='k')
#
# # Create the animation
# animation = FuncAnimation(fig, update, frames=num_epochs, interval=200, repeat=False)
#
# # Display the animation
# plt.show()


# with Streamlit front end (without live arrangement )

import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import streamlit as st

# Generate sample data
data = np.random.rand(100, 5)  # Assuming 100 data points with 5 features

# Function to get a valid float input from the user
def get_valid_float_input(prompt, key):
    while True:
        try:
            value = float(prompt)
            return value
        except ValueError:
            prompt = st.sidebar.text_input("Invalid input. Please enter a numerical value.", key=key)

# Streamlit sidebar for user inputs
st.sidebar.title("SOM Parameters")
map_size = tuple(int(x) for x in st.sidebar.text_input("Enter the SOM grid size (e.g., 10 10):", key="map_size").split())
sigma = get_valid_float_input(st.sidebar.text_input("Enter the initial neighborhood radius (sigma):", key="sigma"), key="sigma")
learning_rate = get_valid_float_input(st.sidebar.text_input("Enter the initial learning rate:", key="learning_rate"), key="learning_rate")
num_epochs = int(get_valid_float_input(st.sidebar.text_input("Enter the number of training epochs:", key="num_epochs"), key="num_epochs"))

# Define SOM parameters
input_len = data.shape[1]  # Length of input vector (number of features)

# Initialize the SOM
som = MiniSom(map_size[0], map_size[1], input_len, sigma=sigma, learning_rate=learning_rate)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Update function for the plot
def update_plot(epoch):
    # Clear the current plot
    ax.clear()

    # Train the SOM for one epoch
    som.train_random(data, 1)

    # Get the SOM grid coordinates for each data point
    grid_coordinates = som.win_map(data)

    # Get the coordinates of each grid cell in the SOM
    xx, yy = np.meshgrid(range(map_size[0]), range(map_size[1]))
    xx = xx.ravel()
    yy = yy.ravel()

    # Compute the mean distance between the weight vectors and their neighbors
    mean_distances = som.distance_map()

    # Plot the SOM grid
    ax.pcolor(xx.reshape(map_size), yy.reshape(map_size), mean_distances, cmap='bone_r')
    ax.set_title(f'Self-Organizing Map - Epoch: {epoch}/{num_epochs}')

    # Mark the grid cells with the data points
    for i, x in enumerate(data):
        w = som.winner(x)
        ax.plot(w[0]+0.5, w[1]+0.5, 'ro', marker='o', markersize=5, markeredgecolor='k')

    # Refresh the plot
    plt.pause(0.001)

# Train the SOM for the specified number of epochs
for epoch in range(num_epochs):
    update_plot(epoch + 1)

# Display the plot within Streamlit using st.pyplot
st.pyplot(fig)

# # with Streamlit front end (with live arrangement)

import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import streamlit as st

# Generate sample data
data = np.random.rand(100, 5)  # Assuming 100 data points with 5 features

# Function to get a valid float input from the user
def get_valid_float_input(prompt, key):
    while True:
        try:
            value = float(prompt)
            return value
        except ValueError:
            prompt = st.sidebar.text_input("Invalid input. Please enter a numerical value.", key=key)

# Streamlit sidebar for user inputs
st.sidebar.title("SOM Parameters")
map_size = tuple(int(x) for x in st.sidebar.text_input("Enter the SOM grid size (e.g., 10 10):", key="map_size").split())
sigma = get_valid_float_input(st.sidebar.text_input("Enter the initial neighborhood radius (sigma):", key="sigma"), key="sigma")
learning_rate = get_valid_float_input(st.sidebar.text_input("Enter the initial learning rate:", key="learning_rate"), key="learning_rate")
num_epochs = int(get_valid_float_input(st.sidebar.text_input("Enter the number of training epochs:", key="num_epochs"), key="num_epochs"))

# Define SOM parameters
input_len = data.shape[1]  # Length of input vector (number of features)

# Initialize the SOM
som = MiniSom(map_size[0], map_size[1], input_len, sigma=sigma, learning_rate=learning_rate)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Update function for the plot
def update_plot(epoch):
    # Clear the current plot
    ax.clear()

    # Train the SOM for one epoch
    som.train_random(data, 1)

    # Get the SOM grid coordinates for each data point
    grid_coordinates = som.win_map(data)

    # Get the coordinates of each grid cell in the SOM
    xx, yy = np.meshgrid(range(map_size[0]), range(map_size[1]))
    xx = xx.ravel()
    yy = yy.ravel()

    # Compute the mean distance between the weight vectors and their neighbors
    mean_distances = som.distance_map()

    # Plot the SOM grid
    ax.pcolor(xx.reshape(map_size), yy.reshape(map_size), mean_distances, cmap='bone_r')
    ax.set_title(f'Self-Organizing Map - Epoch: {epoch}/{num_epochs}')

    # Mark the grid cells with the data points
    for i, x in enumerate(data):
        w = som.winner(x)
        ax.plot(w[0]+0.5, w[1]+0.5, 'ro', marker='o', markersize=5, markeredgecolor='k')

    # Refresh the plot
    plt.pause(0.001)

# Train the SOM for the specified number of epochs
for epoch in range(num_epochs):
    update_plot(epoch + 1)

# Display the plot within Streamlit using st.pyplot
st.pyplot(fig)


