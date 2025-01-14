### CS106EA Exploring Artificial Intelligence
#   Patrick Young, Stanford University
#
#   This code is mean to be loaded by Colab Notebook
#   It's purpose is to hide most of the implementation code
#   making the Colab Notebook easier to follow.

### IMPORTS AND BASIC SETUP

import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, HTML, Label, Dropdown, Button, Output
from IPython.display import clear_output

import sys
import io
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim

import matplotlib.pyplot as plt

import numpy as np
import time

from collections import OrderedDict

from dataclasses import dataclass, field
from typing import Type, Callable, Tuple, Dict
import math
import random

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

### SETUP CSS STYLES
#   These will only be used if you if you explicitly include
#   them in a given "display()" call

html_style = HTML(
    value="""
<style>
.control-major-label {
    font-size: 1.2em;
    font-weight: bold;
}
.control-label {
    font-size: 1em;
    font-weight: bold;
}
.control-minor-label {
    font-size: 0.9em;
}
.widget-checkbox {
    width: auto !important;  /* Adjust this if necessary */
    /*border: 1px solid blue;*/ /* To see the actual space taken by the checkbox container */
}
.widget-checkbox > label {
    margin: 0 !important;
    padding: 0 !important;
    width: auto !important;
    /*border: 1px solid red;*/ /* To see the space taken by the label */
}
.widget-checkbox input[type="checkbox"] {
    margin: 0 !important;
}
.widget-inline-hbox .widget-label {
    flex: 0 0 auto !important;
}
.widget-inline-hbox {
    align-items: center; /* Align items vertically in the center */
    min-width: 0; /* Helps prevent flex containers from growing too large */
}
.code {
    font-family: 'Courier New', Courier, monospace;
    font-weight: bold;
    line-height: 0.5;
    margin: 0;
    padding: 0;
}
</style>

    """
)

# DEFINE NETWORKS

class DynamicNetwork(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, activation_fn, input_size = 1, output_size = 1):
        """
        Parameters:
        - hidden_size (int): Number of nodes in each hidden layer.
        - num_hidden_layers (int): Number of hidden layers.
        - activation_fn (nn.Module): PyTorch activation function (e.g., nn.ReLU, nn.Tanh).
        - input_size (int): Number of inputs
        - output_size (int): Number of outputs
        """
        super().__init__()

        layers = []

        # initial hidden layer 1 to hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        if activation_fn is not None:
            layers.append(activation_fn())

        # add additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation_fn is not None:
                layers.append(activation_fn())

        # output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # place layers into sequential block
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def create_random_networks(num_layers, num_nodes_per_layer, activation_func=nn.Tanh):
    return (DynamicNetwork(num_nodes_per_layer, num_layers, activation_func),
                DynamicNetwork(num_nodes_per_layer, num_layers, None))

# BASIC PRESET ARCHITECTURE VERSION

basic_go_button = Button(
    description = "Randomize Parameters and Graph",
    layout = Layout(width="300px"),
    button_style = 'info'
)

def create_basic_networks_and_graph(_):
    global basic_activation, basic_no_activation
    basic_activation, basic_no_activation = create_random_networks(
        10,
        10,
        nn.Tanh
    )

    graph_networks(basic_graph_output, basic_activation, basic_no_activation)

basic_go_button.on_click(create_basic_networks_and_graph)

basic_graph_output = Output()

def display_basic_ui():
    display(html_style, 
            VBox([
                basic_go_button,
                basic_graph_output
            ]))
    create_basic_networks_and_graph(None)

# PRINT BASIC ARCHITECTURES

basic_print_output = Output()    

def display_print_basic_architectures():
    display(basic_print_output)

    with basic_print_output:
        clear_output(wait=True)
        display(HTML("<h2>Network with Activation Functions</h2>"))
        print(basic_activation)
        display(HTML("<h2>Network <u>without</u> Activation Functions</h2>"))
        print(basic_no_activation)

# UI FOR USER-DEFINED NETWORKS

ui_activation_label = HTML(
    "<b>Activation Function</b>:",
    layout = Layout(width="125px")
)

activation_dict = {
    "Tanh": nn.Tanh,
    # "Sigmoid": nn.Sigmoid, 
    "ReLU": nn.ReLU
}

ui_choose_activation_dropdown = Dropdown(
    options = list(activation_dict.keys()),
    layout = Layout(width="100px"),
    value = "Tanh"
)

ui_activation_group = HBox([ui_activation_label, ui_choose_activation_dropdown])

ui_nodes_per_layer_label = HTML(
    "<b>Nodes per Layer</b>:",
    layout = Layout(width="125px")
)

ui_choose_nodes_per_layer_dropdown = Dropdown(
    options = [1, 2, 5, 10],
    value = 1,
    layout = Layout(width="45px")
)

ui_choose_nodes_per_layer_group = HBox([ui_nodes_per_layer_label, ui_choose_nodes_per_layer_dropdown])

ui_layers_label = HTML(
    "<b>Total Hidden Layers</b>:",
    layout = Layout(width="125px")
)

ui_choose_layers_dropdown = Dropdown(
    options = [1, 2, 5, 10],
    value = 1,
    layout = Layout(width="45px")
)

ui_choose_layers_group = HBox([ui_layers_label, ui_choose_layers_dropdown])

ui_go_button = Button(
    description = "Randomize Parameters and Graph",
    layout = Layout(width="300px"),
    button_style = 'info'
)

def create_random_networks_and_graph(_):
    global net_with_activation, net_without_activation
    net_with_activation, net_without_activation = create_random_networks(
        ui_choose_layers_dropdown.value,
        ui_choose_nodes_per_layer_dropdown.value,
        activation_dict[ui_choose_activation_dropdown.value]
    )

    graph_networks(ui_graph_output, net_with_activation, net_without_activation)

ui_go_button.on_click(create_random_networks_and_graph)

ui_graph_output = Output()

def display_ui():
    display(html_style, 
            VBox([
                ui_activation_group, 
                ui_choose_nodes_per_layer_group,
                ui_choose_layers_group,
                ui_go_button,
                ui_graph_output
            ]))
    create_random_networks_and_graph(None)

# ACTUAL GRAPH OF NETWORK OUTPUT

def graph_networks(output, network_activation, network_no_activation):
    with output:
        clear_output(wait=True)
        
        # Generate sample inputs
        x_values = np.linspace(-10, 10, 100).reshape(-1, 1)
        x_tensor = torch.tensor(x_values, dtype=torch.float32)
        
        # Get outputs from both networks
        y_with_activation = network_activation(x_tensor).detach().numpy()
        y_without_activation = network_no_activation(x_tensor).detach().numpy()
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_values, y_with_activation, label="With Activation", color="blue")
        plt.title("Network with Activation Functions")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid()
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(x_values, y_without_activation, label="Without Activation", color="red")
        plt.title("Network without Activation Functions")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# UI FOR PRINTING NETWORK ARCHITECTURE

print_btn_network = Button(
    description = "Print Network Architectures",
    layout = Layout(width="300px"),
    button_style = 'info'    
)

print_output = Output()

def print_network_architectures(_):
    with print_output:
        clear_output(wait=True)
        display(HTML("<h2>Network with Activation Functions</h2>"))
        print(net_with_activation)
        display(HTML("<h2>Network <u>without</u> Activation Functions</h2>"))
        print(net_without_activation)
        
print_btn_network.on_click(print_network_architectures)

def display_network_printouts():
    display(html_style, VBox([print_btn_network, print_output]))
    print_network_architectures(None)

    