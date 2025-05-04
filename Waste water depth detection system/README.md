# Wastewater GNN Project

This project implements a Graph Neural Network (GNN) for analyzing wastewater system data using Graph Attention Networks (GAT) with PyTorch. The goal is to predict water depth and flow rates at various nodes in the wastewater system and compare the performance of the GAT model with baseline models such as Linear Regression and Support Vector Machine (SVM).

## Project Structure

```
wastewater-gnn
├── data
│   ├── WW01_node.csv          # Node data for the wastewater system
│   ├── WW01_edge.csv          # Edge data representing connections between nodes
│   ├── Flow_depth_v3.xls      # Time-series data of water depth measurements
│   └── Flow_rate_v3.xls       # Time-series data of total inflow measurements
├── models
│   ├── __init__.py            # Marks the models directory as a package
│   ├── gat.py                  # Implementation of the Graph Attention Network model
│   └── baselines.py            # Implementations of baseline models (Linear Regression, SVM)
├── utils
│   ├── __init__.py            # Marks the utils directory as a package
│   ├── data_preprocessing.py   # Functions for loading and preprocessing data
├── requirements.txt            # Lists the Python dependencies required for the project
|── Main.ipynb              # Notbook file that shows the outputs of the files and code execution
└── README.md                   # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd wastewater-gnn
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data:
   - Ensure that the data files are placed in the `data` directory.

## Usage

- To train the GNN model, run:
  ```
  python train.py
  ```

- To make predictions using the trained model, run:
  ```
  python predict.py
  ```

## Evaluation

The project includes evaluation metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score to compare the performance of the GAT model with baseline models.

## Visualization

Visualization functions are provided to plot training loss curves and scatter plots of actual vs. predicted values.

## Acknowledgments

This project is inspired by the need for efficient wastewater management and the application of advanced machine learning techniques in environmental engineering.