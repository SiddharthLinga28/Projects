import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

class SVMBaseline:
    """
    Support Vector Machine Regression baseline model
    """
    def __init__(self):
        # Initialize SVR with sensible default parameters
        self.model = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    
    def train(self, X, y):
        """
        Train the SVM model
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target values [n_samples]
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Predictions [n_samples]
        """
        return self.model.predict(X)

class LinearRegressionBaseline:
    """
    Linear Regression baseline model
    """
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X, y):
        """
        Train the Linear Regression model
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target values [n_samples]
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Feature matrix [n_samples, n_features]
            
        Returns:
            Predictions [n_samples]
        """
        return self.model.predict(X)
    
def prepare_baseline_features(node_features, adjacency_matrix):
    """
    Prepare features for baseline models by flattening graph structure
    
    Args:
        node_features: Node features tensor
        adjacency_matrix: Graph adjacency matrix
        
    Returns:
        Numpy array of features for traditional ML models
    """
    # Convert to numpy if tensors
    if hasattr(node_features, 'numpy'):
        node_features = node_features.numpy()
    
    # Create features that indirectly capture graph structure
    node_degrees = adjacency_matrix.sum(axis=1).reshape(-1, 1)
    
    # For each node, compute some basic graph metrics
    num_nodes = adjacency_matrix.shape[0]
    neighbor_features = np.zeros((num_nodes, 2))
    
    for i in range(num_nodes):
        # Get neighbors of this node
        neighbors = np.where(adjacency_matrix[i, :] > 0)[0]
        
        # If the node has neighbors, compute aggregate features
        if len(neighbors) > 0:
            # Average of neighbor features
            neighbor_features[i, 0] = np.mean(node_features[neighbors].sum(axis=1))
            # Max of neighbor features
            neighbor_features[i, 1] = np.max(node_features[neighbors].sum(axis=1))
    
    # Combine original features with graph-based features
    X_combined = np.concatenate([node_features, node_degrees, neighbor_features], axis=1)
    
    return X_combined