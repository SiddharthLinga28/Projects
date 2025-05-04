import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class WastewaterGAT(torch.nn.Module):
    """
    Graph Attention Network for wastewater system analysis
    
    This model uses multiple GAT layers with edge features to capture
    the complex relationships in wastewater networks.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=8, dropout=0.5):
        super(WastewaterGAT, self).__init__()
        
        # First GAT layer with edge features
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # Second GAT layer
        self.conv2 = GATConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # Output layer
        self.lin = nn.Linear(hidden_channels * heads, out_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the network
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges] 
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Node predictions [num_nodes, out_channels]
        """
        # First GAT layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.lin(x)
        
        return x