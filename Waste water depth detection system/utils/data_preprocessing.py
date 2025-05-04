import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os
from datetime import datetime, timedelta

def load_and_preprocess_data(node_file, edge_file, depth_file=None, flow_file=None, 
                            lookback_hours=24, feature_agg='mean'):
    """
    Load and preprocess wastewater network data from multiple sources.
    
    Args:
        node_file: Path to node CSV file
        edge_file: Path to edge CSV file
        depth_file: Optional path to flow depth Excel file
        flow_file: Optional path to flow rate Excel file
        lookback_hours: Hours of historical data to include as features
        feature_agg: How to aggregate temporal features ('mean', 'max', 'min', etc.)
    
    Returns:
        torch_geometric.data.Data object and node mapping dictionary
    """
    print("Loading node and edge data...")
    
    # Load static network data
    node_data = pd.read_csv(node_file)
    edge_data = pd.read_csv(edge_file)
    
    # Fix column names with leading spaces (if any)
    node_data.columns = [col.strip() for col in node_data.columns]
    edge_data.columns = [col.strip() for col in edge_data.columns]
    
    # Get the node ID column name
    node_id_col = [col for col in node_data.columns if "Node ID" in col][0]
    
    # Create a mapping from node ID to index
    node_ids = list(node_data[node_id_col]) + ['OF-1']  # Add outfall if needed
    node_mapping = {str(node_id): idx for idx, node_id in enumerate(node_ids)}
    
    # Number of nodes including the outfall
    num_nodes = len(node_mapping)
    print(f"Total nodes in graph: {num_nodes} (including outfall)")
    
    # Selected node features (modify as needed based on domain knowledge)
    node_features = []
    
    # Extract relevant static features for each node
    for idx, node_id in enumerate(node_ids):
        if idx < len(node_data):  # For regular nodes in the node_data
            row = node_data.iloc[idx]
            # Select relevant features (adjust based on your analysis)
            features = [
                row['X-Coordinate'],
                row['Y-Coordinate'],
                row['Invert Elev. (ft)'],
                row['Rim Elev. (ft)'], 
                row['Depth (ft)'],
                row['Avg. Depth (ft)'],
                row['Max. Depth (ft)'],
                row['Max. Lat. Inflow (cfs)'],
                row['Max. Total Inflow (cfs)'],
                row['Total inflow (MG)']
            ]
        else:  # For outfall node which might not be in node_data
            # Use default values or consider using mean values from other nodes
            features = [0.0] * 10  # Placeholder values
            
        node_features.append(features)
    
    # Convert to tensor
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Process edge data
    edge_index = []
    edge_attr = []
    
    print("Processing edge connections...")
    
    # For each edge, find source and target node indices
    for _, row in edge_data.iterrows():
        source = str(row['Inlet Node'])
        target = str(row['Outlet Node'])
        
        # Skip edges where source or target is not in our node mapping
        if source not in node_mapping or target not in node_mapping:
            continue
        
        source_idx = node_mapping[source]
        target_idx = node_mapping[target]
        
        edge_index.append([source_idx, target_idx])
        
        # Extract edge attributes (modify based on engineering requirements)
        attrs = [
            float(row['Length (ft)']),
            float(row['Roughness']),
            float(row['Slope (ft/ft)']),
            float(row['Max. |Flow| (cfs)']),
            float(row['Max. |Velocity| (ft/s)']),
            float(row['Max/Full Flow']),
            float(row['Max/Full Depth'])
        ]
        edge_attr.append(attrs)
    
    # Convert to tensors
    if not edge_index:  # Check if empty
        raise ValueError("No valid edges found. Check that node IDs in edge file match those in node file.")
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    print(f"Graph created with {len(edge_attr)} edges")
    
    # Load temporal data if provided and add to node features
    if depth_file and flow_file and os.path.exists(depth_file) and os.path.exists(flow_file):
        print("Loading temporal depth and flow data...")
        
        try:
            depth_data = pd.read_excel(depth_file)
            flow_data = pd.read_excel(flow_file)
            
            # Fix column names with spaces or special characters
            depth_data.columns = [str(col).strip() for col in depth_data.columns]
            flow_data.columns = [str(col).strip() for col in flow_data.columns]
            
            # Ensure datetime format for Date/Time column
            if 'Date/Time' in depth_data.columns:
                depth_data['Date/Time'] = pd.to_datetime(depth_data['Date/Time'])
                
            if 'Date/Time' in flow_data.columns:
                flow_data['Date/Time'] = pd.to_datetime(flow_data['Date/Time'])
                
            # Sort by date/time
            depth_data = depth_data.sort_values('Date/Time')
            flow_data = flow_data.sort_values('Date/Time')
            
            # Add temporal features to node features
            temporal_features = process_temporal_data(
                depth_data, flow_data, node_mapping, lookback_hours, feature_agg
            )
            
            # Combine static and temporal features
            if temporal_features is not None:
                combined_features = torch.cat([node_features, temporal_features], dim=1)
                print(f"Added temporal features: node features shape is now {combined_features.shape}")
                node_features = combined_features
                
        except Exception as e:
            print(f"Warning: Error processing temporal data: {e}")
            print("Proceeding with only static features.")
    
    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    
    # Normalize node features
    data.x = normalize_features(data.x)
    
    return data, node_mapping

def process_temporal_data(depth_df, flow_df, node_mapping, lookback_hours=24, 
                          feature_agg='mean'):
    """
    Process temporal depth and flow data to create node features
    
    Args:
        depth_df: DataFrame with depth readings
        flow_df: DataFrame with flow readings
        node_mapping: Dictionary mapping node IDs to indices
        lookback_hours: Number of hours to look back for temporal features
        feature_agg: Aggregation method for temporal features
        
    Returns:
        Tensor of temporal features for each node
    """
    if depth_df.empty or flow_df.empty:
        return None
        
    # Get the latest timestamp
    latest_time = max(depth_df['Date/Time'].max(), flow_df['Date/Time'].max())
    
    # Define the lookback period
    lookback_start = latest_time - timedelta(hours=lookback_hours)
    
    # Filter data for the lookback period
    depth_period = depth_df[depth_df['Date/Time'] >= lookback_start]
    flow_period = flow_df[flow_df['Date/Time'] >= lookback_start]
    
    # Initialize temporal features tensor
    num_nodes = len(node_mapping)
    # 8 features: mean/max depth and flow for each node in recent history
    temporal_features = torch.zeros((num_nodes, 8), dtype=torch.float)
    
    # For each node, calculate temporal features
    for node_id, node_idx in node_mapping.items():
        if node_id in depth_period.columns:
            # Get depth data for this node
            node_depth = depth_period[node_id].values
            
            # Calculate aggregated statistics 
            if feature_agg == 'mean':
                depth_mean = np.nanmean(node_depth) if len(node_depth) > 0 else 0
                depth_max = np.nanmax(node_depth) if len(node_depth) > 0 else 0
                depth_min = np.nanmin(node_depth) if len(node_depth) > 0 else 0
                depth_std = np.nanstd(node_depth) if len(node_depth) > 0 else 0
            else:
                # Default to mean if agg method not recognized
                depth_mean = np.nanmean(node_depth) if len(node_depth) > 0 else 0
                depth_max = np.nanmax(node_depth) if len(node_depth) > 0 else 0
                depth_min = np.nanmin(node_depth) if len(node_depth) > 0 else 0
                depth_std = np.nanstd(node_depth) if len(node_depth) > 0 else 0
            
            # Set temporal features for this node
            temporal_features[node_idx, 0] = depth_mean
            temporal_features[node_idx, 1] = depth_max
            temporal_features[node_idx, 2] = depth_min
            temporal_features[node_idx, 3] = depth_std
                
        if node_id in flow_period.columns:
            # Get flow data for this node
            node_flow = flow_period[node_id].values
            
            # Calculate aggregated statistics
            if feature_agg == 'mean':
                flow_mean = np.nanmean(node_flow) if len(node_flow) > 0 else 0
                flow_max = np.nanmax(node_flow) if len(node_flow) > 0 else 0
                flow_min = np.nanmin(node_flow) if len(node_flow) > 0 else 0
                flow_std = np.nanstd(node_flow) if len(node_flow) > 0 else 0
            else:
                # Default to mean
                flow_mean = np.nanmean(node_flow) if len(node_flow) > 0 else 0
                flow_max = np.nanmax(node_flow) if len(node_flow) > 0 else 0
                flow_min = np.nanmin(node_flow) if len(node_flow) > 0 else 0
                flow_std = np.nanstd(node_flow) if len(node_flow) > 0 else 0
                
            # Set temporal features for this node
            temporal_features[node_idx, 4] = flow_mean  
            temporal_features[node_idx, 5] = flow_max
            temporal_features[node_idx, 6] = flow_min
            temporal_features[node_idx, 7] = flow_std
    
    return temporal_features

def normalize_features(features):
    """
    Normalize node or edge features using z-score normalization
    
    Args:
        features: Tensor of features
        
    Returns:
        Normalized features tensor
    """
    # Calculate mean and std for each feature across nodes
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    
    # Replace zero std with 1 to avoid division by zero
    std[std == 0] = 1.0
    
    # Normalize features
    normalized_features = (features - mean) / std
    
    # Handle NaN values that might appear after normalization
    normalized_features[torch.isnan(normalized_features)] = 0.0
    
    return normalized_features

def create_time_series_data(depth_df, flow_df, node_mapping, seq_length=12, stride=1, 
                            prediction_horizon=1, target_column=None):
    """
    Create time series dataset from depth and flow data for sequence learning
    
    Args:
        depth_df: DataFrame with depth readings
        flow_df: DataFrame with flow readings
        node_mapping: Dictionary mapping node IDs to indices
        seq_length: Number of time steps to include in each sequence
        stride: Stride for sliding window
        prediction_horizon: How many steps ahead to predict
        target_column: Target column to predict
        
    Returns:
        List of (sequence, target) pairs
    """
    if depth_df.empty or flow_df.empty:
        return None
        
    # Ensure data is sorted by time
    depth_df = depth_df.sort_values('Date/Time')
    flow_df = flow_df.sort_values('Date/Time')
    
    # Align depth and flow data by timestamp
    merged_data = pd.merge(
        depth_df, flow_df, on='Date/Time', 
        suffixes=('_depth', '_flow')
    )
    
    sequences = []
    targets = []
    
    # Total number of time steps
    n_steps = len(merged_data) - seq_length - prediction_horizon + 1
    
    for i in range(0, n_steps, stride):
        # Extract sequence window
        seq_data = merged_data.iloc[i:i+seq_length]
        
        # Extract target (future value)
        target_idx = i + seq_length + prediction_horizon - 1
        if target_idx < len(merged_data):
            target_data = merged_data.iloc[target_idx]
            
            # Create features for each time step in the sequence
            seq_features = []
            
            for _, row in seq_data.iterrows():
                # Extract features for each node at this time step
                step_features = []
                
                for node_id in node_mapping.keys():
                    # Find columns for this node
                    depth_col = node_id
                    flow_col = node_id
                    
                    # Extract values (with error handling)
                    try:
                        depth_val = row.get(depth_col, 0)
                        flow_val = row.get(flow_col, 0)
                        step_features.extend([depth_val, flow_val])
                    except:
                        step_features.extend([0, 0])
                
                seq_features.append(step_features)
            
            # Extract target values
            if target_column:
                target_vals = []
                for node_id in node_mapping.keys():
                    target_val = target_data.get(f"{node_id}_{target_column}", 0)
                    target_vals.append(target_val)
            else:
                # Default: predict future depth
                target_vals = []
                for node_id in node_mapping.keys():
                    target_val = target_data.get(node_id, 0) 
                    target_vals.append(target_val)
            
            # Add to sequences and targets
            sequences.append(seq_features)
            targets.append(target_vals)
    
    return sequences, targets

def prepare_baseline_features(node_features, adjacency_matrix=None):
    """
    Prepare features for baseline models (SVM, Linear Regression)
    that don't use graph structure directly
    
    Args:
        node_features: Node feature tensor
        adjacency_matrix: Optional adjacency matrix to include connectivity info
        
    Returns:
        Numpy array of features for traditional ML models
    """
    features = node_features.numpy()
    
    if adjacency_matrix is not None:
        # Add graph connectivity information to node features
        # This helps baseline models somewhat account for graph structure
        node_degrees = adjacency_matrix.sum(axis=1).reshape(-1, 1)
        clustering = np.zeros((adjacency_matrix.shape[0], 1))
        
        # Simple measure of local clustering
        for i in range(adjacency_matrix.shape[0]):
            neighbors = np.where(adjacency_matrix[i] > 0)[0]
            if len(neighbors) > 1:
                # Count connections among neighbors
                connections = 0
                for j in neighbors:
                    for k in neighbors:
                        if j != k and adjacency_matrix[j, k] > 0:
                            connections += 1
                possible = len(neighbors) * (len(neighbors) - 1)
                clustering[i] = connections / possible if possible > 0 else 0
                
        # Concatenate with original features
        features = np.concatenate([features, node_degrees, clustering], axis=1)
    
    return features