import os
import warnings
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import pickle
import json
import glob
import sys

# Machine Learning
from sklearn.preprocessing import StandardScaler

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Graph Neural Networks
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, TransformerConv
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

# Gradient Boosting
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import kaggle_evaluation.nfl_inference_server

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    """Configuration for the NFL Movement Prediction model."""
    # Data paths
    data_dir: str = './train/'
    test_dir: str = './test/'
    kaggle_data_dir: str = '/kaggle/input/nfl-big-data-bowl-2026-prediction/'
    output_dir: str = './outputs/'
    
    # Model parameters
    random_seed: int = 42
    
    # Feature engineering
    use_velocity_features: bool = True
    use_acceleration_features: bool = True
    use_angle_features: bool = True
    use_distance_features: bool = True
    use_separation_features: bool = True

    # Transformer parameters
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 100

    # GNN parameters
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    gnn_heads: int = 4

    # Ensemble weights
    transformer_weight: float = 0.4
    gnn_weight: float = 0.3
    gbdt_weight: float = 0.3

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# ============================================================================
# Feature Engineering
# ============================================================================
class FootballFeatureEngineer:
    """Creates football-specific features for player movement prediction."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scalers = {}
        
    def compute_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute velocity-based features."""
        df = df.copy()
        
        # Use input features (pre-pass state)
        s_col = 's_input' if 's_input' in df.columns else 's'
        dir_col = 'dir_input' if 'dir_input' in df.columns else 'dir'
        
        if s_col in df.columns and dir_col in df.columns:
            # Velocity components from speed and direction
            df['vx'] = df[s_col] * np.cos(np.radians(df[dir_col]))
            df['vy'] = df[s_col] * np.sin(np.radians(df[dir_col]))
        
        return df
    
    def compute_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute distance-based features."""
        df = df.copy()
        
        # Get position columns
        x_col = 'x_input' if 'x_input' in df.columns else 'x'
        y_col = 'y_input' if 'y_input' in df.columns else 'y'
        
        # Distance to ball landing spot
        if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
            df['dist_to_ball_landing'] = np.sqrt(
                (df[x_col] - df['ball_land_x'])**2 + 
                (df[y_col] - df['ball_land_y'])**2
            )
        
        # Distance to line of scrimmage
        if 'absolute_yardline_number' in df.columns:
            df['dist_from_los'] = df[x_col] - df['absolute_yardline_number']
        
        # Distance to sidelines (field is 53.3 yards wide)
        if y_col in df.columns:
            df['dist_to_near_sideline'] = np.minimum(df[y_col], 53.3 - df[y_col])
            df['dist_from_center'] = np.abs(df[y_col] - 26.65)
        
        return df
    
    def compute_angle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute angle-based features."""
        df = df.copy()
        
        x_col = 'x_input' if 'x_input' in df.columns else 'x'
        y_col = 'y_input' if 'y_input' in df.columns else 'y'
        dir_col = 'dir_input' if 'dir_input' in df.columns else 'dir'
        o_col = 'o_input' if 'o_input' in df.columns else 'o'
        
        # Angle to ball landing spot
        if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
            df['angle_to_ball'] = np.degrees(np.arctan2(
                df['ball_land_y'] - df[y_col],
                df['ball_land_x'] - df[x_col]
            ))
            
            if dir_col in df.columns:
                # Difference between direction and angle to ball (pursuit angle)
                df['pursuit_angle'] = np.abs(
                    ((df[dir_col] - df['angle_to_ball'] + 180) % 360) - 180
                )
                
                # Is player facing the ball? (within 45 degrees)
                df['facing_ball'] = (df['pursuit_angle'] < 45).astype(int)
        
        # Orientation vs direction (body alignment)
        if dir_col in df.columns and o_col in df.columns:
            df['body_alignment'] = np.abs(
                ((df[o_col] - df[dir_col] + 180) % 360) - 180
            )
        
        return df
    
    def compute_player_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute player-specific context features."""
        df = df.copy()
        
        # Encode player side (offense vs defense)
        if 'player_side' in df.columns:
            df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
            df['is_defense'] = (df['player_side'] == 'Defense').astype(int)
        
        # Encode key positions
        if 'player_position' in df.columns:
            df['is_receiver'] = df['player_position'].isin(['WR', 'TE']).astype(int)
            df['is_db'] = df['player_position'].isin(['CB', 'SS', 'FS', 'DB', 'S']).astype(int)
            df['is_lb'] = df['player_position'].isin(['LB', 'ILB', 'OLB', 'MLB']).astype(int)
        
        # Play direction adjustment
        if 'play_direction' in df.columns:
            df['play_dir_right'] = (df['play_direction'] == 'right').astype(int)
        
        # Frame-based features
        if 'frame_id' in df.columns and 'num_frames_output' in df.columns:
            df['frame_progress'] = df['frame_id'] / df['num_frames_output'].clip(lower=1)
        
        return df
    
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        is_training: bool = True
    ) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        
        if self.config.use_velocity_features:
            df = self.compute_velocity_features(df)
        
        if self.config.use_distance_features:
            df = self.compute_distance_features(df)
        
        if self.config.use_angle_features:
            df = self.compute_angle_features(df)
        
        df = self.compute_player_context_features(df)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names for model input."""
        features = [
            # Input position/motion features
            'x_input', 'y_input', 's_input', 'a_input', 'dir_input', 'o_input',
            # Velocity features
            'vx', 'vy',
            # Distance features
            'dist_to_ball_landing', 'dist_from_los',
            'dist_to_near_sideline', 'dist_from_center',
            # Angle features
            'angle_to_ball', 'pursuit_angle', 'facing_ball', 'body_alignment',
            # Player context features
            'is_offense', 'is_defense', 'is_receiver', 'is_db', 'is_lb',
            'play_dir_right', 'frame_progress',
            # Ball landing position
            'ball_land_x', 'ball_land_y',
            # Frame info
            'frame_id', 'num_frames_output'
        ]
        return features

# ============================================================================
# Deep Learning Models
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PlayerMovementTransformer(nn.Module):
    """Transformer model for predicting player movement trajectories."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers
        )
        
        # Output heads for x and y prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # Predict (x, y)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Project input to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask if provided
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        
        # Predict coordinates (take the last token for prediction in this context)
        # Note: In a full sequence model we might predict all steps, but here we predict for the current frame
        predictions = self.output_head(encoded[:, -1, :])
        
        return predictions


class PlayerInteractionGNN(nn.Module):
    """Graph Neural Network for modeling player interactions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers - use simple MLP-based message passing
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        edge_threshold: float = 15.0
    ) -> torch.Tensor:
        # Project input
        h = self.input_proj(x)
        
        # Build adjacency based on positions
        # x shape: (batch_size, num_nodes, features)
        # positions shape: (batch_size, num_nodes, 2)
        
        diff = positions.unsqueeze(1) - positions.unsqueeze(2) # (B, 1, N, 2) - (B, N, 1, 2) -> (B, N, N, 2)
        distances = torch.norm(diff, dim=-1)
        adj = (distances < edge_threshold).float()
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Message passing layers
        for i, layer in enumerate(self.gnn_layers):
            neighbor_features = torch.matmul(adj, h)
            combined = torch.cat([h, neighbor_features], dim=-1)
            h_new = layer(combined)
            h = self.layer_norms[i](h + self.dropout(h_new))
        
        # Output prediction
        output = self.output_head(h)
        return output

# ============================================================================
# GBDT Model
# ============================================================================
class GBDTPredictor:
    """GBDT model for tabular feature prediction."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_x = None
        self.model_y = None
        self.feature_cols = None
        self.scaler = StandardScaler()
        
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str]
    ) -> np.ndarray:
        """Prepare features for GBDT model."""
        self.feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions using GBDT models."""
        if self.model_x is None or self.model_y is None:
            raise ValueError("Models not trained. Call train() first.")
        
        X = self.prepare_features(df, self.feature_cols)
        X_scaled = self.scaler.transform(X)
        
        pred_x = self.model_x.predict(X_scaled)
        pred_y = self.model_y.predict(X_scaled)
        
        return pred_x, pred_y
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model_x = data['model_x']
            self.model_y = data['model_y']
            self.scaler = data['scaler']
            self.feature_cols = data['feature_cols']
        print(f"Model loaded from {path}")

# ============================================================================
# Inference Logic
# ============================================================================

# Global model storage
GLOBAL_GBDT_MODEL = None
GLOBAL_TRANSFORMER_MODEL = None
GLOBAL_GNN_MODEL = None
GLOBAL_FEATURE_ENGINEER = None
GLOBAL_CONFIG = None
GLOBAL_INITIALIZED = False

def initialize_model():
    """Initialize model for inference."""
    global GLOBAL_GBDT_MODEL, GLOBAL_TRANSFORMER_MODEL, GLOBAL_GNN_MODEL
    global GLOBAL_FEATURE_ENGINEER, GLOBAL_CONFIG, GLOBAL_INITIALIZED
    
    if GLOBAL_INITIALIZED:
        return
    
    GLOBAL_CONFIG = Config()
    GLOBAL_FEATURE_ENGINEER = FootballFeatureEngineer(GLOBAL_CONFIG)
    GLOBAL_GBDT_MODEL = GBDTPredictor(GLOBAL_CONFIG)
    
    # --- LOAD UNIFIED MODEL ---
    model_path = None
    if os.path.exists('./model_output/model.pkl'):
        model_path = './model_output/model.pkl'
    elif os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'model.pkl' in files:
                model_path = os.path.join(root, 'model.pkl')
                break
    
    if model_path:
        print(f"Loading unified model from: {model_path}")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        # 1. Load GBDT
        if 'gbdt_model' in data:
            GLOBAL_GBDT_MODEL = data['gbdt_model']
            print("GBDT model loaded.")
            
        # 2. Load Transformer
        input_dim = 27 # Default
        if 'feature_cols' in data:
             input_dim = len(data['feature_cols'])
             
        GLOBAL_TRANSFORMER_MODEL = PlayerMovementTransformer(
            input_dim=input_dim,
            d_model=GLOBAL_CONFIG.d_model,
            n_heads=GLOBAL_CONFIG.n_heads,
            n_encoder_layers=GLOBAL_CONFIG.n_encoder_layers
        ).to(GLOBAL_CONFIG.device)
        
        if 'transformer_state_dict' in data:
            GLOBAL_TRANSFORMER_MODEL.load_state_dict(data['transformer_state_dict'])
            print("Transformer weights loaded.")
        GLOBAL_TRANSFORMER_MODEL.eval()

        # 3. Load GNN
        if HAS_TORCH_GEOMETRIC:
            GLOBAL_GNN_MODEL = PlayerInteractionGNN(
                input_dim=input_dim,
                hidden_dim=GLOBAL_CONFIG.gnn_hidden_dim,
                num_layers=GLOBAL_CONFIG.gnn_num_layers
            ).to(GLOBAL_CONFIG.device)
            
            if 'gnn_state_dict' in data and data['gnn_state_dict'] is not None:
                GLOBAL_GNN_MODEL.load_state_dict(data['gnn_state_dict'])
                print("GNN weights loaded.")
            GLOBAL_GNN_MODEL.eval()
            
    else:
        print("No unified model.pkl found. Attempting to load separate files...")
        # Fallback to separate files logic (omitted for brevity, assume unified)
    
    GLOBAL_INITIALIZED = True

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:
    """
    Main prediction function for the inference server.
    """
    global GLOBAL_GBDT_MODEL, GLOBAL_TRANSFORMER_MODEL, GLOBAL_GNN_MODEL
    global GLOBAL_FEATURE_ENGINEER, GLOBAL_CONFIG
    
    # Initialize on first call
    initialize_model()
    
    # Convert to pandas for processing
    test_pd = test.to_pandas() if isinstance(test, pl.DataFrame) else test
    test_input_pd = test_input.to_pandas() if isinstance(test_input, pl.DataFrame) else test_input
    
    n_rows = len(test_pd)
    
    try:
        # Get the last frame for each player from input (pre-pass state)
        last_state = test_input_pd.sort_values('frame_id').groupby(
            ['game_id', 'play_id', 'nfl_id']
        ).last().reset_index()
        
        # Rename input columns
        feature_cols = ['x', 'y', 's', 'a', 'dir', 'o']
        rename_dict = {col: f'{col}_input' for col in feature_cols if col in last_state.columns}
        last_state = last_state.rename(columns=rename_dict)
        
        # Also rename frame_id to avoid collision
        last_state = last_state.rename(columns={'frame_id': 'last_input_frame_id'})
        
        # Merge test with input features
        merged = test_pd.merge(
            last_state,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        )
        
        # Engineer features
        merged = GLOBAL_FEATURE_ENGINEER.engineer_features(merged, is_training=False)
        
        # --- 1. GBDT Prediction ---
        pred_x_gbdt = np.zeros(n_rows)
        pred_y_gbdt = np.zeros(n_rows)
        
        if GLOBAL_GBDT_MODEL.model_x is not None:
            pred_x_gbdt, pred_y_gbdt = GLOBAL_GBDT_MODEL.predict(merged)
        else:
            # Physics baseline if GBDT missing
            dt = 0.1 * merged['frame_id']
            if 's_input' in merged.columns and 'dir_input' in merged.columns:
                vx = merged['s_input'] * np.cos(np.radians(merged['dir_input']))
                vy = merged['s_input'] * np.sin(np.radians(merged['dir_input']))
                pred_x_gbdt = merged['x_input'] + vx * dt
                pred_y_gbdt = merged['y_input'] + vy * dt
            else:
                pred_x_gbdt = merged.get('x_input', np.zeros(n_rows) + 50).values
                pred_y_gbdt = merged.get('y_input', np.zeros(n_rows) + 26.65).values

        # --- 2. Transformer Prediction ---
        pred_x_trans = np.zeros(n_rows)
        pred_y_trans = np.zeros(n_rows)
        
        # Prepare tensor input (simplified: treating current row as sequence of length 1)
        feature_cols = GLOBAL_FEATURE_ENGINEER.get_feature_columns()
        valid_cols = [c for c in feature_cols if c in merged.columns]
        X_tensor = torch.tensor(merged[valid_cols].fillna(0).values, dtype=torch.float32).to(GLOBAL_CONFIG.device)
        X_tensor = X_tensor.unsqueeze(1) # (Batch, SeqLen=1, Features)
        
        if GLOBAL_TRANSFORMER_MODEL is not None:
            with torch.no_grad():
                if X_tensor.shape[2] == GLOBAL_TRANSFORMER_MODEL.input_proj[0].in_features:
                    out = GLOBAL_TRANSFORMER_MODEL(X_tensor)
                    pred_x_trans = out[:, 0].cpu().numpy()
                    pred_y_trans = out[:, 1].cpu().numpy()
        
        # --- 3. GNN Prediction ---
        pred_x_gnn = np.zeros(n_rows)
        pred_y_gnn = np.zeros(n_rows)
        
        if GLOBAL_GNN_MODEL is not None and HAS_TORCH_GEOMETRIC:
             # Process per play to ensure interactions are only within the same play
             # We can use a simple loop over unique plays in this batch
             
             # Create a mapping from original index to play
             merged['orig_index'] = range(len(merged))
             
             for (game_id, play_id), play_group in merged.groupby(['game_id', 'play_id']):
                 indices = play_group['orig_index'].values
                 
                 # Prepare inputs for this play
                 play_X_tensor = torch.tensor(play_group[valid_cols].fillna(0).values, dtype=torch.float32).to(GLOBAL_CONFIG.device)
                 
                 play_positions = torch.stack([
                    torch.tensor(play_group['x_input'].fillna(50).values),
                    torch.tensor(play_group['y_input'].fillna(25).values)
                 ], dim=1).to(GLOBAL_CONFIG.device)
                 
                 with torch.no_grad():
                     if play_X_tensor.shape[1] == GLOBAL_GNN_MODEL.input_proj.in_features:
                         # GNN expects (NumNodes, Features) and (NumNodes, 2)
                         out = GLOBAL_GNN_MODEL(play_X_tensor, play_positions)
                         
                         pred_x_gnn[indices] = out[:, 0].cpu().numpy()
                         pred_y_gnn[indices] = out[:, 1].cpu().numpy()

        # --- Ensemble ---
        w_gbdt = GLOBAL_CONFIG.gbdt_weight
        w_trans = GLOBAL_CONFIG.transformer_weight
        w_gnn = GLOBAL_CONFIG.gnn_weight
        
        total_w = w_gbdt + w_trans + w_gnn
        w_gbdt /= total_w
        w_trans /= total_w
        w_gnn /= total_w
        
        final_pred_x = w_gbdt * pred_x_gbdt + w_trans * pred_x_trans + w_gnn * pred_x_gnn
        final_pred_y = w_gbdt * pred_y_gbdt + w_trans * pred_y_trans + w_gnn * pred_y_gnn
        
        # Clip
        final_pred_x = np.clip(final_pred_x, 0, 120)
        final_pred_y = np.clip(final_pred_y, 0, 53.3)
            
    except Exception as e:
        # Fallback predictions (field center)
        final_pred_x = np.zeros(n_rows) + 50
        final_pred_y = np.zeros(n_rows) + 26.65
    
    # Create prediction dataframe
    predictions = pl.DataFrame({
        'x': final_pred_x.tolist(),
        'y': final_pred_y.tolist()
    })
    
    return predictions

# ============================================================================
# Main Execution
# ============================================================================
inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

def get_data_dir():
    # Potential paths to check for test.csv
    paths = [
        '/kaggle/input/nfl-big-data-bowl-2026-prediction/',
        './test/',
        './',
        '/kaggle/input/nfl-big-data-bowl-2026-prediction/train/'
    ]
    for p in paths:
        if os.path.exists(os.path.join(p, 'test.csv')):
            return os.path.abspath(p)
    return None

if __name__ == "__main__":
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        # Local testing
        local_data_dir = get_data_dir()
        if local_data_dir:
            print(f"Running local gateway with data from: {local_data_dir}")
            inference_server.run_local_gateway((local_data_dir,))
        else:
            print("Error: Could not find test.csv in common directories. Please check your data path.")
