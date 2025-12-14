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

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Graph Neural Networks
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, TransformerConv
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("torch_geometric not available. GNN features will be disabled.")

# Gradient Boosting
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available. Will use XGBoost or skip GBDT.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available.")

warnings.filterwarnings('ignore')

# Configuration
@dataclass
class Config:
    """Configuration for the NFL Movement Prediction model."""
    # Data paths - support both local and Kaggle environments
    data_dir: str = './train/'  # Local training data directory
    test_dir: str = './test/'   # Local test data directory
    kaggle_data_dir: str = '/kaggle/input/nfl-big-data-bowl-2026-prediction/'
    output_dir: str = './outputs/'
    
    # Model parameters
    random_seed: int = 42
    n_folds: int = 5
    
    # Transformer parameters
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 100  # Maximum frames to consider
    
    # GNN parameters
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    gnn_heads: int = 4
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    
    # Feature engineering
    use_velocity_features: bool = True
    use_acceleration_features: bool = True
    use_angle_features: bool = True
    use_distance_features: bool = True
    use_separation_features: bool = True
    
    # Ensemble weights (will be tuned)
    transformer_weight: float = 0.4
    gnn_weight: float = 0.3
    gbdt_weight: float = 0.3
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Loader
class NFLDataLoader:
    """Handles loading and preprocessing of NFL tracking data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.train_dir = Path(config.data_dir)
        self.test_dir = Path(config.test_dir)
        
    def load_training_data(self, weeks: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data from input and output files."""
        input_dfs = []
        output_dfs = []
        
        # Find all input files
        input_pattern = str(self.train_dir / 'input_2023_w*.csv')
        input_files = sorted(glob.glob(input_pattern))
        
        if not input_files:
            print(f"No input files found at {input_pattern}")
            return None, None
        
        print(f"Found {len(input_files)} input files")
        
        for input_file in input_files:
            # Extract week number
            week_str = Path(input_file).stem.split('_w')[-1]
            week = int(week_str)
            
            if weeks is not None and week not in weeks:
                continue
            
            # Load input file
            input_df = pd.read_csv(input_file)
            input_df['week'] = week
            input_dfs.append(input_df)
            print(f"  Loaded {Path(input_file).name}: {len(input_df):,} rows")
            
            # Load corresponding output file
            output_file = str(self.train_dir / f'output_2023_w{week_str}.csv')
            if os.path.exists(output_file):
                output_df = pd.read_csv(output_file)
                output_df['week'] = week
                output_dfs.append(output_df)
        
        if not input_dfs:
            print("No data loaded!")
            return None, None
        
        all_input = pd.concat(input_dfs, ignore_index=True)
        all_output = pd.concat(output_dfs, ignore_index=True) if output_dfs else None
        
        print(f"\\nTotal input rows: {len(all_input):,}")
        if all_output is not None:
            print(f"Total output rows: {len(all_output):,}")
        
        return all_input, all_output
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load test data."""
        test_file = self.test_dir / 'test.csv'
        test_input_file = self.test_dir / 'test_input.csv'
        
        test_df = None
        test_input_df = None
        
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            print(f"Loaded test.csv: {len(test_df):,} rows")
        
        if test_input_file.exists():
            test_input_df = pd.read_csv(test_input_file)
            print(f"Loaded test_input.csv: {len(test_input_df):,} rows")
        
        return test_df, test_input_df
    
    def create_training_samples(
        self, 
        input_df: pd.DataFrame, 
        output_df: pd.DataFrame,
        include_all_players: bool = False
    ) -> pd.DataFrame:
        """Create training samples by pairing input features with output targets."""
        # Get last known state for each player before pass
        last_state = input_df.sort_values('frame_id').groupby(
            ['game_id', 'play_id', 'nfl_id']
        ).last().reset_index()
        
        # Rename input columns to avoid confusion with targets
        feature_cols = ['x', 'y', 's', 'a', 'dir', 'o']
        rename_dict = {col: f'{col}_input' for col in feature_cols}
        last_state = last_state.rename(columns=rename_dict)
        
        # Also rename frame_id from input to avoid collision
        last_state = last_state.rename(columns={'frame_id': 'last_input_frame_id'})
        
        # Merge with targets
        training_df = output_df.merge(
            last_state,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        )
        
        # Filter to only players we need to predict if requested
        if not include_all_players and 'player_to_predict' in training_df.columns:
            training_df = training_df[training_df['player_to_predict'] == True]
        
        # Create unique identifier
        training_df['id'] = (
            training_df['game_id'].astype(str) + '_' + 
            training_df['play_id'].astype(str) + '_' + 
            training_df['nfl_id'].astype(str) + '_' + 
            training_df['frame_id'].astype(str)
        )
        
        return training_df

# Feature Engineer
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
    
    def compute_target_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features relative to target position."""
        df = df.copy()
        
        x_col = 'x_input' if 'x_input' in df.columns else 'x'
        y_col = 'y_input' if 'y_input' in df.columns else 'y'
        
        # Distance and angle to target (post-pass position we're predicting)
        if 'x' in df.columns and 'y' in df.columns and x_col in df.columns:
            df['target_displacement_x'] = df['x'] - df[x_col]
            df['target_displacement_y'] = df['y'] - df[y_col]
            df['target_displacement_dist'] = np.sqrt(
                df['target_displacement_x']**2 + df['target_displacement_y']**2
            )
        
        return df
    
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        is_training: bool = True
    ) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        print("Engineering features...")
        
        if self.config.use_velocity_features:
            df = self.compute_velocity_features(df)
        
        if self.config.use_distance_features:
            df = self.compute_distance_features(df)
        
        if self.config.use_angle_features:
            df = self.compute_angle_features(df)
        
        df = self.compute_player_context_features(df)
        
        if is_training:
            df = self.compute_target_relative_features(df)
        
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

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model]
        # We assume batch_first=True, so x: [batch_size, seq_len, d_model]
        
        # Actually, pe is [max_len, 1, d_model]. 
        # If batch_first=True, we want [1, max_len, d_model] broadcasted.
        # Let's adjust:
        pe_slice = self.pe[:x.size(1), 0, :].unsqueeze(0) # [1, seq_len, d_model]
        x = x + pe_slice
        return self.dropout(x)

# Transformer Model
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
        
        # Predict coordinates
        predictions = self.output_head(encoded)
        
        return predictions

# GNN Model
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
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
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

# GBDT Predictor
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
    
    def train(
        self, 
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col_x: str = 'x',
        target_col_y: str = 'y',
        val_df: Optional[pd.DataFrame] = None
    ):
        """Train GBDT models for x and y prediction."""
        X = self.prepare_features(df, feature_cols)
        y_x = df[target_col_x].values
        y_y = df[target_col_y].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare validation data if provided
        X_val_scaled = None
        y_val_x = None
        y_val_y = None
        if val_df is not None:
            X_val = self.prepare_features(val_df, feature_cols)
            X_val_scaled = self.scaler.transform(X_val)
            y_val_x = val_df[target_col_x].values
            y_val_y = val_df[target_col_y].values
        
        if HAS_LIGHTGBM:
            print("Training LightGBM models...")
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': -1,
                'random_state': self.config.random_seed
            }
            
            # Train model for X
            train_data_x = lgb.Dataset(X_scaled, label=y_x)
            valid_sets_x = [train_data_x]
            if X_val_scaled is not None:
                valid_data_x = lgb.Dataset(X_val_scaled, label=y_val_x)
                valid_sets_x.append(valid_data_x)
            
            self.model_x = lgb.train(
                params,
                train_data_x,
                num_boost_round=1000,
                valid_sets=valid_sets_x,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            # Train model for Y
            train_data_y = lgb.Dataset(X_scaled, label=y_y)
            valid_sets_y = [train_data_y]
            if X_val_scaled is not None:
                valid_data_y = lgb.Dataset(X_val_scaled, label=y_val_y)
                valid_sets_y.append(valid_data_y)
            
            self.model_y = lgb.train(
                params,
                train_data_y,
                num_boost_round=1000,
                valid_sets=valid_sets_y,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
        elif HAS_XGBOOST:
            print("Training XGBoost models...")
            
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.config.random_seed,
                'n_jobs': -1
            }
            
            eval_set_x = [(X_scaled, y_x)]
            eval_set_y = [(X_scaled, y_y)]
            if X_val_scaled is not None:
                eval_set_x.append((X_val_scaled, y_val_x))
                eval_set_y.append((X_val_scaled, y_val_y))
            
            self.model_x = xgb.XGBRegressor(**params)
            self.model_x.fit(
                X_scaled, y_x,
                eval_set=eval_set_x,
                verbose=100
            )
            
            self.model_y = xgb.XGBRegressor(**params)
            self.model_y.fit(
                X_scaled, y_y,
                eval_set=eval_set_y,
                verbose=100
            )
        else:
            print("No GBDT library available. Skipping GBDT training.")
            return
        
        print("GBDT training complete.")
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions using GBDT models."""
        if self.model_x is None or self.model_y is None:
            raise ValueError("Models not trained. Call train() first.")
        
        X = self.prepare_features(df, self.feature_cols)
        X_scaled = self.scaler.transform(X)
        
        pred_x = self.model_x.predict(X_scaled)
        pred_y = self.model_y.predict(X_scaled)
        
        return pred_x, pred_y
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained models."""
        if self.model_x is None:
            return pd.DataFrame()
        
        if HAS_LIGHTGBM:
            importance_x = self.model_x.feature_importance(importance_type='gain')
            importance_y = self.model_y.feature_importance(importance_type='gain')
        elif HAS_XGBOOST:
            importance_x = self.model_x.feature_importances_
            importance_y = self.model_y.feature_importances_
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance_x': importance_x,
            'importance_y': importance_y,
            'importance_avg': (importance_x + importance_y) / 2
        }).sort_values('importance_avg', ascending=False)
        
        return importance_df
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_x': self.model_x,
                'model_y': self.model_y,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model_x = data['model_x']
            self.model_y = data['model_y']
            self.scaler = data['scaler']
            self.feature_cols = data['feature_cols']
        print(f"Model loaded from {path}")

def ensure_numeric_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Ensure all feature columns are numeric."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

def train_gbdt_model(
    training_df: pd.DataFrame,
    feature_engineer: FootballFeatureEngineer,
    config: Config,
    val_ratio: float = 0.2
) -> GBDTPredictor:
    """Train the GBDT model on prepared training data."""
    
    print("\n" + "="*60)
    print("Training GBDT Model")
    print("="*60)
    
    # Get feature columns
    feature_cols = feature_engineer.get_feature_columns()
    available_features = [c for c in feature_cols if c in training_df.columns]
    print(f"\\nUsing {len(available_features)} features")
    
    # Split by plays for proper validation
    unique_plays = training_df[['game_id', 'play_id']].drop_duplicates()
    train_plays, val_plays = train_test_split(
        unique_plays, 
        test_size=val_ratio, 
        random_state=config.random_seed
    )
    
    train_df = training_df.merge(train_plays, on=['game_id', 'play_id'])
    val_df = training_df.merge(val_plays, on=['game_id', 'play_id'])
    
    print(f"\\nTrain size: {len(train_df):,}, Validation size: {len(val_df):,}")
    
    # Initialize and train GBDT
    gbdt = GBDTPredictor(config)
    gbdt.train(
        train_df, 
        available_features,
        target_col_x='x',
        target_col_y='y',
        val_df=val_df
    )
    
    # Evaluate on validation set
    print("\\n=== Validation Evaluation ===")
    pred_x, pred_y = gbdt.predict(val_df)
    
    rmse_x = np.sqrt(mean_squared_error(val_df['x'], pred_x))
    rmse_y = np.sqrt(mean_squared_error(val_df['y'], pred_y))
    rmse_combined = np.sqrt(0.5 * (rmse_x**2 + rmse_y**2))
    
    print(f"Validation RMSE X: {rmse_x:.4f}")
    print(f"Validation RMSE Y: {rmse_y:.4f}")
    print(f"Validation RMSE Combined: {rmse_combined:.4f}")
    
    # Feature importance
    importance_df = gbdt.get_feature_importance()
    print(f"\\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    return gbdt, val_df, pred_x, pred_y

# ============================================================================
# Transformer Training Logic
# ============================================================================

class PlayerSequenceDataset(Dataset):
    """Dataset for Transformer model sequences."""
    def __init__(self, input_df: pd.DataFrame, training_df: pd.DataFrame, feature_cols: List[str], config: Config):
        self.config = config
        self.feature_cols = feature_cols
        
        # Group input history by player-play
        # We assume input_df is sorted by frame_id
        print("Indexing history data... (this may take a moment)")
        self.history_data = {}
        # Pre-convert to float32 to save time during training
        for name, group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
            self.history_data[name] = group[feature_cols].values.astype(np.float32)
        print(f"Indexed {len(self.history_data)} player histories.")
        
        # Training samples (targets)
        self.training_samples = training_df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.training_samples)
    
    def __getitem__(self, idx):
        row = self.training_samples.iloc[idx]
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        
        # Get history
        if key in self.history_data:
            history = self.history_data[key]
        else:
            # Should not happen if data is consistent
            history = np.zeros((1, len(self.feature_cols)), dtype=np.float32)
            
        # Create query token from target row (using input features)
        query_features = row[self.feature_cols].values.reshape(1, -1)
        
        # Combine history + query
        sequence = np.vstack([history, query_features])
        
        # Pad/Truncate
        if len(sequence) > self.config.max_seq_len:
            sequence = sequence[-self.config.max_seq_len:]
        else:
            pad_len = self.config.max_seq_len - len(sequence)
            sequence = np.pad(sequence, ((pad_len, 0), (0, 0)), mode='constant')
            
        # Target
        target = row[['x', 'y']].values.astype(np.float32)
        
        # Target
        target = row[['x', 'y']].values.astype(np.float32)
        
        try:
            return torch.FloatTensor(sequence.astype(np.float32)), torch.FloatTensor(target)
        except Exception as e:
            print(f"Error converting sequence to tensor at idx {idx}")
            print(f"Sequence shape: {sequence.shape}")
            print(f"Sequence dtype: {sequence.dtype}")
            print(f"First row: {sequence[0]}")
            raise e

def train_transformer_model(
    training_df: pd.DataFrame,
    input_df: pd.DataFrame,
    feature_engineer: FootballFeatureEngineer,
    config: Config,
    val_ratio: float = 0.2
):
    print("\n" + "="*60)
    print("Training Transformer Model")
    print("="*60)
    
    feature_cols = feature_engineer.get_feature_columns()
    available_features = [c for c in feature_cols if c in training_df.columns]
    input_dim = len(available_features)
    print(f"Input dimension: {input_dim}")
    
    # Debug: Check dtypes
    print("\nFeature column dtypes:")
    print(training_df[available_features].dtypes)
    
    # Split plays
    unique_plays = training_df[['game_id', 'play_id']].drop_duplicates()
    train_plays, val_plays = train_test_split(unique_plays, test_size=val_ratio, random_state=config.random_seed)
    
    train_df = training_df.merge(train_plays, on=['game_id', 'play_id'])
    val_df = training_df.merge(val_plays, on=['game_id', 'play_id'])
    
    # Create Datasets
    train_dataset = PlayerSequenceDataset(input_df, train_df, available_features, config)
    val_dataset = PlayerSequenceDataset(input_df, val_df, available_features, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Model
    model = PlayerMovementTransformer(
        input_dim=input_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_encoder_layers=config.n_encoder_layers,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    ).to(config.device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            output = model(X) # (Batch, Seq, 2)
            # Loss on last token
            pred = output[:, -1, :]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1} Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}", end='\r')
        
        print() # New line after batch loop
            
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.device), y.to(config.device)
                output = model(X)
                pred = output[:, -1, :]
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            
    print("Transformer training complete.")
    return best_state_dict

# ============================================================================
# GNN Training Logic
# ============================================================================

class PlayerGraphDataset(Dataset):
    """Dataset for GNN model."""
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], config: Config):
        self.config = config
        self.feature_cols = feature_cols
        
        # Group by frame (game, play, frame)
        self.groups = list(df.groupby(['game_id', 'play_id', 'frame_id']))
        
    def __len__(self):
        return len(self.groups)
        
    def __getitem__(self, idx):
        keys, group = self.groups[idx]
        
        # Node features
        x = torch.FloatTensor(group[self.feature_cols].values)
        
        # Positions for edge construction
        pos_cols = ['x_input', 'y_input']
        pos = torch.FloatTensor(group[pos_cols].values)
        
        # Targets
        y = torch.FloatTensor(group[['x', 'y']].values)
        
        return x, pos, y

def collate_gnn(batch):
    # Batch size 1 for simplicity with custom GNN
    return batch[0]

def train_gnn_model(
    training_df: pd.DataFrame,
    feature_engineer: FootballFeatureEngineer,
    config: Config,
    val_ratio: float = 0.2
):
    if not HAS_TORCH_GEOMETRIC:
        print("torch_geometric not available. Skipping GNN training.")
        return None

    print("\n" + "="*60)
    print("Training GNN Model")
    print("="*60)
    
    feature_cols = feature_engineer.get_feature_columns()
    available_features = [c for c in feature_cols if c in training_df.columns]
    input_dim = len(available_features)
    
    # Split plays
    unique_plays = training_df[['game_id', 'play_id']].drop_duplicates()
    train_plays, val_plays = train_test_split(unique_plays, test_size=val_ratio, random_state=config.random_seed)
    
    train_df = training_df.merge(train_plays, on=['game_id', 'play_id'])
    val_df = training_df.merge(val_plays, on=['game_id', 'play_id'])
    
    train_dataset = PlayerGraphDataset(train_df, available_features, config)
    val_dataset = PlayerGraphDataset(val_df, available_features, config)
    
    # Batch size 1 because custom GNN implementation handles one graph at a time
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_gnn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_gnn)
    
    model = PlayerInteractionGNN(
        input_dim=input_dim,
        hidden_dim=config.gnn_hidden_dim,
        num_layers=config.gnn_num_layers,
        dropout=config.dropout
    ).to(config.device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for x, pos, y in train_loader:
            x, pos, y = x.to(config.device), pos.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            output = model(x, pos)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, pos, y in val_loader:
                x, pos, y = x.to(config.device), pos.to(config.device), y.to(config.device)
                output = model(x, pos)
                loss = criterion(output, y)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()
            
    print("GNN training complete.")
    return best_state_dict

if __name__ == "__main__":
    # Initialize config
    config = Config()
    
    # Initialize data loader
    data_loader = NFLDataLoader(config)
    
    # Load training data
    print("\n" + "="*60)
    print("Loading training data...")
    print("="*60)
    input_df, output_df = data_loader.load_training_data()
    
    if input_df is not None and output_df is not None:
        # Initialize feature engineer
        feature_engineer = FootballFeatureEngineer(config)
        
        print("\n" + "="*60)
        print("Creating training samples...")
        print("="*60)
        
        # Create training samples by merging input (pre-pass) with output (targets)
        training_df = data_loader.create_training_samples(
            input_df, 
            output_df,
            include_all_players=False  # Only predict for marked players
        )
        
        print("Engineering features for training samples...")
        training_df = feature_engineer.engineer_features(training_df, is_training=True)
        
        # Ensure numeric features
        feature_cols = feature_engineer.get_feature_columns()
        training_df = ensure_numeric_features(training_df, feature_cols)
        
        # 1. Train GBDT
        gbdt_model, val_df, val_pred_x, val_pred_y = train_gbdt_model(
            training_df, 
            feature_engineer, 
            config
        )
        
        # 2. Train Transformer
        print("\nEngineering features for input_df (history)...")
        input_df_engineered = feature_engineer.engineer_features(input_df, is_training=False)
        
        # Rename columns to match expected input features
        rename_dict = {
            'x': 'x_input', 'y': 'y_input', 
            's': 's_input', 'a': 'a_input', 
            'dir': 'dir_input', 'o': 'o_input'
        }
        input_df_engineered = input_df_engineered.rename(columns=rename_dict)
        
        # Ensure numeric features for history
        input_df_engineered = ensure_numeric_features(input_df_engineered, feature_cols)
        
        transformer_state_dict = train_transformer_model(
            training_df,
            input_df_engineered,
            feature_engineer,
            config
        )
        
        # 3. Train GNN
        gnn_state_dict = train_gnn_model(
            training_df,
            feature_engineer,
            config
        )
        
        # Save Unified Model
        os.makedirs('./model_output', exist_ok=True)
        save_path = './model_output/model.pkl'
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'gbdt_model': gbdt_model,
                'transformer_state_dict': transformer_state_dict,
                'gnn_state_dict': gnn_state_dict,
                'feature_cols': gbdt_model.feature_cols,
                'scaler': gbdt_model.scaler
            }, f)
            
        print(f"\nUnified model saved to {save_path}")
        
    else:
        print("No training data available.")
