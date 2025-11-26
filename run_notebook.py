#!/usr/bin/env python3
"""
Run the NFL Movement Prediction notebook code with bug fixes applied.
"""

# ============================================================================
# Section 1: Imports and Configuration
# ============================================================================

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

config = Config()
print(f"Using device: {config.device}")
print(f"torch_geometric available: {HAS_TORCH_GEOMETRIC}")
print(f"LightGBM available: {HAS_LIGHTGBM}")
print(f"XGBoost available: {HAS_XGBOOST}")


# ============================================================================
# Section 2: Data Loading and Exploration
# ============================================================================

class NFLDataLoader:
    """Handles loading and preprocessing of NFL tracking data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.train_dir = Path(config.data_dir)
        self.test_dir = Path(config.test_dir)
        
    def load_training_data(self, weeks: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training data from input and output files.
        
        Args:
            weeks: List of week numbers to load (1-18). If None, load all.
            
        Returns:
            Tuple of (input_df, output_df)
        """
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
        
        print(f"\nTotal input rows: {len(all_input):,}")
        if all_output is not None:
            print(f"Total output rows: {len(all_output):,}")
        
        return all_input, all_output
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load test data.
        
        Returns:
            Tuple of (test_df, test_input_df)
        """
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
        """
        Create training samples by pairing input features with output targets.
        
        Args:
            input_df: Pre-pass tracking data
            output_df: Post-pass target positions
            include_all_players: If False, only include players marked for prediction
        """
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


# Initialize data loader
data_loader = NFLDataLoader(config)

# Load training data
print("\n" + "="*60)
print("Loading training data...")
print("="*60)
input_df, output_df = data_loader.load_training_data()

if input_df is not None:
    print(f"\nInput columns: {input_df.columns.tolist()}")
    print(f"\nInput sample:")
    print(input_df.head())


# Explore the data structure
if input_df is not None:
    print("\n" + "="*60)
    print("INPUT DATA EXPLORATION")
    print("="*60)
    
    print(f"\nShape: {input_df.shape}")
    
    print(f"\nUnique values:")
    print(f"  Games: {input_df['game_id'].nunique()}")
    print(f"  Plays: {input_df.groupby('game_id')['play_id'].nunique().sum()}")
    print(f"  Players: {input_df['nfl_id'].nunique()}")
    
    print(f"\nPlayer positions: {input_df['player_position'].value_counts().head(10).to_dict()}")
    print(f"\nPlayer sides: {input_df['player_side'].value_counts().to_dict()}")
    print(f"\nPlayer roles: {input_df['player_role'].value_counts().to_dict()}")
    
    print(f"\nPlayers to predict per play:")
    pred_counts = input_df.groupby(['game_id', 'play_id'])['player_to_predict'].sum()
    print(f"  Mean: {pred_counts.mean():.2f}")
    print(f"  Min: {pred_counts.min()}, Max: {pred_counts.max()}")

if output_df is not None:
    print("\n" + "="*60)
    print("OUTPUT DATA EXPLORATION")
    print("="*60)
    
    print(f"\nShape: {output_df.shape}")
    print(f"\nColumns: {output_df.columns.tolist()}")
    print(f"\nSample:")
    print(output_df.head())
    
    print(f"\nFrames per player per play:")
    frame_counts = output_df.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].count()
    print(f"  Mean: {frame_counts.mean():.2f}")
    print(f"  Min: {frame_counts.min()}, Max: {frame_counts.max()}")


# ============================================================================
# Section 3: Feature Engineering
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
            print("  Computing velocity features...")
            df = self.compute_velocity_features(df)
        
        if self.config.use_distance_features:
            print("  Computing distance features...")
            df = self.compute_distance_features(df)
        
        if self.config.use_angle_features:
            print("  Computing angle features...")
            df = self.compute_angle_features(df)
        
        print("  Computing player context features...")
        df = self.compute_player_context_features(df)
        
        if is_training:
            print("  Computing target relative features...")
            df = self.compute_target_relative_features(df)
        
        print(f"  Feature engineering complete. Shape: {df.shape}")
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
    
    def get_target_columns(self) -> List[str]:
        """Return list of target column names."""
        return ['x', 'y']


# Initialize feature engineer
feature_engineer = FootballFeatureEngineer(config)


# Create training dataset with features
if input_df is not None and output_df is not None:
    print("\n" + "="*60)
    print("Creating training samples...")
    print("="*60)
    
    # Create training samples by merging input (pre-pass) with output (targets)
    training_df = data_loader.create_training_samples(
        input_df, 
        output_df,
        include_all_players=False  # Only predict for marked players
    )
    
    print(f"\nTraining samples created: {len(training_df):,} rows")
    print(f"Columns: {training_df.columns.tolist()}")
    
    # Engineer features
    training_df = feature_engineer.engineer_features(training_df, is_training=True)
    
    print(f"\nAfter feature engineering: {training_df.shape}")
    
    # Show sample
    feature_cols = feature_engineer.get_feature_columns()
    available_features = [c for c in feature_cols if c in training_df.columns]
    print(f"\nAvailable features ({len(available_features)}): {available_features}")
    
    print(training_df.head())


# ============================================================================
# Section 6: GBDT Model for Tabular Features
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


# ============================================================================
# Section 7: Training Pipeline
# ============================================================================

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
    print(f"\nUsing {len(available_features)} features")
    
    # Split by plays for proper validation
    unique_plays = training_df[['game_id', 'play_id']].drop_duplicates()
    train_plays, val_plays = train_test_split(
        unique_plays, 
        test_size=val_ratio, 
        random_state=config.random_seed
    )
    
    train_df = training_df.merge(train_plays, on=['game_id', 'play_id'])
    val_df = training_df.merge(val_plays, on=['game_id', 'play_id'])
    
    print(f"\nTrain size: {len(train_df):,}, Validation size: {len(val_df):,}")
    
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
    print("\n=== Validation Evaluation ===")
    pred_x, pred_y = gbdt.predict(val_df)
    
    rmse_x = np.sqrt(mean_squared_error(val_df['x'], pred_x))
    rmse_y = np.sqrt(mean_squared_error(val_df['y'], pred_y))
    rmse_combined = np.sqrt(0.5 * (rmse_x**2 + rmse_y**2))
    
    print(f"Validation RMSE X: {rmse_x:.4f}")
    print(f"Validation RMSE Y: {rmse_y:.4f}")
    print(f"Validation RMSE Combined: {rmse_combined:.4f}")
    
    # Feature importance
    importance_df = gbdt.get_feature_importance()
    print(f"\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    return gbdt, val_df, pred_x, pred_y


# Train the model
if 'training_df' in dir() and training_df is not None and len(training_df) > 0:
    gbdt_model, val_df, val_pred_x, val_pred_y = train_gbdt_model(
        training_df, 
        feature_engineer, 
        config
    )
    
    # Save the model
    os.makedirs('./model_output', exist_ok=True)
    gbdt_model.save('./model_output/gbdt_model.pkl')
else:
    print("No training data available. Please load data first.")


# ============================================================================
# Section 8: Inference and Submission
# ============================================================================

# Global model storage
GLOBAL_GBDT_MODEL = None
GLOBAL_FEATURE_ENGINEER = None
GLOBAL_CONFIG = None
GLOBAL_INITIALIZED = False


def initialize_model():
    """Initialize model for inference."""
    global GLOBAL_GBDT_MODEL, GLOBAL_FEATURE_ENGINEER, GLOBAL_CONFIG, GLOBAL_INITIALIZED
    
    if GLOBAL_INITIALIZED:
        return
    
    GLOBAL_CONFIG = Config()
    GLOBAL_FEATURE_ENGINEER = FootballFeatureEngineer(GLOBAL_CONFIG)
    GLOBAL_GBDT_MODEL = GBDTPredictor(GLOBAL_CONFIG)
    
    # Try to load saved model
    model_path = './model_output/gbdt_model.pkl'
    if os.path.exists(model_path):
        GLOBAL_GBDT_MODEL.load(model_path)
        print("Model loaded successfully.")
    else:
        print("No saved model found. Using physics-based baseline.")
    
    GLOBAL_INITIALIZED = True


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:
    """
    Main prediction function for the inference server.
    
    Args:
        test: DataFrame with rows to predict (id, game_id, play_id, nfl_id, frame_id)
        test_input: DataFrame with input features (pre-pass tracking data)
        
    Returns:
        DataFrame with 'x' and 'y' predictions
    """
    global GLOBAL_GBDT_MODEL, GLOBAL_FEATURE_ENGINEER, GLOBAL_CONFIG
    
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
        
        # Try to use GBDT model
        if GLOBAL_GBDT_MODEL.model_x is not None:
            pred_x, pred_y = GLOBAL_GBDT_MODEL.predict(merged)
        else:
            # Physics-based baseline
            dt = 0.1 * merged['frame_id']  # Time since pass release
            
            if 's_input' in merged.columns and 'dir_input' in merged.columns:
                vx = merged['s_input'] * np.cos(np.radians(merged['dir_input']))
                vy = merged['s_input'] * np.sin(np.radians(merged['dir_input']))
                pred_x = merged['x_input'] + vx * dt
                pred_y = merged['y_input'] + vy * dt
            else:
                pred_x = merged.get('x_input', np.zeros(n_rows) + 50).values
                pred_y = merged.get('y_input', np.zeros(n_rows) + 26.65).values
            
            pred_x = np.array(pred_x)
            pred_y = np.array(pred_y)
        
        # Clip predictions to field boundaries
        pred_x = np.clip(pred_x, 0, 120)
        pred_y = np.clip(pred_y, 0, 53.3)
            
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback predictions (field center)
        pred_x = np.zeros(n_rows) + 50
        pred_y = np.zeros(n_rows) + 26.65
    
    # Create prediction dataframe
    predictions = pl.DataFrame({
        'x': pred_x.tolist(),
        'y': pred_y.tolist()
    })
    
    assert len(predictions) == len(test), f"Predictions length {len(predictions)} != test length {len(test)}"
    return predictions


print("\nInference function defined.")


# ============================================================================
# Section 9: Test on Local Test Data
# ============================================================================

print("\n" + "="*60)
print("Development Mode - Testing with Local Test Data")
print("="*60)

# Load test data
test_df, test_input_df = data_loader.load_test_data()

if test_df is not None and test_input_df is not None:
    print(f"\nTest rows to predict: {len(test_df):,}")
    print(f"Test input rows: {len(test_input_df):,}")
    
    # Convert to polars for prediction function
    test_pl = pl.DataFrame(test_df)
    test_input_pl = pl.DataFrame(test_input_df)
    
    # Run prediction
    predictions = predict(test_pl, test_input_pl)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"\nSample predictions:")
    print(predictions.head())
    
    # Save predictions
    pred_df = predictions.to_pandas()
    pred_df['id'] = test_df['id']
    pred_df = pred_df[['id', 'x', 'y']]
    pred_df.to_csv('./test_predictions.csv', index=False)
    print(f"\nPredictions saved to ./test_predictions.csv")
else:
    print("No test data available.")


# ============================================================================
# Section 10: Visualization and Analysis
# ============================================================================

def plot_prediction_errors(val_df, pred_x, pred_y, save_path='./error_distribution.png'):
    """Plot prediction error distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    error_x = pred_x - val_df['x'].values
    error_y = pred_y - val_df['y'].values
    error_dist = np.sqrt(error_x**2 + error_y**2)
    
    axes[0].hist(error_x, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_xlabel('X Error (yards)')
    axes[0].set_title(f'X Error\nMean: {error_x.mean():.2f}, Std: {error_x.std():.2f}')
    
    axes[1].hist(error_y, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Y Error (yards)')
    axes[1].set_title(f'Y Error\nMean: {error_y.mean():.2f}, Std: {error_y.std():.2f}')
    
    axes[2].hist(error_dist, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[2].set_xlabel('Distance Error (yards)')
    axes[2].set_title(f'Distance Error\nMean: {error_dist.mean():.2f}, Std: {error_dist.std():.2f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved error distribution plot to {save_path}")
    plt.close()
    return fig


def plot_feature_importance(importance_df, top_n=15, save_path='./feature_importance.png'):
    """Plot feature importance."""
    if len(importance_df) == 0:
        print("No feature importance data available.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance_avg'], align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature importance plot to {save_path}")
    plt.close()
    return fig


# Create visualizations if we have validation data
if 'val_df' in dir() and 'val_pred_x' in dir():
    print("\n" + "="*60)
    print("Model Analysis")
    print("="*60)
    
    # Plot error distributions
    plot_prediction_errors(val_df, val_pred_x, val_pred_y)
    
    # Plot feature importance
    if 'gbdt_model' in dir():
        importance_df = gbdt_model.get_feature_importance()
        plot_feature_importance(importance_df)


print("\n" + "="*60)
print("DONE!")
print("="*60)
