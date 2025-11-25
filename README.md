# NFL Big Data Bowl 2026 — Dataset Description

This repository contains the data and supporting files for the **2026 NFL Big Data Bowl** competition.  
Below is a complete summary of the dataset structure, key variables, and file descriptions.

---

## 📌 Competition Phases

The competition consists of **two phases**:

### **1. Model Training Phase**
You will build and train your models using **historic game data** provided in the `train/` directory.

### **2. Forecasting Phase**
You will make predictions on a **test set** containing games played *after* the submission deadline.

- The scored portion of the forecasting test set will be similar in size to the scored portion of the phase 1 test set.
- The number of plays may vary due to natural variation in NFL games.
- During this phase, the **evaluation API** will return only previously unseen games.

---

## 📁 Files Overview

### **`train/`**
Historic play-by-play tracking data split by week.

#### **`input_2023_w[01-18].csv`**
Tracking data **before the pass is thrown**.

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier (numeric) |
| `play_id` | Play identifier, not unique across games (numeric) |
| `player_to_predict` | Whether this player's x/y will be scored (bool) |
| `nfl_id` | Unique player identifier (numeric) |
| `frame_id` | Frame number within game/play/file type (numeric) |
| `play_direction` | Offense direction (`left` or `right`) |
| `absolute_yardline_number` | Distance from the end zone for the possession team |
| `player_name` | Player name (text) |
| `player_height` | Height (ft-in) |
| `player_weight` | Weight (lbs) |
| `player_birth_date` | Date of birth (yyyy-mm-dd) |
| `player_position` | Player's position on the field |
| `player_side` | `Offense` or `Defense` |
| `player_role` | Player's role on the play |
| `x` | Player long-axis field position (0–120 yards) |
| `y` | Player short-axis field position (0–53.3 yards) |
| `s` | Speed (yards/sec) |
| `a` | Acceleration (yards/sec²) |
| `o` | Orientation (degrees) |
| `dir` | Motion direction (degrees) |
| `num_frames_output` | Total frames to predict for this player |
| `ball_land_x` | Ball landing long-axis position |
| `ball_land_y` | Ball landing short-axis position |

---

#### **`output_2023_w[01-18].csv`**
Tracking data **after the pass is thrown**.

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `play_id` | Play identifier |
| `nfl_id` | Player identifier |
| `frame_id` | Frame number (matches range from corresponding `input` file) |
| `x` | **TARGET** long-axis player position |
| `y` | **TARGET** short-axis player position |

---

### **`test_input.csv`**
A convenience file containing player tracking at the time of prediction.  
The actual test data is served by the evaluation API.

### **`test.csv`**
Mock test set with rows specifying prediction targets:

- `game_id`
- `play_id`
- `nfl_id`
- `frame_id`

This matches the structure of the real unseen test set.

---

## 📁 `kaggle_evaluation/`
Contains all files required by the Kaggle evaluation API.  
Refer to the **demo submission** for example usage.

---

## 📊 Additional Notes

- You may use supplemental datasets from the analytics competition to explore plays and game context.
- The **future rerun dataset** will be similar in size to the **public leaderboard dataset (~60k rows)**.

---

## 📘 Summary

You now have:

- Pre-pass tracking data (`input_*`)
- Post-pass tracking targets (`output_*`)
- Mock test sets (`test_input.csv`, `test.csv`)
- Evaluation utilities (`kaggle_evaluation/`)

This dataset supports building player movement forecasting models to predict precise x/y locations during passing plays.
![Diagram](field.png)


