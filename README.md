## 🏈 Overview

The **downfield pass** is one of the most iconic plays in American football. Once the ball is in the air, the outcome becomes unpredictable—touchdowns, interceptions, contested catches, and incompletions all hang in the balance. This suspense is part of what makes the sport so thrilling.

The **2026 Big Data Bowl** aims to deepen the NFL’s understanding of **player movement during pass plays**, specifically focusing on the period from the quarterback’s throw to the moment the ball is caught or ruled incomplete.

### Offensive Perspective  
The **targeted receiver** must move efficiently toward the ball’s landing point to complete the catch.

### Defensive Perspective  
Multiple defenders may pursue the ball simultaneously, attempting to:
- prevent the catch  
- intercept the pass  
- contest the receiver  

The challenge is to model and predict these movements.

---

## 🎯 Prediction Competition Goal

Participants must **predict player movement** *while the ball is in the air*.  
The provided data includes:

- Pre-pass **Next Gen Stats** tracking data  
- Identification of the **targeted receiver**  
- **Ball landing location**  

When the ball is released, the input ends — the rest must be predicted.

The goal:  
> **Generate accurate frame-by-frame x,y movement predictions for every player in the play.**

---

## 📌 Competition Specifics

### Frame Rate
- NFL tracking data captures **10 frames per second**.  
- Example: A 2.5-second pass → **25 frames to predict**.

### Data Filtering
These plays **are excluded**:
- Quick passes (< 0.5s in the air)
- Deflected passes
- Throwaway passes

### Evaluation Windows
- **Training phase:** evaluated using historical data.  
- **Leaderboard:** evaluated on future data — specifically, a *live leaderboard* for the last **five weeks of the 2025 NFL season**.

---

## 📊 Evaluation Metric

Submissions are scored using:

### **Root Mean Squared Error (RMSE)**  
Measured between:
- Predicted player x/y trajectory  
- Actual observed trajectory  

Official metric details can be found on the competition page.

---

## 📤 Submission Requirements

Participants must submit predictions **via the provided evaluation API**, which processes **one play at a time**.

Your model must output:

```
x, y
```

for every row in the **test dataframe**, where each row corresponds to:

- `game_id`
- `play_id`
- `nfl_id`
- `frame_id`

Accurate frame-by-frame player movement prediction is the objective.

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


