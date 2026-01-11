# Thermal Runaway Prediction using Continuous Time Markov Chain (CTMC)

A machine learning project that predicts thermal runaway in battery systems using a Continuous Time Markov Chain (CTMC) model. The system monitors temperature data from multiple thermocouples and assesses thermal risk levels through state transitions and probability predictions.

## 🎯 Overview

This project implements a CTMC-based approach to predict and monitor thermal runaway events in battery systems. The model analyzes temperature data from 6 thermocouples, maps temperatures to discrete states, and predicts future state probabilities to compute a Thermal Risk Index (TRI).

## 🔬 Features

- **State-based Modeling**: Maps temperature data to 4 discrete states (S0-S3) based on thermal thresholds
- **Probabilistic Predictions**: Predicts future state probabilities using CTMC transitions
- **Thermal Risk Index (TRI)**: Computes a risk score to assess thermal runaway likelihood
- **Real-time Simulation**: Supports real-time monitoring with configurable simulation speed
- **Visualization**: Generates plots for temperature, TRI, and runaway probability over time

## 📊 System States

The system defines 4 states based on maximum temperature:

- **S0** (Normal): Temperature < 17°C
- **S1** (Low Risk): 17°C ≤ Temperature < 25°C
- **S2** (Moderate Risk): 25°C ≤ Temperature < 40°C
- **S3** (High Risk/Thermal Runaway): Temperature ≥ 40°C (Absorbing State)

## 🔧 Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- matplotlib (for plotting)

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Master
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Master/
├── preprocess.py              # Data preprocessing and state assignment
├── ctmc_model.py             # CTMC model implementation
├── train.py                  # Model training script
├── simulate.py               # Basic simulation script
├── ctmc_simulate_with_output.py  # Advanced simulation with visualization
├── requirements.txt          # Python dependencies
├── train.csv                 # Training dataset
├── validate.csv              # Validation dataset
├── test.csv                  # Test dataset
├── simulate.csv              # Simulation input data
├── ctmc_model.pkl           # Trained model (generated after training)
└── ctmc_output.csv          # Simulation output (generated after simulation)
```

## 🚀 Usage

### 1. Data Preparation

Ensure your CSV files contain the following columns:
- `Test Time [s]`: Time stamps in seconds
- `TC1 near positive terminal [C]` through `TC6 below punch [C]`: Temperature readings from 6 thermocouples

The preprocessing module automatically computes the maximum temperature and assigns states.

### 2. Model Training

Train the CTMC model using training and validation datasets:

```bash
python train.py
```

This will:
- Load `train.csv` and `validate.csv`
- Compute transition counts and time spent in each state
- Build the rate matrix Q
- Save the trained model as `ctmc_model.pkl`

### 3. Simulation

#### Basic Simulation

Run a simple simulation:

```bash
python simulate.py
```

#### Advanced Simulation with Visualization

Run the full simulation with output generation and plots:

```bash
python ctmc_simulate_with_output.py
```

**Command-line options:**
```bash
python ctmc_simulate_with_output.py --simulate simulate.csv --model ctmc_model.pkl --output ctmc_output.csv --speed 0.2
```

**Parameters:**
- `--simulate`: Path to simulation input CSV (default: `simulate.csv`)
- `--model`: Path to trained model file (default: `ctmc_model.pkl`)
- `--output`: Output CSV file path (default: `ctmc_output.csv`)
- `--speed`: Delay per row in seconds for real-time simulation (default: 0.2)

### 4. Output

The simulation generates:

- **CSV Output**: Contains columns for time, state, state probabilities (P_S0, P_S1, P_S2, P_S3), TRI, status, and max temperature
- **Plots**:
  - `plot_max_temp.png`: Maximum temperature over time
  - `plot_tri.png`: Thermal Risk Index over time
  - `plot_s3_probability.png`: Probability of thermal runaway (S3 state)

## 📈 Thermal Risk Index (TRI)

The TRI is computed using weighted state probabilities:

```
TRI = 0.0 × P(S0) + 0.3 × P(S1) + 0.7 × P(S2) + 1.0 × P(S3)
```

**Risk Levels:**
- **NORMAL**: TRI < 0.2
- **CAUTION**: 0.2 ≤ TRI < 0.5
- **HIGH RISK**: TRI ≥ 0.5 (Possible thermal runaway)

## 🧮 Model Details

### CTMC Rate Matrix

The model builds a 4×4 rate matrix Q where:
- `Q[i,j]` = transition rate from state i to state j (i ≠ j)
- `Q[i,i]` = -Σ(Q[i,j]) for j ≠ i (diagonal elements)
- `Q[3,:]` = [0, 0, 0, 0] (S3 is absorbing)

### Probability Prediction

Future state probabilities are computed using matrix exponentiation:

```
P(t) = P(0) × exp(Q × t)
```

where `P(0)` is the initial state distribution and `t` is the prediction horizon (default: 300 seconds / 5 minutes).

## 🔍 Key Functions

### `preprocess.py`
- `load_dataset(path)`: Loads CSV and computes max temperature and states
- `assign_state(temp)`: Maps temperature to state (0-3)
- `compute_transition_counts(df)`: Computes transition matrix Nij and time spent Ti

### `ctmc_model.py`
- `CTMC(Nij, Ti)`: CTMC model class
- `build_rate_matrix()`: Constructs the rate matrix Q
- `predict_probs(initial_state, t_seconds)`: Predicts future probabilities
- `compute_TRI(prob)`: Computes Thermal Risk Index

## 📝 Example Workflow

```python
from preprocess import load_dataset, compute_transition_counts
from ctmc_model import CTMC
import pickle

# Load and preprocess data
train_data = load_dataset("train.csv")
Nij, Ti = compute_transition_counts(train_data)

# Train model
model = CTMC(Nij, Ti)
pickle.dump(model, open("ctmc_model.pkl", "wb"))

# Make predictions
probs = model.predict_probs(initial_state=1, t_seconds=300)
tri = model.compute_TRI(probs)
print(f"TRI: {tri}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is part of a final year academic project.

## 👤 Authors

- [Your Name/Team]

## 🙏 Acknowledgments

- Academic project for battery thermal management research

