import pandas as pd
import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)
    
    # Extract all temperature columns
    temp_cols = [
        "TC1 near positive terminal [C]",
        "TC2 near negative terminal [C]",
        "TC3 bottom - bottom [C]",
        "TC4 bottom - top [C]",
        "TC5 above punch [C]",
        "TC6 below punch [C]"
    ]
    
    df["MAX_TEMP"] = df[temp_cols].max(axis=1)
    
    df["STATE"] = df["MAX_TEMP"].apply(assign_state)
    
    return df

def assign_state(temp):
    if temp < 17:
        return 0  # S0
    elif temp < 25:
        return 1  # S1
    elif temp < 40:
        return 2  # S2
    else:
        return 3  # S3 (absorbing)

def compute_transition_counts(df):
    states = df["STATE"].values
    times = df["Test Time [s]"].values

    Nij = np.zeros((4, 4))
    Ti = np.zeros(4)

    for i in range(len(states)-1):
        curr = states[i]
        nxt = states[i+1]
        
        dt = times[i+1] - times[i]
        
        Ti[curr] += dt
        
        if curr != nxt:
            Nij[curr][nxt] += 1

    return Nij, Ti
