#!/usr/bin/env python3
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from preprocess import load_dataset
from ctmc_model import CTMC

def run_simulation(sim_file, model_file, out_file, speed):

    # Load CTMC model
    model = pickle.load(open(model_file, "rb"))

    # Load simulation dataset
    df = load_dataset(sim_file)

    # Create output DataFrame
    results = {
        "Time[s]": [],
        "State": [],
        "P_S0": [],
        "P_S1": [],
        "P_S2": [],
        "P_S3": [],
        "TRI": [],
        "STATUS": [],
        "MAX_TEMP": [],
    }

    print("\nStarting simulation...\n")

    # Loop over each row
    for idx in range(len(df)):
        state = df.loc[idx, "STATE"]
        max_temp = df.loc[idx, "MAX_TEMP"]
        t = df.loc[idx, "Test Time [s]"]

        # Predict next 300 seconds (5 minutes)
        t_future = 300
        probs = model.predict_probs(state, t_future)
        tri = model.compute_TRI(probs)

        # Determine status
        if tri < 0.2:
            status = "NORMAL"
        elif tri < 0.5:
            status = "CAUTION"
        else:
            status = "HIGH RISK"

        # Print to terminal
        print(f"\nTime = {t}")
        print("Initial State:", state)
        print("Future Probabilities:", probs)
        print("TRI =", tri)
        print("Status:", status)

        # Store results
        results["Time[s]"].append(t)
        results["State"].append(state)
        results["P_S0"].append(probs[0])
        results["P_S1"].append(probs[1])
        results["P_S2"].append(probs[2])
        results["P_S3"].append(probs[3])
        results["TRI"].append(tri)
        results["STATUS"].append(status)
        results["MAX_TEMP"].append(max_temp)

        # Simulation delay
        time.sleep(speed)

    # Save CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_file, index=False)
    print(f"\nSaved output to {out_file}")

    # Generate plots
    generate_plots(out_df)


def generate_plots(df):

    # Plot MAX Temperature
    plt.figure(figsize=(10,5))
    plt.plot(df["Time[s]"], df["MAX_TEMP"], label="MAX Temperature", color="red")
    plt.title("MAX Temperature over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)
    plt.savefig("plot_max_temp.png")
    plt.close()

    # Plot TRI
    plt.figure(figsize=(10,5))
    plt.plot(df["Time[s]"], df["TRI"], label="TRI", color="blue")
    plt.title("Thermal Risk Index (TRI) over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("TRI")
    plt.grid(True)
    plt.savefig("plot_tri.png")
    plt.close()

    # Plot S3 Probabilities
    plt.figure(figsize=(10,5))
    plt.plot(df["Time[s]"], df["P_S3"], label="P(S3 - Runaway)", color="purple")
    plt.title("Probability of Thermal Runaway (S3)")
    plt.xlabel("Time [s]")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.savefig("plot_s3_probability.png")
    plt.close()

    print("Generated plots: plot_max_temp.png, plot_tri.png, plot_s3_probability.png")


# ------------------------ CLI Interface ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTMC Simulation with Output Storage")

    parser.add_argument("--simulate", default="simulate.csv", help="Path to simulate.csv")
    parser.add_argument("--model", default="ctmc_model.pkl", help="CTMC model file")
    parser.add_argument("--output", default="ctmc_output.csv", help="Output CSV file")
    parser.add_argument("--speed", type=float, default=0.2, help="Delay per row (seconds)")

    args = parser.parse_args()

    run_simulation(
        sim_file=args.simulate,
        model_file=args.model,
        out_file=args.output,
        speed=args.speed
    )
