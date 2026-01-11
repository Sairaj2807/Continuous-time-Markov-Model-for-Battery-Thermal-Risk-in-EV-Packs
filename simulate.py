import pickle
from preprocess import load_dataset
from ctmc_model import CTMC

model = pickle.load(open("ctmc_model.pkl", "rb"))

df = load_dataset("simulate.csv")

for idx in range(len(df)):
    state = df.loc[idx, "STATE"]
    t_future = 30  # predict for next 5 minutes

    probs = model.predict_probs(state, t_future)
    tri = model.compute_TRI(probs)

    print(f"\nTime = {df.loc[idx, 'Test Time [s]']}")
    print("Initial State:", state)
    print("Future Probabilities:", probs)
    print("Thermal Risk Index (TRI):", tri)
    
    if tri < 0.2:
        print("Status: NORMAL")
    elif tri < 0.5:
        print("Status: CAUTION")
    else:
        print("Status: HIGH RISK (possible thermal runaway)")
