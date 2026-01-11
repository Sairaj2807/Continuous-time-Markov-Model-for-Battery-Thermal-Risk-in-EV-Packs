from preprocess import *
from ctmc_model import CTMC
import pickle

train_path = "train.csv"
val_path = "validate.csv"

train = load_dataset(train_path)
val = load_dataset(val_path)

Nij1, Ti1 = compute_transition_counts(train)
Nij2, Ti2 = compute_transition_counts(val)

Nij = Nij1 + Nij2
Ti = Ti1 + Ti2

model = CTMC(Nij, Ti)

pickle.dump(model, open("ctmc_model.pkl", "wb"))

print("Training completed.")
print("Rate matrix Q:\n", model.Q)
