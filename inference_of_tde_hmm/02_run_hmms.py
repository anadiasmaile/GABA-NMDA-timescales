"""Train an HMM on time-delay embedded/PCA data.
"""

from sys import argv

if len(argv) != 3:
    print("Please pass the number of states and run id, e.g. python 1_train_hmm.py 8 1")
    exit()
n_states = int(argv[1])
run = int(argv[2])

#%% Import packages

print("Importing packages")

import pickle

from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.analysis import spectral
from osl_dynamics.inference import modes
import numpy as np
from glob import glob


#%% Load data

# Get file names - sorted by condition
files = sorted(glob(f"data/training_dataset/*.npy"))

# Load data
data = Data(
    files,
    store_dir=f"tmp_{n_states:02d}_{run}",
    n_jobs=15,
)


#%% Setup model

# Settings
config = Config(
    n_states=n_states,
    n_channels=data.n_channels,
    sequence_length=2000,
    learn_means=False,
    learn_covariances=True,
    batch_size=32,
    learning_rate=0.01,
    n_epochs=20,
)


# Create model
model = Model(config)
model.summary()


#%% Training

# Initialisation
init_history = model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)

# Full training
history = model.fit(data)

# Save trained model
model_dir = f"inf/{n_states:02d}_states/run{run:02d}/"
model.save(model_dir)

# Calculate the free energy
free_energy = model.free_energy(data)
history["free_energy"] = free_energy

# Save training history and free energy
pickle.dump(init_history, open(f"inf/{n_states:02d}_states/run{run:02d}/init_history.pkl", "wb"))
pickle.dump(history, open(f"inf/{n_states:02d}_states/run{run:02d}/history.pkl", "wb"))


#%% Get inferred parameters

# State probabilities
alp = model.get_alpha(data)

pickle.dump(alp, open(f"inf/{n_states:02d}_states/run{run:02d}/alp.pkl", "wb"))

# Observation model parameters
means, covs = model.get_means_covariances()

np.save(f"inf/{n_states:02d}_states/run{run:02d}/means.npy", means)
np.save(f"inf/{n_states:02d}_states/run{run:02d}/covs.npy", covs)

#%% Calculate State Metrics
print("Calculating summary stats")

# Calculate state time course
stc = modes.argmax_time_courses(alp)

# Calculate summary stats
fo = modes.fractional_occupancies(stc)
lt = modes.mean_lifetimes(stc, sampling_frequency=250) * 1e3
intv = modes.mean_intervals(stc, sampling_frequency=250)
sr = modes.switching_rates(stc, sampling_frequency=250)

print(np.min(fo,axis=0))
print(np.max(fo,axis=0))

# Save
np.save(f"inf/{n_states:02d}_states/run{run:02d}/fo.npy", fo)
np.save(f"inf/{n_states:02d}_states/run{run:02d}/lt.npy", lt)
np.save(f"inf/{n_states:02d}_states/run{run:02d}/intv.npy", intv)
np.save(f"inf/{n_states:02d}_states/run{run:02d}/sr.npy", sr)


data.delete_dir()
