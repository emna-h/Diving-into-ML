# Inverse Design of a Circular Antenna Array with a Neural Network
### A complete beginner's guide, file by file

---

## Table of contents

0. [What problem does this project solve?](#0-what-problem-does-this-project-solve)
1. [The big picture](#1-the-big-picture)
2. [Physics background](#2-physics-background)
3. [Machine-learning background](#3-machine-learning-background)
4. [How to run the project](#4-how-to-run-the-project)
5. [File 0: `circular_antenna_array.py`](#5-file-0-circular_antenna_arraypy)
6. [File 1: `dataset.py`](#6-file-1-datasetpy)
7. [File 2: `repartition_training_test.py`](#7-file-2-repartition_training_testpy)
8. [File 3: `FFNN_training.py`](#8-file-3-ffnn_trainingpy)
9. [File 4: `test_model_FFNN.py`](#9-file-4-test_model_ffnnpy)
10. [Data shapes cheat sheet](#10-data-shapes-cheat-sheet)
11. [Known issues and improvements](#11-known-issues-and-improvements)
12. [Glossary](#12-glossary)

---

## 0. What problem does this project solve?

### The engineer's real question

Suppose you are building a Wi-Fi access point, a radar, or a 5G base station. You need an antenna array that meets a spec such as:

- "The beam must point at 60°."
- "The beam must be narrower than 20°, so it doesn't interfere with neighbors."
- "Unwanted side lobes must be at least 10 dB down."
- "The gain must be at least X dB."

Now you must decide: **how many elements on each ring?** Each of the 5 rings can hold 0 to 10 elements, which gives 11⁵ = **161,051 possible layouts**, and the steering angle changes the result again.

### Why this is hard

The physics only works in one direction. Given a layout, you can compute the pattern easily. Given a wanted pattern, there is **no formula** that gives you the layout. The relationship is a tangle of interfering waves, and changing one element on one ring changes the whole pattern in a way that is hard to predict by intuition.

### What engineers do without this project

They use trial and error, or a search algorithm (genetic algorithms, particle swarm optimization and similar):

1. Guess a layout.
2. Run the physics and check the result.
3. Adjust and repeat, often thousands of times.

That works, but it is **slow**, and it must be **restarted from scratch for every new specification**. If a customer changes the wanted beam width, you optimize again.

### What this project does instead

It pays the cost once, up front:

1. **Simulate 1000 random arrays** (fast, because the forward physics is cheap).
2. **Train a neural network** on those examples, so it learns the reverse mapping: performance → layout.
3. **Afterwards, any new specification gets an answer in a fraction of a millisecond**, with no search at all. You type in the wanted gain, side lobe, beam width and angle, and the network outputs a layout.

This is called **inverse design** with a **surrogate model**, and it is used in many fields: antennas, optics, materials, aerodynamics and chip design.

### What it is good for in practice

| Use | Benefit |
|---|---|
| **Fast design exploration** | Try hundreds of "what if" specs in seconds |
| **Starting point for an optimizer** | The network's answer is a good first guess, and a search can refine it, so it converges much faster |
| **Real-time reconfiguration** | A system that can switch elements on and off could pick a new layout instantly when the target direction or the requirements change |
| **Cheaper development** | Less time spent running heavy simulations for every design change |

### Honest limits

- It is a **learning and research demo**, not a finished product. The physics is simplified (ideal point sources, a 2D cut, no coupling between elements), and the network is small.
- The network gives an **approximate** answer. That is why the test script feeds its predicted layout back through the physics and measures the error, rather than trusting it blindly.
- Some specs are **impossible** (for example, a very narrow beam with very few elements), and the network will still return something, so its answer must always be verified.
- Only 1000 samples are used. A real design tool would use far more data and would be checked against a full electromagnetic simulator.

### The one-sentence summary

The project replaces a slow, repeated trial-and-error search for antenna layouts with a trained neural network that predicts a good layout instantly from the performance you want.

---

## 1. The big picture

### Normal direction (forward problem)
You choose an antenna layout, and physics tells you how it performs.

```
layout (elements per ring) + steering angle  ──physics──▶  gain, side lobe, beam width
```

### This project (inverse problem)
You choose the performance you want, and a neural network proposes a layout.

```
gain, side lobe, beam width, steering angle  ──neural network──▶  layout (elements per ring)
```

The inverse direction is hard because there is no simple formula for it. So the project:

1. Uses the **forward physics** to create 1000 examples (random layout → performance).
2. **Flips** the examples: performance becomes the network's input and layout becomes its output.
3. **Trains** a neural network on 90% of the examples.
4. **Tests** it on the other 10%. The predicted layout is fed back through the physics to see if it really delivers the wanted performance.

### Pipeline

| Step | File | Reads | Writes |
|---|---|---|---|
| 0 | `circular_antenna_array.py` | nothing | plots (prototype only) |
| 1 | `dataset.py` | nothing | `Minput.npy/.mat`, `Moutput.npy/.mat` |
| 2 | `repartition_training_test.py` | `Minput.npy`, `Moutput.npy` | `*_training.npy/.mat`, `*_test.npy/.mat` |
| 3 | `FFNN_training.py` | `*_training.npy` | `ffnn_model.pth`, `normalization_params.npz` |
| 4 | `test_model_FFNN.py` | `*_test.npy`, `Minput.npy`, model, norm params | plots and printed errors |

Steps 1 to 4 must be run **in order**, because each one needs the files the previous one created.

---

## 2. Physics background

### 2.1 Antenna and array
An antenna converts an electrical signal into a radio wave. One small antenna radiates in many directions at once, like a bare light bulb, so power is wasted. An **antenna array** is several antennas ("elements") working together so that the radiation is concentrated where you want it.

### 2.2 Interference: why arrays work
When waves overlap they add:

- crest + crest = bigger wave (**constructive interference**)
- crest + trough = cancellation (**destructive interference**)

In an array, each element radiates a wave. In some directions the waves add up (strong signal: a **lobe**), and in others they cancel (a **null**).

### 2.3 Wavelength and wave number
- Wavelength: `λ = c / f = 3×10⁸ / 2.45×10⁹ ≈ 0.122 m` (12.2 cm).
- Wave number: `k = 2π / λ`. Moving a distance `d` changes a wave's phase by `k·d` radians.

Antenna sizes are always measured in wavelengths. Two elements 0.5λ apart are always in opposite phase, whatever the frequency.

### 2.4 Phase and phase shift
**Phase** is where you are in a wave's cycle (like a clock hand). A **phase shift** delays a signal slightly. Electronically this is easy and cheap.

### 2.5 Beam steering
Elements are at different positions, so their waves take different paths to a receiver in direction θ₀ and arrive out of step. If each element is pre-delayed by exactly its path difference toward θ₀, then in that direction all waves arrive together and add up to the maximum. The beam points at θ₀ with **no moving parts**. Radar and 5G use this.

### 2.6 Circular ring geometry
Elements sit on concentric rings. For a ring of radius `a` with `N` elements:

- angles: `φn = 2π·n / N`, for `n = 0 … N-1` (evenly spaced)
- position: `(a·cos φn, a·sin φn)`

Example, `N = 4`: 0°, 90°, 180°, 270°.

### 2.7 The array factor (AF)
For each observation direction, AF is the sum of all elements' contributions:

```
AF(θ) = Σ over all elements  exp( j · phase )

phase = k · a · ( sinθ · cos(φ − φn)  −  sinθ0 · cos(φ0 − φn) )
                 └──── geometry ────┘   └───── steering ─────┘
```

- **Geometry term:** the element's natural phase as seen from direction θ.
- **Steering term:** the delay you add so that at θ = θ₀ the two terms cancel and every element has phase 0.

**Important consequence:** at θ = θ₀ every element contributes `exp(0) = 1`, so

```
|AF(θ₀)| = N_total          (total number of elements)
main lobe gain (un-normalized) = 20·log10(N_total)
```

So the "main lobe gain" input feature is really a disguised **total element count**. With 25 elements: 20·log10(25) ≈ 27.96 dB.

### 2.8 Performance metrics used in this project

| Metric | Meaning | Why it matters |
|---|---|---|
| **Main lobe gain** | Peak strength, `20·log10(max\|AF\|)` (un-normalized) | More elements → higher gain |
| **HPBW** (Half-Power Beam Width) | Angular width of the main lobe where power is within 3 dB of the peak | Smaller = sharper beam |
| **SSL** (Side Lobe Level) | Level (in dB below the main lobe) of the strongest lobe that is not a main lobe | Smaller (more negative) = less wasted or interfering energy |
| **θ₀** | Steering direction | Where the beam points |

### 2.9 Decibels
`dB = 20·log10(amplitude ratio)`. Normalizing by the maximum makes the peak 0 dB and everything else negative. Every 20 dB is a factor of 10 in amplitude.

### 2.10 Simplifications made by the code
- Only **one 2D cut** of the 3D pattern is computed (`phi = 0`, θ from 0 to 360°).
- Elements are ideal **point sources** (equal radiation in every direction).
- All elements have **equal amplitude** (only phase is controlled).
- No mutual coupling between elements.

---

## 3. Machine-learning background

### 3.1 Supervised learning
You give the model many (input, correct output) pairs. It adjusts internal numbers (**weights**) until its predictions match the correct outputs. Then you check it on pairs it has never seen.

### 3.2 Neural network (FFNN)
A **Feed-Forward Neural Network** passes data in one direction through layers. Each layer computes:

```
output = activation( W · input + b )
```

- `W` (weights) and `b` (biases) are learned.
- The **activation** (here ReLU, `max(0, x)`) adds non-linearity. Without it, many layers would collapse into a single linear one and could not learn curves.

### 3.3 Training in five steps
Repeated for every mini-batch:

1. **Zero the gradients** (clear old information).
2. **Forward pass:** compute predictions.
3. **Loss:** measure the error (here MSE = mean of (prediction − truth)²).
4. **Backward pass (backpropagation):** compute the **gradient**, which says how each weight should change to reduce the loss.
5. **Optimizer step:** update the weights.

One pass through all the data is an **epoch**.

### 3.4 Normalization
Inputs have very different scales (HPBW up to 180, gain around 20). Min-Max normalization rescales each column to 0–1 so that all features count equally.

```
x_norm = (x − x_min) / (x_max − x_min)
x      = x_norm · (x_max − x_min) + x_min        (de-normalization)
```

The min and max **must come from the training data** and be reused unchanged at test time.

### 3.5 Train / test split and overfitting
A network can memorize the training data (**overfitting**). Holding out 10% as a **test set** that is never used for training reveals whether it truly generalizes.

---

## 4. How to run the project

Install: `pip install numpy scipy matplotlib torch`

Put all `.py` files in one folder, then run in this order:

```
python circular_antenna_array.py      # optional: see one example pattern
python dataset.py                     # creates Minput/Moutput
python repartition_training_test.py   # creates training/test files
python FFNN_training.py               # trains the network (~seconds)
python test_model_FFNN.py             # evaluates one test sample
```

Every script saves and loads files **in its own folder** (`os.path.dirname(os.path.abspath(__file__))`), so they must stay together.

---

## 5. File 0: `circular_antenna_array.py`

**Purpose:** a prototype that computes and plots the radiation pattern of one fixed array. Files 1 and 4 reuse its physics.

### 5.1 Imports
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
```
| Import | Use |
|---|---|
| `numpy` | array math |
| `matplotlib.pyplot` | plotting |
| `find_peaks` | finds local maxima (lobes) |
| `Axes3D` | 3D plotting, **imported but never used** |

### 5.2 Constants
```python
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
```
The frequency is 2.45 GHz (Wi-Fi band). `lambda_` has a trailing underscore because `lambda` is a reserved Python keyword.

### 5.3 The array definition
```python
rings = 5
radii = np.linspace(0.2 * lambda_, 2.2 * lambda_, rings)
elements_per_ring = [2, 9, 4, 5, 5]
assert len(radii) == len(elements_per_ring)
```
- `np.linspace(start, stop, n)` gives `n` evenly spaced values → radii of 0.2λ, 0.7λ, 1.2λ, 1.7λ, 2.2λ.
- The elements per ring are 2, 9, 4, 5, 5, so **25 elements** in total.
- `assert` halts the program with an error if the lists differ in length.

### 5.4 Steering
```python
theta0deg = 45
theta0 = np.deg2rad(theta0deg)
phi0 = 0
```
The wanted beam direction. NumPy trigonometric functions use radians, hence `deg2rad`.

### 5.5 Angle grid and accumulator
```python
theta = np.linspace(0, 2 * np.pi, 1000)
phi = 0
AF_az = np.zeros_like(theta, dtype=complex)
```
- 1000 observation angles between 0° and 360° (resolution ≈ 0.36°).
- `AF_az` starts as 1000 complex zeros and accumulates the sum of all element waves. It must be complex because waves have amplitude **and** phase.

### 5.6 The double loop
```python
for ring in range(rings):
    a = radii[ring]
    N = elements_per_ring[ring]
    if N == 0:
        continue
    phi_n = 2 * np.pi * np.arange(N) / N
    for n in range(N):
        phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n])
                         - np.sin(theta0) * np.cos(phi0 - phi_n[n]))
        AF_az += np.exp(1j * phase)
```
Step by step:

1. **Outer loop:** take one ring, with radius `a` and `N` elements.
2. **`if N == 0: continue`** skips empty rings.
3. **`phi_n`:** the element angles, `np.arange(N)` gives `[0,1,…,N-1]`, scaled by `2π/N`.
4. **Inner loop:** for each element compute `phase` (a vector of 1000 values, one per observation angle).
5. **`np.exp(1j*phase)`:** turns the phase into a unit-length complex number (Euler's formula: `e^(jx) = cos x + j sin x`).
6. **`+=`:** adds it to the running total. That sum **is** the interference.

### 5.7 Normalization and dB
```python
AF_norm_az = np.abs(AF_az) / np.max(np.abs(AF_az))
AF_dB_az   = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
AF_dB_az[AF_dB_az < -40] = -40
theta_deg  = np.rad2deg(theta)
```
- `np.abs` gives the magnitude of a complex number.
- Dividing by the maximum makes the peak = 1 (0 dB).
- `eps` (≈ 2.2×10⁻¹⁶) avoids `log10(0) = -∞`.
- `AF_dB_az[AF_dB_az < -40] = -40` is **boolean indexing**: every element below −40 is replaced by −40 (a floor for readable plots).

### 5.8 Un-normalized gain
```python
maxVal = np.max(AF_abs_az)
maxVal_non_norm = 20 * np.log10(maxVal + eps)
```
The real peak before normalization. As explained in §2.7, this is `20·log10(total number of elements)`, so it encodes array size.

### 5.9 The plots
- **Polar plot:** `plt.subplot(111, polar=True)` creates a circular chart; `ax.set_rlim([-40, 0])` sets the radial range.
- **Cartesian plot:** angle on X, dB on Y, with limits `[0,360]` and `[-40,0]` and a grid.

> **Note on labels:** θ here is the angle measured from the z-axis, with `phi = 0`, so this is a vertical cut (the x–z plane), i.e. an *elevation* cut. The polar plot title ("coupe en élévation") is right. The Cartesian axis label "Azimut" is the one that is misleading. Also, in the first explanation of this project the title was wrongly called a mistake; the polar title is the correct one.

### 5.10 Why there are two main lobes
The array lies in the x–y plane, so the pattern is symmetric about it: `sin θ = sin(180° − θ)`. A beam at 45° therefore has an identical mirror lobe at 135°. That is why the code later treats several peaks within 1 dB of the top as "main lobes" (plural).

### 5.11 HPBW calculation
```python
maxVal_dB = np.max(AF_dB_az)
maxIdx = np.argmax(AF_dB_az)
halfPower = maxVal_dB - 3
```
- `argmax` returns the **index** of the peak (the first one if there are ties).
- `halfPower` is the −3 dB level (half of the power).

**Wrap-around trick:**
```python
AF_dB_ext     = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
maxIdx_ext    = maxIdx + len(theta_deg)
```
The curve is tripled, with angles shifted by −360, 0 and +360. If the beam is near 0° or 360°, walking left or right from the peak would fall off the array. With three copies, the search can cross the boundary naturally. The peak in the middle copy is at `maxIdx + 1000`.

**Finding the crossings:**
```python
leftIdx_ext  = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1]   # last one before the peak
rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0][0] + maxIdx_ext  # first after
HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]
```
- `np.where(condition)[0]` returns the indices where the condition is true.
- `[-1]` = the last such index on the left; `[0]` = the first on the right.
- If either side is not found, `HPBW = 180` (a fallback meaning "very wide beam").

### 5.12 Side lobe detection
```python
peaks, _ = find_peaks(responseLin, distance=5)
pk = responseLin[peaks]
sorted_idx    = np.argsort(pk)[::-1]
sorted_pk     = pk[sorted_idx]
sorted_angles = theta_deg[peaks][sorted_idx]
```
1. `find_peaks` returns the indices of local maxima (`distance=5`: peaks must be at least 5 samples apart).
2. `pk` are their heights.
3. `argsort` gives the ordering from small to large; `[::-1]` reverses it to **descending**.
4. `sorted_angles` gives the angle of each peak in that order.

```python
threshold_dB = 1
main_lobes_idx = np.where(20*np.log10(sorted_pk) >= 20*np.log10(sorted_pk[0]) - threshold_dB)[0]
side_lobe_idx  = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)
```
- Every peak within 1 dB of the highest is a **main lobe**.
- `setdiff1d` = "all indices that are not main lobes" = **side lobes**.
- Since the list is sorted, `side_lobe_idx[0]` is the **strongest** side lobe, the SSL.

> A limitation: `find_peaks` never counts the very first or last sample as a peak. A lobe exactly at 0° or 360° could be missed.

### 5.13 Marking the peaks
Red dots mark the main lobes, a black dot marks the SSL, then `plt.legend` and `plt.show()`.

---

## 6. File 1: `dataset.py`

**Purpose:** repeat the physics of File 0 **1000 times with random layouts** and store the results. This is the training data factory.

### 6.1 Imports and helper
The imports are the same as File 0, plus `os` (paths) and `scipy.io` (`.mat` files). `platform`, `sys` and the second `import os` are unused leftovers.

```python
def clear_console():
    print("\n" + "="*80 + ...)
clear_console()
```
Prints a divider so each run is easy to spot in the console.

### 6.2 Parameters
```python
nb_samples = 1000
max_rings = 5
max_elements = 10
theta0_max_deg = 180
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
np.random.seed(46)
```
| Parameter | Meaning |
|---|---|
| `nb_samples` | number of random arrays to generate |
| `max_rings` | fixed number of rings (5) |
| `max_elements` | maximum elements on one ring (0–10 possible) |
| `theta0_max_deg` | steering angle drawn from 0° to 180° |
| `r0`, `delta_r` | first radius and spacing: radii = 0.2λ, 0.7λ, 1.2λ, 1.7λ, 2.2λ |
| `seed(46)` | makes the "random" numbers the same on every run (reproducibility) |

### 6.3 Empty matrices
```python
Moutput = np.zeros((max_rings, nb_samples))   # 5 × 1000
Minput  = np.zeros((4, nb_samples))           # 4 × 1000
```
**Each column is one sample.**

- `Minput` rows: `[main lobe gain, SSL, HPBW, θ₀]`: what the **network receives**.
- `Moutput` rows: elements on ring 1…5: what the **network must predict**.

The names are from the network's point of view, which is the reverse of the physics direction.

### 6.4 The main loop, one sample at a time

**(a) Random layout**
```python
while True:
    elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
    if np.sum(elements_per_ring > 0) > 0:
        break
Moutput[:, sample_idx] = elements_per_ring
```
- `randint(0, 11, size=5)` gives 5 random integers from 0 to 10.
- `elements_per_ring > 0` gives an array of True/False; `np.sum` counts the True values (rings that are used).
- If all five rings are 0 there is no antenna, so `while True` tries again. `break` exits the loop once at least one ring is non-empty.
- `Moutput[:, sample_idx] = ...` writes the layout into that column.

**(b) Random steering angle**
```python
theta0deg = np.random.uniform(0, theta0_max_deg)
```
A random decimal between 0 and 180.

**(c) Physics.** Everything from File 0 (radii, AF loops, normalization, dB, HPBW, peaks) is repeated, with two extra safeties:
- `eps` added in the normalization division, so an all-zero AF cannot cause division by zero.
- `if len(pk) == 0:` handles the case where no peak is found (then SSL = 0).

**(d) Store the four numbers**
```python
Minput[0, sample_idx] = main_lobe_gain
Minput[1, sample_idx] = true_SSL_gain
Minput[2, sample_idx] = HPBW
Minput[3, sample_idx] = theta0deg
```
If there is no side lobe, `true_SSL_gain = 0` is stored, which looks the same as "a side lobe at 0 dB". This is a mild ambiguity in the data.

### 6.5 Preview and saving
- Prints the shape and first five columns (`np.round(..., 4)`; `.astype(int)` for layouts).
- `script_dir = os.path.dirname(os.path.abspath(__file__))` is the folder containing the script.
- `os.path.join(script_dir, 'Minput.npy')` builds a full path safely on any operating system.
- `np.save` writes `.npy` (what the next scripts load). `scipy.io.savemat` writes `.mat` (for MATLAB users).

---

## 7. File 2: `repartition_training_test.py`

**Purpose:** split the 1000 samples into **900 training** and **100 test** samples.

### 7.1 Load
```python
Minput  = np.load('.../Minput.npy')     # 4 × 1000
Moutput = np.load('.../Moutput.npy')    # 5 × 1000
```

### 7.2 Random split
```python
total_cols = Minput.shape[1]                       # 1000
rand_indices = np.random.permutation(total_cols)   # shuffled 0…999
nb_test = round(0.10 * total_cols)                 # 100
test_indices  = rand_indices[:nb_test]             # first 100
train_indices = rand_indices[nb_test:]             # last 900
```
- `.shape[1]` = number of columns.
- `permutation(n)` gives a random ordering of `0…n-1`, so the split is random.
- Slicing: `[:100]` first 100 items, `[100:]` all the rest.

### 7.3 Select columns
```python
Minput_test     = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test    = Moutput[:, test_indices]
Moutput_training= Moutput[:, train_indices]
```
`[:, idx]` means "all rows, only these columns". Using the **same indices** for input and output keeps every pair matched.

### 7.4 Save
Four datasets are saved, each in `.mat` and `.npy`. The repeated `script_dir = ...` and `file_path = ...` lines do the same job many times; they could be a small function, but they work.

---

## 8. File 3: `FFNN_training.py`

**Purpose:** train the neural network to map performance → layout.

### 8.1 Load and orient the data
```python
Minput_training  = np.load(...)    # 4 × 900
Moutput_training = np.load(...)    # 5 × 900
if Minput_training.shape[0] < Minput_training.shape[1]:
    Minput_training = Minput_training.T     # → 900 × 4
```
Files store *features × samples*, but PyTorch wants *samples × features*, so the arrays are **transposed** (`.T`). The `if` only flips when there are fewer rows than columns.

### 8.2 Normalization
```python
Minput_min = Minput_training.min(axis=0)
Minput_max = Minput_training.max(axis=0)
Minput_norm = (Minput_training - Minput_min) / (Minput_max - Minput_min + 1e-8)
```
- `axis=0` means "compute down the rows", so you get **one min/max per feature column**.
- `+ 1e-8` prevents dividing by zero if a column is constant.
- The same is done for the outputs, so the layout numbers (0–10) become 0–1.

### 8.3 Tensors and DataLoader
```python
X = torch.tensor(Minput_norm,  dtype=torch.float32)
Y = torch.tensor(Moutput_norm, dtype=torch.float32)
dataset    = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```
- A **tensor** is PyTorch's array (can also track gradients and run on GPUs).
- `float32` is the standard precision for networks.
- `TensorDataset` pairs each `X[i]` with `Y[i]`.
- `DataLoader` serves **mini-batches** of 64 in a new random order every epoch. With 900 samples that is 15 batches per epoch (14 full + 1 of 4 samples).

### 8.4 The model class
```python
class FFNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size, dropout_rate=0.0):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        ...
        self.fc4 = nn.Linear(hidden3, output_size)
```
- A **class** is a blueprint for objects. `class FFNN(nn.Module)` says "my network *is a* PyTorch module", inheriting all the training machinery.
- `__init__` runs once when you create the model, and here it **creates the layers**.
- `super().__init__()` initializes the parent class. Forgetting it breaks PyTorch.
- `nn.Linear(a, b)` is a dense layer with `a` inputs and `b` outputs (`a×b` weights + `b` biases).
- `nn.ReLU()` is `max(0, x)`.
- `nn.Dropout(p)` randomly zeroes a fraction `p` of neurons **during training only** to reduce overfitting. Here `p = 0`, so it is switched off.

**Architecture:**

```
input (4) → Linear → 14 → ReLU
          → Linear → 20 → ReLU
          → Linear →  8 → ReLU
          → Linear →  5   (no activation = raw regression output)
```
Trainable parameters: (4·14+14) + (14·20+20) + (20·8+8) + (8·5+5) = 70 + 300 + 168 + 45 = **583**. That is tiny, which is fine for such a small problem.

```python
def forward(self, x):
    x = self.relu1(self.fc1(x))
    x = self.dropout1(x)
    ...
    return self.fc4(x)
```
`forward` defines how data flows. You never call it directly: `model(x)` calls it for you.

### 8.5 Model, loss, optimizer
```python
model = FFNN(input_size=4, hidden1=14, hidden2=20, hidden3=8, output_size=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```
- **MSE:** `mean((prediction − truth)²)`. Squaring punishes large errors more.
- **Adam:** a popular optimizer that adapts the step size per weight.
- **`lr` (learning rate):** step size. Too large → unstable; too small → slow. (A comment in the file mentions 0.001, but the real value is **0.005**.)
- `model.parameters()` gives the optimizer all the weights and biases to update.

### 8.6 The training loop
```python
for epoch in range(100):
    total_loss = 0.0
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_list.append(total_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
```
| Line | What it does |
|---|---|
| `zero_grad()` | PyTorch **accumulates** gradients, so clear the old ones first |
| `model(batch_X)` | forward pass: predictions |
| `criterion(...)` | scalar error for this batch |
| `loss.backward()` | backpropagation: fills in the gradient of every weight |
| `optimizer.step()` | moves every weight a small step downhill |
| `loss.item()` | converts the 1-value tensor to a normal Python float |

> `total_loss` is the **sum** of the 15 batch losses, not their average. It still decreases as the model learns, but the number is 15× larger than a per-batch loss.

Timing is done with `time.time()` before and after. The loss curve is plotted with `plt.plot(loss_list)`; a healthy curve falls quickly then flattens.

### 8.7 Saving
```python
torch.save(model.state_dict(), 'ffnn_model.pth')
np.savez('normalization_params.npz', Minput_min=..., Minput_max=..., Moutput_min=..., Moutput_max=...)
```
- `state_dict()` is a dictionary of all learned weights (only weights, **not** the class definition).
- The normalization parameters must be saved: the test script needs to scale new data with these exact numbers.

---

## 9. File 4: `test_model_FFNN.py`

**Purpose:** take one unseen test sample, predict a layout, then check by physics whether the predicted layout performs like the wanted one.

### 9.1 Physical parameters
```python
carrierFreq = 2.45e9; c = 3e8; lambda_ = c/carrierFreq
r0 = 0.2*lambda_; delta_r = 0.5*lambda_; max_rings = 5
```
The comment stresses these **must equal** the values in `dataset.py`. Otherwise the "reference" and "predicted" patterns are computed in a different world from the training data.

### 9.2 Loading
```python
Minput_FFNN_test  = np.load('Minput_test.npy').T     # 100 × 4
Moutput_FFNN_test = np.load('Moutput_test.npy').T    # 100 × 5
Minput_complete   = np.load('Minput.npy')            # 4 × 1000
norm_params = np.load('normalization_params.npz')
```
- `.T` restores *samples × features*.
- `Minput_complete` (all 1000 samples) is only used later to find the largest values for error scaling.
- The `.npz` file behaves like a dictionary: `norm_params['Minput_min']`.

```python
Minput_norm_test = (Minput_FFNN_test - Minput_min) / (Minput_max - Minput_min + 1e-8)
```
Test inputs are normalized with the **training** min/max, never with their own. This simulates real use, where new data arrives without any knowledge of its statistics.

### 9.3 Choose a sample
```python
indice_test = 10
x_test = torch.tensor(Minput_norm_test[indice_test], dtype=torch.float32)
```
Indexing starts at 0, so 10 is the 11th test sample. Try any value from 0 to 99.

### 9.4 Rebuild the model and load weights
The `FFNN` class is copied exactly, because `.pth` only stores weights. Then:
```python
model = FFNN(input_size, 14, 20, 8, output_size, 0.0)
model.load_state_dict(torch.load('ffnn_model.pth'))
model.eval()
```
- Sizes must match training, or loading fails with a shape error.
- `eval()` switches off training-only behaviors such as dropout.

### 9.5 Prediction
```python
with torch.no_grad():
    y_pred_norm = model(x_test)
y_pred = y_pred_norm.numpy() * (Moutput_max - Moutput_min + 1e-8) + Moutput_min
architecture_predicted = np.round(y_pred).astype(int)
architecture_reference = Moutput_FFNN_test[indice_test].astype(int)
```
1. `no_grad()` disables gradient tracking (faster, less memory; not needed when not training).
2. `.numpy()` converts the tensor to a NumPy array.
3. **De-normalization** converts the 0–1 output back to element counts.
4. `np.round(...).astype(int)` makes whole numbers, since fractions of an antenna don't exist.

### 9.6 Physics as a function
```python
def calcul_AF_performance_metrics(elements_per_ring):
    ...
    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain
```
It is the same computation as in Files 0 and 1, wrapped in a function so it can be called twice (reference and prediction). It uses the global `theta0` taken from the test sample:

```python
theta0deg = Minput_FFNN_test.T[3, indice_test]
```
`Minput_FFNN_test.T` is 4 × 100, so `[3, indice_test]` = 4th feature (θ₀) of the chosen sample.

### 9.7 Comparison
The function runs for `architecture_reference` and for `architecture_predicted`, both with the same θ₀. Then:

- **Plots:** reference (solid) and prediction (dashed) overlaid in polar and Cartesian form. The closer they are, the better the network.
- **Absolute errors:** `|pred − ref|` for gain (dB), SSL (dB) and HPBW (degrees).
- **Relative errors (%)**, each divided by a natural scale:

| Metric | Divided by |
|---|---|
| Main gain | max main gain in the dataset |
| SSL | max \|SSL\| in the dataset |
| HPBW | 180° |

- **Global weighted error:**
```
error = 0.33·err_gain + 0.33·err_SSL + 0.34·err_HPBW
```
The weights (which sum to 1) let you decide which metric matters most.

> The variable names say "MAE", but with a single sample it is just an absolute error. A true MAE averages over many samples.

---

## 10. Data shapes cheat sheet

| Variable | Shape | Meaning |
|---|---|---|
| `Minput` | 4 × 1000 | rows: gain, SSL, HPBW, θ₀; columns: samples |
| `Moutput` | 5 × 1000 | rows: elements on ring 1…5 |
| `Minput_training` | 4 × 900 → 900 × 4 after `.T` | network inputs |
| `Moutput_training` | 5 × 900 → 900 × 5 after `.T` | network targets |
| `Minput_test` | 4 × 100 → 100 × 4 after `.T` | unseen inputs |
| `Moutput_test` | 5 × 100 → 100 × 5 after `.T` | unseen true layouts |
| `theta` | 1000 | observation angles |
| `AF_az` | 1000 (complex) | array factor per angle |

---

## 11. Known issues and improvements

1. **Not fully reproducible.** `np.random.seed` seeds only NumPy. Add `torch.manual_seed(46)` in the training script so weight initialization and batch shuffling repeat.
2. **Possible crash or garbage at test time.** After rounding, a predicted layout may contain negative numbers, values above 10, or all zeros. Add:
   ```python
   architecture_predicted = np.clip(np.round(y_pred), 0, 10).astype(int)
   ```
   and handle the all-zero case (the physics would divide by zero).
3. **The inverse problem is not one-to-one.** Many layouts give almost the same performance, and MSE on layouts penalizes the network for choosing a *different but equally good* layout. Comparing performance (as File 4 does) is the fair check.
4. **Only one test sample is evaluated.** Loop over all 100 and average the errors to get a meaningful score.
5. **Ambiguous SSL = 0.** "No side lobe" and "side lobe at 0 dB" look the same. Consider a flag or a different placeholder.
6. **The main gain feature is redundant** with total element count (`20·log10(N_total)`), so the network may be able to infer total elements directly from it.
7. **No validation set.** Hyperparameters (layer sizes, learning rate, epochs) are tuned by hand. A validation split, or comparing training and test loss, helps to detect overfitting.
8. **Code duplication.** The physics is copy-pasted in three files. Put it in one shared module and import it, so a change cannot leave the files inconsistent.
9. **Unused imports** (`Axes3D`, `platform`, `sys`, duplicate `os`) can be removed.
10. **Newer PyTorch:** `torch.load(path, weights_only=True)` avoids a security warning.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Element** | one small antenna in the array |
| **Ring** | a circle on which elements are evenly spaced |
| **λ (wavelength)** | distance over which the wave repeats (≈ 12.2 cm here) |
| **k (wave number)** | `2π/λ`, converts distance to phase |
| **Phase** | position within a wave's cycle |
| **Array factor (AF)** | total signal of all elements for each direction |
| **Beam steering** | pointing the beam by choosing per-element delays |
| **Lobe / null** | direction of strong / near-zero radiation |
| **Main lobe** | the intended beam |
| **Side lobe (SSL)** | unwanted secondary lobe; the strongest one is reported |
| **HPBW** | width of the main lobe at −3 dB |
| **dB** | logarithmic unit, `20·log10(amplitude ratio)` |
| **FFNN** | feed-forward neural network |
| **Weight / bias** | learned parameters of a layer |
| **ReLU** | activation `max(0, x)` |
| **Loss (MSE)** | measure of prediction error |
| **Gradient** | direction in which the loss increases; used to update weights |
| **Backpropagation** | algorithm that computes gradients through all layers |
| **Optimizer (Adam)** | rule for updating weights from gradients |
| **Learning rate** | size of each update step |
| **Epoch** | one full pass through the training data |
| **Batch** | a small group of samples processed together |
| **Overfitting** | memorizing training data instead of generalizing |
| **Normalization** | rescaling features to a common range (0–1) |
| **Tensor** | PyTorch's array type |
| **`.npy` / `.npz` / `.mat` / `.pth`** | NumPy array / NumPy multi-array archive / MATLAB matrix / PyTorch weights |
