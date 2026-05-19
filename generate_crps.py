import random
import numpy as np
import csv
import sys
from itertools import product

# =========================
# PARAMETERS
# =========================
mean_fall, std_fall = 4.36061, 2.69344
lower_fall, upper_fall = 0.09119, 17.63640

mean_rise, std_rise = 4.94125, 2.15374
lower_rise, upper_rise = 0.56923, 15.07770

challenge_length = int(sys.argv[1])
num_chains = int(sys.argv[2])
num_repeats = int(sys.argv[3]) 
filename = sys.argv[4]


# =========================
# DELAY GENERATION
# =========================
def generate_delay(mean, std, low, high):
    while True:
        v = random.gauss(mean, std)
        if low <= v <= high:
            return v


# =========================
# BUILD PUF (STATIC STRUCTURE)
# =========================
def build_puf():
    return [
        [
            [
                generate_delay(mean_fall, std_fall, lower_fall, upper_fall),
                generate_delay(mean_rise, std_rise, lower_rise, upper_rise),
            ]
            for _ in range(challenge_length)
        ]
        for _ in range(num_chains)
    ]


# =========================
# SINGLE EVALUATION
# =========================
def evaluate_once(puf, challenge):

    chain_deltas = []

    for chain in puf:

        delay_a = 0.0
        delay_b = 0.0
        imbalance = 0.0

        for i, sel in enumerate(challenge):

            sel_n = 1 - sel

            base_a = chain[i][sel]
            base_b = chain[i][sel_n]

            weight = 1.0 + (i * 0.03)
            bit_bias = 1 if sel == 1 else -1

            imbalance += bit_bias * weight

            delay_a += base_a * (1 + 0.01 * imbalance)
            delay_b += base_b * (1 - 0.01 * imbalance)

        chain_deltas.append(delay_b - delay_a)

    # collapse chains → single CRP output
    return np.mean(chain_deltas)


# =========================
# REPEATED MEASUREMENTS (KEY PART)
# =========================
def evaluate_crp(puf, challenge, repeats):

    deltas = []

    for _ in range(repeats):
        deltas.append(evaluate_once(puf, challenge))

    return np.array(deltas)


# =========================
# CHALLENGES
# =========================
def all_bits(n):
    return list(product([0, 1], repeat=n))


# =========================
# GENERATE DATASET
# =========================
puf = build_puf()
challenges = all_bits(challenge_length)

dataset = []

for ch in challenges:

    deltas = evaluate_crp(puf, ch, num_repeats)

    dataset.append((ch, deltas))


# =========================
# SAVE
# =========================
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)

    # each row now contains a distribution of Δ
    writer.writerow(["challenge", "deltas"])

    for ch, deltas in dataset:
        ch_str = "".join(map(str, ch))

        delta_str = ",".join(f"{d:.6f}" for d in deltas)

        writer.writerow([ch_str, delta_str])