import pandas as pd
import re
import sys

# =========================
# USAGE:
# python convert_mc.py input.csv output.csv
# =========================

input_file = sys.argv[1]
output_file = sys.argv[2]

# Read everything as strings (important for mixed CSV)
df = pd.read_csv(input_file, header=None, dtype=str, keep_default_na=False)

data = {}

current_bits = None


# =========================
# SEL PARSER
# =========================
def parse_sel_bits(param_str):
    bits = re.findall(r"Sel(\d)=([0-9.]+|[0-9]+m)", param_str)
    bits_sorted = sorted(bits, key=lambda x: int(x[0]))

    result = []
    for _, val in bits_sorted:
        v = val.strip().lower()

        # handle "900m" or "0.9"
        if v.endswith("m"):
            v = float(v.replace("m", "")) / 1000.0
        else:
            v = float(v)

        result.append("1" if v >= 0.9 else "0")

    return "".join(result)


# =========================
# PARSE FILE
# =========================
for i in range(len(df)):
    row = df.iloc[i].tolist()

    cell0 = str(row[0]) if len(row) > 0 else ""
    cell1 = str(row[1]) if len(row) > 1 else ""

    # -------------------------
    # Parameter row
    # -------------------------
    if "Parameters:" in cell0:
        current_bits = parse_sel_bits(cell0)
        continue

    # -------------------------
    # Data row
    # -------------------------
    if "CapacitivePUF" in cell1:
        try:
            delay = float(f"{float(row[7]) * 1e9:.6f}")
        except:
            continue

        if current_bits is None:
            continue

        # store only ONE value per challenge (overwrite or keep first)
        data[current_bits] = delay


# =========================
# BUILD OUTPUT (10x repetition)
# =========================
records = []

for challenge, delay in data.items():
    deltas = ",".join([str(delay)] * 10)
    records.append((challenge, deltas))


# =========================
# SAVE
# =========================
out_df = pd.DataFrame(records, columns=["challenge", "deltas"])
out_df.to_csv(output_file, index=False, quoting=1)

print(f"Done → {output_file}")