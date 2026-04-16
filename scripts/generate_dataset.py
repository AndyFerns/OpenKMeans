import csv
import random
import os

OUTPUT = "data/patients_large.csv"
ROWS = 200_000   # adjust (100k–300k is sweet spot)

def generate_row():
    age = random.randint(20, 95)
    bp = random.randint(100, 200)
    glucose = random.randint(70, 180)
    bmi = round(random.uniform(18.0, 40.0), 1)

    # Add slight correlation (makes clusters meaningful)
    if age > 50:
        bp += random.randint(5, 15)
        glucose += random.randint(5, 20)

    return [age, bp, glucose, bmi]


def main():
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Age", "BloodPressure", "Glucose", "BMI"])

        for i in range(ROWS):
            writer.writerow(generate_row())

            if i % 50000 == 0:
                print(f"[+] Generated {i} rows...")

    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\n✅ Done: {OUTPUT}")
    print(f"📦 Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()