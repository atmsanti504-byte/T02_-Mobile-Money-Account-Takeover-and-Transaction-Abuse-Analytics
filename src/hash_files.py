from pathlib import Path
import hashlib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FILES_TO_HASH = [
    DATA_DIR / "raw" / "accounts.csv",
    DATA_DIR / "raw" / "authentication.csv",
    DATA_DIR / "raw" / "transactions.csv",
    DATA_DIR / "raw" / "device_sim.csv",
    DATA_DIR / "raw" / "geolocation.csv",
    DATA_DIR / "raw" / "support_tickets.csv",
    DATA_DIR / "features" / "ato_features.csv",
    DATA_DIR / "ato_risk_scores.csv",
    DATA_DIR / "adversarial_test_results.csv",
    DATA_DIR / "incident_timeline.csv",
    DATA_DIR / "simulation_results.csv",
    DATA_DIR / "simulation_summary.csv",
]


def sha256_file(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


print("=" * 60)
print("SHA-256 DATA INTEGRITY HASHES")
print("=" * 60)

results = []

for path in FILES_TO_HASH:

    if not path.exists():
        print(f"SKIPPED — not found: {path}")
        continue

    file_hash = sha256_file(path)

    results.append({
        "file": str(path.relative_to(BASE_DIR)),
        "sha256": file_hash
    })

    print(f"\n{path.name}")
    print(file_hash)

output = DATA_DIR / "file_hashes.txt"
with open(output, "w", encoding="utf-8") as f:
    f.write("SHA-256 DATA INTEGRITY HASHES\n")
    f.write("=" * 70 + "\n\n")
    for result in results:
        f.write(f"FILE: {result['file']}\n")
        f.write(f"SHA256: {result['sha256']}\n\n")

print("\n" + "=" * 60)
print("HASHING COMPLETE")
print("=" * 60)
print(f"Saved: {output}")