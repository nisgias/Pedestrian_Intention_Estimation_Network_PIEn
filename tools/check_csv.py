import pandas as pd
from pathlib import Path

def check_titan_csv():
    titan_path = Path("/data/TITAN/titan_data/dataset")
    
    # Βρίσκουμε όλα τα CSV, αγνοώντας τον φάκελο imu_data
    csv_files = [p for p in titan_path.rglob("*.csv") if "imu_data" not in str(p)]
    
    if not csv_files:
        print("❌ Δεν βρέθηκαν αρχεία CSV.")
        return
        
    first_csv = csv_files[0]
    print(f"✅ Βρέθηκε αρχείο Annotation: {first_csv}")
    
    # Διαβάζουμε τις πρώτες 5 γραμμές
    df = pd.read_csv(first_csv, nrows=5)
    
    print("\n=== ΟΝΟΜΑΤΑ ΣΤΗΛΩΝ (COLUMNS) ===")
    for col in df.columns:
        print(f" - {col}")
    
    print("\n=== ΠΡΩΤΗ ΓΡΑΜΜΗ ΔΕΔΟΜΕΝΩΝ ===")
    for key, value in df.iloc[0].to_dict().items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    check_titan_csv()