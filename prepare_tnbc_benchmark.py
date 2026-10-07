"""Reconstruct the TNBC data file from chunks."""
import os
from pathlib import Path

def main():
    # Путь до папки с частями
    data_dir = Path(__file__).resolve().parent / "data" / "tnbc"
    target_file = data_dir / "tnbc.h5ad"
    parts = sorted(data_dir.glob("tnbc.h5ad.part_*"))

    if target_file.exists():
        print(f"{target_file} already exists. Ready to go!")
        return

    if not parts:
        print(f"Error: Could not find chunk files in {data_dir}")
        return

    print(f"Reconstructing {target_file.name} from {len(parts)} chunks...")
    with open(target_file, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                outfile.write(infile.read())
    print("TNBC dataset successfully prepared!")

if __name__ == "__main__":
    main()