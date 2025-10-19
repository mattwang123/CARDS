from datasets import load_dataset
import os

os.makedirs("dataset/opc", exist_ok=True)

ds = load_dataset("INSAIT-Institute/OPC")

for split_name, split_data in ds.items():
    split_data.to_json(f"dataset/opc/OPC_{split_name}.json")
    print(f"Saved dataset/opc/OPC_{split_name}.json")