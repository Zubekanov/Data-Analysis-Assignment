import os
import pandas as pd

def dir_heads(relative_path: str, n: int = 5):
	"""
	Displays basic info and the first n rows of all CSVs in the targeted directory.
	"""
	if not os.path.isdir(relative_path):
		print(f"Could not find directory: {relative_path}")
		return
	
	for filename in os.listdir(relative_path):
		if filename.lower().endswith(".csv"):
			filepath = os.path.join(relative_path, filename)
			try:
				df = pd.read_csv(filepath)
				print(f"\n--- {filename} ---")
				# Shape
				rows, cols = df.shape
				print(f"Shape: {rows} rows, {cols} columns")
				# Null counts
				null_counts = df.isna().sum()
				print("Missing values per column:")
				for col, count in null_counts.items():
					print(f"\t{col}: {count} ({int((count/rows) * 100)}% missing)")
				# Display head
				print(f"\nFirst {n} rows:")
				print(df.head(n))
			except Exception as e:
				print(f"Could not read {filepath}, {e}")

if __name__ == "__main__":
	dir_heads("data_raw")
