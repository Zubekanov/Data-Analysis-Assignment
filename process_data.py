import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dir_describe_csvs(relative_path: str, n: int = 5):
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

def txt_to_csv(filename: str):
	"""
	Read a whitespace-delimited text file from data_raw and save it as a CSV in data_cleaned.
	Assumes the first line is the header and subsequent lines are data.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))

	raw_dir = os.path.join(script_dir, "data_raw")
	clean_dir = os.path.join(script_dir, "data_cleaned")
	os.makedirs(clean_dir, exist_ok=True)

	input_path = os.path.join(raw_dir, filename)
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Could not find file: {input_path}")

	df = pd.read_csv(input_path, sep=r"\s+", engine="python")

	base_name = os.path.splitext(filename)[0]
	output_name = f"{base_name}.csv"
	output_path = os.path.join(clean_dir, output_name)

	df.to_csv(output_path, index=False)

def keep_values(filename: str, values: dict):
	"""
	Filter rows in the CSV `filename` in data_raw to keep only those where *any* column matches 
	a value from the provided `values` dictionary. The dictionary maps column headers to lists 
	of allowed values. The resulting CSV is saved to data_cleaned with the same name.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	raw_path = os.path.join(script_dir, "data_raw", filename)
	output_dir = os.path.join(script_dir, "data_cleaned")
	os.makedirs(output_dir, exist_ok=True)

	df = pd.read_csv(raw_path)

	# Build mask: keep any row that satisfies any condition
	mask = pd.Series([False] * len(df))
	for col, allowed in values.items():
		if col in df.columns:
			mask |= df[col].isin(allowed)
		else:
			print(f"Warning: column '{col}' not found in {filename}")

	filtered_df = df[mask]

	# Save result
	output_path = os.path.join(output_dir, filename)
	filtered_df.to_csv(output_path, index=False)

def compute_cumulative_carbon(emissions_file: str):
    """
    Reads CO₂ emissions from data_cleaned/emissions_file,
    converts to carbon (x 0.272921), computes cumulative carbon (1880→year),
    and forces the 1880-value of Cumulative_Carbon to zero.
    Saves result as data_cleaned/Cumulative_Carbon.csv.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "data_cleaned", emissions_file)

    df = pd.read_csv(path)
    df = df.sort_values("Year").copy()

    df["Annual_Carbon"] = df["Annual CO₂ emissions"] * 0.272921

    df = df[df["Year"] >= 1880].reset_index(drop=True)
    df["Cumulative_Carbon"] = df["Annual_Carbon"].cumsum()
    first_cum = df["Cumulative_Carbon"].iloc[0]
    df["Cumulative_Carbon"] = df["Cumulative_Carbon"] - first_cum

    out_path = os.path.join(script_dir, "data_cleaned", "Cumulative_Carbon.csv")
    df.to_csv(out_path, index=False, columns=["Year", "Cumulative_Carbon"])


def join_temp_and_cumulative(temp_file: str, cumulative_file: str):
	"""
	Joins temperature anomaly and cumulative Carbon on Year,
	and saves result as data_cleaned/Temp_And_Cumulative.csv
	Also scales first item to 0 to account for paper taking industrial revolution baseline.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	temp_path = os.path.join(script_dir, "data_cleaned", temp_file)
	co2_path = os.path.join(script_dir, "data_cleaned", cumulative_file)

	temp_df = pd.read_csv(temp_path)
	co2_df = pd.read_csv(co2_path)

	merged = pd.merge(temp_df, co2_df, on="Year")

	# Subtract 1880 anomaly so that baseline = 0
	baseline = merged.loc[merged["Year"] == 1880, "Temperature_Anomaly"].iloc[0]
	merged["Temperature_Anomaly"] = merged["Temperature_Anomaly"] - baseline

	out_path = os.path.join(script_dir, "data_cleaned", "Temp_And_Cumulative.csv")
	merged.to_csv(out_path, index=False, columns=["Year", "Temperature_Anomaly", "Cumulative_Carbon"])

def plot_temperature_anomaly():
	"""
	Plots Temperature Anomaly over time and saves to plots/temperature_anomaly.png
	"""
	import os
	import pandas as pd
	import matplotlib.pyplot as plt

	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(script_dir, "data_cleaned", "NASA_Temperature_Anomaly.csv")
	plot_dir = os.path.join(script_dir, "plots")
	os.makedirs(plot_dir, exist_ok=True)

	df = pd.read_csv(data_path)

	plt.figure(figsize=(10, 6))
	plt.plot(df["Year"], df["Temperature_Anomaly"], color="tab:red")
	plt.xlabel("Year")
	plt.ylabel("Temperature Anomaly (°C)")
	plt.title("Global Temperature Anomaly Over Time")
	plt.grid(True)
	plt.tight_layout()

	out_path = os.path.join(plot_dir, "temperature_anomaly.png")
	plt.savefig(out_path)
	plt.close()

def plot_cumulative_carbon():
	"""
	Plots Cumulative Carbon (in trillion tonnes) over time and saves to plots/Cumulative_Carbon.png
	"""
	import os
	import pandas as pd
	import matplotlib.pyplot as plt

	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(script_dir, "data_cleaned", "Cumulative_Carbon.csv")
	plot_dir = os.path.join(script_dir, "plots")
	os.makedirs(plot_dir, exist_ok=True)

	df = pd.read_csv(data_path)
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12

	plt.figure(figsize=(10, 6))
	plt.plot(df["Year"], df["Cumulative_Carbon_Tt"], color="tab:blue")
	plt.xlabel("Year")
	plt.ylabel("Cumulative Carbon Emissions (Trillion Tonnes of Carbon)")
	plt.title("Cumulative Global Carbon Emissions Over Time")
	plt.grid(True)
	plt.tight_layout()

	out_path = os.path.join(plot_dir, "Cumulative_Carbon.png")
	plt.savefig(out_path)
	plt.close()


def plot_temp_and_cumulative_carbon_timeseries():
	"""
	Plots Temperature Anomaly and Cumulative Carbon (in trillion tonnes) over time,
	with both y‐axes using the same scale.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(script_dir, "data_cleaned", "Temp_And_Cumulative.csv")
	plot_dir = os.path.join(script_dir, "plots")
	os.makedirs(plot_dir, exist_ok=True)

	df = pd.read_csv(data_path)
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12

	fig, ax1 = plt.subplots(figsize=(10, 6))

	ax1.set_xlabel("Year")
	ax1.set_ylabel("Temperature Anomaly (°C)", color="tab:red")
	ax1.plot(
		df["Year"],
		df["Temperature_Anomaly"],
		color="tab:red",
		label="Temperature Anomaly"
	)
	ax1.tick_params(axis="y", labelcolor="tab:red")

	ax2 = ax1.twinx()
	ax2.set_ylabel("Cumulative Carbon Emissions (TtC)", color="tab:blue")
	ax2.plot(
		df["Year"],
		df["Cumulative_Carbon_Tt"],
		color="tab:blue",
		label="Cumulative Carbon"
	)
	ax2.tick_params(axis="y", labelcolor="tab:blue")

	# Find a shared y‐range
	ymin = min(df["Temperature_Anomaly"].min(), df["Cumulative_Carbon_Tt"].min())
	ymax = max(df["Temperature_Anomaly"].max(), df["Cumulative_Carbon_Tt"].max())
	ax1.set_ylim(ymin, ymax)
	ax2.set_ylim(ymin, ymax)

	plt.title("Global Temperature Anomaly and Cumulative Carbon Over Time")
	fig.tight_layout()

	output_path = os.path.join(plot_dir, "temp_and_co2_timeseries.png")
	plt.savefig(output_path)
	plt.close()

def plot_temp_vs_cumulative_carbon():
	"""
	Creates a scatter plot of Temperature Anomaly vs. Cumulative Carbon (in trillion tonnes),
	using data from data_cleaned/csv_file, and saves the plot to plots/temp_vs_cumulative.png.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(script_dir, "data_cleaned", "Temp_And_Cumulative.csv")
	plot_dir = os.path.join(script_dir, "plots")
	os.makedirs(plot_dir, exist_ok=True)

	df = pd.read_csv(data_path)

	# Convert to trillions of tonnes
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12

	plt.figure(figsize=(8, 6))
	plt.scatter(df["Cumulative_Carbon_Tt"], df["Temperature_Anomaly"], s=15)
	plt.xlabel("Cumulative Carbon Emissions (TtC)")
	plt.ylabel("Temperature Anomaly (°C)")
	plt.title("Temperature Anomaly vs. Cumulative Carbon")
	plt.grid(True)
	plt.tight_layout()

	output_path = os.path.join(plot_dir, "temp_vs_cumulative.png")
	plt.savefig(output_path)
	plt.close()

def plot_scatter_with_regression():
	"""
	Performs a linear regression on Temperature Anomaly vs Cumulative CO₂ (in Tt),
	saves regression stats to plots/info.txt, and saves a scatter plot with the fitted line.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(script_dir, "data_cleaned", "Temp_And_Cumulative.csv")
	plot_dir = os.path.join(script_dir, "plots")
	os.makedirs(plot_dir, exist_ok=True)

	df = pd.read_csv(data_path)

	x = (df["Cumulative_Carbon"] / 1e12).values
	y = df["Temperature_Anomaly"].values

	m = np.sum(x * y) / np.sum(x * x)
	y_pred = m * x
	# m, b = np.polyfit(x, y, 1)
	# y_pred = m * x + b

	ss_res = np.sum((y - y_pred) ** 2)
	ss_tot = np.sum((y - np.mean(y)) ** 2)
	r2 = 1 - ss_res / ss_tot

	corr_matrix = np.corrcoef(x, y)
	r = corr_matrix[0, 1]

	info_path = os.path.join(plot_dir, "info.txt")
	with open(info_path, "w") as f:
		f.write(f"Slope: {m:.6f}\n")
		f.write(f"R-squared: {r2:.6f}\n")
		f.write(f"Correlation coefficient: {r:.6f}\n")

	plt.figure(figsize=(8, 6))
	plt.scatter(x, y, s=20, label="Data points")
	x_line = np.linspace(x.min(), x.max(), 100)
	plt.plot(x_line, m * x_line, color="tab:green", linewidth=2, label=f"Linear fit: CCR = {m:.1f}")
	plt.xlabel("Cumulative Carbon Emissions (TtC)", fontsize=12)
	plt.ylabel("Temperature Anomaly (°C)", fontsize=12)
	plt.title("Temperature Anomaly vs. Cumulative Carbon with Linear Fit", fontsize=14, weight="bold")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.3)
	plt.tight_layout()

	out_path = os.path.join(plot_dir, "scatter_with_regression.png")
	plt.savefig(out_path, dpi=300)
	plt.close()

def plot_ccr_over_time():
	"""
	Plots the time series of CCR (°C per Tt C) calculated as
	Temperature_Anomaly / Cumulative_Carbon (in Tt C) and saves to plots/ccr_over_time.png.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	data_path = os.path.join(script_dir, "data_cleaned", "Temp_And_Cumulative.csv")
	plot_dir = os.path.join(script_dir, "plots")
	os.makedirs(plot_dir, exist_ok=True)

	# Load data
	df = pd.read_csv(data_path)

	# Convert cumulative carbon (tonnes of C) to trillion tonnes of C
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12

	# Compute CCR = ΔT / Ccum (°C per Tt C)
	# Skip rows where Cumulative_Carbon_Tt is zero to avoid division by zero
	df = df[df["Cumulative_Carbon_Tt"] > 0].copy()
	df["CCR"] = df["Temperature_Anomaly"] / df["Cumulative_Carbon_Tt"]

	# Plot
	plt.figure(figsize=(10, 6))
	plt.plot(
		df["Year"],
		df["CCR"],
		color="tab:purple",
		marker="o",
		markersize=4,
		linewidth=1.8
	)
	plt.xlabel("Year")
	plt.ylabel("CCR (°C per Tt C)")
	plt.title("CCR vs. Time")
	plt.grid(True, linestyle="--", alpha=0.3)
	plt.ylim(0, 10)
	plt.tight_layout()

	out_path = os.path.join(plot_dir, "ccr_over_time.png")
	plt.savefig(out_path, dpi=300)
	plt.close()

if __name__ == "__main__":
	txt_to_csv("NASA_Temperature_Anomaly_nosmooth.txt")
	keep_values("CO2_Emissions.csv", {"Entity":["World"]})
	compute_cumulative_carbon("CO2_Emissions.csv")
	join_temp_and_cumulative("NASA_Temperature_Anomaly.csv", "Cumulative_Carbon.csv")
	plot_temperature_anomaly()
	plot_cumulative_carbon()
	plot_temp_and_cumulative_carbon_timeseries()
	plot_temp_vs_cumulative_carbon()
	plot_scatter_with_regression()
	plot_ccr_over_time()

	dir_describe_csvs("data_cleaned")
