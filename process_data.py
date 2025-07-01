import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "data_raw")
CLEAN_DIR = os.path.join(SCRIPT_DIR, "data_cleaned")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

HEAD_N = 5

NASA_TEMP_LOWESS_TXT = "NASA_Temperature_Anomaly_nosmooth.txt"
NASA_TEMP_CSV = "NASA_Temperature_Anomaly.csv"
SCALED_NASA_TEMP_CSV = "Scaled_Temperature_Anomaly.csv"
CO2_EMISSIONS_CSV = "CO2_Emissions.csv"
GHG_EMISSIONS_CSV = "total_ghg_emissions.csv"
CUMULATIVE_CARBON_CSV = "Cumulative_Carbon.csv"
TEMP_AND_CUMULATIVE_CSV = "Temp_And_Cumulative.csv"

KEEP_FILTERS = {
	CO2_EMISSIONS_CSV: {"Entity": ["World"]},
	GHG_EMISSIONS_CSV: {"Entity": ["World"]},
}

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def dir_describe_csvs():
	"""Displays basic info and the first HEAD_N rows of all CSVs in CLEAN_DIR."""
	if not os.path.isdir(CLEAN_DIR):
		print(f"Could not find directory: {CLEAN_DIR}")
		return
	for filename in os.listdir(CLEAN_DIR):
		if filename.lower().endswith(".csv"):
			filepath = os.path.join(CLEAN_DIR, filename)
			try:
				df = pd.read_csv(filepath)
				print(f"\n--- {filename} ---")
				rows, cols = df.shape
				print(f"Shape: {rows} rows, {cols} columns")
				null_counts = df.isna().sum()
				print("Missing values per column:")
				for col, count in null_counts.items():
					pct = int((count / rows) * 100)
					print(f"\t{col}: {count} ({pct}% missing)")
				print(f"\nFirst {HEAD_N} rows:")
				print(df.head(HEAD_N))
			except Exception as e:
				print(f"Could not read {filepath}, {e}")

def txt_to_csv():
	"""Reads NASA_TEMP_TXT from RAW_DIR and saves as NASA_TEMP_CSV in CLEAN_DIR."""
	input_path = os.path.join(RAW_DIR, NASA_TEMP_LOWESS_TXT)
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Could not find file: {input_path}")
	df = pd.read_csv(input_path, sep=r"\s+", engine="python")
	output_path = os.path.join(CLEAN_DIR, NASA_TEMP_CSV)
	df.to_csv(output_path, index=False)

def keep_values():
	"""Filters CSVs in RAW_DIR based on KEEP_FILTERS and saves results to CLEAN_DIR."""
	for filename, filters in KEEP_FILTERS.items():
		input_path = os.path.join(RAW_DIR, filename)
		if not os.path.isfile(input_path):
			print(f"Could not find file: {input_path}")
			continue
		df = pd.read_csv(input_path)
		mask = pd.Series(False, index=df.index)
		for col, allowed in filters.items():
			if col in df.columns:
				mask |= df[col].isin(allowed)
			else:
				print(f"Warning: column '{col}' not found in {filename}")
		filtered_df = df[mask]
		output_path = os.path.join(CLEAN_DIR, filename)
		filtered_df.to_csv(output_path, index=False)

def compute_cumulative_carbon():
	"""Reads CO2_EMISSIONS_CSV from CLEAN_DIR, converts to carbon, and saves CUMULATIVE_CARBON_CSV."""
	path = os.path.join(CLEAN_DIR, CO2_EMISSIONS_CSV)
	df = pd.read_csv(path).sort_values("Year").copy()
	df["Annual_Carbon"] = df["Annual CO₂ emissions"] * 0.272921
	df = df[df["Year"] >= 1880].reset_index(drop=True)
	df["Cumulative_Carbon"] = df["Annual_Carbon"].cumsum()
	output_path = os.path.join(CLEAN_DIR, CUMULATIVE_CARBON_CSV)
	df.to_csv(output_path, index=False, columns=["Year", "Cumulative_Carbon"])

def join_temp_and_cumulative():
	"""Joins NASA_TEMP_CSV and CUMULATIVE_CARBON_CSV on Year, adjusts baseline, and saves TEMP_AND_CUMULATIVE_CSV. (Changed to use scaled temperature anomaly)"""
	temp_path = os.path.join(CLEAN_DIR, SCALED_NASA_TEMP_CSV)
	co2_path = os.path.join(CLEAN_DIR, CUMULATIVE_CARBON_CSV)
	temp_df = pd.read_csv(temp_path)
	co2_df = pd.read_csv(co2_path)
	merged = pd.merge(temp_df, co2_df, on="Year")
	baseline = merged.loc[merged["Year"] == 1880, "Temperature_Anomaly"].iloc[0]
	merged["Temperature_Anomaly"] -= baseline
	output_path = os.path.join(CLEAN_DIR, TEMP_AND_CUMULATIVE_CSV)
	merged.to_csv(output_path, index=False, columns=["Year", "Temperature_Anomaly", "Cumulative_Carbon"])

def plot_carbon_vs_ghg():
	"""Plotting carbon vs total GHG emissions."""
	co2_path = os.path.join(CLEAN_DIR, CO2_EMISSIONS_CSV)
	ghg_path = os.path.join(CLEAN_DIR, GHG_EMISSIONS_CSV)
	co2_df = pd.read_csv(co2_path)
	ghg_df = pd.read_csv(ghg_path)
	co2_df = co2_df[co2_df["Entity"] == "World"].reset_index(drop=True)
	ghg_df = ghg_df[ghg_df["Entity"] == "World"].reset_index(drop=True)
	ghg_df["Total_GHG_Emissions_Tt"] = ghg_df["Annual greenhouse gas emissions in CO₂ equivalents"] / 1e12
	merged = pd.merge(co2_df, ghg_df, on="Year", how="inner")
	plt.figure(figsize=(10, 6))
	plt.xlabel("Year")
	plt.ylabel("Emissions (Tt)")
	plt.plot(merged["Year"], merged["Annual CO₂ emissions"] / 1e12, label="Annual CO₂ Emissions (TtC)")
	plt.plot(merged["Year"], merged["Total_GHG_Emissions_Tt"], label="Annual GHG Emissions (TtCO₂)")
	plt.title("Cumulative Carbon vs Total GHG Emissions Over Time")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, "carbon_vs_ghg.png"))

def save_scaled_temperature_anomaly():
	"""Scales the yearly temperature anomaly by the ratio of carbon to total GHG yearly emissions."""
	co2_path = os.path.join(CLEAN_DIR, CO2_EMISSIONS_CSV)
	ghg_path = os.path.join(CLEAN_DIR, GHG_EMISSIONS_CSV)
	temp_path = os.path.join(CLEAN_DIR, NASA_TEMP_CSV)
	co2_df = pd.read_csv(co2_path)
	ghg_df = pd.read_csv(ghg_path)
	temp_df = pd.read_csv(temp_path)
	co2_df = co2_df[co2_df["Entity"] == "World"].reset_index(drop=True)
	ghg_df = ghg_df[ghg_df["Entity"] == "World"].reset_index(drop=True)
	ghg_df["Total_GHG_Emissions"] = ghg_df["Annual greenhouse gas emissions in CO₂ equivalents"]
	merged = pd.merge(co2_df, ghg_df, on="Year", how="inner")
	merged = pd.merge(merged, temp_df, on="Year", how="inner")
	merged["Scaled_Temperature_Anomaly"] = merged["Temperature_Anomaly"] * (merged["Annual CO₂ emissions"] / merged["Total_GHG_Emissions"])
	merged["Temperature_Anomaly"] = merged["Scaled_Temperature_Anomaly"]
	output_path = os.path.join(CLEAN_DIR, "Scaled_Temperature_Anomaly.csv")
	merged.to_csv(output_path, index=False, columns=["Year", "Temperature_Anomaly"])

def plot_temperature_anomaly():
	"""Plots Temperature Anomaly over time."""
	df = pd.read_csv(os.path.join(CLEAN_DIR, NASA_TEMP_CSV))
	plt.figure(figsize=(10, 6))
	plt.plot(df["Year"], df["Temperature_Anomaly"])
	plt.xlabel("Year")
	plt.ylabel("Temperature Anomaly (°C)")
	plt.title("Global Temperature Anomaly Over Time")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, "temperature_anomaly.png"))
	plt.close()

def plot_cumulative_carbon():
	"""Plots cumulative carbon over time."""
	df = pd.read_csv(os.path.join(CLEAN_DIR, CUMULATIVE_CARBON_CSV))
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12
	plt.figure(figsize=(10, 6))
	plt.plot(df["Year"], df["Cumulative_Carbon_Tt"])
	plt.xlabel("Year")
	plt.ylabel("Cumulative Carbon Emissions (TtC)")
	plt.title("Cumulative Global Carbon Emissions Over Time")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, "Cumulative_Carbon.png"))
	plt.close()

def plot_temp_and_cumulative_carbon_timeseries():
	"""Plots temperature anomaly and cumulative carbon with twin axes."""
	df = pd.read_csv(os.path.join(CLEAN_DIR, TEMP_AND_CUMULATIVE_CSV))
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12
	fig, ax1 = plt.subplots(figsize=(10, 6))
	ax1.set_xlabel("Year")
	ax1.set_ylabel("Temperature Anomaly (°C)")
	ax1.plot(df["Year"], df["Temperature_Anomaly"], label="Temp Anomaly")
	ax2 = ax1.twinx()
	ax2.set_ylabel("Cumulative Carbon Emissions (TtC)")
	ax2.plot(df["Year"], df["Cumulative_Carbon_Tt"], label="Cumulative Carbon")
	ymin = min(df["Temperature_Anomaly"].min(), df["Cumulative_Carbon_Tt"].min())
	ymax = max(df["Temperature_Anomaly"].max(), df["Cumulative_Carbon_Tt"].max())
	ax1.set_ylim(ymin, ymax)
	ax2.set_ylim(ymin, ymax)
	plt.title("Global Temperature Anomaly and Cumulative Carbon Over Time")
	fig.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, "temp_and_co2_timeseries.png"))
	plt.close()

def plot_temp_vs_cumulative_carbon():
	"""Plots temperature anomaly (red) and cumulative carbon (blue) with twin axes."""
	df = pd.read_csv(os.path.join(CLEAN_DIR, TEMP_AND_CUMULATIVE_CSV))
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12
	fig, ax1 = plt.subplots(figsize=(10, 6))
	ax1.set_xlabel("Year")
	ax1.set_ylabel("Temperature Anomaly (°C)", color="tab:red")
	ax1.plot(
		df["Year"],
		df["Temperature_Anomaly"],
		color="tab:red",
		linewidth=2,
		label="Scaled Temp Anomaly",
	)
	ax1.tick_params(axis="y", labelcolor="tab:red")
	ax2 = ax1.twinx()
	ax2.set_ylabel("Cumulative Carbon Emissions (Tt C)", color="tab:blue")
	ax2.plot(
		df["Year"],
		df["Cumulative_Carbon_Tt"],
		color="tab:blue",
		linewidth=2,
		label="Cumulative Carbon",
	)
	ax2.tick_params(axis="y", labelcolor="tab:blue")
	plt.title("Global Scaled Temperature Anomaly (red) and Cumulative Carbon (blue) Over Time")
	ymin = min(df["Temperature_Anomaly"].min(), df["Cumulative_Carbon_Tt"].min())
	ymax = max(df["Temperature_Anomaly"].max(), df["Cumulative_Carbon_Tt"].max())
	ax1.set_ylim(ymin, ymax)
	ax2.set_ylim(ymin, ymax)
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines + lines2, labels + labels2, loc="upper left")
	fig.tight_layout()
	plt.grid(False)
	plt.savefig(os.path.join(PLOTS_DIR, "temp_and_cumulative_colored.png"), dpi=300)
	plt.close()

def plot_scatter_with_regression():
	df = pd.read_csv(os.path.join(CLEAN_DIR, TEMP_AND_CUMULATIVE_CSV))
	x = df["Cumulative_Carbon"].values / 1e12
	y = df["Temperature_Anomaly"].values
	w = x.copy()
	w /= w.max()
	# polyfit with weights: it minimizes sum( w_i^2 * (y_i - (m*x_i + b))^2 )
	m, b = np.polyfit(x, y, 1, w=w)
	y_pred = m*x + b
	mse    = np.mean((y - y_pred)**2)
	mse_w  = np.average((y - y_pred)**2, weights=w)
	print(f"Slope: {m:.4f}, Intercept: {b:.4f}")
	print(f"MSE: {mse:.6f}, Weighted MSE: {mse_w:.6f}")
	plt.figure(figsize=(10, 6))
	plt.scatter(x, y, s=20, c="k", alpha=0.6)
	xs = np.linspace(x.min(), x.max(), 200)
	plt.plot(xs, m*xs + b, c="C1", lw=2, label=f"CCR={m:.2f}")
	plt.xlabel("Cumulative Carbon (Tt C)")
	plt.ylabel("Temp Anomaly (°C)")
	plt.title("Temperature Anomaly vs Cumulative Carbon \n Weighted Linear Regression")  
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, "weighted_regression.png"), dpi=300)
	plt.close()

def plot_ccr():
	"""Plots all metrics on same axes."""
	df = pd.read_csv(os.path.join(CLEAN_DIR, TEMP_AND_CUMULATIVE_CSV))
	df["Cumulative_Carbon_Tt"] = df["Cumulative_Carbon"] / 1e12
	df = df[df["Cumulative_Carbon_Tt"] > 0].copy()
	df["CCR"] = df["Temperature_Anomaly"] / df["Cumulative_Carbon_Tt"]
	plt.figure(figsize=(10, 6))
	plt.plot(df["Year"], df["Temperature_Anomaly"], label="Temperature Anomaly")
	plt.plot(df["Year"], df["Cumulative_Carbon_Tt"], label="Cumulative Carbon")
	plt.plot(df["Year"], df["CCR"], label="CCR")
	plt.xlabel("Year")
	plt.ylabel("Value")
	plt.title("Temperature Anomaly, Cumulative Carbon & CCR Over Time")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.ylim(-1, 3)
	plt.savefig(os.path.join(PLOTS_DIR, "all_metrics.png"), dpi=300)
	plt.close()

if __name__ == "__main__":
	txt_to_csv()
	keep_values()
	compute_cumulative_carbon()
	plot_carbon_vs_ghg()
	save_scaled_temperature_anomaly()
	join_temp_and_cumulative()
	plot_temperature_anomaly()
	plot_cumulative_carbon()
	plot_temp_and_cumulative_carbon_timeseries()
	plot_temp_vs_cumulative_carbon()
	plot_scatter_with_regression()
	plot_ccr()
	# dir_describe_csvs()
