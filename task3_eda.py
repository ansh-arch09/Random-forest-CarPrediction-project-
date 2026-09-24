import os
import pandas as pd
import matplotlib.pyplot as plt

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/car_dataset.csv")

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

# ---------------------------------------
# Histogram : Selling Price
# ---------------------------------------

plt.figure(figsize=(8,5))
plt.hist(df["Selling_Price"], bins=20)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.savefig("outputs/histogram_selling_price.png")
plt.show()

# ---------------------------------------
# Box Plot : Selling Price
# ---------------------------------------

plt.figure(figsize=(6,5))
plt.boxplot(df["Selling_Price"])
plt.title("Boxplot of Selling Price")
plt.savefig("outputs/boxplot_selling_price.png")
plt.show()

# ---------------------------------------
# Scatter Plot
# ---------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df["Present_Price"], df["Selling_Price"])
plt.title("Present Price vs Selling Price")
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.savefig("outputs/scatter_plot.png")
plt.show()

# ---------------------------------------
# Correlation Heatmap
# ---------------------------------------

numeric_df = df.select_dtypes(include=["int64","float64"])

correlation = numeric_df.corr()

plt.figure(figsize=(8,6))

plt.imshow(correlation, cmap="coolwarm")

plt.colorbar()

plt.xticks(range(len(correlation.columns)),
           correlation.columns,
           rotation=45)

plt.yticks(range(len(correlation.columns)),
           correlation.columns)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/heatmap.png")

plt.show()

print("\nEDA Completed Successfully")