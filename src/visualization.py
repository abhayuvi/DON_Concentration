import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load dataset."""
    return pd.read_csv(filepath)

def plot_histograms(df):
    """Plot histograms of spectral reflectance values."""
    df.iloc[:, :-1].hist(figsize=(12, 8), bins=30)
    plt.suptitle("Histograms of Spectral Reflectance Values")
    plt.show()

def plot_boxplots(df):
    """Plot boxplots to detect outliers."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.iloc[:, :-1])
    plt.xticks(rotation=90)
    plt.title("Boxplots of Spectral Features")
    plt.show()

def plot_average_reflectance(df):
    """Plot average reflectance over wavelengths."""
    avg_reflectance = df.iloc[:, :-1].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(avg_reflectance, marker='o')
    plt.title("Average Spectral Reflectance Across Wavelengths")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Reflectance")
    plt.grid()
    plt.show()

def visualize_data(filepath):
    """Run all visualization functions."""
    df = load_data(filepath)
    plot_histograms(df)
    plot_boxplots(df)
    plot_average_reflectance(df)

if __name__ == "__main__":
    input_file = input_file = "/Users/apple/DON_Concentration/data/MLE-Assignment.csv"
    visualize_data(input_file)
