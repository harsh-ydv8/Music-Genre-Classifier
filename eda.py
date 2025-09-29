import pandas as pd     #type:ignore
import matplotlib.pyplot as plt     #type:ignore
import seaborn as sns       #type:ignore

CSV_PATH = "features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)

    print("DataFrame loaded successfully!")
    print("\nFirst 5 rows of the dataset:")
    print(features_df.head())

    print("\n--- DataFrame Info ---")
    features_df.info()

    print("\n--- DataFrame Statistical Summary ---")
    print(features_df.describe())

    print("\n--- Missing Values Check ---")
    missing_values_count = features_df.isnull().sum()
    
    print("Number of missing values per column:")
    print(missing_values_count)
    
    if missing_values_count.sum() == 0:
        print("\nConclusion: The dataset has no missing values. No handling is required.")
    else:
        print("\nConclusion: The dataset contains missing values. Handling is required.")

    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.countplot(x='genre_label', data=features_df, palette='viridis')

    ax.set_title('Distribution of Music Genres in the Dataset', fontsize=16)
    ax.set_xlabel('Genre', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)

    ax.set_xticklabels(genre_names, rotation=30)
    plt.tight_layout()
    plt.show()



    print("\n--- Generating Box Plot for Spectral Centroid ---")
    plt.figure(figsize=(14, 7))

    box_ax = sns.boxplot(x='genre_label', y='25', data=features_df, palette='cubehelix')

    box_ax.set_title('Spectral Centroid Distribution Across Genres', fontsize=18)
    box_ax.set_xlabel('Genre', fontsize=14)
    box_ax.set_ylabel('Spectral Centroid', fontsize=14)

    box_ax.set_xticklabels(genre_names, rotation=30, ha="right") 

    plt.tight_layout()
    plt.show()


    print("\n--- Generating Violin Plot for First MFCC (Column 0) ---")
    plt.figure(figsize=(14, 7))

    violin_ax = sns.violinplot(x='genre_label', y='0', data=features_df, palette='Spectral')
    violin_ax.set_title('First MFCC (Timbre/Energy) Distribution Across Genres', fontsize=18)
    violin_ax.set_xlabel('Genre', fontsize=14)
    violin_ax.set_ylabel('MFCC 1 Value', fontsize=14)

    violin_ax.set_xticklabels(genre_names, rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


    print("\\n--- Computing Correlation Matrix ---")
    correlation_matrix = features_df.corr()
    print("Correlation matrix computed successfully.")
    print("Top 5 rows of the correlation matrix:")
    print(correlation_matrix.head())

    print("\n--- Generating Heatmap of Feature Correlations ---")
    plt.figure(figsize=(18, 15))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)

    plt.title('Correlation Matrix of Music Features', fontsize=20)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred while loading the DataFrame: {e}")