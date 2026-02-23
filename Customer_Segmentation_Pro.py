import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerSegmentationSystem:
    """
    A professional system to perform Customer Segmentation using K-Means Clustering.
    This system includes Data Wrangling, Optimal Cluster Detection, and Insight Visualization.
    """
    
    def __init__(self, data_path):
        """Initializes the system with dataset path and ML components."""
        self.data_path = data_path
        self.df = None
        self.X_scaled = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.centroids = None
        self.cluster_map = {}

    def load_and_preprocess(self):
        """Phase 1: Data Wrangling and Feature Scaling[cite: 4, 9]."""
        try:
            self.df = pd.read_csv(self.data_path)
            # Selecting Annual Income (k$) and Spending Score (1-100) for analysis [cite: 8]
            X = self.df.iloc[:, [3, 4]].values
            
            # Feature Scaling: Essential for distance-based algorithms like K-Means [cite: 5]
            self.X_scaled = self.scaler.fit_transform(X)
            print("[STATUS] Data loaded and preprocessed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")

    def detect_optimal_k(self):
        """Phase 2: Pattern Discovery using the Elbow Method[cite: 4, 9]."""
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42)
            km.fit(self.X_scaled)
            wcss.append(km.inertia_)
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), wcss, marker='o', color='#2c3e50', linewidth=2)
        plt.title('Analysis 1: Elbow Method for Optimal K Detection', fontsize=14)
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS (Inertia)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('Task3_Elbow_Graph.png')
        print("[REPORT] Elbow graph saved as 'Task3_Elbow_Graph.png'.")

    def train_model(self, k=5):
        """Phase 3: Training the K-Means Model and Assigning Clusters[cite: 4, 9]."""
        self.kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        self.df['Cluster'] = self.kmeans.fit_predict(self.X_scaled)
        
        # Inversing scaling for the centroids to interpret real-world values 
        self.centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self._map_cluster_insights()
        print(f"[STATUS] Model trained successfully with {k} clusters.")

    def _map_cluster_insights(self):
        """Internal Logic: Mapping numeric clusters to business-driven labels[cite: 6]."""
        for i, (income, spending) in enumerate(self.centroids):
            if income > 70 and spending > 70:
                self.cluster_map[i] = {'label': 'Target/VIP', 'color': '#2ecc71'} # High Income, High Spend
            elif income > 70 and spending < 40:
                self.cluster_map[i] = {'label': 'Careful', 'color': '#e74c3c'}    # High Income, Low Spend
            elif income < 40 and spending > 70:
                self.cluster_map[i] = {'label': 'Spendthrifts', 'color': '#9b59b6'} # Low Income, High Spend
            elif income < 40 and spending < 40:
                self.cluster_map[i] = {'label': 'Sensible', 'color': '#f1c40f'}   # Low Income, Low Spend
            else:
                self.cluster_map[i] = {'label': 'Standard', 'color': '#3498db'}   # Mid Income, Mid Spend

    def generate_analytical_summary(self):
        """Phase 4: Numerical summary for business decision-making[cite: 6, 9]."""
        print("\n" + "="*70)
        print("                 CUSTOMER SEGMENTATION ANALYTICAL SUMMARY")
        print("="*70)
        print(f"{'Segment Category':<18} | {'Count':<8} | {'Avg. Income':<15} | {'Avg. Spend Score'}")
        print("-" * 70)
        
        summary_data = []
        for i in range(len(self.centroids)):
            cluster_subset = self.df[self.df['Cluster'] == i]
            label = self.cluster_map[i]['label']
            count = len(cluster_subset)
            avg_inc = cluster_subset.iloc[:, 3].mean()
            avg_spd = cluster_subset.iloc[:, 4].mean()
            
            print(f"{label:<18} | {count:<8} | ${avg_inc:<14.2f} | {avg_spd:.2f}")
            summary_data.append([label, count, avg_inc, avg_spd])
        
        # Exporting summary to CSV for documentation
        summary_df = pd.DataFrame(summary_data, columns=['Segment', 'Count', 'Average_Income', 'Average_Spend_Score'])
        summary_df.to_csv('Task3_Segment_Summary.csv', index=False)
        print("="*70)
        print("[REPORT] Numerical summary exported to 'Task3_Segment_Summary.csv'.")

    def visualize_final_segments(self):
        """Phase 5: High-fidelity visualization of segmented clusters."""
        plt.figure(figsize=(12, 8))
        
        for i in range(len(self.centroids)):
            subset = self.df[self.df['Cluster'] == i]
            plt.scatter(subset.iloc[:, 3], subset.iloc[:, 4], 
                        s=100, c=self.cluster_map[i]['color'], 
                        label=self.cluster_map[i]['label'], edgecolors='black', alpha=0.75)

        # Highlighting the Cluster Centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=350, 
                    c='white', marker='X', label='Cluster Centers', edgecolors='black')
        
        plt.title('Analysis 2: Final Customer Segmentation Segments', fontsize=16)
        plt.xlabel('Annual Income (k$)', fontsize=12)
        plt.ylabel('Spending Score (1-100)', fontsize=12)
        plt.legend(title="Customer Segments", bbox_to_anchor=(1.05, 1), loc='upper left', shadow=True)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.savefig('Task3_Final_Analysis.png', bbox_inches='tight')
        print("[REPORT] Final analysis plot saved as 'Task3_Final_Analysis.png'.")
        plt.show()

# --- Professional Execution ---
if __name__ == "__main__":
    # Task 3: Building algorithms to drive data-driven solutions [cite: 6]
    analyst = CustomerSegmentationSystem('Mall_Customers.csv')
    
    # Execution Flow
    analyst.load_and_preprocess()
    analyst.detect_optimal_k()
    analyst.train_model(k=5)
    analyst.generate_analytical_summary()
    analyst.visualize_final_segments()
    
    print("\n[SUCCESS] Task 3 finalized for submission.")