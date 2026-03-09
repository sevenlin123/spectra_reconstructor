import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob
import numpy as np

class PCASpectrumGenerator:
    def __init__(self, real_spectra, labels, n_components=5):
        """
        real_spectra: Array of shape (36, 100) - Your real TNO spectra
        n_components: Number of PCA components to keep (3-5 is usually enough for TNOs)
        """
        self.pca = PCA(n_components=n_components)
        
        # 1. Compress the real spectra into the "Latent Space"
        self.latent_data = self.pca.fit_transform(real_spectra)
        
        print(f"PCA Variance Explained: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # 2. Learn the shape of the data in this compressed space
        # We use Kernel Density Estimation (KDE) to create a probability map
        # bandwidth=1.0 controls how 'smooth' the distribution is. 
        # Increase bandwidth to generate more diverse (but riskier) samples.
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        self.kde.fit(self.latent_data)
        #[cliff_methanol, cliff_no_methanol, dd, bowl] = [0, 1, 2, 3]
        self.labels = labels
        self.colors = []
        for label in labels:
            if label == 0:
                self.colors.append('darkviolet')
            elif label == 1:
                self.colors.append('red')
            elif label == 2:
                self.colors.append('darkorange')
            else:
                self.colors.append('dodgerblue')
                
    def generate(self, n_samples=1000):
        """
        Generates new synthetic spectra.
        """
        # 3. Sample random points from the KDE probability map
        # These are new sets of PCA coefficients
        new_latent_vectors = self.kde.sample(n_samples)
        
        # 4. Decompress back to full spectra
        mock_spectra = self.pca.inverse_transform(new_latent_vectors)
        
        return mock_spectra

    def visualize_latent_space(self, n_samples=1000):
        """
        Creates a corner plot comparing the Real Data (Black) vs KDE-Generated Data (Red).
        This verifies that the Generator is sampling from the correct region of PC space.
        """
        # Generate sample latent vectors from the KDE
        mock_vectors = self.kde.sample(n_samples)
        real_vectors = self.latent_data
        
        n_dims = self.pca.n_components
        
        fig, axes = plt.subplots(n_dims, n_dims, figsize=(3 * n_dims, 3 * n_dims))
        fig.suptitle("Latent Space: Real (Black) vs Mock (Red Contours)", fontsize=16)
        
        # Handle case where n_dims=1
        if n_dims == 1:
            axes = np.array([[axes]])
            
        for i in range(n_dims):
            for j in range(n_dims):
                ax = axes[i, j]
                
                # Diagonal: Density Histogram (1D)
                if i == j:
                    ax.hist(real_vectors[:, i], bins=10, density=True, 
                            color='grey', alpha=0.5, label='Real')
                    ax.hist(mock_vectors[:, i], bins=30, density=True, 
                            color='salmon', alpha=0.3, label='Mock')
                    ax.set_xlabel(f"PC {i+1}")
                    if i == 0:
                        ax.legend(loc='upper right')
                        
                # Lower Triangle: Contours (2D)
                elif i > j:
                    # Calculate 2D KDE for the mock data visualization
                    x = mock_vectors[:, j]
                    y = mock_vectors[:, i]
                    
                    # Define grid limits with some padding
                    xmin, xmax = x.min(), x.max()
                    ymin, ymax = y.min(), y.max()
                    dx = (xmax - xmin) * 0.1
                    dy = (ymax - ymin) * 0.1
                    xmin -= dx; xmax += dx
                    ymin -= dy; ymax += dy
                    
                    # Evaluate KDE on a grid
                    X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    values = np.vstack([x, y])
                    kernel = gaussian_kde(values)
                    Z = np.reshape(kernel(positions).T, X.shape)
                    
                    # Plot Mock Data as Contours
                    ax.contourf(X, Y, Z, cmap='Reds', alpha=0.3, levels=6)
                    ax.contour(X, Y, Z, colors='red', alpha=0.5, levels=6, linewidths=0.5)

                    # Real in foreground (Black dots)
                    ax.scatter(real_vectors[:, j], real_vectors[:, i], 
                               s=20, color=self.colors, alpha=0.8, marker='o')
                    ax.set_xlabel(f"PC {j+1}")
                    ax.set_ylabel(f"PC {i+1}")
                    
                # Upper Triangle: Hide
                else:
                    ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

