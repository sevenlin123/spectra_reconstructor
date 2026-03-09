import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

class TNOpcClassifier:
    def __init__(self, pca , model_save_path):
        self.n_components = pca.n_components
        self.pca = pca
        self.predictor = None
        self.train_df = None # Store training data for visualization
        self.model_save_path = model_save_path
        
    def prepare_and_train(self, real_spectra, real_labels, augment_factor=50):
        """
        1. Fits PCA on real spectra.
        2. Augments data (generates synthetic samples PER CLASS).
        3. Trains AutoGluon classifier.
        """
        # --- Step 1: PCA Compression ---
        # Transform 100-point spectra into 5-point PC vectors
        print("Fitting PCA and transforming real spectra...")
        real_pcs = self.pca.transform(real_spectra) # Shape: (n_samples, 5)
        
        # --- Step 2: Per-Class Augmentation ---
        # We need to balloon the dataset from 36 -> ~2000 for stable classification
        print(f"Augmenting data (Target: {augment_factor}x per class)...")
        
        aug_pcs_list = []
        aug_labels_list = []
        
        # Convert to DF for easy grouping
        df_real = pd.DataFrame(real_pcs, columns=[f'PC{i}' for i in range(self.n_components)])
        df_real['label'] = real_labels
        
        unique_classes, counts = np.unique(df_real['label'], return_counts=True)
        factors = counts.sum() / counts
        factors /= factors.min()  # Normalize to minimum class 
        factors = np.ceil(factors).astype(int)  # Round up to nearest integer  
        

        for n, cls in enumerate(unique_classes):
            # Get real samples for this specific class
            class_subset = df_real[df_real['label'] == cls].drop(columns=['label']).values
            n_real = len(class_subset)
            
            # Calculate stats for this class cluster
            mean_vec = np.mean(class_subset, axis=0)
            
            # If we have only 1 sample, we can't compute std, so we assume a small noise
            if n_real > 1:
                std_vec = np.std(class_subset, axis=0)
            else:
                std_vec = np.ones(self.n_components) * 0.1 # Default noise cloud
                
            # Generate synthetic samples specifically for this class
            # We use a Normal distribution around the class mean
            n_synthetic = n_real * factors[n] * augment_factor
            
            # Shape: (n_synthetic, n_components)
            # We add randomness scaled by the class's natural spread (or default noise)
            noise = np.random.randn(n_synthetic, self.n_components) * (std_vec + 1e-6)
            synthetic_pcs = mean_vec + noise
            
            aug_pcs_list.append(synthetic_pcs)
            aug_labels_list.extend([cls] * n_synthetic)
            
        # Combine everything
        X_aug = np.vstack(aug_pcs_list)
        y_aug = np.array(aug_labels_list)
        
        # Create final Training DataFrame
        self.train_df = pd.DataFrame(X_aug, columns=[f'PC{i}' for i in range(self.n_components)])
        self.train_df['class_label'] = y_aug
        
        print(f"Training set size after augmentation: {len(self.train_df)} samples")
        
        # --- Step 3: Train AutoGluon ---


        print("\n--- Starting AutoGluon Classification ---")
        self.predictor = TabularPredictor(label='class_label', verbosity=2, path=self.model_save_path).fit(
            train_data=self.train_df,
            presets='medium_quality', # Fast and accurate enough for this 
            time_limit=60 # Train for 1 minute
        )
        
    def classify_reconstruction(self, predicted_pcs):
        """
        Args:
            predicted_pcs: The output from your previous Reconstruction Model
                           Shape (N, 5)
        Returns:
            DataFrame with 'class', 'probability'
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet!")
            
        # Convert input numpy array to DataFrame with correct column names
        input_df = pd.DataFrame(predicted_pcs, columns=[f'PC{i}' for i in range(self.n_components)])
        
        # Predict Class
        y_pred = self.predictor.predict(input_df)
        
        # Predict Probabilities (Confidence)
        y_prob = self.predictor.predict_proba(input_df)
        
        return pd.DataFrame({'predicted_class': y_pred}).join(y_prob)

    def visualize_classification_space(self, new_object_pcs, predicted_classes=None):
        """
        Generates a corner plot showing:
        1. The Augmented Training Data (Contours Colored by Class)
        2. The New Objects (Black Stars)
        """
        if self.train_df is None:
            print("No training data found. Cannot plot.")
            return

        n_dims = self.n_components
        #classes = self.train_df['class_label'].unique()
        classes = ['cliff_methanol', 'cliff_no_methanol', 'dd', 'bowl']
        #colors = plt.get_cmap('tab10', len(classes))
        colors = ['darkviolet', 'red', 'darkorange', 'dodgerblue']
        cmaps = ['Purples', 'Reds', 'Oranges', 'Blues']
        if predicted_classes is not None:
            predicted_colors = []
            for cls in predicted_classes:
                if cls == 'cliff_methanol':
                    predicted_colors.append('darkviolet')
                elif cls == 'cliff_no_methanol':
                    predicted_colors.append('red')
                elif cls == 'dd':
                    predicted_colors.append('darkorange')
                else:
                    predicted_colors.append('dodgerblue')
        
        fig, axes = plt.subplots(n_dims, n_dims, figsize=(3 * n_dims, 3 * n_dims))
        fig.suptitle("Classification Space: Training Contours", fontsize=16)
        
        # Handle case where n_dims=1
        if n_dims == 1:
            axes = np.array([[axes]])
            
        for i in range(n_dims):
            for j in range(n_dims):
                ax = axes[i, j]
                
                # Diagonal: Density Histogram
                if i == j:
                    for k, cls in enumerate(classes):
                        subset = self.train_df[self.train_df['class_label'] == cls]
                        ax.hist(subset[f'PC{i}'], bins=20, alpha=0.5, density=True, 
                                label=cls, color=colors[k])
                    ax.set_xlabel(f"PC {i+1}")
                    if i == 0:
                        ax.legend(fontsize='x-small')
                        
                # Lower Triangle: Contours
                elif i > j:
                    for k, cls in enumerate(classes):
                        subset = self.train_df[self.train_df['class_label'] == cls]
                        x = subset[f'PC{j}'].values
                        y = subset[f'PC{i}'].values
                        
                        # Only plot contours if we have enough variance/points
                        if len(x) > 2 and (x.max() != x.min()) and (y.max() != y.min()):
                            try:
                                # Define grid limits with padding
                                xmin, xmax = x.min(), x.max()
                                ymin, ymax = y.min(), y.max()
                                dx = (xmax - xmin) * 0.2
                                dy = (ymax - ymin) * 0.2
                                xmin -= dx; xmax += dx
                                ymin -= dy; ymax += dy
                                
                                X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
                                positions = np.vstack([X.ravel(), Y.ravel()])
                                values = np.vstack([x, y])
                                kernel = gaussian_kde(values)
                                Z = np.reshape(kernel(positions).T, X.shape)
                                
                                # Plot Contours (Lines + Light Fill)
                                ax.contour(X, Y, Z, colors=[colors[k]], levels=5, linewidths=1.0, alpha=.8)
                                #ax.contourf(X, Y, Z, colors=[colors[k]], alpha=0.15, levels=5)
                                ax.scatter(x, y, s=1, alpha=0.1, color=colors[k])
                            except np.linalg.LinAlgError:
                                # Fallback to scatter if points are collinear
                                ax.scatter(x, y, s=1, alpha=0.1, color=colors[k])
                        else:
                             ax.scatter(x, y, s=1, alpha=0.1, color=colors[k])

                    # Overlay New Objects
                    if predicted_classes is not None:
                        ax.scatter(new_object_pcs[:, j], new_object_pcs[:, i], 
                                   s=100, color=predicted_colors, marker='.', alpha=0.8, label='New Obj')
                    else:   
                        ax.scatter(new_object_pcs[:, j], new_object_pcs[:, i], 
                                   s=100, color='k', marker='.', alpha=0.8, label='New Obj')
                               
                    ax.set_xlabel(f"PC {j+1}")
                    ax.set_ylabel(f"PC {i+1}")
                    
                # Upper Triangle: Hide
                else:
                    ax.axis('off')
                    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()