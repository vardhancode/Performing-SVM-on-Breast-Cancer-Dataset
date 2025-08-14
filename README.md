# Performing-SVM-on-Breast-Cancer-Dataset
#Project Overview
The goal is to classify tumors as Malignant (M) or Benign (B) using SVM with different kernels, visualize decision boundaries in 2D, tune hyperparameters, and evaluate model performance using cross-validation.

# Steps Performed

 1. Load and Prepare the Dataset
- Imported dataset from Kaggle
- Dropped unnecessary columns like `id` and `Unnamed: 32` if present.
- Converted target column:
  - `M` → 1 (Malignant)
  - `B` → 0 (Benign)
- Split into features (X) and labels (y).

 2. Train-Test Split & Scaling
- Used an 80/20 train-test split with `stratify=y` to preserve class proportions.
- Standardized features using `StandardScaler` (fit on train data only).

 3. Train SVM Models
- Linear kernel: `SVC(kernel='linear', C=1)`
- RBF kernel: `SVC(kernel='rbf', C=1, gamma='scale')`
- Compared accuracy, precision, recall, and F1-score on the test set.

4. 2D Decision Boundary Visualization
- Selected first two features for visualization.
- Trained separate SVM models on these two features.
- Plotted decision boundaries for both:
  - Linear Kernel
  - RBF Kernel
- Used `matplotlib` to display class separation.

5. Hyperparameter Tuning
- Used `GridSearchCV` with a `Pipeline` to combine scaling and SVM training.
- Tuned over:
  - `C`: [0.1, 1, 10, 100]
  - `gamma`: [1, 0.1, 0.01, 0.001]
  - Kernel: 'rbf'
- Selected the best parameters based on 5-fold cross-validation accuracy.

6. Final Model Evaluation
- Evaluated the best model on the test set.
- Performed 5-fold cross-validation on the full dataset to estimate generalization accuracy.
