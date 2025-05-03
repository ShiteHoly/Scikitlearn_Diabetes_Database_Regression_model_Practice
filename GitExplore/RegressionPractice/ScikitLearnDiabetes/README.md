# Diabetes Progression Prediction with PyTorch

This project uses a simple neural network built with PyTorch to predict diabetes progression based on baseline patient data. It employs K-Fold Cross-Validation for hyperparameter tuning and evaluates the final model on a held-out test set.

## Dataset

The project utilizes the **Diabetes dataset** available through scikit-learn (`sklearn.datasets.load_diabetes`).

* **Source:** Originally from Bradley Efron, Trevor Hastie, Iain Johnstone, and Robert Tibshirani (2004). *Least Angle Regression*, Annals of Statistics. [Dataset Link](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)
* **Instances:** 442
* **Features:** 10 baseline variables:
    * `age`: age in years
    * `sex`
    * `bmi`: body mass index
    * `bp`: average blood pressure
    * `s1`: tc, total serum cholesterol
    * `s2`: ldl, low-density lipoproteins
    * `s3`: hdl, high-density lipoproteins
    * `s4`: tch, total cholesterol / HDL
    * `s5`: ltg, possibly log of serum triglycerides level
    * `s6`: glu, blood sugar level
* **Target:** A quantitative measure of disease progression one year after baseline.

## Methodology

1.  **Preprocessing:**
    * The dataset is loaded using `sklearn.datasets.load_diabetes`.
    * Features are scaled using `sklearn.preprocessing.StandardScaler` to have zero mean and unit variance.
    * The data is split into a training/validation set (80%) and a final test set (20%).

2.  **Model Architecture:**
    * A feed-forward neural network defined using `torch.nn.Sequential`.
    * Architecture: Linear(10 -> Hidden Size) -> ReLU -> Linear(Hidden Size -> 32) -> ReLU -> Linear(32 -> 1).
    * The initial hidden layer size is treated as a hyperparameter.

3.  **Hyperparameter Tuning & Cross-Validation:**
    * A grid search is performed over:
        * Learning Rate (`lr`): [0.001, 0.01, 0.1]
        * Hidden Layer Size (`hidden_size`): [32, 64, 128]
    * 5-Fold Cross-Validation (`sklearn.model_selection.KFold`) is used on the training/validation set to evaluate each hyperparameter combination.
    * The best combination is chosen based on the lowest average Mean Squared Error (MSE) across the folds.

4.  **Final Training & Evaluation:**
    * A new model is instantiated using the best hidden layer size found during cross-validation.
    * This final model is trained on the *entire* training/validation set using the best learning rate.
    * The performance of the final trained model is evaluated on the held-out test set using MSE.

5.  **Model Saving:**
    * The `state_dict` (weights and biases) of the final trained model is saved to a file named `best_diabetes_model.pth` using `torch.save()`.

## Dependencies

You'll need the following Python libraries:

* Python (>= 3.x recommended)
* PyTorch (`torch`)
* scikit-learn (`sklearn`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)

You can install them using pip:
```bash
pip install torch scikit-learn numpy matplotlib