# Employee Attrition and Department Prediction

This project builds a machine learning model to predict two key outcomes based on employee data:
1. Employee attrition (whether an employee is likely to leave the company).
2. Department classification (assigning an employee to a specific department).

The dataset and methodology used are designed to handle these multi-output tasks, with a focus on addressing data imbalances and optimizing model performance.

---

## Files Included
- `attrition.ipynb`: The main Jupyter notebook containing data preprocessing, model building, training, and evaluation steps.

---

## Model Overview

### Inputs
- The model takes a set of features describing employees (e.g., job role, tenure, salary) as input.

### Outputs
1. **Attrition**: A binary classification output (True/False).
2. **Department**: A multi-class classification output (e.g., Research & Development, Sales, Human Resources).

### Architecture
- **Shared Layers**: Two fully connected layers with 64 and 128 neurons respectively, using ReLU activation.
- **Output Layers**:
  - **Sigmoid activation** for attrition prediction.
  - **Softmax activation** for department classification.

### Loss Functions
- **Binary Crossentropy** for attrition.
- **Categorical Crossentropy** for department.

---

## Data Imbalance
The dataset has significant imbalances in both outputs:
- **Attrition**: 1,233 negatives (employees staying) vs. 237 positives (employees leaving).
- **Department**: Research & Development (961 employees) vs. Sales (446) vs. Human Resources (63).

Techniques to mitigate imbalance include:
- **Oversampling** using SMOTE for attrition.
- **Class weighting** for department.

---

## How to Use

### Prerequisites
- Python 3.x
- Required libraries: `tensorflow`, `pandas`, `numpy`, `scikit-learn`

### Steps
1. Open `attrition.ipynb` in a Jupyter environment.
2. Run the notebook cells to:
   - Load and preprocess the data.
   - Train the multi-output model.
   - Evaluate the model's performance.
3. Review the evaluation metrics and consider applying the suggested improvements.

## Conclusion
This project demonstrates a robust approach to handling multi-output classification tasks, focusing on imbalanced datasets and the importance of appropriate metrics. Future work could include refining the model further with advanced techniques like ensemble learning or alternative architectures.
