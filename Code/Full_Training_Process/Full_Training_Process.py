"""

This file show training progress before getting the model.

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from linear_model_function import find_best_linear_model
from ANN_function import ANN_model

def main():
    #TODO: Data Preprocessing 
    #read file
    file_path  = os.path.join(os.getcwd(), 'dataset', 'HR_comma_sep.csv')
    try:
        with open(file_path, 'rb') as file:
            df0 =  pd.read_csv(file_path)
    except Exception as e:
        print("Error loading model:", e)
    df0 = pd.read_csv(file_path)

    #compare table
    print("This table show parameters between left and stay employee")
    compare_table = df0.drop(["Department","salary"], axis="columns").groupby("left").mean()
    display(compare_table)
    print("\n")

    #Show impact of salary on employee
    print("Show impact of salary on employee")
    ax = pd.crosstab(df0["salary"], df0["left"]).plot(kind='bar', color=['green', 'red'])
    ax.set_xlabel("Salary")
    ax.set_ylabel("Count")
    plt.title('Impact of Salary on Employee Retention')
    plt.legend(labels=['Not Left', 'Left'])
    plt.show()
    print("\n")

    #Show No. of employee left from each department
    print("Number of employee left from each department")
    pd.crosstab(df0.Department,df0.left).plot(kind='bar', color=['green', 'red'])
    ax.set_xlabel("Department")
    ax.set_ylabel("Count")
    plt.title('Employee Retention from each Department')
    plt.legend(labels=['Not Left', 'Left'])
    plt.show()
    print("\n")

    #TODO: Training Model
    print("Waiting for model training...")
    print("\n")
    #train linear model
    df_linear = pd.get_dummies(df0, dtype=float, drop_first=True)
    X = df_linear.drop(["left"],axis="columns")
    y = df_linear["left"]
    df_result, linear_models = find_best_linear_model(X, y)

    #train ANN model
    df_ann = pd.get_dummies(df0, dtype=float, drop_first=False)
    X = df_ann.drop(["left"],axis="columns")
    y = df_ann["left"]
    ann_model = ANN_model()
    cros_val_score, trained_models =  ann_model.find_best_ANN_model(X,y)

    return df_result, linear_models, cros_val_score, trained_models


if __name__ == "__main__":
    df_result, linear_models, cros_val_score, trained_models = main()
    print("\n")
    print("Finish Training and Saving Model.")

