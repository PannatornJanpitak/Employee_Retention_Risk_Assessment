""""
This file contain code for building dropdown table.
"""
import os
import tkinter as tk
from tkinter import ttk
import json
import numpy as np
from joblib import load

class EmployeeRetentionApp: 
    
    def __init__(self) -> None:
        self.model = self.get_model()
        self.feature_name, self.columns_name = self.get_JSON_file()
        self.create_UI()

    #load model
    def get_model(self):
        model_path = os.path.join(os.getcwd(), 'model', 'Linear_model', 'linear_best_model.pkl')
        try:
            with open(model_path, 'rb') as model_file:
                model  = load(model_file)
            return model
        except Exception as e:
            print("Error loading model:", e)

    #load JSON file
    def get_JSON_file(self):
        json_path = os.path.join(os.getcwd(), 'json_file', 'columns.json')
        try:
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
            feature_name = json_data["feature_name"]
            columns_name = json_data["columns_name"]
            return feature_name, columns_name
        except Exception as e:
            print("Error loading JSON file:", e)
    
    #design drop down table
    def create_dropdown(self, values, default_value, row):
        dropdown = ttk.Combobox(self.root, values=values, width=15)
        dropdown.grid(row=row, column=1, padx=10, pady=5, sticky="w")
        dropdown.current(0)
        dropdown.set(default_value)
        return dropdown

    #create UI dropdown table
    def create_UI(self):
        self.root = tk.Tk() #create tkinter window 
        self.root.title(" Employee Retention Classification")
        self.root.resizable(False, False) 
        self.root.configure(bg='#f0f0f0')  # Set background color
        

        #create label name for each drop down table
        table_name = [
            'Satisfaction Level',
            'Last Evaluation', 
            'Number Project', 
            'Average Monthly Hours',
            'Time Spend Company',
            'Work Accident',
            'Promotion last 5 years',
            'Department',
            'Salary']  

        #add values to drop down table
        self.dropdown_table = []
        default_values = ['0.38', '0.53', '2', '157', '3'] 
        for idx, feature in enumerate(table_name):
            tk.Label(self.root, text=feature,bg='#f0f0f0').grid(row=idx, column=0, padx=5, pady=5, sticky="e") #first 4 box will get any input from user 
            if idx < 5: #box 5 to end will give a choice to user  
                dropdown = tk.Entry(self.root, width=15)
                dropdown.insert(0, default_values[idx])
                dropdown.grid(row=idx, column=1, padx=10, pady=5, sticky="w")
            elif idx == 5 or idx == 6:
                dropdown = self.create_dropdown(["Yes", "No"], "No", idx)
            elif idx == 7:
                dropdown = self.create_dropdown(self.columns_name["Department"], self.columns_name["Department"][0], idx)
            elif idx == 8:
                dropdown = self.create_dropdown(self.columns_name["salary"], self.columns_name["salary"][0], idx)
            self.dropdown_table.append(dropdown)

        #create exit button
        exit_button = tk.Button(self.root, text="Exit", command=self.exit_app)
        exit_button.grid(row=len(table_name), column=2, padx=5, pady=10, sticky="e")

        #create button for prediction
        predict_button = tk.Button(self.root, text="Predict Retention", bg='#4CAF50', fg='white',font=("Helvetica", 10, "bold") , command=self.predict_retention)
        predict_button.grid(row=len(table_name), column=0, columnspan=2, sticky="we")

        #assign displaying prediction Text
        self.output_label = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"))
        self.output_label.grid(row=len(table_name)+1, column=0, columnspan=3, pady=5, sticky="we")
        # self.output_label = tk.Label(self.root, text="", bg='#f0f0f0', fg='#4CAF50', font=("Helvetica", 12, "bold"))

        #run tkinter UI
        self.root.mainloop()

    # Exit application
    def exit_app(self):
        self.root.destroy()

    def validate_input(self, input_data):
        """
        Validate input data:
        - Ensure values for boxes 1-4
        """
        for idx, value in enumerate(input_data[:5]):
            if idx==0 or idx==1:
                try:
                    value = float(value)
                    if not (0 <= value <= 1):
                        return False, f"Please enter values between 0-1 for boxes {idx+1}"
                except ValueError:
                    return False, f"Please enter numeric values for boxes {idx+1}"
            elif idx==2 or idx==4:
                try:
                    value = int(value)
                    if not (0 <= value <= 20):
                        return False, f"Please enter integer values between 0-20 for boxes {idx+1}"
                except ValueError:
                    return False, f"Please enter integer values for boxes {idx+1}"
            elif idx==3:
                try:
                    value = float(value)
                    if not (0 <= value <= 500):
                        return False, f"Please enter values between 0-500 for boxes {idx+1}"
                except ValueError:
                    return False, f"Please enter numeric values for boxes {idx+1}"
        return True, ""
 
    #Predict Employee Retention from user input
    def predict_retention(self):
        #get input data from user
        input_data = [box.get() for box in self.dropdown_table]
        # Manipulate input index [1,4]
        is_valid, validation_msg = self.validate_input(input_data)
        if not is_valid:
            self.output_label.config(text=validation_msg, fg="red")
            return 
        #manipulate input index [5,6]
        input_data = [1 if x == "Yes" else 0 if x == "No" else x for x in  input_data]
        #manipulate input index [7,8]
        for idx in range(7,9):
            try: 
                input_data[idx] = self.feature_name.index(list(self.columns_name.keys())[idx]+'_'+input_data[idx])
            except:
                input_data[idx] = -1

        #convert index to onehot for pass through model
        all_index = np.zeros(len(self.feature_name)) #record index after convrt to onehot

        for idx, data in enumerate(input_data):
            if idx <= 6:
                all_index[idx] = data
            else:
                if input_data[idx] >= 0:
                    all_index[input_data[idx]] = 1

        all_index = all_index.reshape(1, len(all_index))

        #prediction
        prediction = self.model.predict(all_index)
        prediction = "Retain" if prediction == 0 else "Left"
        if prediction == "Retain":
            self.output_label.config(text=f"Employee: {prediction} ", fg="green")  
        else:
            self.output_label.config(text=f"Employee: {prediction} ", fg="red")  
        print(f"Employee choose to {prediction} ")