
"""

This file contain all function for creating Artificial neural network(ANN) model for [Data Science Job Salaries 2020 - 2024] dataset

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

class ANN_model:
    
    def __init__(self) -> None:
        self.model = self.create_ANN_model() 

    #Create ANN model
    def create_ANN_model(self):
        """
        #This function contain [ANN model] architechure 
        input = []
        output = ANN model 
        """
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_shape=(20,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer= Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['binary_accuracy'])
        return model


    #train ANN model
    def find_best_ANN_model(self,X,y):
        """
        #This function perform K-fold cross validation to [ANN model] 
        #defualt K = 10
        input = (X,y) = (your data feature, your data target)
        output = dataframe result of this model, all model from K-fold cross validation
        """
        # Initialize k-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cros_val_score = []  # store cross validation value
        trained_models = []  # store trained models

        #Scale the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform k-fold cross-validation
        for train_index, test_index in kf.split(X_scaled ):
            X_train_scaled, X_test_scaled = X_scaled [train_index], X_scaled [test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Build the model
            model = self.create_ANN_model()
            
            # Train the model
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train_scaled, y_train, epochs=1000, batch_size=64, validation_split=0.15, verbose=1, callbacks=[early_stop])

            # Store the trained model
            trained_models.append(model)

            # Evaluate the model on validation data
            _, mae = model.evaluate(X_test_scaled, y_test)
            cros_val_score.append(mae)
        
        save_best_ann_model(cros_val_score, trained_models) #save best model

        return cros_val_score, trained_models
    
#save ANN model
def save_best_ann_model(cros_val_score, trained_models):
    best_ann_model =  trained_models[cros_val_score.index(min(cros_val_score))]
    best_ann_model.save("ANN_best_model.h5")
