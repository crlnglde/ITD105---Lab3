import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt1

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib



# Sidebar with Classification and Regression navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Task", ["Classification", "Regression"])


if option == "Classification":
    best_model_instance = None
    # Step 1: Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
       
        # Initialize session state for pagination
        if 'start_idx' not in st.session_state:
            st.session_state.start_idx = 0

        # Display the first 10 rows of the current section
        st.write("Data Preview:")
        st.write(df.iloc[st.session_state.start_idx:st.session_state.start_idx+10])


        button_container = st.container()
        with button_container:
            # Create two buttons in the same container with a gap
            col1, col2, col3 = st.columns([1, 1, 1])
              
            with col2:  # Place the buttons in the middle column
                prev_button, next_button = st.columns([1, 1])  # Two equal columns for the buttons side by side

                with prev_button:
                    # Create the 'Previous' button
                    prev_button = st.button('Previous')
                    if prev_button and st.session_state.start_idx - 10 >= 0:
                        st.session_state.start_idx -= 10

                with next_button:
                    # Create the 'Next' button
                    next_button = st.button('Next')
                    if next_button and st.session_state.start_idx + 10 < len(df):
                        st.session_state.start_idx += 10
            
        # Step 2: Preprocessing
        X = df.drop(columns=['Heart Disease'])  # Features
        y = df['Heart Disease']  # Target
        
        # Step 3: Train/Test Split (80:20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Machine Learning Algorithms
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        # Step 5: Train models and evaluate accuracy
        results = []
        highest_accuracy_model = None
        highest_accuracy_value = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((name, accuracy))

            # Track the highest accuracy model
            if accuracy > highest_accuracy_value:
                highest_accuracy_value = accuracy
                highest_accuracy_model = name
                best_model_instance = model

        # Step 6: Display Results in a Table for Comparison
        results_df = pd.DataFrame(results, columns=['ML Algorithm', 'Accuracy'])
        
        def highlight_max_min(row):
            
            if row['Accuracy'] == results_df['Accuracy'].max():
                return ['background-color: lightblue; color: black' for _ in row]  
            elif row['Accuracy'] == results_df['Accuracy'].min():
                return ['background-color: lightcoral; color: black' for _ in row]  
            else:
                return ['' for _ in row]  

        
        st.write("Machine Learning Algorithms and their Accuracy:")
        st.dataframe(results_df.style.apply(highlight_max_min, axis=1))

        # Step 7: Generate a Graph for Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['lightblue' if acc == max(results_df['Accuracy']) else 'lightcoral' if acc == min(results_df['Accuracy']) else 'gray' for acc in results_df['Accuracy']]
        ax.barh(results_df['ML Algorithm'], results_df['Accuracy'], color=colors)
        ax.set_xlabel('Accuracy')
        ax.set_title('ML Algorithms Accuracy Comparison')
        st.pyplot(fig)

        # Hyperparameter grids for tuning
        param_grids = {
            "Decision Tree": {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]},
            "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]},
            "Logistic Regression": {'C': [0.01, 0.1, 1, 10]},
            "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7, 9]},
            "Naive Bayes": {},  # No hyperparameters to tune
            "SVM": {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            "Gradient Boosting": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
        }

        # Step 8: Hyperparameter Tuning with GridSearchCV
        tuning_results = []
        best_models = {}
        
        
        if highest_accuracy_model:
            model = models[highest_accuracy_model]
            if highest_accuracy_model in param_grids and param_grids[highest_accuracy_model]:
                grid_search = GridSearchCV(model, param_grids[highest_accuracy_model], cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                results = pd.DataFrame(grid_search.cv_results_)
                results['Accuracy'] = results['mean_test_score']  # Adding accuracy column for clarity
                results['Hyperparameters'] = results['params']  # Adding hyperparameters column
                # Create a list of model names for each iteration
                results['Model'] = [f"Model {i+1}" for i in range(len(results))]
                tuning_results.append((highest_accuracy_model, results, grid_search.best_estimator_))
                best_model_instance = grid_search.best_estimator_
            else:
                # For models without tunable parameters
                model.fit(X_train, y_train)
                accuracy = accuracy_score(y_test, model.predict(X_test))
                best_models[name] = model
                results = pd.DataFrame({
                    'Model': [name],
                    'Parameters': ["N/A"],
                    'Accuracy': [accuracy],
                })
                tuning_results.append((name, results, model))

        # Step 9: Display Hyperparameter Tuning Results with Hyperparameters Column
        for model_name, results_df, best_model in tuning_results:
            if not results_df.empty:
                st.write(f"**Hyperparameter Tuning Results for {model_name}:**")
                
                # Normalize the parameter dictionary into separate columns
                results_df = results_df.rename(columns={'params': 'Parameters'})
                parameter_columns = pd.json_normalize(results_df['Parameters']).columns.tolist()

                # Add the model parameters to the table
                tuning_table = pd.concat([results_df[['Model', 'Accuracy']], pd.json_normalize(results_df['Parameters'])], axis=1)

                # Reorder columns: Model Iteration -> Model -> Parameters -> Accuracy
                all_columns = ['Model'] + [col for col in tuning_table.columns if col not in ['Model', 'Accuracy']] + ['Accuracy']
                tuning_table = tuning_table[all_columns]
                
                 # Find the highest accuracy in the 'Accuracy' column
                max_accuracy = tuning_table['Accuracy'].max()
        
                # Highlight the highest accuracy in the tuning table
                def highlight_max_accuracy(row):
                    color = ['background-color: skyblue ; color: black' if row['Accuracy'] == results_df['Accuracy'].max() else '' for _ in row]
                    return color

                # Display the table without sorting by Accuracy
                st.dataframe(tuning_table.style.apply(highlight_max_accuracy, axis=1))

                
        # Step 10: Download the Best Model
        if st.button("Save the Most Accurate Model"):
            if best_model_instance:
                # Save the model using joblib
                model_filename = f"{highest_accuracy_model}_best_model.pkl"
                joblib.dump(best_model_instance, model_filename)
                st.success(f"Model '{highest_accuracy_model}' with the best hyperparameters saved successfully! You can download it below.")
                
                # Provide a download button for the saved model
                with open(model_filename, "rb") as file:
                    st.download_button(label="Download Model", data=file, file_name=model_filename)
            else:
                st.warning("No model available for download.")


if option == "Regression":

    # Step 1: Upload CSV file for Regression
    uploaded_file = st.file_uploader("Upload your CSV file for Regression", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
       
        # Initialize session state for pagination
        if 'start_idx' not in st.session_state:
            st.session_state.start_idx = 0

        # Display the first 10 rows of the current section
        st.write("Data Preview:")
        st.write(df.iloc[st.session_state.start_idx:st.session_state.start_idx+10])


        button_container = st.container()
        with button_container:
            # Create two buttons in the same container with a gap
            col1, col2, col3 = st.columns([1, 1, 1])
              
            with col2:  # Place the buttons in the middle column
                prev_button, next_button = st.columns([1, 1])  # Two equal columns for the buttons side by side

                with prev_button:
                    # Create the 'Previous' button
                    prev_button = st.button('Previous')
                    if prev_button and st.session_state.start_idx - 10 >= 0:
                        st.session_state.start_idx -= 10

                with next_button:
                    # Create the 'Next' button
                    next_button = st.button('Next')
                    if next_button and st.session_state.start_idx + 10 < len(df):
                        st.session_state.start_idx += 10

        # Step 2: Preprocessing
        X = df.drop(columns=['Moisture_Content_Percent'])  # Features
        y = df['Moisture_Content_Percent']  # Target (regression label)

        # Step 3: K-Fold Cross Validation (k=10)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        # Step 4: Machine Learning Algorithms for Regression
        models = {
            "CART": RandomForestRegressor(),
            "Elastic Net": ElasticNet(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge(),
            "Linear Regression": LinearRegression(),
            "Multi-Layer Perceptron": MLPRegressor(max_iter=1000),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Machines": SVR()
        }

        # Step 5: Train models and evaluate MAE using K-fold cross-validation
        results = []
        best_models = {}  # Store trained models
        for name, model in models.items():
            neg_mae_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
            mae = -neg_mae_scores.mean()
            results.append((name, mae))
            best_models[name] = model.fit(X, y)  # Train and store the model

        # Step 6: Display Results in a Table
        results_df = pd.DataFrame(results, columns=['ML Algorithm', 'MAE'])

        # Find the lowest MAE in the 'MAE' column
        min_mae = results_df['MAE'].min()
        max_mae = results_df['MAE'].max()

        def highlight_mae(row):
            # Compare the MAE values for the row
            if row['MAE'] == min_mae:
                return ['background-color: lightblue; color: black' for _ in row]  # Highlight the lowest MAE (best model) in light blue
            elif row['MAE'] == max_mae:
                return ['background-color: lightcoral; color: black' for _ in row]  # Highlight the highest MAE (worst model) in light red
            else:
                return ['' for _ in row]  # No highlighting for other rows


        st.write("Machine Learning Algorithms and their MAE:")
        st.dataframe(results_df.style.apply(highlight_mae, axis=1))  # Highlight the row with the minimum MAE

        # Step 7: Generate a Graph for Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['lightblue' if mae == min(results_df['MAE']) else 'lightcoral' if mae == max(results_df['MAE']) else 'gray' for mae in results_df['MAE']]
        ax.barh(results_df['ML Algorithm'], results_df['MAE'], color=colors)

        # Set labels and title
        ax.set_xlabel('Mean Absolute Error (MAE)')
        ax.set_title('ML Algorithms MAE Comparison')

        st.pyplot(fig)

        # Step 8: Hyperparameter Tuning with GridSearchCV for the best model (Lowest MAE)
        tuning_results = []
        best_models_for_tuning = {}

        # Identify the model with the lowest MAE
        best_model_name = results_df.loc[results_df['MAE'] == min_mae, 'ML Algorithm'].values[0]
        best_model = best_models[best_model_name]

        # Hyperparameter grids for tuning (For illustration purposes, we'll define grids for a few models)
        param_grids = {
            "CART": {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]},
            "Elastic Net": {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.1, 0.5, 0.9]},
            "Gradient Boosting": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
            "K-Nearest Neighbors": {'n_neighbors': [3, 5, 7, 9]},
            "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]},
        }

        if best_model_name in param_grids:
            grid_search = GridSearchCV(best_model, param_grids[best_model_name], cv=5, n_jobs=-1)
            grid_search.fit(X, y)
            tuning_results_df = pd.DataFrame(grid_search.cv_results_)
            tuning_results_df['MAE'] = tuning_results_df['mean_test_score']  # Adding MAE column for clarity
            tuning_results_df['Hyperparameters'] = tuning_results_df['params']  # Adding hyperparameters column
            tuning_results_df['Model'] = [f"Model {i+1}" for i in range(len(tuning_results_df))]
            tuning_results.append((best_model_name, tuning_results_df, grid_search.best_estimator_))

        # Step 9: Display Hyperparameter Tuning Results with Hyperparameters Column
        for model_name, results_df, best_model in tuning_results:
            if not results_df.empty:
                st.write(f"**Hyperparameter Tuning Results for {model_name}:**")
                
                # Normalize the parameter dictionary into separate columns
                results_df = results_df.rename(columns={'params': 'Parameters'})
                parameter_columns = pd.json_normalize(results_df['Parameters']).columns.tolist()

                # Add the model parameters to the table
                tuning_table = pd.concat([results_df[['Model', 'MAE']], pd.json_normalize(results_df['Parameters'])], axis=1)

                # Reorder columns: Model Iteration -> Model -> Parameters -> MAE
                all_columns = ['Model'] + [col for col in tuning_table.columns if col not in ['Model', 'MAE']] + ['MAE']
                tuning_table = tuning_table[all_columns]

                # Find the lowest MAE in the 'MAE' column
                min_mae_tuning = tuning_table['MAE'].min()

                # Highlight the row with the lowest MAE
                def highlight_min_mae_tuning(row):
                    return ['background-color: lightblue ; color: black' if row['MAE'] == min_mae_tuning else '' for _ in row]

                # Display the table and highlight the row with the lowest MAE
                st.dataframe(tuning_table.style.apply(highlight_min_mae_tuning, axis=1))  # Highlight the best tuning row

        # Step 10: Download the Best Model
        if st.button("Save the Most Accurate Model"):
            
            best_model_instance = best_models[best_model_name]  # Retrieve the best model instance

            if best_model_instance:
                # Save the model using joblib
                model_filename = f"{best_model_name}_best_model.pkl"
                joblib.dump(best_model_instance, model_filename)
                st.success(f"Model '{best_model_name}' with the best hyperparameters saved successfully! You can download it below.")
                
                # Provide a download button for the saved model
                with open(model_filename, "rb") as file:
                    st.download_button(label="Download Model", data=file, file_name=model_filename)
            else:
                st.warning("No model available for download.")