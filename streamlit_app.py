import streamlit as st
import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import seaborn as sns
import statsmodels.api as sm
import plotly.express as px
import matplotlib.pyplot as plt


def drop_features_with_missing_values(data):
    threshold = 0.1
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    data = data.drop(columns=columns_to_drop)
    return data


def perform_hyperparameter_tuning_IsolationForest(data):
    parameters = {
        'n_estimators': [50, 100, 150],
        'contamination': [0.01, 0.05, 0.1]
    }

    isolation_forest = IsolationForest()
    grid_search = GridSearchCV(isolation_forest, parameters, scoring='accuracy', cv=5)
    grid_search.fit(data)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    return best_params, best_estimator

def perform_hyperparameter_tuning_EllipticEnvelope(data):
    parameters = {
        'contamination': [0.01, 0.05, 0.1],
        'support_fraction': [0.8, 0.9, 0.95]
    }

    elliptic_envelope = EllipticEnvelope()
    grid_search = GridSearchCV(elliptic_envelope, parameters, scoring='accuracy', cv=5)
    grid_search.fit(data)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    return best_params, best_estimator

def perform_hyperparameter_tuning_LocalOutlierFactor(data):
    parameters = {
        'contamination': [0.01, 0.05, 0.1],
        'n_neighbors': [5, 10, 15]
    }

    local_outlier_factor = LocalOutlierFactor()
    grid_search = GridSearchCV(local_outlier_factor, parameters, scoring='accuracy', cv=5)
    grid_search.fit(data)
    
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    return best_params, best_estimator


def perform_hyperparameter_tuning_GaussianMixture(data):
    parameters = {
        'n_components': [2, 3, 4],
        'covariance_type': ['full', 'tied', 'diag', 'spherical']
    }

    gaussian_mixture = GaussianMixture()
    grid_search = GridSearchCV(gaussian_mixture, parameters, scoring='accuracy', cv=5)
    grid_search.fit(data)

    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    return best_params, best_estimator

def apply_anomaly_detection_IsolationForest(data, best_model):
    if 'Anomaly' in data.columns:
        data_without_anomaly = data.drop(columns=['Anomaly'])
    else:
        data_without_anomaly = data.copy()
    data['Anomaly'] = best_model.fit_predict(data_without_anomaly)
    # Mapping -1 to 1 for anomalies and 0 to inliers
    data['Anomaly'] = data['Anomaly'].map({-1: 1, 1: 0})
    return data


def apply_anomaly_detection_EllipticEnvelope(data, best_model):
    if 'Anomaly' in data.columns:
        data_without_anomaly = data.drop(columns=['Anomaly'])
    else:
        data_without_anomaly = data.copy()
    data['Anomaly'] = best_model.fit_predict(data_without_anomaly)
    # Mapping -1 to 1 for anomalies and 0 to inliers
    data['Anomaly'] = data['Anomaly'].map({-1: 1, 1: 0})
    return data

def apply_anomaly_detection_LocalOutlierFactor(data, best_model):
    if 'Anomaly' in data.columns:
        data_without_anomaly = data.drop(columns=['Anomaly'])
    else:
        data_without_anomaly = data.copy()
    data['Anomaly'] = best_model.fit_predict(data_without_anomaly)
    # Mapping -1 to 1 for anomalies and 0 to inliers
    data['Anomaly'] = data['Anomaly'].map({-1: 1, 1: 0})
    return data

def apply_anomaly_detection_GaussianMixture(data, best_model):
    if 'Anomaly' in data.columns:
        data_without_anomaly = data.drop(columns=['Anomaly'])
    else:
        data_without_anomaly = data.copy()
    data['Anomaly'] = best_model.fit_predict(data_without_anomaly)
    # Mapping -1 to 1 for anomalies and 0 to inliers
    data['Anomaly'] = data['Anomaly'].map({-1: 1, 1: 0})
    return data

def main():
    st.title("Anomaly Detection App")

    # Anomaly detection section
    selected_anomalyAlgorithm = st.selectbox("Select Anomaly Detection Algorithm", ["Isolation Forest", "Elliptic Envelope", "Local Outlier Factor", "Gaussian Mixture"])

    st.markdown("<h2 style='font-size: 24px; color: blue;'>Upload Dataset, Empower Machine Learning Algorithms!</h2>", unsafe_allow_html=True)
    data_file = st.file_uploader("Upload File", type=["csv", "xlsx", "XLSX"])

    if data_file is not None:
        file_extension = data_file.name.split(".")[-1]
        if file_extension == "csv":
            data = pd.read_csv(data_file, encoding='ISO-8859-1', low_memory= False)
        elif file_extension in ["xlsx", "XLSX"]:
            data = pd.read_excel(data_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")

        copy_data = data.copy()

        # Preprocessing steps
        st.write("Dealing with missing values:")
        data = drop_features_with_missing_values(data)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')  # You can use other strategies like 'median', 'most_frequent', etc.
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


        st.write("Dealing with duplicate values...")
        num_duplicates = data.duplicated().sum()
        data_unique = data.drop_duplicates()
        st.write(f"Number of duplicate rows: {num_duplicates}")
        st.write("Dealing done with duplicates.")

        st.write("Performing categorical feature encoding...")
        categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
        data_encoded = data_unique.copy()
        for feature in categorical_features:
            labels_ordered = data_unique.groupby([feature]).size().sort_values().index
            labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
            data_encoded[feature] = data_encoded[feature].map(labels_ordered)
        data = data_encoded

        st.write("Performing feature scaling...")
        numeric_columns = data.select_dtypes(include=["int", "float"]).columns

        if len(numeric_columns) > 0:
            scaler = MinMaxScaler()
            data_scaled = data.copy()
            data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
            data = data_scaled
            st.write("Feature scaling performed successfully.")
        else:
            st.write("No numeric columns found.")

        st.write("Downloading the dataset...")
        modified_dataset_filename = "modified_dataset.csv"
        st.write(data.head())
        st.write(data.shape)

        # Performing hyperparameter tuning...
        st.write("Performing hyperparameter tuning...")

        if selected_anomalyAlgorithm == "Isolation Forest":
            best_params, best_model = perform_hyperparameter_tuning_IsolationForest(data)
        elif selected_anomalyAlgorithm == "Elliptic Envelope":
            best_params, best_model = perform_hyperparameter_tuning_EllipticEnvelope(data)
        elif selected_anomalyAlgorithm == "Local Outlier Factor":
            best_params, best_model = perform_hyperparameter_tuning_LocalOutlierFactor(data)
        elif selected_anomalyAlgorithm == "Gaussian Mixture":
            best_params, best_model = perform_hyperparameter_tuning_GaussianMixture(data)

        # Displaying the best hyperparameters
        st.write(f"Best Hyperparameters for {selected_anomalyAlgorithm}: {best_params}")

        # Applying the selected anomaly detection algorithm
        if selected_anomalyAlgorithm == "Isolation Forest":
            data_with_anomalies = apply_anomaly_detection_IsolationForest(data, best_model)
        elif selected_anomalyAlgorithm == "Elliptic Envelope":
            data_with_anomalies = apply_anomaly_detection_EllipticEnvelope(data, best_model)
        elif selected_anomalyAlgorithm == "Local Outlier Factor":
            data_with_anomalies = apply_anomaly_detection_LocalOutlierFactor(data, best_model)
        elif selected_anomalyAlgorithm == "Gaussian Mixture":
            data_with_anomalies = apply_anomaly_detection_GaussianMixture(data, best_model)


        original_data_with_anomalies = pd.concat([copy_data, data_with_anomalies], axis=1)
        original_data_with_anomalies['PointColor'] = 'Inlier'
        original_data_with_anomalies.loc[original_data_with_anomalies['Anomaly'] == 1, 'PointColor'] = 'Outlier'


        #st.subheader("Data with Anomalies")
        # Debugging statements
        #st.write("Columns in data_with_anomalies:", data_with_anomalies.columns)
       # subset_columns = ['Anomaly', 'PointColor']
       # st.write(f"Subset of columns to be selected: {subset_columns}")
       # st.write(f"Columns available in data_with_anomalies: {data_with_anomalies.columns}")
        #st.write(f"Columns present in subset: {[col for col in subset_columns if col in data_with_anomalies.columns]}")

        # Concatenation
        # Concatenation
      #  common_columns = set(copy_data.columns) & set(data_with_anomalies.columns)
       # final_data = pd.concat([copy_data[common_columns], data_with_anomalies[subset_columns].copy()], axis=1)
        data_with_anomalies['PointColor'] = 'Inlier'
        data_with_anomalies.loc[data_with_anomalies['Anomaly'] == -1, 'PointColor'] = 'Outlier'

        st.subheader("Data with Anomalies")
        final_data = pd.concat([copy_data, data_with_anomalies[['Anomaly', 'PointColor']]], axis=1)
        st.write(final_data.head(5))

        st.write(final_data.head(5))

        st.subheader("Visualize anomalies")
        selected_option = st.radio("Please select the type of plot:", ["2D ScatterPlot", "Density Plot", "Parallel Coordinates Plot", "QQ-Plot"])

        if selected_option == "QQ-Plot":
            selected_x_col = st.selectbox("Select X-axis column", data.columns)
            selected_data = data_with_anomalies[selected_x_col]
            sm.qqplot(selected_data, line='s')
            plt.xlabel('Theoretical Quantiles')
            plt.ylabel(f'Quantiles of {selected_x_col}')
            plt.title(f'QQ-Plot of {selected_x_col}')
            plt.gca().set_facecolor('#F1F6F5')
            st.pyplot(plt)

        elif selected_option == "Density Plot":
            selected_x_col = st.selectbox("Select X-axis column", data.columns)
            sns.kdeplot(data_with_anomalies[selected_x_col], shade=True)
            plt.xlabel(f'{selected_x_col} Label')
            plt.ylabel('Density')
            plt.title(f'Density Plot of {selected_x_col}')
            plt.gca().set_facecolor('#F1F6F5')
            st.pyplot(plt)

        elif selected_option == "Parallel Coordinates Plot":
            selected_columns = st.multiselect("Select columns for Parallel Coordinates Plot", data.columns)
            if len(selected_columns) > 0:
                parallel_data = final_data[selected_columns + ["PointColor"]]
                fig = px.parallel_coordinates(
                    parallel_data,
                    color="PointColor",
                    color_continuous_scale=["blue", "red"],
                    labels={"PointColor": "Anomaly"},
                )
                fig.update_layout(
                    title="Parallel Coordinates Plot",
                    paper_bgcolor='#F1F6F5',
                    plot_bgcolor='white',
                )
                st.plotly_chart(fig)
            else:
                st.warning("Please select at least one column for the Parallel Coordinates Plot.")

        elif selected_option == "2D ScatterPlot":
            selected_x_col = st.selectbox("Select X-axis column", data.columns)
            selected_y_col = st.selectbox("Select Y-axis column", data.columns)
            fig = px.scatter(
                data_with_anomalies,
                x=selected_x_col,
                y=selected_y_col,
                color="PointColor",
                color_discrete_map={"Inlier": "blue", "Outlier": "red"},
                title=f'{selected_anomalyAlgorithm} Anomaly Detection',
                labels={selected_x_col: selected_x_col, selected_y_col: selected_y_col, "Anomaly": 'Anomaly', "PointColor": "Data Type"},
            )
            fig.update_traces(
                marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
                selector=dict(mode='markers+text')
            )
            fig.update_layout(
                legend=dict(
                    itemsizing='constant',
                    title_text='',
                    font=dict(family='Arial', size=12),
                    borderwidth=2
                ),
                xaxis=dict(
                    title_text=selected_x_col,
                    title_font=dict(size=14),
                    showgrid=False,
                    showline=True,
                    linecolor='lightgray',
                    linewidth=2,
                    mirror=True
                ),
                yaxis=dict(
                    title_text=selected_y_col,
                    title_font=dict(size=14),
                    showgrid=False,
                    showline=True,
                    linecolor='lightgray',
                    linewidth=2,
                    mirror=True
                ),
                title_font=dict(size=18, family='Arial'),
                paper_bgcolor='#F1F6F5',
                plot_bgcolor='white',
                margin=dict(l=80, r=80, t=50, b=80),
            )
            st.plotly_chart(fig)

            import time
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success('Done!')



            #st.download_button(
            #    label=f"Download Plot ({selected_anomalyAlgorithm} - HTML)",
            #    data=plotly.offline.plot(fig, include_plotlyjs=True, output_type='div'),
            #    file_name=f"{selected_anomalyAlgorithm}Anomaly.html",
            #    mime="text/html"
           # )


        # ... (other options for visualization)

        st.write("Download the data with anomaly indicator")
        st.download_button(
            label="Download",
            data=final_data.to_csv(index=False),
            file_name=f"{selected_anomalyAlgorithm}Anomaly.csv",
            mime="text/csv"
        )

        filtered_data = final_data[final_data['Anomaly'] == -1]
        st.write("Download the dataset where all observations are labeled as anomalies")
        st.download_button(
            label="Download",
            data=filtered_data.to_csv(index=False),
            file_name=f"{selected_anomalyAlgorithm}OnlyAnomaly.csv",
            mime="text/csv"
        )

        num_anomalies = (data_with_anomalies['Anomaly'] == -1).sum()
        total_data_points = len(data_with_anomalies)
        percentage_anomalies = (num_anomalies / total_data_points) * 100

        st.write(f"Number of anomalies: {num_anomalies}")
        st.write(f"Percentage of anomalies: {percentage_anomalies:.2f}%")

# Run the Streamlit app
if __name__ == "__main__":
    main()
