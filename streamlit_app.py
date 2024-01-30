#file for streamlit for anomalies
#Isolation Forest
elif selected_anomalyAlgorithm == "Isolation Forest":
            st.markdown(
                "<h2 style='font-size: 24px; color: blue;'>Upload Dataset, Empower Machine Learning Algorithms!</h2>",
                unsafe_allow_html=True)
            data_file = st.file_uploader("Upload File", type=["csv", "xlsx", "XLSX"])

            if data_file is not None:
                file_extension = data_file.name.split(".")[-1]
                if file_extension == "csv":
                    data = pd.read_csv(data_file,encoding='ISO-8859-1')
                    
                elif file_extension in ["xlsx", "XLSX"]:
                    data = pd.read_excel(data_file)
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")






                copy_data=data.copy()
                st.write("Dealing with missing values:")
                threshold = 0.1  # Set the threshold to 10% (0.1)
                missing_percentages = data.isnull().mean()  # Calculate the percentage of missing values in each column
                columns_to_drop = missing_percentages[missing_percentages > threshold].index  # Get the columns exceeding the threshold
                data = data.drop(columns=columns_to_drop)  # Drop the columns
                st.write(f"Features with more than {threshold*100:.2f}% missing values dropped successfully.")



                data = drop_features_with_missing_values(data)




                st.write("Dealing with duplicate values...")
                num_duplicates = data.duplicated().sum()  # Count the number of duplicate rows
                data_unique = data.drop_duplicates()  # Drop the duplicate rows
                st.write(f"Number of duplicate rows: {num_duplicates}")
                st.write("Dealing done with duplicates.")

                st.write("Performing categorical feature encoding...")
                # creating the copy of the original dataset over here
                # st.write(copy_data)
                categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
                data_encoded = data_unique.copy()
                for feature in categorical_features:
                    labels_ordered = data_unique.groupby([feature]).size().sort_values().index
                    labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
                    data_encoded[feature] = data_encoded[feature].map(labels_ordered)
                data = data_encoded  # Update the original dataset with encoded features
                st.write("Categorical features encoded successfully.")

                st.write("Performing feature scaling...")
                numeric_columns = data.select_dtypes(include=["int", "float"]).columns

                if len(numeric_columns) == 0:
                    st.write("No numeric columns found.")
                else:
                    scaler = MinMaxScaler()
                    data_scaled = data.copy()
                    data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
                    data = data_scaled  # Update the original dataset with scaled features
                    st.write("Feature scaling performed successfully.")

                st.write("Downloading the dataset...")

                # Save the modified dataset to a file
                modified_dataset_filename = "modified_dataset.csv"
                # data.to_csv(modified_dataset_filename, index=False)
                st.write(data.head())
                st.write(data.shape)

                # select_PO_Number = st.selectbox("Select a PO_Number:", data.columns)
                
                # select_PO_Value = st.selectbox("Select a PO_Value:", data.columns)



             


   

                # grouped_data = (data.groupby(select_PO_Number)
                # .agg({select_PO_Value: 'sum'})
                # .reset_index())

                
                

                






                # Applying the anomaly detection
                
                data_with_anomalies_IsolationForest = apply_anomaly_detection_IsolationForest(data)


                data_with_anomalies_IsolationForest['PointColor'] = 'Inlier'
                data_with_anomalies_IsolationForest.loc[data_with_anomalies_IsolationForest['Anomaly_IF'] == 1, 'PointColor'] = 'Outlier'

                AnomalyFeature=data_with_anomalies_IsolationForest[["Anomaly_IF"]]
                # st.write(AnomalyFeature)
                
                st.subheader("Data with Anomalies")
                final_data=pd.concat([copy_data,AnomalyFeature],axis=1)
                st.write(final_data.head(5))


                
                st.subheader("Visualize anomalies")
                selected_option = st.radio("Please select the type of plot:", ["2D ScatterPlot", "3D ScatterPlot","Density Plot","Parallel Coordinates Plot","QQ-Plot"])

                if selected_option == "QQ-Plot":
                    selected_x_col = st.selectbox("Select X-axis column", data.columns)
                    
                    # Filter the data to get the selected column
                    selected_data = data_with_anomalies_IsolationForest[selected_x_col]
                    
                    # Create a QQ-Plot
                    sm.qqplot(selected_data, line='s')  # 's' indicates a standardized line
                    plt.xlabel('Theoretical Quantiles')
                    plt.ylabel(f'Quantiles of {selected_x_col}')
                    plt.title(f'QQ-Plot of {selected_x_col}')
                    plt.gca().set_facecolor('#F1F6F5')
                    st.pyplot(plt)
                
                elif selected_option=="Density Plot":
                    selected_x_col = st.selectbox("Select X-axis column", data.columns)
                    sns.kdeplot(data_with_anomalies_IsolationForest[selected_x_col], shade=True)
                    plt.xlabel(f'{selected_x_col} Label')
                    plt.ylabel('Density')
                    plt.title(f'Density Plot of {selected_x_col}')
                    plt.gca().set_facecolor('#F1F6F5')
                    st.pyplot(plt)

                elif selected_option == "Parallel Coordinates Plot":
                    selected_columns = st.multiselect("Select columns for Parallel Coordinates Plot", data.columns)
                    
                    if len(selected_columns) > 0:
                        # Filter the data based on selected columns
                        parallel_data = final_data[selected_columns + ["Anomaly"]]
                        
                        # Create a Parallel Coordinates Plot
                        fig = px.parallel_coordinates(
                            parallel_data,
                            color="Anomaly",
                            color_continuous_scale=["blue", "red"],  # Color scale for anomalies
                            labels={"Anomaly": "Anomaly"},
                        )
                        
                        # Customize the plot layout
                        fig.update_layout(
                            title="Parallel Coordinates Plot",
                            paper_bgcolor='#F1F6F5',
                            plot_bgcolor='white',
                        )
                        
                        # Display the plot using Streamlit's st.plotly_chart() function
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least one column for the Parallel Coordinates Plot.")
          


                elif selected_option=="2D ScatterPlot":
                    selected_x_col = st.selectbox("Select X-axis column", data.columns)
                    selected_y_col = st.selectbox("Select Y-axis column", data.columns)

                    # Create a scatter plot using Plotly
                    fig = px.scatter(
                        data_with_anomalies_IsolationForest,
                        x=selected_x_col,
                        y=selected_y_col,
                        color="PointColor",
                        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
                        title='Isolation Forest Anomaly Detection',
                        labels={selected_x_col: selected_x_col, "Anomaly_IF": 'Anomaly_IF', "PointColor": "Data Type"},
                    )

                    # Update the trace styling
                    fig.update_traces(
                        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
                        selector=dict(mode='markers+text')
                    )

                    # Update layout with custom styling
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

                    # Display the Plotly figure using Streamlit's st.plotly_chart() function
                    st.plotly_chart(fig)


                    import time
                    with st.spinner('Wait for it...'):
                        time.sleep(3)
                    st.success('Done!')

                    st.download_button(
                    label="Download Plot (HTML)",
                    data=plotly.offline.plot(fig, include_plotlyjs=True, output_type='div'),
                    file_name="IsolationForestAnomaly.html",
                    mime="text/html"
                    )

        
                

                # elif selected_option == "2D ScatterPlot":
                #     selected_x_col = st.selectbox("Select X-axis column", data.columns)
                #     selected_y_col = st.selectbox("Select Y-axis column", data.columns)

                #     # Create a scatter plot using Seaborn
                #     plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
                #     sns.scatterplot(data=data_with_anomalies_IsolationForest, x=selected_x_col, y=selected_y_col, hue="PointColor", palette={"Inlier": "blue", "Outlier": "red"})
                #     plt.title('Isolation Forest Anomaly Detection (2D Scatter Plot)')
                #     plt.xlabel(selected_x_col)
                #     plt.ylabel(selected_y_col)
                #     plt.gca().set_facecolor('#F1F6F5')

                #     # Display the Seaborn scatter plot using Streamlit's st.pyplot() function
                #     st.pyplot(plt)

                elif selected_option=="3D ScatterPlot":
                    selected_x_col = st.selectbox("Select X-axis column", data.columns)
                    selected_y_col = st.selectbox("Select Y-axis column", data.columns)
                    selected_z_col = st.selectbox("Select Z-axis column", data.columns)  # Add this line to select the Z-axis column

                    # Create a 3D scatter plot using Plotly
                    fig = px.scatter_3d(
                        data_with_anomalies_IsolationForest,
                        x=selected_x_col,
                        y=selected_y_col,
                        z=selected_z_col,  # Add the Z-axis
                        color="PointColor",
                        color_discrete_map={"Inlier": "blue", "Outlier": "red"},
                        title='Isolation Forest Anomaly Detection (3D Scatter Plot)',
                        labels={selected_x_col: selected_x_col, selected_y_col: selected_y_col, selected_z_col: selected_z_col, "Anomaly": 'Anomaly', "PointColor": "Data Type"},
                    )

                    # Update the trace styling
                    fig.update_traces(
                        marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
                        selector=dict(mode='markers+text')
                    )

                    # Update layout with custom styling
                    fig.update_layout(
                        legend=dict(
                            itemsizing='constant',
                            title_text='',
                            font=dict(family='Arial', size=12),
                            borderwidth=2
                        ),
                        scene=dict(
                            xaxis=dict(
                                title_text=selected_x_col,
                                title_font=dict(size=14),
                            ),
                            yaxis=dict(
                                title_text=selected_y_col,
                                title_font=dict(size=14),
                            ),
                            zaxis=dict(
                                title_text=selected_z_col,  # Add the Z-axis title
                                title_font=dict(size=14),
                            ),
                        ),
                        title_font=dict(size=18, family='Arial'),
                        paper_bgcolor='#F1F6F5',
                        plot_bgcolor='white',
                        margin=dict(l=80, r=80, t=50, b=80),
                    )

                    # Display the Plotly 3D scatter plot using Streamlit's st.plotly_chart() function
                    st.plotly_chart(fig)


                import time
                with st.spinner('Wait for it...'):
                        time.sleep(3)
                st.success('Done!')

                st.write("Download the data with anomaly indicator")
                st.download_button(
                    label="Download",
                    data=final_data.to_csv(index=False),
                    file_name="IsolationForestAnomaly.csv",
                    mime="text/csv"
                )

                filtered_data = final_data[final_data['Anomaly_IF'] == 1]
                st.write("Download the dataset where all observations are labeled as anomalies")
                st.download_button(
                    label="Download",
                    data=filtered_data.to_csv(index=False),
                    file_name="IsolationForestOnlyAnomaly.csv",
                    mime="text/csv"
                )




                # Count the number of anomalies
                num_anomalies = data_with_anomalies_IsolationForest['Anomaly_IF'].sum()

                # Total number of data points
                total_data_points = len(data_with_anomalies_IsolationForest)

                # Calculate the percentage of anomalies
                percentage_anomalies = (num_anomalies / total_data_points) * 100

                st.write(f"Number of anomalies: {num_anomalies}")
                st.write(f"Percentage of anomalies: {percentage_anomalies:.2f}%")
