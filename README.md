Churn Prediction App
Welcome to the Churn Prediction App! Developed by Hung Que Dang, this application leverages machine learning models to predict customer churn based on historical data. Built with Python, Streamlit, and key libraries such as Scikit-Learn, Pandas, and XGBoost, this app provides an interactive and insightful experience for analyzing and predicting customer retention.

Features
Interactive Data Visualization: Visualize customer data with dynamic charts, providing clear insights into key metrics.
Model Comparison: Compare multiple machine learning models (e.g., XGBoost, Random Forest, KNN) to assess their accuracy and performance.
Real-Time Predictions: Enter customer data to receive immediate predictions on the likelihood of churn.
User-Friendly Interface: Streamlit-powered UI designed for intuitive navigation and accessibility.


Setup Instructions
To run the app locally, follow these steps:

bash
Copy code
# Clone the repository
git clone https://github.com/alexdng10/customerchurnproj1.git

# Change into the project directory
cd churn-prediction

# Install the required packages
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
Model Details
This app utilizes several machine learning models to predict customer churn, including:

Random Forest: An ensemble method for robust classification.
XGBoost: Highly efficient for large datasets and complex patterns.
K-Nearest Neighbors (KNN): A straightforward and effective classification algorithm.
Support Vector Machines (SVM): Ideal for high-dimensional data.
Naive Bayes: A simple, probabilistic approach for classification.
Each model is fine-tuned for optimal performance using a customer churn dataset from Kaggle.

Technologies Used
Streamlit: Builds an interactive and accessible web interface.
Scikit-Learn: Implements various machine learning algorithms.
Pandas: Manages and analyzes data effectively.
Plotly: Creates dynamic and visually engaging data visualizations.
XGBoost: Provides advanced gradient-boosted decision trees.
The app integrates with the Groq API, which accelerates inference speed and supports scalable deployment. While models are trained using standard libraries such as Scikit-Learn and XGBoost, Groq API enables rapid real-time predictions, making this app efficient for both small and large datasets in production.

License
This project is licensed under the MIT License. For more information, see the LICENSE file.