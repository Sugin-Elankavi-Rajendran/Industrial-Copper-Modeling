Industrial Copper Modeling Application
======================================

This application provides predictive analytics for copper-related industrial data, allowing users to predict the selling price of copper items and the status (won/lost) of deals. It uses machine learning models trained on historical data to make these predictions.

Table of Contents
-----------------

1.  [Getting Started](#getting-started)
    
2.  [Dependencies](#dependencies)
    
3.  [Data Preprocessing](#data-preprocessing)
    
4.  [Model Training](#model-training)
    
    *   [Regression Model for Selling Price](#regression-model-for-selling-price)
        
    *   [Classification Model for Deal Status](#classification-model-for-deal-status)
        
5.  [Streamlit Application](#streamlit-application)
    
    *   [Predict Selling Price](#predict-selling-price)
        
    *   [Predict Status](#predict-status)
        
6.  [Usage](#usage)
    
7.  [Creator](#creator)
    

Getting Started
---------------

To get started, ensure you have all the dependencies installed and the necessary data files. You can run the application locally using Streamlit.

Dependencies
------------

Ensure you have the following Python libraries installed:

*   pandas
    
*   numpy
    
*   re
    
*   seaborn
    
*   matplotlib
    
*   scikit-learn
    
*   streamlit
    
*   pickle
    

You can install these using pip:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   shCopy codepip install pandas numpy seaborn matplotlib scikit-learn streamlit   `

Data Preprocessing
------------------

The data preprocessing steps include:

1.  Loading data from an Excel file.
    
2.  Handling missing values and incorrect data types.
    
3.  Feature engineering (e.g., creating log-transformed features).
    
4.  One-hot encoding categorical variables.
    
5.  Standardizing numerical features.
    

Model Training
--------------

### Regression Model for Selling Price

A DecisionTreeRegressor is used to predict the selling price of copper items. The model is trained using log-transformed features and optimized using GridSearchCV. The best model is saved for later use.

### Classification Model for Deal Status

A DecisionTreeClassifier is used to predict the status of a deal (won/lost). The model uses log-transformed features and is trained on a subset of the data where the status is either 'Won' or 'Lost'. The trained model is saved for later use.

Streamlit Application
---------------------

The Streamlit application has two main functionalities:

### Predict Selling Price

Users can input various features of a copper item and get a predicted selling price.

1.  **Inputs**:
    
    *   Status
        
    *   Item Type
        
    *   Country
        
    *   Application
        
    *   Product Reference
        
    *   Quantity Tons
        
    *   Thickness
        
    *   Width
        
    *   Customer ID
        
2.  **Output**:
    
    *   Predicted Selling Price
        

### Predict Status

Users can input various features of a deal and get a predicted status (won/lost).

1.  **Inputs**:
    
    *   Quantity Tons
        
    *   Thickness
        
    *   Width
        
    *   Customer ID
        
    *   Selling Price
        
    *   Item Type
        
    *   Country
        
    *   Application
        
    *   Product Reference
        
2.  **Output**:
    
    *   Predicted Status (Won/Lost)
        

Usage
-----

1.  Clone the repository.
    
2.  Ensure all dependencies are installed.
    
3.  Place the required data files in the appropriate directories.
    
4.  shCopy codestreamlit run app.py
    
5.  Use the web interface to input data and get predictions.
    

Creator
-------

App Created by Sugin Elankavi

This application leverages machine learning to provide valuable predictions for the copper industry. It is a demonstration of the power of predictive analytics in industrial applications.

For any inquiries or further information, please contact Sugin Elankavi.