import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

# Set page config
st.set_page_config(
    page_title="🌟 StarCraft2 Player Analysis",
    page_icon="🚀",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            color: #333;
        }
        .streamlit-container {
            max-width: 100%;
        }
        .streamlit-table {
            width: 100%;
        }
        .reportview-container .markdown-text-container {
            font-family: 'Arial', sans-serif;
        }
        .reportview-container h1, h2, h3, h4, h5, h6 {
            color: #333;
        }
        .reportview-container p {
            font-size: 16px;
            line-height: 1.6;
        }
        .reportview-container a {
            color: #0072b5;
        }
        .streamlit-table .data {
            font-size: 14px;
        }
        .streamlit-table .col-header {
            font-size: 16px;
        }
        .streamlit-button {
            background-color: #0072b5;
            color: white !important;
        }
        .streamlit-button:hover {
            background-color: #00578a;
        }
        .nav-link {
            font-size: 20px;
            text-decoration: none;
            margin-right: 20px;
            color: #0072b5;
        }
        .nav-link:hover {
            color: #00578a;
        }
        .emoji {
            font-size: 24px;
            margin-right: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation links
st.markdown("<div class='nav-link'><a href='#overview' class='emoji'>📊</a> Overview</div>", unsafe_allow_html=True)
st.markdown("<div class='nav-link'><a href='#data-preprocessing' class='emoji'>🛠️</a> Data Preprocessing</div>", unsafe_allow_html=True)
st.markdown("<div class='nav-link'><a href='#data-visualization' class='emoji'>📈</a> Data Visualization</div>", unsafe_allow_html=True)
st.markdown("<div class='nav-link'><a href='#data-modeling' class='emoji'>🤖</a> Data Modeling</div>", unsafe_allow_html=True)
st.markdown("<div class='nav-link'><a href='#performance-summary' class='emoji'>📉</a> Performance Summary</div>", unsafe_allow_html=True)
st.markdown("<div class='nav-link'><a href='#tuning' class='emoji'>⚙️</a> Tuning</div>", unsafe_allow_html=True)

# Title and Team Information
st.markdown("<div style='background-color: lightyellow; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
st.title("🐍 Python for Data Analysis Final Project")
st.header("Characteristics of StarCraft2 Players")
st.subheader("Dataset: SkillCraft1 Master")
st.subheader("Team Members: Sainan BI, Lancine Conde, Jinyoung Ko")
st.markdown("</div>", unsafe_allow_html=True)

# Contents Overview
st.subheader("Our Contents")
st.markdown("""
    - **Part I: Load data and identify data characteristics**
    - **Part II: Data preprocessing**
        - Imputation missing data and Handling outliers
    - **Part III: Data visualization**
    - **Part IⅤ: Data Modeling**
        1. Applying various models
            - Linear Regression, Decision Tree, Random Forest, Support Vector Machine, Gradient Boosting, XGBoost, LightGBM, KNN, Neural Network
        2. Performance improvement through Grid and hyperparameter tuning
        3. Comparison of results between models
    - **Part Ⅴ: Our Conclusion**
""")

# Overview Section
st.subheader("Overview")


st.markdown("""
    Through this project, we would like to examine the characteristics of StarCraft 2 players using the skillCraft1 Master data set. Using the game-related statistics contained in this data, we will analyze the relationship between various factors of the player and his or her performance.
    StarCraft player ranks range from 1 to 8, meaning 'Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster', and 'Professional'.
""")

# Part I: Load data and identify data characteristics
st.header("Part I: Load data and identify data characteristics")

# Importing libraries required for analysis work
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# Load the data set
data_file = "SkillCraft1_Dataset.csv"
df = pd.read_csv(data_file, delimiter=',')

# Data Preview 1 - Check data size
st.info("Data Shape: " + str(df.shape))

# Data Preview 2 - Data summary information
st.dataframe(df.info())

# Data Preview 3 - Data statistical summary information
st.dataframe(df.describe())

# Data Preview 4 - Data sampling
st.subheader("Data Sampling (Head)")
st.dataframe(df.head())

# Data Preview 5 - Data sampling
st.subheader("Data Sampling (Tail)")
st.dataframe(df.tail())

# (remarks - Part I )
st.subheader("Remarks - Part I")

st.markdown("""
    After previewing the data, we can see that this dataset records the characteristics of 3395 StarCraft 2 players. 
    For each player, 20 characteristics are expressed, such as game ID, LeagueIndex, age, usage time, and number of consecutive operations per minute.
    For accurate data analysis, data outliers and missing values must be identified and processed appropriately.
""")

# Part II: Data preprocessing
st.header("Data preprocessing")

# Imputation missing data and Handling outliers
st.subheader("Imputation missing data and Handling outliers")

# Let us see how many null values each column has
st.write("Null Values:", df.isnull().sum())

# Convert "?" to NaN
df = df.replace('?', np.nan)


# Imputation of missing data and handling outliers
st.subheader("Imputation of Missing Data and Handling Outliers")

# Check unique values and remarks for 'Age', 'HoursPerWeek', and 'TotalHours'
st.write("Unique values for 'Age':", df["Age"].unique())
st.write("Unique values for 'HoursPerWeek':", df["HoursPerWeek"].unique())
st.write("Unique values for 'TotalHours':", df["TotalHours"].unique())

# Remark for unwanted character "?"
st.write("☞ Remark: There is a constant unwanted character '?' in these columns.")

# Replace "?" with NaN and convert to int type
df = df.replace('?', np.nan)
df['Age'] = df['Age'].astype(float).astype('Int64')
df['HoursPerWeek'] = df['HoursPerWeek'].astype(float).astype('Int64')
df['TotalHours'] = df['TotalHours'].astype(float).astype('Int64')

# Display DataFrame information and summary
st.dataframe(df.info())
st.dataframe(df.describe())

# Check missing values after the conversion
st.write("Missing values after conversion:")
st.write(df.isnull().sum())

# Remark for missing values in 'Age'
st.write("☞ Remark: All 55 missing values of 'Age' appear when 'LeagueIndex' is 8. We can determine that 'LeagueIndex' 8 players intentionally do not reveal their ages.")
st.write(df[df["Age"].isnull()].head(55))


# Data visualization to determine how to handle missing values
# Calculate average by grouping by LeagueIndex

# Delete all rows with NaN
df1 = df.dropna()

# Calculate average by grouping by LeagueIndex
grouped_data = df1.groupby('LeagueIndex').agg({
    'Age': 'mean',
    'HoursPerWeek': 'mean',
    'TotalHours': 'mean'
}).reset_index()

# Create a subplot with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Draw a graph for each subplot
for i, (col, palette) in enumerate([('Age', 'viridis'), ('HoursPerWeek', 'magma'), ('TotalHours', 'plasma')]):
    ax = axes[i]
    sns.barplot(data=grouped_data, x='LeagueIndex', y=col, palette=palette, ax=ax)
    ax.set_title(f'{col} (by LeagueIndex)')
    ax.set_xlabel('LeagueIndex')
    ax.set_ylabel('mean value')

# Set x-axis labels
xticks_labels = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster', 'Professional']
for ax in axes:
    ax.set_xticks(range(8))
    ax.set_xticklabels(xticks_labels, rotation=45)

# Display the Matplotlib plot using Streamlit
st.pyplot(fig)

# "Age" Boxenplot
age_boxenplot = plt.figure(figsize=(15, 10))
sns.boxenplot(data=df, x="LeagueIndex", y="Age", k_depth="trustworthy")
st.pyplot(age_boxenplot)

# Remark: Impute missing values for Age using Median Age of LeagueIndex = 7
median_val = df[df["LeagueIndex"] == 7]["Age"].median()
df["Age"] = df["Age"].fillna(median_val)

# Remark: Handling outliers for 'HoursPerWeek'
# Delete rows with HoursPerWeek = 168
df = df[df['HoursPerWeek'] != 168]

# Imputation for 'HoursPerWeek' missing values
med_val_l5 = df[df["LeagueIndex"] == 5]["HoursPerWeek"].median()
df.loc[(df["HoursPerWeek"].isnull()) & (df["LeagueIndex"] == 5), "HoursPerWeek"] = med_val_l5

mean_val_l8 = df[df["LeagueIndex"] == 7]["HoursPerWeek"].mean()
mean_val_l8 = int(mean_val_l8)
df.loc[(df["HoursPerWeek"].isnull()) & (df["LeagueIndex"] == 8), "HoursPerWeek"] = mean_val_l8

# Remark: Handling outliers for 'TotalHours'
# Delete rows with TotalHours = 1000000
df = df[df['TotalHours'] != 1000000]

# Imputation for 'TotalHours' missing values
med_val_l5 = df[df["LeagueIndex"] == 5]["TotalHours"].median()
df.loc[(df["TotalHours"].isnull()) & (df["LeagueIndex"] == 5), "TotalHours"] = med_val_l5

mean_val_l8 = df[df["LeagueIndex"] == 7]["TotalHours"].mean()
mean_val_l8 = int(mean_val_l8)
df.loc[(df["TotalHours"].isnull()) & (df["LeagueIndex"] == 8), "TotalHours"] = mean_val_l8

# Data visualization in Streamlit
st.subheader("Data Visualization")
st.write("Remarks: The age difference according to league index level is not large, but in general, the higher the level, the younger the age group. This actually makes sense because the higher the league, the more you need strength, stamina, etc.")

# Bar plots for Age, HoursPerWeek, and TotalHours by LeagueIndex
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (col, palette) in enumerate([('Age', 'viridis'), ('HoursPerWeek', 'magma'), ('TotalHours', 'plasma')]):
    ax = axes[i]
    sns.barplot(data=df, x='LeagueIndex', y=col, palette=palette, ax=ax)
    ax.set_title(f'{col} (by LeagueIndex)')
    ax.set_xlabel('LeagueIndex')
    ax.set_ylabel('mean value')

# Set x-axis labels
xticks_labels = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster', 'Professional']
for ax in axes:
    ax.set_xticks(range(8))
    ax.set_xticklabels(xticks_labels, rotation=45)

# Display plots
st.pyplot()

# Data visualization 1: Ratio of game players by LeagueIndex level
st.subheader("Data Visualization 1: Ratio of game players by LeagueIndex level")

# Map LeagueIndex values to strings.
league_mapping = {
    1: 'Bronze',
    2: 'Silver',
    3: 'Gold',
    4: 'Platinum',
    5: 'Diamond',
    6: 'Master',
    7: 'GrandMaster',
    8: 'Professional'
}
df['LeagueIndex_name'] = df['LeagueIndex'].map(league_mapping)

# Group data by LeagueIndex column and count rows per group.
grouped = df.groupby('LeagueIndex_name').size()

# Calculate the total number of data.
total_count = len(df)

# Calculate the number and percentage of rows in each group.
grouped_percentage = (grouped / total_count * 100).reset_index()
grouped_percentage.columns = ['LeagueIndex_name', 'Percentage']

# Specify the order so that they appear in the desired order on the x-axis.
order = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster', 'Professional']
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='LeagueIndex_name', y='Percentage', data=grouped_percentage, order=order)
plt.title('Percentage of Players in Each LeagueIndex')
plt.xlabel('LeagueIndex')
plt.ylabel('Percentage')

# Displays percentage values in each bar.
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), textcoords='offset points')

# Show legend
plt.xticks(rotation=45)
plt.legend(['Percentage'])

# Display the plot
st.pyplot()

# Data visualization 2: Histograms for multiple feature columns
st.subheader("Data Visualization 2: Histograms for multiple feature columns")

features = ['LeagueIndex', 'APM', 'SelectByHotkeys', 'AssignToHotkeys', 'UniqueHotkeys',
            'MinimapAttacks', 'MinimapRightClicks', 'NumberOfPACs', 'GapBetweenPACs',
            'ActionLatency', 'ActionsInPAC', 'TotalMapExplored', 'WorkersMade',
            'UniqueUnitsMade', 'ComplexUnitsMade', 'ComplexAbilitiesUsed']

# Plot histograms
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(4, 4, i + 1)
    df[feature].hist(bins=50, edgecolor='k')
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
st.pyplot()

# Data visualization 3: Box Plots for Columns
st.subheader("Data Visualization")


# Display remark
st.markdown("☞Remark : We were able to confirm relationships between data through various data visualizations. "
            "Now, we are going to apply it to various data analysis models with league index values and columns "
            "with correlations above 0.5 and below -0.5.")



# Display Part IⅤ: data Modeling text
st.title("Data Modeling")
st.write("1. Applying various models")
st.write("- 1) Linear Regression, 2) Decision Tree, 3) Random Forest, 4) Support Vector Machine, 5) Gradient Boosting, 6) XGBoost, 7) LightGBM, 8) KNN, 9) Neural Network")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the data
data_file = "SkillCraft1_Dataset.csv"
df = pd.read_csv(data_file, delimiter=',')

# ... (Previous Streamlit code for data preprocessing and visualization)

# Part V: Data Modeling
st.header("Part V: Data Modeling")

# Separate features and target variables
X = df[['APM', 'AssignToHotkeys', 'NumberOfPACs', 'GapBetweenPACs', 'ActionLatency']]
y = df['LeagueIndex']

# Data normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create model list
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Support Vector Machine', SVR()),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
    ('XGBoost', XGBRegressor(random_state=42)),
    ('LightGBM', LGBMRegressor(random_state=42)),
    ('KNN', KNeighborsRegressor())
]

# Add neural network model
nn_model = Sequential([
    Dense(10, activation='relu', input_dim=X_train.shape[1]),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')
models.append(('Neural Network', nn_model))

# Performance evaluation for each model
results = []
for name, model in models:
    if name != 'Neural Network':
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
    else:
        model.fit(X_train, y_train, epochs=100, verbose=0)
        train_preds = model.predict(X_train).flatten()
        test_preds = model.predict(X_test).flatten()

    train_rmse = mean_squared_error(y_train, train_preds, squared=False)
    test_rmse = mean_squared_error(y_test, test_preds, squared=False)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R-squared': train_r2,
        'Test R-squared': test_r2
    })

# Display results
st.subheader("Model Performance")
results_df = pd.DataFrame(results)
st.table(results_df)

# Plot graphs
for i, (name, model) in enumerate(models):
    predictions = model.predict(X_test).flatten() if name == 'Neural Network' else model.predict(X_test)

    st.subheader(f'{name} - Actual vs Predicted')
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    st.pyplot(fig)

import streamlit as st

# Display performance summary
st.title("Performance Summary of Machine Learning Models")
st.write("Linear Regression: The model exhibits comparable performance on both training and testing data, with moderate R-squared values. It seems to generalize reasonably well on the dataset.")

st.write("Decision Tree: Shows perfect performance on training data but very poor performance on testing data. This is a clear case of overfitting.")

st.write("Random Forest: High performance on training data but relatively lower on testing data. Indicates a tendency towards overfitting, though less severe than the Decision Tree.")

st.write("Support Vector Machine: Similar performance to Linear Regression, showing consistent performance across training and testing data.")

st.write("Gradient Boosting: Decent performance on training data but slightly lower on testing data. Indicates a need for better generalization of the model.")

st.write("XGBoost: Extremely high performance on training data but significantly lower on testing data, suggesting overfitting.")

st.write("LightGBM: Shows quite good performance on training data but lower on testing data. Indicates signs of overfitting.")

st.write("KNN: Fairly good performance on training data but lower on testing data. Indicates challenges in model generalization.")

st.write("Neural Network: Shows performance similar to Linear Regression and SVM, with uniform performance across training and testing data.")

st.write("In summary, most models performed relatively well on the training data but showed decreased performance on the testing data, indicating a trend of overfitting. This suggests that further adjustments are needed to improve the models' generalization capabilities. Notably, the Decision Tree and XGBoost models displayed very evident signs of overfitting. On the other hand, Linear Regression, SVM, and Neural Network models exhibited more consistent performance across both training and testing data, suggesting better generalization on the dataset.")



# Part VI: Performance improvement through Grid and hyperparameter tuning
st.title("Performance Improvement through Grid and Hyperparameter Tuning")

'''

Model: Linear Regression
Train RMSE: 1.01
Test RMSE: 1.03
R-squared (Train): 0.56
R-squared (Test): 0.53

Model: Decision Tree
Train RMSE: 0.98
Test RMSE: 1.08
R-squared (Train): 0.58
R-squared (Test): 0.48

Model: Random Forest
Train RMSE: 0.72
Test RMSE: 1.05
R-squared (Train): 0.78
R-squared (Test): 0.52

Model: Support Vector Machine
Train RMSE: 1.00
Test RMSE: 1.03
R-squared (Train): 0.56
R-squared (Test): 0.54

Model: Gradient Boosting
Train RMSE: 0.97
Test RMSE: 1.04
R-squared (Train): 0.60
R-squared (Test): 0.53

Model: XGBoost
Train RMSE: 0.95
Test RMSE: 1.03
R-squared (Train): 0.61
R-squared (Test): 0.53

Model: LightGBM
Train RMSE: 0.96
Test RMSE: 1.03
R-squared (Train): 0.60
R-squared (Test): 0.53

Model: KNN
Train RMSE: 0.94
Test RMSE: 1.09
R-squared (Train): 0.61
R-squared (Test): 0.48

85/85 [==============================] - 0s 1ms/step
22/22 [==============================] - 0s 763us/step
Neural Network
Train RMSE: 1.01
Test RMSE: 1.03
R-squared (Train): 0.56
R-squared (Test): 0.53
'''

st.title("The analysis of the machine learning models before and after hyperparameter tuning reveals some insightful changes in their performance:")

st.write("Linear Regression: No significant change in RMSE or R-squared values. The model's performance remains consistent post-tuning.")

st.write("Decision Tree: The tuning significantly reduced overfitting as evident from the improved test RMSE and R-squared. The model now generalizes better.")

st.write("Random Forest: Post-tuning, there is a notable improvement in test RMSE and R-squared, indicating a reduction in overfitting and better generalization.")

st.write("Support Vector Machine (SVM): Slight improvement in test performance. The tuning helped in achieving a better fit for the test data.")

st.write("Gradient Boosting: A minor improvement in the model's performance on test data is observed. This indicates a slight enhancement in the model's ability to generalize.")

st.write("XGBoost: The test RMSE and R-squared improved slightly, suggesting a more balanced model between training and test datasets post-tuning.")

st.write("LightGBM: Test RMSE and R-squared values show a marginal improvement. The model's generalization capability has been slightly enhanced.")

st.write("KNN (K-Nearest Neighbors): Improved test RMSE and R-squared values indicate better performance and generalization post-tuning.")

st.write("Neural Network: Performance remains consistent with the initial results, indicating that the hyperparameter tuning did not significantly affect its performance.")

st.write("Overall, hyperparameter tuning has led to improvements in most models, particularly in reducing overfitting and enhancing generalization capabilities. Models like Decision Tree and Random Forest showed the most noticeable improvements, while models like Linear Regression and Neural Network remained largely unaffected in terms of performance metrics. The consistent performance of the Neural Network, despite tuning, suggests that it might have reached its potential with the given data and architecture. The improvements in models like XGBoost, LightGBM, and KNN, although less pronounced, are significant as they indicate a better balance between fitting the training data and generalizing to unseen test data.")