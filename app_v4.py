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
st.set_option('deprecation.showPyplotGlobalUse', False)



# Set page config
st.set_page_config(
    page_title="🌟 StarCraft2 Player Analysis",
    page_icon="🚀",
    layout="wide",
)

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
        .reportview-container h1, .reportview-container h2, .reportview-container h3, .reportview-container h4, .reportview-container h5, .reportview-container h6 {
            color: #00578a; /* Adjusted title color */
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


st.markdown("### Variable Information")

variable_info = """
- **GameID:** Unique ID number for each game (integer)
- **LeagueIndex:** Bronze, Silver, Gold, Platinum, Diamond, Master, GrandMaster, and Professional leagues coded 1-8 (Ordinal)
- **Age:** Age of each player (integer)
- **HoursPerWeek:** Reported hours spent playing per week (integer)
- **TotalHours:** Reported total hours spent playing (integer)
- **APM:** Action per minute (continuous)
- **SelectByHotkeys:** Number of unit or building selections made using hotkeys per timestamp (continuous)
- **AssignToHotkeys:** Number of units or buildings assigned to hotkeys per timestamp (continuous)
- **UniqueHotkeys:** Number of unique hotkeys used per timestamp (continuous)
- **MinimapAttacks:** Number of attack actions on minimap per timestamp (continuous)
- **MinimapRightClicks:** Number of right-clicks on minimap per timestamp (continuous)
- **NumberOfPACs:** Number of PACs per timestamp (continuous)
- **GapBetweenPACs:** Mean duration in milliseconds between PACs (continuous)
- **ActionLatency:** Mean latency from the onset of PACs to their first action in milliseconds (continuous)
- **ActionsInPAC:** Mean number of actions within each PAC (continuous)
- **TotalMapExplored:** The number of 24x24 game coordinate grids viewed by the player per timestamp (continuous)
- **WorkersMade:** Number of SCVs, drones, and probes trained per timestamp (continuous)
- **UniqueUnitsMade:** Unique units made per timestamp (continuous)
- **ComplexUnitsMade:** Number of ghosts, infestors, and high templars trained per timestamp (continuous)
- **ComplexAbilitiesUsed:** Abilities requiring specific targeting instructions used per timestamp (continuous)
"""

st.markdown(variable_info)


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
st.write("Imputation missing data and Handling outliers¶")

# Imputation missing data and Handling outliers
st.subheader("Imputation missing data and Handling outliers")

# Let us see how many null values each column has
st.write("Null Values:", df.isnull().sum())
st.markdown("### Remark")
st.write("☞Although there are no explicit null values, it is necessary to check why the values 'Age', 'HoursPerWeek', and 'TotalHours' are of object type even though they are numeric features.")
# Convert "?" to NaN
df = df.replace('?', np.nan)


# Imputation of missing data and handling outliers
st.subheader("Imputation of Missing Data and Handling Outliers")

# Check unique values and remarks for 'Age', 'HoursPerWeek', and 'TotalHours'
st.write("Unique values for 'Age':", df["Age"].unique())
st.write("Unique values for 'HoursPerWeek':", df["HoursPerWeek"].unique())
st.write("Unique values for 'TotalHours':", df["TotalHours"].unique())

# Remark for unwanted character "?"
st.write("☞Remark : There is a constant unwanted character that appears in these three (3) columns \"?\". We will replace \"?\" with NaN and convert the columns 'Age', 'HoursPerWeek', and 'TotalHours' to type int.")


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



st.write(df[df["Age"].isnull()].head(55))


# Remark for missing values in 'Age'
st.write("☞ Remark: All 55 missing values of 'Age' appear when 'LeagueIndex' is 8. We can determine that 'LeagueIndex' 8 players intentionally do not reveal their ages.")




# Data visualization to determine how to handle missing values
# Calculate average by grouping by LeagueIndex
st.title("Let's visualize")
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


st.code("""
# "Age" Boxenplot
age_boxenplot = plt.figure(figsize=(15, 10))
sns.boxenplot(data=df, x="LeagueIndex", y="Age", k_depth="trustworthy")
st.pyplot(age_boxenplot)
""")


# "Age" Boxenplot
age_boxenplot = plt.figure(figsize=(15, 10))
sns.boxenplot(data=df, x="LeagueIndex", y="Age", k_depth="trustworthy")
st.pyplot(age_boxenplot)


st.write("☞Remark : The graph shows that high-league players spend more time playing, and that some data represents outliers.")
st.write("Therefore, we can use the median of 'LeagueIndex' = 7 to impute the missing value of 'HoursPerWeek' for players with 'LeagueIndex' = 8.")
st.write("And outlier data can be removed.")

st.write("# outlier handling- HoursPerWeek max values")
top_10_hours_per_week = df.nlargest(10, 'HoursPerWeek')
st.write(top_10_hours_per_week)

st.write("☞Remark : Since one week is 168 hours (7 days x 24 hours), the playtime of 168 hours per week can be judged as an outlier. Delete rows with this value.")

# Delete outlier rows
max_hours_index = df['HoursPerWeek'].idxmax()
df = df.drop(max_hours_index)
st.code("""
# Delete outlier rows
max_hours_index = df['HoursPerWeek'].idxmax()
df = df.drop(max_hours_index)
""", language='python')

top_10_hours_per_week = df.nlargest(10, 'HoursPerWeek')
st.write(top_10_hours_per_week)


st.title("Let's now check the distribution of Total hours for different leagueindex")
total_hours_boxenplot = plt.figure(figsize=(15, 10))
sns.boxenplot(data=df, x="LeagueIndex", y="TotalHours", k_depth="trustworthy")
st.pyplot(total_hours_boxenplot)
st.write("☞Remark : There is an important trend to note here. After removing outlier values, we will imputate missing values.")


top_10_total_hours = df.nlargest(10, 'TotalHours')
st.write(top_10_total_hours)

st.write("Delete outlier rows that have TotalHours = 1000000") 
max_hours_index = df['TotalHours'].idxmax()
df = df.drop(max_hours_index)


st.write("Check deletion results") 
top_10_hours_per_week = df.nlargest(10, 'TotalHours')
st.code(top_10_hours_per_week)

# Boxenplot for TotalHours by LeagueIndex
total_hours_boxenplot = plt.figure(figsize=(15, 10))
sns.boxenplot(data=df, x="LeagueIndex", y="TotalHours", k_depth="trustworthy")
st.pyplot(total_hours_boxenplot)

st.write("""
# Calculating mean value for TotalHours where LeagueIndex is 7
""")
mean_val_l8 = df[df["LeagueIndex"] == 7]["TotalHours"].mean()
st.write(mean_val_l8)

st.write("""
# Imputation for 'TotalHours' missing values
""")
med_val_l5 = df[df["LeagueIndex"] == 5]["TotalHours"].median()
df.loc[
    (df["TotalHours"].isnull()) & (df["LeagueIndex"] == 5), "TotalHours"
] = med_val_l5

# Now imputing the missing value for players with missing TotalHours for LeagueIndex = 8
mean_val_l8 = df[df["LeagueIndex"] == 7]["TotalHours"].mean()
mean_val_l8 = int(mean_val_l8)
df.loc[
    (df["TotalHours"].isnull()) & (df["LeagueIndex"] == 8), "TotalHours"
] = mean_val_l8


st.write("""
# Checking the last 5 rows of the DataFrame
""")
st.write(df.tail())










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


st.write("""
# Checking for Null Values in the DataFrame
""")
st.write(df.isnull().sum())



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


# Data visualization in Streamlit
st.subheader("Data Visualization")
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








st.title("### Let's check for multicollinearity in our data. For that, we will display the correlation heatmap.")

st.write("""
# Data Visualization 4: Correlation Heatmap
""")

plt.figure(figsize=(20, 15))
corr = df.corr()
sns.heatmap(corr, annot=True)
st.pyplot()


st.write("""
# Data Visualization 3: Box Plots for Columns
""")

# Drawing a Box Plot for Columns
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 15))
for i, feature in enumerate(features):
    row, col = divmod(i, 4)
    ax = axes[row, col]
    df[feature].plot(kind='box', ax=ax)
    ax.set_title(f'Box Plot for {feature}')
    ax.set_xlabel('')

plt.tight_layout()
st.pyplot(fig)



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
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

# Calculate predicted values for each model and draw graphs
for i, (name, model) in enumerate(models):
    predictions = model.predict(X_test).flatten() if name == 'Neural Network' else model.predict(X_test)

    # Subplot settings
    plt.subplot(3, 3, i + 1)
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'{name} - Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

plt.tight_layout()
st.pyplot(fig)

import streamlit as st

# Display performance summary
st.title("Performance Summary")
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
st.header("Performance Improvement through Grid and Hyperparameter Tuning")

# Hyperparameter grid definition
linear_params = {'fit_intercept': [True, False]}
decision_tree_params = {'max_depth': [3, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 4]}
random_forest_params = {'n_estimators': [100, 200],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [10, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 4]}
svr_params = {'C': [1, 10],
              'gamma': ['scale', 'auto'],
              'kernel': ['rbf', 'poly']}
gradient_boosting_params = {'n_estimators': [100, 200],
                            'learning_rate': [0.01, 0.1],
                            'max_depth': [3, 7]}
xgboost_params = {'n_estimators': [100, 200],
                  'learning_rate': [0.01, 0.1],
                  'max_depth': [3, 7]}
lightgbm_params = {'n_estimators': [100, 200],
                   'learning_rate': [0.01, 0.1],
                   'max_depth': [3, 7]}
knn_params = {'n_neighbors': [3, 7],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}


# Model and grid connections
models = [
    ('Linear Regression', LinearRegression(), linear_params),
    ('Decision Tree', DecisionTreeRegressor(random_state=42), decision_tree_params),
    ('Random Forest', RandomForestRegressor(random_state=42), random_forest_params),
    ('Support Vector Machine', SVR(), svr_params),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42), gradient_boosting_params),
    ('XGBoost', XGBRegressor(random_state=42), xgboost_params),
    ('LightGBM', LGBMRegressor(random_state=42), lightgbm_params),
    ('KNN', KNeighborsRegressor(), knn_params)
]

# Save tuning results
tuned_models = {}

# Model tuning with grid search
for model_name, model, params in models:
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    tuned_models[model_name] = best_model
    
    st.write(f"{model_name} best params: {best_params}")

# Performance evaluation for each tuned model
results = []

for name, model in tuned_models.items():
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
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

# Performance evaluation of tuned neural network models
nn_train_preds = nn_model.predict(X_train)
nn_test_preds = nn_model.predict(X_test)
nn_train_rmse = mean_squared_error(y_train, nn_train_preds, squared=False)
nn_test_rmse = mean_squared_error(y_test, nn_test_preds, squared=False)
nn_train_r2 = r2_score(y_train, nn_train_preds)
nn_test_r2 = r2_score(y_test, nn_test_preds)

results.append({
    'Model': 'Neural Network',
    'Train RMSE': nn_train_rmse,
    'Test RMSE': nn_test_rmse,
    'Train R-squared': nn_train_r2,
    'Test R-squared': nn_test_r2
})

# Display results in a table
st.subheader("Tuned Model Performance")
results_df = pd.DataFrame(results)
st.table(results_df)

# Calculate predicted values for each tuned model and draw graphs
tuned_models['Neural Network'] = nn_model
plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(tuned_models.items()):
    # model prediction
    if name != 'Neural Network':
        predictions = model.predict(X_test)
    else:
        predictions = nn_model.predict(X_test).flatten()  # Neural network model prediction

    # Subplot settings
    plt.subplot(3, 3, i + 1)
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'{name} - Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

plt.tight_layout()
st.pyplot(plt)



