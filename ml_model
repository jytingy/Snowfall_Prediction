import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

#read in data
csv_file = "weather.csv"
df = pd.read_csv(csv_file)

#clean out non rain/snow days
exclude_values = ['sun', 'drizzle', 'fog']
df = df[~df['weather'].isin(exclude_values)]
df['weather'] = df['weather'].map({'rain': 0, 'snow': 1})
input_variables = list(df.select_dtypes(include = np.number).columns)
input_variables.remove('weather')

#checking distribution
plt.subplots(figsize=(15,8))
for i, col in enumerate(input_variables):
    plt.subplot(2, 2, i+1)
    sb.distplot(df[col])
plt.tight_layout()
# plt.show()

#checking outliers //not removing outliers bc natural occurrence in population
df = df.reset_index(drop=True)
for i, col in enumerate(input_variables):
    plt.subplot(2, 2, i + 1)
    sb.boxplot(x=df[col])
plt.tight_layout()
plt.show()

#check for highly correlated features; remove max/min temp and date columns (date shouldn't matter)
plt.figure(figsize=(10,10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

df.drop(['date','temp_max', 'temp_min'], axis=1, inplace=True)
variables_df = df.drop(['weather'], axis=1)
target = df.drop(['precipitation', 'wind'], axis=1)

#balancing dataset
x_train, x_test, y_train, y_test = train_test_split(variables_df, target, test_size=0.2, stratify=target, random_state=2)
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)  #majority of data is rain class so we need to oversample snow class
x, y = ros.fit_resample(x_train, y_train) #instance of RandomOverSampler 

#normalize variables (mean of 0, sd of 1)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x, y)
training_predictions  = model.predict_proba(x)
print('Accuracy:', metrics.roc_auc_score(y, training_predictions[:,1]))
value_predictions = model.predict_proba(x_test)
print('Validation Accuracy:', metrics.roc_auc_score(y_test, value_predictions[:,1]))
print()
print(metrics.classification_report(y_test, model.predict(x_test)))
