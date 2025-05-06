import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import streamlit as st

# Function to prepare data for a given product
def prepare_data(grocery, product_name):
    product_df = grocery[grocery['product_name'] == product_name]
    product_df.drop(columns=['product_name', 'category', 'price', 'product_id', 'sales_time', 'buyer_gender'], inplace=True)
    product_df['holiday'] = product_df['holiday'].astype(int)
    product_df['day_of_week'] = product_df['sales_date'].dt.dayofweek
    group_columns = ['sales_date', 'day_of_week', 'holiday', 'month', 'day_of_year']
    product_df = product_df.groupby(group_columns).agg({
        'sales': 'sum',
        'total_revenue': 'sum'
    }).reset_index()
    product_df.set_index('sales_date', inplace=True)
    return product_df

# Function to train and forecast sales using XGBoost
def forecast_sales(product_df, train_cutoff='2023-08-31'):
    train = product_df.loc[product_df.index < train_cutoff]
    test = product_df.loc[product_df.index >= train_cutoff]

    X = ['day_of_week', 'holiday', 'month', 'day_of_year']
    y = 'sales'

    X_train = train[X]
    X_test = test[X]
    y_train = train[y]
    y_test = test[y]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)
    
    fi = pd.DataFrame(data=reg.feature_importances_,
                      index=reg.feature_names_in_,
                      columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    
    test['prediction'] = reg.predict(X_test)
    product_df = product_df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    
    score = np.sqrt(mean_squared_error(test['sales'], test['prediction']))
    st.write(f'RMSE Score on Test set: {score:0.2f}')
    
    return product_df, reg

# Function to forecast future sales
def forecast_future_sales(model, future_dates):
    future_df = pd.DataFrame(index=future_dates)
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['holiday'] = 0
    future_df['month'] = future_df.index.month
    future_df['day_of_year'] = future_df.index.dayofyear
    future_df['forecast'] = model.predict(future_df)
    return future_df

# Load your data
grocery = pd.read_csv('Grocery_sales_dataset.csv')
grocery.rename(columns={'number_of_items_sold':'sales'}, inplace=True)
grocery['sales_date'] = pd.to_datetime(grocery['sales_date'])
grocery['month'] = grocery['sales_date'].dt.month
grocery['day_of_year'] = grocery['sales_date'].dt.dayofyear

# Streamlit app starts here
st.title('Grocery Sales Forecasting App')

# Select box to choose the product
product_name = st.selectbox('Select a product', grocery['product_name'].unique())

# Prepare the data for the selected product
product_df = prepare_data(grocery, product_name)

#Forecast the sales
product_df, model = forecast_sales(product_df)

# Forecast future sales
future_dates = pd.date_range(start='2023-12-01', periods=st.slider('Select number of days to forecast', 10, 60), freq='D')
future_sales = forecast_future_sales(model, future_dates)


if(st.button("Predict", type="primary")):# Display future sales forecast
    st.write(f'Future Sales Forecast For {product_name}' )
    st.line_chart(future_sales['forecast'])
