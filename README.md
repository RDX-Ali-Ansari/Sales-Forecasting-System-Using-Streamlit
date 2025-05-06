# 🛒 Grocery Sales Forecasting App

A Streamlit-based machine learning application that forecasts sales for individual grocery products using the **XGBoost** regression algorithm. It enables users to select a product, visualize feature importance, evaluate model performance, and generate future sales forecasts.

---

## 📊 Project Overview

This project demonstrates how machine learning, particularly **XGBoost**, can be used to forecast retail sales based on historical sales data and calendar features such as day of the week, holidays, and day of the year.

The user can:
- Select a specific grocery product.
- View a model’s performance and feature importance.
- Forecast future sales over a customizable date range.
- Visualize predicted sales trends.

---

## 🔍 Features

- Interactive product selection.
- Automatic feature engineering (e.g., extracting day of week, month, etc.).
- XGBoost regression modeling with early stopping.
- RMSE performance metric.
- Feature importance plot.
- Dynamic forecasting for future dates.
- Sales forecast visualization.

---

## 🧠 Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **XGBoost**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Streamlit**

---

## 🗃️ Dataset

The model uses a CSV file: `Grocery_sales_dataset.csv`.

**Expected columns**:
- `sales_date` (datetime)
- `product_name` (string)
- `category`, `price`, `product_id`, `sales_time`, `buyer_gender` (optional but removed)
- `number_of_items_sold` → renamed to `sales`
- `total_revenue`
- `holiday` (binary/int)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/grocery-sales-forecast.git
cd grocery-sales-forecast
