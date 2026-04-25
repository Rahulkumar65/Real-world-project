# 🛒 Retail Sales Data – End-to-End Data Science Project

> **Domain:** Retail | **Skills:** EDA, Data Visualization, Machine Learning  
> **Dataset:** 5,000 retail transactions across 5 categories, 5 regions (2022–2024)

---

## 📋 Project Overview

This project performs a complete data science pipeline on a retail sales dataset:
- **Exploratory Data Analysis** – trends, seasonality, top products
- **Feature Engineering** – label encoding, scaling
- **ML Prediction** – 3 models to predict sales revenue
- **Visualizations** – dashboards, heatmaps, and model evaluation plots

---

## 📁 Repository Structure

```
retail-sales-project/
│
├── retail_sales_data.csv          # Dataset (5,000 records)
├── Retail_Sales_Analysis.ipynb    # Main Jupyter Notebook
├── generate_data.py               # Data generation script
├── analysis.py                    # Full analysis script
│
├── figures/
│   ├── fig1_dashboard.png         # Sales Overview Dashboard
│   ├── fig2_eda.png               # EDA Deep Dive
│   └── fig3_ml.png                # ML Evaluation Charts
│
└── README.md
```

---

## 📊 Dataset Description

| Column | Description |
|---|---|
| `OrderID` | Unique order identifier |
| `Date` | Transaction date |
| `Category` | Product category (Electronics, Clothing, etc.) |
| `Product` | Product name |
| `Region` | Sales region (North/South/East/West/Central) |
| `CustomerSegment` | Regular / Premium / New |
| `PaymentMethod` | Credit Card, UPI, Cash, etc. |
| `UnitPrice` | Price per unit (₹) |
| `Quantity` | Units ordered |
| `Discount` | Discount applied (0–20%) |
| `TotalSales` | Final revenue (₹) |
| `Profit` | Profit earned (₹) |
| `ProfitMargin` | Profit as fraction of sales |

---

## 🔍 Key Findings

### Business Insights
- 📈 **Nov–Dec** consistently peak months (+40% vs average) due to festive season
- 💻 **Electronics** is the top revenue category; **Laptops** lead individual products
- 🛍️ **Weekend orders** show higher average order values
- 👥 **Premium customers** contribute ~35% of total revenue

### Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ₹9,851 | ₹19,148 | 0.846 |
| Random Forest | ₹661 | ₹3,017 | 0.996 |
| **Gradient Boosting** ✅ | **₹900** | **₹2,085** | **0.998** |

**Best Model: Gradient Boosting** (R² = 0.9982)

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-work/retail-sales-project.git
cd retail-sales-project

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Generate the dataset
python generate_data.py

# 4. Run the full analysis
python analysis.py

# 5. Open the Jupyter Notebook
jupyter notebook Retail_Sales_Analysis.ipynb
```

---

## 🛠️ Technologies Used

- **Python 3.10**
- **Pandas** – data manipulation
- **NumPy** – numerical computing
- **Matplotlib / Seaborn** – visualizations
- **Scikit-learn** – ML models (Linear Regression, Random Forest, Gradient Boosting)
- **Jupyter Notebook** – interactive analysis

---

## 📈 Visualizations

### Dashboard
![Dashboard](figures/fig1_dashboard.png)

### EDA
![EDA](figures/fig2_eda.png)

### ML Evaluation
![ML](figures/fig3_ml.png)

---

## 👤 Author

**[Your Name]**  
Data Science Project – 2026  
Due: 21 May 2026
