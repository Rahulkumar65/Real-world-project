import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Configuration
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
n_records = 5000

categories = {
    'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch'],
    'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes'],
    'Food & Beverages': ['Coffee', 'Tea', 'Snacks', 'Juice', 'Energy Drink'],
    'Home & Garden': ['Chair', 'Table Lamp', 'Curtains', 'Pillow', 'Vase'],
    'Sports': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Water Bottle', 'Resistance Band']
}

base_prices = {
    'Laptop': 65000, 'Smartphone': 30000, 'Headphones': 3500, 'Tablet': 22000, 'Smartwatch': 8000,
    'T-Shirt': 500, 'Jeans': 1500, 'Jacket': 2500, 'Dress': 2000, 'Shoes': 2200,
    'Coffee': 250, 'Tea': 150, 'Snacks': 80, 'Juice': 60, 'Energy Drink': 100,
    'Chair': 5000, 'Table Lamp': 1200, 'Curtains': 800, 'Pillow': 400, 'Vase': 350,
    'Yoga Mat': 900, 'Dumbbells': 2500, 'Running Shoes': 3000, 'Water Bottle': 350, 'Resistance Band': 450
}

regions = ['North', 'South', 'East', 'West', 'Central']
payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Cash', 'Net Banking']
customer_segments = ['Regular', 'Premium', 'New']

dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(n_records)]
dates.sort()

records = []
for i, date in enumerate(dates):
    cat = random.choice(list(categories.keys()))
    product = random.choice(categories[cat])
    base_price = base_prices[product]

    # Seasonal effect
    month = date.month
    seasonal_factor = 1.0
    if month in [11, 12]:  # Festive season
        seasonal_factor = 1.4
    elif month in [6, 7]:  # Summer
        seasonal_factor = 1.1
    elif month in [1, 2]:  # Post-festive dip
        seasonal_factor = 0.85

    price = round(base_price * seasonal_factor * np.random.uniform(0.9, 1.1), 2)
    quantity = np.random.randint(1, 6)
    discount = round(random.choice([0, 0, 0, 5, 10, 15, 20]) / 100, 2)
    total_sales = round(price * quantity * (1 - discount), 2)
    profit_margin = round(np.random.uniform(0.12, 0.35), 3)
    profit = round(total_sales * profit_margin, 2)

    records.append({
        'OrderID': f'ORD{10000 + i}',
        'Date': date.strftime('%Y-%m-%d'),
        'Year': date.year,
        'Month': date.month,
        'Quarter': f'Q{(date.month - 1) // 3 + 1}',
        'DayOfWeek': date.strftime('%A'),
        'Category': cat,
        'Product': product,
        'Region': random.choice(regions),
        'CustomerSegment': random.choice(customer_segments),
        'PaymentMethod': random.choice(payment_methods),
        'UnitPrice': price,
        'Quantity': quantity,
        'Discount': discount,
        'TotalSales': total_sales,
        'Profit': profit,
        'ProfitMargin': profit_margin
    })

df = pd.DataFrame(records)
df.to_csv('/home/claude/retail_project/retail_sales_data.csv', index=False)
print(f"Dataset created: {len(df)} records")
print(df.head())
print(df.describe())
