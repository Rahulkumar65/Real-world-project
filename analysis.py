import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Styling ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f9f9f9',
})
COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
PALETTE = sns.color_palette(COLORS)

df = pd.read_csv('/home/claude/retail_project/retail_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("=" * 60)
print("RETAIL SALES DATA - SUMMARY")
print("=" * 60)
print(f"Total Records     : {len(df):,}")
print(f"Total Revenue     : ₹{df['TotalSales'].sum():,.0f}")
print(f"Total Profit      : ₹{df['Profit'].sum():,.0f}")
print(f"Avg Order Value   : ₹{df['TotalSales'].mean():,.0f}")
print(f"Overall Profit Margin: {df['ProfitMargin'].mean()*100:.1f}%")
print(f"Date Range        : {df['Date'].min().date()} → {df['Date'].max().date()}")


# ── FIGURE 1: Overview Dashboard ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Retail Sales Analysis Dashboard (2022–2024)', fontsize=20, fontweight='bold', y=1.01)

# 1a. Monthly Revenue Trend
monthly = df.groupby(['Year', 'Month'])['TotalSales'].sum().reset_index()
monthly['Period'] = pd.to_datetime(monthly[['Year', 'Month']].assign(day=1))
ax = axes[0, 0]
ax.plot(monthly['Period'], monthly['TotalSales'] / 1e6, color='#2196F3', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(monthly['Period'], monthly['TotalSales'] / 1e6, alpha=0.15, color='#2196F3')
ax.set_title('Monthly Revenue Trend', fontsize=13, fontweight='bold')
ax.set_ylabel('Revenue (₹ Millions)')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=30)

# 1b. Sales by Category
cat_sales = df.groupby('Category')['TotalSales'].sum().sort_values()
ax = axes[0, 1]
bars = ax.barh(cat_sales.index, cat_sales.values / 1e6, color=COLORS)
ax.set_title('Revenue by Category', fontsize=13, fontweight='bold')
ax.set_xlabel('Revenue (₹ Millions)')
for bar, val in zip(bars, cat_sales.values / 1e6):
    ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f'₹{val:.1f}M', va='center', fontsize=9)

# 1c. Region Performance
region_sales = df.groupby('Region')['TotalSales'].sum().sort_values(ascending=False)
ax = axes[0, 2]
bars = ax.bar(region_sales.index, region_sales.values / 1e6, color=COLORS, edgecolor='white', linewidth=1.5)
ax.set_title('Revenue by Region', fontsize=13, fontweight='bold')
ax.set_ylabel('Revenue (₹ Millions)')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f'₹{bar.get_height():.1f}M', ha='center', fontsize=9, fontweight='bold')

# 1d. Quarterly Profit
quarterly = df.groupby(['Year', 'Quarter'])['Profit'].sum().reset_index()
quarterly['Label'] = quarterly['Year'].astype(str) + ' ' + quarterly['Quarter']
ax = axes[1, 0]
colors_q = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'] * 3
bars = ax.bar(quarterly['Label'], quarterly['Profit'] / 1e6,
              color=colors_q[:len(quarterly)], edgecolor='white')
ax.set_title('Quarterly Profit', fontsize=13, fontweight='bold')
ax.set_ylabel('Profit (₹ Millions)')
ax.tick_params(axis='x', rotation=45)

# 1e. Customer Segment Distribution
seg = df.groupby('CustomerSegment')['TotalSales'].sum()
ax = axes[1, 1]
wedges, texts, autotexts = ax.pie(seg.values, labels=seg.index, autopct='%1.1f%%',
                                   colors=COLORS[:3], startangle=90,
                                   wedgeprops={'edgecolor': 'white', 'linewidth': 2})
[at.set_fontsize(10) for at in autotexts]
ax.set_title('Revenue by Customer Segment', fontsize=13, fontweight='bold')

# 1f. Payment Method
pay = df.groupby('PaymentMethod')['TotalSales'].sum().sort_values(ascending=False)
ax = axes[1, 2]
bars = ax.bar(pay.index, pay.values / 1e6, color=COLORS, edgecolor='white')
ax.set_title('Revenue by Payment Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Revenue (₹ Millions)')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('/home/claude/retail_project/fig1_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 1 saved")


# ── FIGURE 2: EDA Deep Dive ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Exploratory Data Analysis – Sales Patterns', fontsize=18, fontweight='bold')

# 2a. Seasonal Heatmap
pivot = df.groupby(['Year', 'Month'])['TotalSales'].sum().reset_index()
pivot_table = pivot.pivot(index='Year', columns='Month', values='TotalSales') / 1e6
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
pivot_table.columns = month_names
ax = axes[0, 0]
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Revenue (₹M)'})
ax.set_title('Monthly Revenue Heatmap (₹ Millions)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Year')

# 2b. Top 10 Products
top_products = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False).head(10)
ax = axes[0, 1]
colors_p = plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1]
bars = ax.barh(top_products.index, top_products.values / 1e6, color=colors_p)
ax.set_title('Top 10 Products by Revenue', fontsize=13, fontweight='bold')
ax.set_xlabel('Revenue (₹ Millions)')
for bar, val in zip(bars, top_products.values / 1e6):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f'₹{val:.1f}M', va='center', fontsize=9)

# 2c. Day-of-week Sales Pattern
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_sales = df.groupby('DayOfWeek')['TotalSales'].mean().reindex(dow_order)
ax = axes[1, 0]
bar_colors = ['#FF5722' if d in ['Saturday', 'Sunday'] else '#2196F3' for d in dow_order]
ax.bar(dow_sales.index, dow_sales.values / 1000, color=bar_colors, edgecolor='white')
ax.set_title('Avg Daily Sales by Day of Week', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Sales (₹ Thousands)')
ax.tick_params(axis='x', rotation=30)
weekend_patch = mpatches.Patch(color='#FF5722', label='Weekend')
weekday_patch = mpatches.Patch(color='#2196F3', label='Weekday')
ax.legend(handles=[weekday_patch, weekend_patch])

# 2d. Profit Margin by Category
cat_margin = df.groupby('Category')['ProfitMargin'].mean() * 100
ax = axes[1, 1]
bars = ax.bar(cat_margin.index, cat_margin.values, color=COLORS, edgecolor='white')
ax.set_title('Average Profit Margin by Category', fontsize=13, fontweight='bold')
ax.set_ylabel('Profit Margin (%)')
ax.tick_params(axis='x', rotation=15)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/retail_project/fig2_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved")


# ── FIGURE 3: ML Prediction ──────────────────────────────────────────────────
# Feature Engineering
df_ml = df.copy()
le = LabelEncoder()
for col in ['Category', 'Region', 'CustomerSegment', 'PaymentMethod', 'DayOfWeek', 'Quarter']:
    df_ml[col + '_enc'] = le.fit_transform(df_ml[col])

features = ['Month', 'Year', 'Category_enc', 'Region_enc', 'CustomerSegment_enc',
            'PaymentMethod_enc', 'UnitPrice', 'Quantity', 'Discount',
            'DayOfWeek_enc', 'Quarter_enc', 'ProfitMargin']
target = 'TotalSales'

X = df_ml[features]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    X_tr = X_train_s if name == 'Linear Regression' else X_train
    X_te = X_test_s if name == 'Linear Regression' else X_test
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    results[name] = {
        'model': model,
        'preds': preds,
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    }
    print(f"{name:25s} | MAE: ₹{results[name]['MAE']:>10,.0f} | RMSE: ₹{results[name]['RMSE']:>10,.0f} | R²: {results[name]['R2']:.4f}")

best_name = max(results, key=lambda k: results[k]['R2'])
best = results[best_name]
print(f"\n✓ Best model: {best_name} (R² = {best['R2']:.4f})")

# Feature Importance (RF)
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Machine Learning – Sales Prediction Analysis', fontsize=18, fontweight='bold')

# 3a. Model Comparison
model_names = list(results.keys())
r2_vals = [results[m]['R2'] for m in model_names]
rmse_vals = [results[m]['RMSE'] / 1e3 for m in model_names]
ax = axes[0, 0]
x = np.arange(len(model_names))
bars1 = ax.bar(x - 0.2, r2_vals, 0.35, label='R² Score', color='#2196F3', edgecolor='white')
ax2 = ax.twinx()
bars2 = ax2.bar(x + 0.2, rmse_vals, 0.35, label='RMSE (₹K)', color='#FF5722', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=10, fontsize=10)
ax.set_ylabel('R² Score', color='#2196F3')
ax2.set_ylabel('RMSE (₹ Thousands)', color='#FF5722')
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
for bar, val in zip(bars1, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f'{val:.3f}', ha='center', fontsize=9)
lines = [mpatches.Patch(color='#2196F3', label='R²'), mpatches.Patch(color='#FF5722', label='RMSE')]
ax.legend(handles=lines, loc='upper left')
ax.set_facecolor('#f9f9f9')

# 3b. Actual vs Predicted (Best Model)
ax = axes[0, 1]
sample_idx = np.random.choice(len(y_test), 200, replace=False)
y_test_arr = y_test.values
ax.scatter(y_test_arr[sample_idx], best['preds'][sample_idx],
           alpha=0.5, color='#2196F3', edgecolors='white', linewidth=0.5, s=40)
lims = [0, min(y_test_arr.max(), best['preds'].max()) * 1.05]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Sales (₹)')
ax.set_ylabel('Predicted Sales (₹)')
ax.set_title(f'Actual vs Predicted – {best_name}\n(R² = {best["R2"]:.4f})', fontsize=12, fontweight='bold')
ax.legend()

# 3c. Feature Importance
ax = axes[1, 0]
colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi)))[::-1]
ax.barh(fi.index, fi.values * 100, color=colors_fi)
ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance (%)')
for i, (val, name) in enumerate(zip(fi.values * 100, fi.index)):
    ax.text(val + 0.1, i, f'{val:.1f}%', va='center', fontsize=8)

# 3d. Residuals
ax = axes[1, 1]
residuals = y_test_arr - best['preds']
ax.hist(residuals, bins=50, color='#4CAF50', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_title(f'Residual Distribution – {best_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Prediction Error (₹)')
ax.set_ylabel('Frequency')
ax.text(0.97, 0.97, f'Mean Error:\n₹{residuals.mean():,.0f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/claude/retail_project/fig3_ml.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved")

# ── Save model metrics ───────────────────────────────────────────────────────
metrics_data = {name: {k: v for k, v in res.items() if k != 'model' and k != 'preds'}
                for name, res in results.items()}
import json
with open('/home/claude/retail_project/model_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2, default=float)

print("\n✓ All figures and metrics saved!")
print(f"\nKey Stats:")
print(f"  Total Revenue: ₹{df['TotalSales'].sum()/1e6:.2f}M")
print(f"  Total Profit:  ₹{df['Profit'].sum()/1e6:.2f}M")
print(f"  Avg Margin:    {df['ProfitMargin'].mean()*100:.1f}%")
print(f"  Best ML Model: {best_name} (R²={best['R2']:.4f})")
