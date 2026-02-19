import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('sample_sales_data.csv')

print("="*80)
print("EXPLORATORY DATA ANALYSIS (EDA) - SALES DATASET")
print("="*80)

# ============================================================================
# 1. MEANINGFUL QUESTIONS ABOUT THE DATASET
# ============================================================================
print("\n" + "="*80)
print("1. KEY QUESTIONS TO EXPLORE")
print("="*80)
questions = [
    "What is the overall sales performance across different product categories?",
    "Which regions generate the most revenue?",
    "What are the customer demographics (age, gender) and purchasing patterns?",
    "Are there any seasonal trends in sales?",
    "What is the preferred payment method among customers?",
    "Are there any data quality issues (missing values, duplicates, outliers)?"
]
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

# ============================================================================
# 2. DATA STRUCTURE EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("2. DATA STRUCTURE EXPLORATION")
print("="*80)

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Statistical Summary ---")
print(df.describe())

# ============================================================================
# 3. DATA QUALITY CHECKS
# ============================================================================
print("\n" + "="*80)
print("3. DATA QUALITY CHECKS")
print("="*80)

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing_Count'] > 0])

print("\n--- Duplicate Records ---")
duplicates = df.duplicated().sum()
print(f"Total Duplicates: {duplicates}")
if duplicates > 0:
    print("\nDuplicate Rows:")
    print(df[df.duplicated(keep=False)])

print("\n--- Data Type Issues ---")
df['Date'] = pd.to_datetime(df['Date'])
print("Date column converted to datetime")

print("\n--- Outlier Detection (Numerical Columns) ---")
numerical_cols = ['Quantity', 'Unit_Price', 'Customer_Age']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers detected")

# ============================================================================
# 4. TRENDS, PATTERNS AND ANOMALIES
# ============================================================================
print("\n" + "="*80)
print("4. TRENDS, PATTERNS AND ANOMALIES")
print("="*80)

# Calculate revenue
df['Revenue'] = df['Quantity'] * df['Unit_Price']

print("\n--- Revenue by Product Category ---")
category_revenue = df.groupby('Product_Category')['Revenue'].sum().sort_values(ascending=False)
print(category_revenue)

print("\n--- Revenue by Region ---")
region_revenue = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print(region_revenue)

print("\n--- Top 5 Products by Revenue ---")
product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head()
print(product_revenue)

print("\n--- Customer Demographics ---")
print(f"Average Customer Age: {df['Customer_Age'].mean():.2f}")
print("\nGender Distribution:")
print(df['Customer_Gender'].value_counts())

print("\n--- Payment Method Preferences ---")
print(df['Payment_Method'].value_counts())

print("\n--- Monthly Sales Trend ---")
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Revenue'].sum()
print(monthly_sales)

# ============================================================================
# 5. HYPOTHESIS TESTING AND VALIDATION
# ============================================================================
print("\n" + "="*80)
print("5. HYPOTHESIS TESTING AND VALIDATION")
print("="*80)

print("\n--- Hypothesis 1: Is there a significant difference in spending between genders? ---")
male_revenue = df[df['Customer_Gender'] == 'M']['Revenue']
female_revenue = df[df['Customer_Gender'] == 'F']['Revenue']
t_stat, p_value = stats.ttest_ind(male_revenue, female_revenue)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
print(f"Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (α=0.05)")

print("\n--- Hypothesis 2: Correlation between Customer Age and Purchase Amount ---")
correlation = df['Customer_Age'].corr(df['Revenue'])
print(f"Correlation coefficient: {correlation:.4f}")
print(f"Result: {'Weak' if abs(correlation) < 0.3 else 'Moderate' if abs(correlation) < 0.7 else 'Strong'} correlation")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("6. GENERATING VISUALIZATIONS")
print("="*80)

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

# Plot 1: Revenue by Category
plt.subplot(3, 3, 1)
category_revenue.plot(kind='bar', color='skyblue')
plt.title('Revenue by Product Category')
plt.xlabel('Category')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)

# Plot 2: Revenue by Region
plt.subplot(3, 3, 2)
region_revenue.plot(kind='bar', color='lightcoral')
plt.title('Revenue by Region')
plt.xlabel('Region')
plt.ylabel('Revenue ($)')

# Plot 3: Customer Age Distribution
plt.subplot(3, 3, 3)
plt.hist(df['Customer_Age'].dropna(), bins=15, color='lightgreen', edgecolor='black')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Plot 4: Gender Distribution
plt.subplot(3, 3, 4)
df['Customer_Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Gender Distribution')
plt.ylabel('')

# Plot 5: Payment Method Distribution
plt.subplot(3, 3, 5)
df['Payment_Method'].value_counts().plot(kind='bar', color='gold')
plt.title('Payment Method Preferences')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 6: Monthly Sales Trend
plt.subplot(3, 3, 6)
monthly_sales.plot(kind='line', marker='o', color='purple')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)

# Plot 7: Quantity vs Unit Price Scatter
plt.subplot(3, 3, 7)
plt.scatter(df['Unit_Price'], df['Quantity'], alpha=0.5, color='teal')
plt.title('Quantity vs Unit Price')
plt.xlabel('Unit Price ($)')
plt.ylabel('Quantity')

# Plot 8: Missing Values Heatmap
plt.subplot(3, 3, 8)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')

# Plot 9: Box Plot for Outliers
plt.subplot(3, 3, 9)
df.boxplot(column='Revenue', by='Product_Category', ax=plt.gca())
plt.title('Revenue Distribution by Category')
plt.suptitle('')
plt.xlabel('Category')
plt.ylabel('Revenue ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'eda_visualizations.png'")

# ============================================================================
# 7. KEY FINDINGS AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("7. KEY FINDINGS AND RECOMMENDATIONS")
print("="*80)

print("\n--- Key Findings ---")
print(f"1. Total Revenue: ${df['Revenue'].sum():,.2f}")
print(f"2. Average Order Value: ${df['Revenue'].mean():,.2f}")
print(f"3. Most Profitable Category: {category_revenue.index[0]} (${category_revenue.iloc[0]:,.2f})")
print(f"4. Best Performing Region: {region_revenue.index[0]} (${region_revenue.iloc[0]:,.2f})")
print(f"5. Data Quality Issues: {missing.sum()} missing values, {duplicates} duplicates")

print("\n--- Recommendations ---")
print("1. Focus marketing efforts on Electronics category (highest revenue)")
print("2. Investigate and address missing values in Product and Customer_Age columns")
print("3. Remove duplicate records to ensure data accuracy")
print("4. Consider targeted campaigns for underperforming regions")
print("5. Analyze customer age groups for personalized marketing strategies")

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*80)
