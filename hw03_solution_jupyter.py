#!/usr/bin/env python3
"""
Homework 03 - PySpark Data Analysis (Jupyter Version)
This version is configured to work with the Docker Jupyter environment
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, avg, round, isnan, when
import os

# Initialize Spark session - Local mode in Jupyter
spark = SparkSession.builder \
    .appName("HW03_JupyterAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Set log level to reduce output
spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("HOMEWORK 03 - PySpark Data Analysis (Jupyter Version)")
print("=" * 60)

# Define paths to CSV files - Using Jupyter mount path
base_path = "/home/jovyan/project/hw-03"
users_path = f"{base_path}/users.csv"
products_path = f"{base_path}/products.csv"
purchases_path = f"{base_path}/purchases.csv"

# Task 1: Load and read each CSV file as a separate DataFrame
print("\n📚 Task 1: Loading CSV files as DataFrames")
print("-" * 40)

# Load users DataFrame
users_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(users_path)

print(f"✅ Users DataFrame loaded: {users_df.count()} rows")
users_df.printSchema()
users_df.show(5)

# Load products DataFrame
products_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(products_path)

print(f"✅ Products DataFrame loaded: {products_df.count()} rows")
products_df.printSchema()
products_df.show(5)

# Load purchases DataFrame
purchases_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(purchases_path)

print(f"✅ Purchases DataFrame loaded: {purchases_df.count()} rows")
purchases_df.printSchema()
purchases_df.show(5)

# Summary of loaded data
print("\n📊 Data Loading Summary:")
print(f"  • Users: {users_df.count()} records")
print(f"  • Products: {products_df.count()} records")
print(f"  • Purchases: {purchases_df.count()} records")

# Check for data quality
print("\n🔍 Data Quality Check:")
print(f"  • Users with null values: {users_df.filter(col('user_id').isNull() | col('age').isNull()).count()}")
print(f"  • Products with null values: {products_df.filter(col('product_id').isNull() | col('price').isNull()).count()}")
print(f"  • Purchases with null values: {purchases_df.filter(col('purchase_id').isNull() | col('user_id').isNull() | col('product_id').isNull()).count()}")

print("\n✅ Task 1 completed successfully!")
print("=" * 60)

# Task 2: Clean the data by removing any rows with missing values
print("\n🧹 Task 2: Cleaning data by removing rows with missing values")
print("-" * 40)

# Check for nulls before cleaning
print("\nBefore cleaning:")
print(f"  • Users with nulls: {users_df.filter(col('user_id').isNull() | col('age').isNull()).count()}")
print(f"  • Products with nulls: {products_df.filter(col('product_id').isNull() | col('product_name').isNull() | col('category').isNull() | col('price').isNull()).count()}")
print(f"  • Purchases with nulls: {purchases_df.filter(col('purchase_id').isNull() | col('user_id').isNull() | col('product_id').isNull() | col('date').isNull() | col('quantity').isNull()).count()}")

# Clean users DataFrame
users_clean = users_df.dropna()
print(f"\n✅ Users cleaned: {users_df.count()} → {users_clean.count()} rows")

# Clean products DataFrame
products_clean = products_df.dropna()
print(f"✅ Products cleaned: {products_df.count()} → {products_clean.count()} rows")

# Clean purchases DataFrame
purchases_clean = purchases_df.dropna()
print(f"✅ Purchases cleaned: {purchases_df.count()} → {purchases_clean.count()} rows")

print("\n📊 Cleaning Summary:")
print(f"  • Users removed: {users_df.count() - users_clean.count()} rows")
print(f"  • Products removed: {products_df.count() - products_clean.count()} rows")
print(f"  • Purchases removed: {purchases_df.count() - purchases_clean.count()} rows")

# Task 3: Determine the total purchase amount for each product category
print("\n💰 Task 3: Total purchase amount for each product category")
print("-" * 40)

# Join purchases with products to get category and price
purchases_with_products = purchases_clean.join(
    products_clean,
    purchases_clean.product_id == products_clean.product_id,
    "inner"
)

# Calculate total amount (price * quantity) for each purchase
purchases_with_amount = purchases_with_products.withColumn(
    "purchase_amount",
    col("price") * col("quantity")
)

# Group by category and sum the purchase amounts
total_by_category = purchases_with_amount.groupBy("category") \
    .agg(spark_sum("purchase_amount").alias("total_amount")) \
    .orderBy(col("total_amount").desc())

print("\nTotal purchase amount by category:")
total_by_category.show()

# Task 4: Purchase amount for age group 18-25
print("\n👥 Task 4: Purchase amount for age group 18-25")
print("-" * 40)

# Filter users for age group 18-25
users_18_25 = users_clean.filter((col("age") >= 18) & (col("age") <= 25))
print(f"\nUsers in age group 18-25: {users_18_25.count()} users")

# Join with purchases to get purchases for this age group
purchases_18_25 = purchases_clean.join(
    users_18_25,
    purchases_clean.user_id == users_18_25.user_id,
    "inner"
)

# Join with products to get category and price
purchases_18_25_with_products = purchases_18_25.join(
    products_clean,
    purchases_18_25.product_id == products_clean.product_id,
    "inner"
)

# Calculate purchase amount
purchases_18_25_with_amount = purchases_18_25_with_products.withColumn(
    "purchase_amount",
    col("price") * col("quantity")
)

# Group by category
amount_by_category_18_25 = purchases_18_25_with_amount.groupBy("category") \
    .agg(spark_sum("purchase_amount").alias("total_amount_18_25")) \
    .orderBy(col("total_amount_18_25").desc())

print("\nPurchase amount by category for age 18-25:")
amount_by_category_18_25.show()

# Task 5: Share of purchases for age group 18-25
print("\n📊 Task 5: Share of purchases for age group 18-25")
print("-" * 40)

# Calculate total spending for age group 18-25
total_spending_18_25 = purchases_18_25_with_amount.agg(
    spark_sum("purchase_amount").alias("total")
).collect()[0]["total"]

print(f"\nTotal spending for age 18-25: ${total_spending_18_25:.2f}")

# Calculate percentage share for each category
share_by_category_18_25 = amount_by_category_18_25.withColumn(
    "percentage_share",
    round((col("total_amount_18_25") / total_spending_18_25 * 100), 2)
).orderBy(col("percentage_share").desc())

print("\nShare of purchases by category for age 18-25:")
share_by_category_18_25.show()

# Task 6: Select top 3 categories for age 18-25
print("\n🏆 Task 6: Top 3 categories for age 18-25")
print("-" * 40)

top_3_categories = share_by_category_18_25.limit(3)

print("\nTop 3 product categories with highest percentage for age 18-25:")
top_3_categories.select("category", "total_amount_18_25", "percentage_share").show()

print("\n" + "="*60)
print("✅ All homework tasks completed successfully!")
print("="*60)

# Note: Not stopping spark session to keep it available for further exploration
# spark.stop()