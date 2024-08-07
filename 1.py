# Import libraries for data processing & transformation
import pandas as pd
import numpy as np

# Import libraries for data visualization process
import seaborn as sns
import matplotlib.pyplot as plt

# Import libraries for modeling process
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the datasets
customer = pd.read_csv('olist_customers_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
order_payments = pd.read_csv('olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
product_translation = pd.read_csv('product_category_name_translation.csv')

# Merging all customer related datasets
A = pd.merge(orders, order_reviews, on='order_id')
A = pd.merge(A, order_payments, on='order_id')
A = pd.merge(A, customer, on='customer_id')

# Merging all seller related datasets
B = pd.merge(order_items, products, on='product_id')
B = pd.merge(B, sellers, on='seller_id')
B = pd.merge(B, product_translation, on='product_category_name')

# Merging customer and seller datasets
df_ecommerce = pd.merge(A, B, on='order_id')

# Choosing only the important columns
df_ecommerce = df_ecommerce[['order_status', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date',
                             'shipping_limit_date', 'payment_sequential', 'payment_type', 'payment_installments', 'payment_value',
                             'price', 'freight_value', 'product_category_name_english', 'product_name_lenght', 'product_description_lenght',
                             'product_photos_qty', 'review_score']]

# Fixing typos and column names
df_ecommerce = df_ecommerce.rename(columns={'product_name_lenght': 'product_name_length', 'product_description_lenght': 'product_description_length',
                                            'product_category_name_english': 'product_category'})

# Detecting NaN values
df_ecommerce.isnull().sum()

# Removing data with NaN values
prev_size = df_ecommerce.shape[0]
df_ecommerce.dropna(how='any', inplace=True)
current_size = df_ecommerce.shape[0]

# Converting the timestamp format data to date data
df_ecommerce['order_purchase_timestamp'] = pd.to_datetime(df_ecommerce['order_purchase_timestamp']).dt.date
df_ecommerce['order_estimated_delivery_date'] = pd.to_datetime(df_ecommerce['order_estimated_delivery_date']).dt.date
df_ecommerce['order_delivered_customer_date'] = pd.to_datetime(df_ecommerce['order_delivered_customer_date']).dt.date
df_ecommerce['shipping_limit_date'] = pd.to_datetime(df_ecommerce['shipping_limit_date']).dt.date

# Calculating delivery, estimated, and shipping days
df_ecommerce['delivery_days'] = (df_ecommerce['order_delivered_customer_date'] - df_ecommerce['order_purchase_timestamp']).dt.days
df_ecommerce['estimated_days'] = (df_ecommerce['order_estimated_delivery_date'] - df_ecommerce['order_purchase_timestamp']).dt.days
df_ecommerce['shipping_days'] = (df_ecommerce['shipping_limit_date'] - df_ecommerce['order_purchase_timestamp']).dt.days

df_ecommerce.drop(['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date',
                   'shipping_limit_date'], axis=1, inplace=True)

# Visualizations
# Top 10 best purchased product by customers
fig = plt.figure(figsize=(20, 8))
ax = plt.axes()
sns.barplot(x=df_ecommerce.product_category.value_counts().index[:10],
            y=df_ecommerce.product_category.value_counts()[:10], ax=ax)
sns.set(font_scale=1)
ax.set_xlabel('Product category', fontsize=16)
ax.set_ylabel('The quantity of order', fontsize=16)
fig.suptitle("Top 10 best purchased product by customers", fontsize=25)
plt.show()

# Payment value by customer based on the payment type
fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="payment_type", y="payment_value", data=df_ecommerce, ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Payment type', fontsize=20)
ax.set_ylabel('Payment value', fontsize=20)
fig.suptitle("Payment value by customer based on the payment type", fontsize=25)
plt.show()

# Customer review based on payment value
fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="review_score", y="payment_value", data=df_ecommerce, ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Review score', fontsize=20)
ax.set_ylabel('Payment value', fontsize=20)
fig.suptitle("Customer review based on payment value", fontsize=25)
plt.show()

# Customer review based on freight value
fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="review_score", y="freight_value", data=df_ecommerce, ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Review score', fontsize=20)
ax.set_ylabel('Freight value', fontsize=20)
fig.suptitle("Customer review based on freight value", fontsize=25)
plt.show()

# Customer review based on price
fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.barplot(x="review_score", y="price", data=df_ecommerce, ax=ax)
sns.set(font_scale=1.75)
ax.set_xlabel('Review score', fontsize=20)
ax.set_ylabel('Price', fontsize=20)
fig.suptitle("Customer review based on price", fontsize=25)
plt.show()

# Correlation between payment value and price
fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.scatterplot(x="payment_value", y="price", hue="review_score", sizes=(40, 400),
                palette=["green", "orange", "blue", "red", "brown"], data=df_ecommerce, ax=ax)
ax.set_xlabel('Payment value', fontsize=20)
ax.set_ylabel('Price', fontsize=20)
fig.suptitle('Correlation between payment value and price', fontsize=25)
plt.show()

# Correlation between delivery days and estimated days
fig = plt.figure(figsize=(15, 8))
ax = plt.axes()
sns.scatterplot(x="delivery_days", y="estimated_days", hue="review_score", sizes=(40, 400),
                palette=["green", "orange", "blue", "red", "brown"], data=df_ecommerce, ax=ax)
ax.set_xlabel('Delivery days', fontsize=20)
ax.set_ylabel('Estimated days', fontsize=20)
fig.suptitle('Correlation between delivery days and estimated days', fontsize=25)
plt.show()

# Creating new column of arrival time
df_ecommerce['arrival_time'] = df_ecommerce['estimated_days'] - df_ecommerce['delivery_days']

# Creating new feature based on the arrival time of each order
delivery_arrival = ['Late' if x <= 0 else 'On time' for x in df_ecommerce['arrival_time']]
df_ecommerce['delivery_arrival'] = delivery_arrival

# Creating new feature based on the review score of each order of good and bad review
df_ecommerce.loc[df_ecommerce['review_score'] < 3, 'Score'] = 0
df_ecommerce.loc[df_ecommerce['review_score'] > 3, 'Score'] = 1

# Removing review with value of 3 because it is a neutral review
df_ecommerce.drop(df_ecommerce[df_ecommerce['review_score'] == 3].index, inplace=True)
df_ecommerce.drop('review_score', axis=1, inplace=True)

# Label and one hot encoding process
# Handling column with 2 distinct values
df_ecommerce['order_status'] = df_ecommerce['order_status'].replace(['canceled', 'delivered'], [0, 1])
df_ecommerce['delivery_arrival'] = df_ecommerce['delivery_arrival'].replace(['Late', 'On time'], [0, 1])

# Handling column with more than 2 distinct values
one_hot_payment_type = pd.get_dummies(df_ecommerce['payment_type'])
df_ecommerce = df_ecommerce.join(one_hot_payment_type)

# Handling column with more than 10 distinct values
top_10_product_category = df_ecommerce['product_category'].value_counts().sort_values(ascending=False).head(10).index
for label in top_10_product_category:
    df_ecommerce[label] = np.where(df_ecommerce['product_category'] == label, 1, 0)

# Dropping any unimportant columns for the predicting process
df_ecommerce.drop(['payment_type', 'product_category'], axis=1, inplace=True)

# Split the data
X = df_ecommerce.drop('Score', axis=1)
y = df_ecommerce['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Build the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Model evaluation
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot()
plt.show()

print(classification_report(y_test, y_pred))
print('Accuracy Score : %.2f' % accuracy_score(y_test, y_pred))
