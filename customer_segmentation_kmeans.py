#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries here
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# In[ ]:


#Import data
df = pd.read_csv("data/SCFP2019.csv.gz")
print("df shape:", df.shape)
df.head()

#percentage of respondents that are business owners
prop_biz_owners = df["HBUS"].value_counts(normalize=True).min()
print("proportion of business owners in df:", prop_biz_owners)
#DataFrame named df_inccat that shows the normalized frequency of income categories for both business owners and non-business owners using the "INCCAT" and "HBUS" columns.
inccat_dict = {
    1: "0-20",
    2: "21-39.9",
    3: "40-59.9",
    4: "60-79.9",
    5: "80-89.9",
    6: "90-100",
}
df_inccat = (
     df["INCCAT"]
    .replace(inccat_dict)
    .groupby(df["HBUS"])
    .value_counts(normalize=True)
    .rename("frequency")
    .to_frame()
    .reset_index()
)

df_inccat


# In[ ]:


#Income Distribution: Business Owners vs. Non-Business Owners
fig, ax = plt.subplots()

sns.barplot(
    data=df_inccat,
    x="INCCAT",
    y="frequency",
    hue="HBUS",
    order=inccat_dict.values()
)
plt.xlabel("Income Category")
plt.ylabel("Frequency (%)")
plt.title("Income Distribution: Business Owners vs. Non-Business Owners");


# In[ ]:


# Plot "HOUSES" vs "DEBT" with hue as business ownership
fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="DEBT",
    y="HOUSES",
    hue="HBUS",
    palette = "deep",
)
plt.xlabel("Household Debt")
plt.ylabel("Home Value")
plt.title("Home Value vs. Household Debt");


# In[ ]:


mask = (df["HBUS"]==1) & (df["INCOME"] < 5e5)
df_small_biz = df[mask] # use the column `mask` defined above
print("df_small_biz shape:", df_small_biz.shape)
df_small_biz.head()

# Plot histogram of "AGE" of small business owners
fig, ax = plt.subplots()
df_small_biz["AGE"].plot(kind="hist", bins=10, title="Small Business Owners: Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency (count)");


# In[ ]:


# Calculate trimmed variance
top_ten_trim_var = (
    df_small_biz.apply(trimmed_var, limits=(0.1, 0.1))
    .sort_values()
    .tail(10)
)

top_ten_trim_var


# In[ ]:


# Calculate variance, get 10 largest features
top_ten_var = df_small_biz.var().sort_values().tail(10)
top_ten_var


# In[ ]:


# Create horizontal bar chart of `top_ten_trim_var`
fig = px.bar(
    x= top_ten_trim_var,
    y= top_ten_trim_var.index,
    title="Small Business Owners: High Variance Features"
)
fig.update_layout(xaxis_title="Trimmed Variance [$]", yaxis_title="Feature")

fig.show()


# In[ ]:


#top 5 features with the highest trimmed variance
high_var_cols = top_ten_trim_var.sort_values(ascending=True).tail(5).index.tolist()
high_var_cols


# In[ ]:


#Split
X = df_small_biz[high_var_cols]
print("X shape:", X.shape)
X.head()


# In[ ]:


#Hyperparameter tuning
n_clusters = range(2, 13)
inertia_errors = []
silhouette_scores = []

# Add `for` loop to train model and calculate inertia, silhouette score.
for k in n_clusters:
    #Make pipeline
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42, n_init=10))
    #fit data to model
    model.fit(X)
    #Calculate the inertia score
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    #Calculate silhouette score
    silhouette_scores.append(silhouette_score(X, model.named_steps["kmeans"].labels_))

print("Inertia:", inertia_errors[:11])
print()
print("Silhouette Scores:", silhouette_scores[:3])


# In[ ]:


# Create line plot of `inertia_errors` vs `n_clusters`
fig = px.line(
    x=n_clusters,
    y=inertia_errors,
    title="K-Means Model: Inertia vs Number of Clusters",
)

fig.update_layout(xaxis_title="Number of Clusters", yaxis_title ="Inertia")#, #yaxis=dict(tickvals=[2000, 4000, 6000]))
fig.show()


# In[ ]:


# Create a line plot of `silhouette_scores` vs `n_clusters`
fig = px.line(
    x=n_clusters,
    y=silhouette_scores,
    title="K-Means Model: Silhouette Score vs Number of Clusters",
)

fig.update_layout(xaxis_title="Number of Clusters", yaxis_title ="Silhouette Score")
fig.show()


# In[ ]:


#Final model
final_model = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42, n_init=10))
final_model.fit(X)


# In[ ]:


#Communication

labels = final_model.named_steps["kmeans"].labels_
xgb = pd.DataFrame(X.groupby(labels).mean())
# Create side-by-side bar chart of `xgb`
fig= px.bar(
    data_frame=xgb,
    barmode="group",
    title="Small Business Owner Finances by Cluster"
)
fig.update_layout(xaxis_title="Cluster", yaxis_title="Value [$]")
fig.show()



# In[ ]:


#Principal Component Analysis

# Instantiate transformer
pca = PCA(n_components=2, random_state=42)

# Transform `X`
X_t = pca.fit_transform(X)

# Put `X_t` into DataFrame
X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])

print("X_pca shape:", X_pca.shape)
X_pca.head()


# In[ ]:


# Create scatter plot of `PC2` vs `PC1`
fig=px.scatter(
    data_frame=X_pca,
    x="PC1",
    y="PC2",
    color=labels.astype(str),
    title ="PCA Representation of Clusters"
)
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
fig.show()


# In[ ]:




