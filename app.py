# import streamlit as st
# import os
# import numpy as np
# from keras.preprocessing.image import load_img
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.models import Model
# import pickle
# from keras.applications.mobilenet_v2 import MobileNetV2
# from keras.applications.mobilenet_v2 import preprocess_input

# # Load image features and clustering information
# with open('features.pkl', 'rb') as file:
#     data = pickle.load(file)

# filenames = np.array(list(data.keys()))
# feat = np.array(list(data.values()))
# feat = feat.reshape(-1, 4096)

# pca = PCA(n_components=100, random_state=22)
# pca.fit(feat)
# x = pca.transform(feat)

# kmeans = KMeans(n_clusters=10, n_jobs=-1, random_state=22)
# kmeans.fit(x)

# groups = {}
# for file, cluster in zip(filenames, kmeans.labels_):
#     if cluster not in groups.keys():
#         groups[cluster] = []
#     groups[cluster].append(file)

# # model = VGG16()
# # model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)


# def extract_features(file, model):
#     img = load_img(file, target_size=(224, 224))
#     img = np.array(img)
#     reshaped_img = img.reshape(1, 224, 224, 3)
#     imgx = preprocess_input(reshaped_img)
#     features = model.predict(imgx, use_multiprocessing=True)
#     return features


# st.title("Image Clustering with Streamlit")

# # Sidebar with cluster selection
# selected_cluster = st.sidebar.selectbox("Select Cluster", list(groups.keys()))

# # Display images in the selected cluster
# st.subheader(f"Images in Cluster {selected_cluster}")
# cluster_images = groups[selected_cluster]
# for image_file in cluster_images:
#     img = load_img(image_file)
#     st.image(img, caption=os.path.basename(image_file), use_column_width=True)

# # Add other components or features as needed
# # You can customize the layout and appearance based on your requirements


import streamlit as st
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

import pickle
# Load saved features
with open('features.pkl', 'rb') as file:
    model = pickle.load(file)

# import pickle

# with open('features.pkl', 'wb') as file:
#     pickle.dump(model, file)

# Function to extract features
def extract_features(file, model):
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features

# Load flower images
file_path = r"/home/topister/Desktop/ML/Data mining/Clustering/newImages/flower_images/flower_images/"

flowers = [file for file in os.listdir(file_path) if file.endswith('.png')]

# Extract features and store in dictionary
# data = {}
# for flower in flowers:
#     try:
#         feat = extract_features(os.path.join(file_path, flower), model)
#         data[flower] = feat
#     except Exception as e:
#         st.error(f"Error extracting features for {flower}: {str(e)}")


data = {}
for flower in flowers:
    try:
        feat = extract_features(os.path.join(file_path, flower), model)
        # Flatten the feature vector before storing in the dictionary
        data[flower] = feat.flatten()
    except Exception as e:
        st.error(f"Error extracting features for {flower}: {str(e)}")


# # Convert data to DataFrame
# df = pd.DataFrame.from_dict(data, orient='index')
# filenames = df.index
# feat = df.values

# # Perform PCA
# pca = PCA(n_components=2, random_state=22)  # Reduced to 2 components for visualization
# pca.fit(feat)
# x = pca.transform(feat)

# # Perform KMeans clustering
# kmeans = KMeans(n_clusters=5, random_state=22)
# kmeans.fit(x)
# df['cluster'] = kmeans.labels_
        
# Convert data to DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Perform PCA
pca = PCA(n_components=2, random_state=22)  # Reduced to 2 components for visualization
pca.fit(df)

# Transform the original data using the fitted PCA model
x = pca.transform(df)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=22)
kmeans.fit(x)
df['cluster'] = kmeans.labels_

# # Streamlit app
# st.title("Image Clustering with Streamlit")

# # Sidebar for selecting cluster
# selected_cluster = st.sidebar.selectbox("Select Cluster", sorted(df['cluster'].unique()))

# # Display images and clustering visualization for the selected cluster
# cluster_images = df[df['cluster'] == selected_cluster].index
# st.subheader(f"Images in Cluster {selected_cluster}")

# # Display images in a row
# # row = st.beta_columns(len(cluster_images))
# row = st.columns(len(cluster_images))

# for col, image_file in zip(row, cluster_images):
#     img = load_img(os.path.join(file_path, image_file))
#     col.image(img, caption=image_file, use_column_width=True)

# # Display clustering visualization
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(x[:, 0], x[:, 1], c=df['cluster'], cmap='viridis', alpha=0.5)
# plt.title("PCA Visualization of Clusters")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")

# # Create a legend
# legend_labels = [f"Cluster {i}" for i in range(len(df['cluster'].unique()))]
# plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clusters", loc="upper right")

# st.pyplot(plt)


# Streamlit app
st.title("Image Clustering with Streamlit")

# Sidebar for selecting cluster
selected_cluster = st.sidebar.selectbox("Select Cluster", sorted(df['cluster'].unique()))

# Display images for the selected cluster
st.subheader(f"Images in Cluster {selected_cluster}")

# Filter images for the selected cluster
cluster_images = df[df['cluster'] == selected_cluster].index

# Display images with 3 images per row
num_images_per_row = 3
num_rows = (len(cluster_images) + num_images_per_row - 1) // num_images_per_row

for i in range(num_rows):
    row_images = cluster_images[i * num_images_per_row: (i + 1) * num_images_per_row]
    row = st.columns(len(row_images))
    
    for col, image_file in zip(row, row_images):
        img = load_img(os.path.join(file_path, image_file))
        col.image(img, use_column_width=True)

# Display clustering visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x[:, 0], x[:, 1], c=df['cluster'], cmap='viridis', alpha=0.5)
plt.title("PCA Visualization of Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Create a legend
legend_labels = [f"Cluster {i}" for i in range(len(df['cluster'].unique()))]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clusters", loc="upper right")

st.pyplot(plt)
