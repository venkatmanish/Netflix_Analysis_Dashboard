import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Load datasets
@st.cache_data
def load_data():
    data = pd.read_csv('Datasets/netflix_data_normalized.csv')
    cast_data = pd.read_csv('Datasets/cast_data.csv')
    type_data = pd.read_csv('Datasets/type_data.csv')
    return data, cast_data, type_data

data, cast_data, type_data = load_data()

# Encode 'Country' and 'Rating' columns as numeric
data['Country'] = data['Country'].astype('category').cat.codes
data['Rating'] = data['Rating'].astype('category').cat.codes

# Sidebar KPIs
st.sidebar.title("Key Metrics")
total_movies = data[data['Category'] == 0].shape[0]
total_tv_shows = data[data['Category'] == 1].shape[0]
avg_duration = data['Duration'].mean()

st.sidebar.metric("Total Movies", total_movies)
st.sidebar.metric("Total TV Shows", total_tv_shows)
st.sidebar.metric("Average Duration (minutes)", f"{avg_duration:.2f}")

# Sidebar Navigation
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio(
    "Select a Section",
    [
        "Overview",
        "Country-Wise Content",
        "Duration Analysis",
        "Clustering Dashboard",
        "Cast Analysis",
        "Genre Distribution",
        "Classification and Regression",
        "Correlation Analysis",
        "Filtered Data Export",
    ],
)

# Section: Overview
if page == "Overview":
    st.title("Netflix Dataset Overview")
    st.write("Explore the dataset and its key statistics.")
    if st.checkbox("Show Dataset Preview"):
        st.dataframe(data)
    st.subheader("Dataset Statistics")
    st.write(data.describe())

# Section: Country-Wise Content
elif page == "Country-Wise Content":
    st.title("Country-Wise Content Analysis")
    top_n = st.slider("Select Top N Countries", min_value=5, max_value=20, value=10)
    
    category_by_country = data.groupby(['Country', 'Category']).size().reset_index(name='Count')
    total_content_by_country = category_by_country.groupby('Country')['Count'].sum().reset_index()
    top_countries = total_content_by_country.sort_values('Count', ascending=False).head(top_n)
    
    fig = px.bar(
        top_countries, x='Country', y='Count',
        title=f"Top {top_n} Countries by Content",
        labels={'Count': 'Number of Movies/TV Shows'}
    )
    st.plotly_chart(fig)

# Section: Duration Analysis
elif page == "Duration Analysis":
    st.title("Duration Analysis")
    selected_rating = st.selectbox("Select Rating", options=data['Rating'].unique())
    filtered_data = data[data['Rating'] == selected_rating]
    
    fig = px.histogram(
        filtered_data, x='Duration', nbins=20,
        title=f"Duration Distribution for Rating: {selected_rating}",
        labels={'Duration': 'Duration (minutes)'}
    )
    st.plotly_chart(fig)

# Section: Clustering Dashboard
elif page == "Clustering Dashboard":
    st.title("Clustering of Netflix Content")
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
    features = ['Duration', 'Rating']
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[features])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(normalized_data)
    data['Cluster'] = clusters
    
    fig = px.scatter(
        data, x='Duration', y='Rating', color='Cluster',
        title="Clustering of Netflix Content",
        labels={'Duration': 'Duration (minutes)', 'Rating': 'Rating'}
    )
    st.plotly_chart(fig)

# Section: Cast Analysis
elif page == "Cast Analysis":
    st.title("Cast Analysis")
    top_n_cast = st.slider("Select Top N Cast Members", min_value=5, max_value=20, value=10)
    
    # Process frequent cast members
    frequent_cast_members = (
        cast_data['Cast_Member']
        .value_counts()
        .head(top_n_cast)
        .reset_index()
        .rename(columns={'index': 'Cast_Member', 'count': 'Appearance_Count'})
    )

    # Debugging: Show the processed DataFrame
    st.write("Processed Data:")
    st.write(frequent_cast_members.head())
    st.write(frequent_cast_members.columns)

    # Plot the data
    fig = px.bar(
        frequent_cast_members,
        x='Cast_Member',
        y='Appearance_Count',
        title=f"Top {top_n_cast} Most Frequent Cast Members",
        labels={'Appearance_Count': 'Number of Appearances'}
    )
    st.plotly_chart(fig)


# Section: Genre Distribution
elif page == "Genre Distribution":
    st.title("Genre Distribution")
    top_n_genres = st.slider("Select Top N Genres", min_value=5, max_value=20, value=10)
    
    # Generate the genre distribution DataFrame
    genre_distribution = (
        type_data['Type']
        .value_counts()
        .head(top_n_genres)
        .reset_index()
    )
    
    # Rename columns explicitly
    genre_distribution.columns = ['Genre', 'Count']
    
    # Debugging: Check the DataFrame structure
    st.write("Genre Distribution DataFrame")
    st.write(genre_distribution)
    
    # Create pie chart
    fig = px.pie(
        genre_distribution,
        names='Genre',  # Match column name
        values='Count',  # Match column name
        title=f"Top {top_n_genres} Genres Distribution"
    )
    st.plotly_chart(fig)

# Section: Classification and Regression
elif page == "Classification and Regression":
    st.title("Classification and Regression Models")

    # Classification: Predict Movie/TV Show
    st.subheader("Predict Content Type (Movie/TV Show)")
    country_map = dict(enumerate(data['Country'].astype('category').cat.categories))
    rating_map = dict(enumerate(data['Rating'].astype('category').cat.categories))
    
    selected_country = st.selectbox("Select Country", options=country_map.values())
    selected_rating = st.selectbox("Select Rating", options=rating_map.values())
    duration_input = st.slider(
        "Duration (minutes)", 
        min_value=int(data['Duration'].min()), 
        max_value=int(data['Duration'].max()),
        value=int(data['Duration'].mean())
    )

    # Map selected values to encoded numeric values
    country_encoded = {v: k for k, v in country_map.items()}[selected_country]
    rating_encoded = {v: k for k, v in rating_map.items()}[selected_rating]

    rf_classifier = RandomForestClassifier(random_state=42)
    X = data[['Rating', 'Country', 'Duration']]
    y = data['Category']
    rf_classifier.fit(X, y)
    
    user_input = pd.DataFrame({'Rating': [rating_encoded], 'Country': [country_encoded], 'Duration': [duration_input]})
    prediction = rf_classifier.predict(user_input)
    st.write("Predicted Category:", "Movie" if prediction[0] == 0 else "TV Show")
    
# Section: Correlation Analysis
elif page == "Correlation Analysis":
    st.title("Correlation Analysis")
    columns_to_analyze = st.multiselect("Select Columns for Correlation", options=data.select_dtypes(include='number').columns)
    
    if len(columns_to_analyze) > 1:
        correlation_matrix = data[columns_to_analyze].corr()
        fig = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

# Section: Filtered Data Export
elif page == "Filtered Data Export":
    st.title("Filtered Data Export")
    selected_column = st.selectbox("Filter by Column", options=data.columns)
    unique_values = data[selected_column].unique()
    selected_value = st.selectbox(f"Filter {selected_column} by Value", unique_values)
    filtered_data = data[data[selected_column] == selected_value]
    st.write("Filtered Data:")
    st.dataframe(filtered_data)
    
    if st.button("Export Filtered Data"):
        filtered_data.to_csv("filtered_data.csv", index=False)
        st.write("Filtered data exported as 'filtered_data.csv'.")
