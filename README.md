# Netflix Data Analysis and Dashboard

## 📌 Project Overview
This project focuses on analyzing Netflix's content data to uncover meaningful insights and develop an interactive dashboard for dynamic exploration. The project includes comprehensive **data preprocessing, database normalization, exploratory data analysis (EDA), clustering analysis, and predictive modeling.**

---

## 🎯 Objectives
- Analyze Netflix's TV shows and movies dataset.
- Handle missing data and preprocess attributes.
- Normalize the data into a structured **SQL database**.
- Perform **exploratory data analysis (EDA)** and clustering analysis.
- Develop a **dashboard** for dynamic insights visualization.
- Build a **predictive model** to classify content as a Movie or TV Show.

---

## 📂 Dataset Details
The dataset contains **7,787 records with 15 columns**, covering:
- **Show Details:** Title, category (Movie/TV Show), director, cast, country, description.
- **Attributes:** Release date, rating, duration, genre.
- **Challenges Identified:** Missing data, inconsistent formats, categorical encoding.

---

## 🔍 Data Preprocessing & SQL Database Design
### 🔹 Handling Missing Values
- Missing values in **Director** and **Cast** were replaced with "Unknown".
- **Country** and **Release Date** were filled with the most frequent value.
- **Duration** was standardized by converting "min" to numeric values.

### 🔹 Categorical Data Encoding
- Categorical columns (Category, Type, Country, Rating) were converted into numeric codes.

### 🔹 Outlier Handling
- **Interquartile Range (IQR)** method was used to replace outliers with median values.

### 🔹 SQL Database Design
- **Normalized to 3NF:**
  - **Main Table:** Stores atomic attributes like Show_ID, Title, Category, Country.
  - **Supporting Tables:** Hold multi-valued fields for Cast and Type.
- **SQLite Implementation:** Structured querying and insights extraction.

---

## 📊 Key Insights from EDA
- **Rating Distribution:** Majority of content is rated **TV-MA, TV-14, and TV-PG**.
- **Duration Trends:**
  - Movies mostly range between **90 to 120 minutes**.
  - TV shows have varied season-based durations.
- **Content by Country:**
  - The **United States** has the highest content production, followed by **India and the UK**.
- **Popular Genres:** **Drama** and **Comedy** dominate Netflix’s catalog.
- **Frequent Directors & Cast:** Specific directors and actors contribute significantly to Netflix's content.

---

## 🤖 Machine Learning Models
### 🔹 Clustering Analysis
- **KMeans clustering** was used to group content based on **duration and rating**, identifying distinct content patterns.

### 🔹 Predictive Model
- **Random Forest Classifier** was built to predict if a content item is a **Movie or TV Show** based on Rating, Country, and Duration.

---

## 📊 Interactive Dashboard
### 🖥 Features
✅ **Dynamic Filters:** Dropdowns and sliders for selecting **countries, ratings, durations**.
✅ **Clustering Visualization:** Discover content patterns visually.
✅ **Predictive Model Integration:** Classify content type based on input parameters.
✅ **Export Functionality:** Filtered datasets can be downloaded for offline analysis.

### 🚀 Impact
- Enables **non-technical users** to explore Netflix’s content interactively.
- Helps stakeholders identify **trends in content production**.

---

## 🛠 Challenges & Solutions
| Challenge | Solution |
|-----------|----------|
| **Missing Data** in Director, Cast | Imputed missing values with placeholders |
| **Inconsistent Duration Formats** | Converted all durations into a unified numeric format |
| **Categorical Data Encoding** | Used Label Encoding for compatibility |
| **SQL Normalization Complexity** | Designed separate supporting tables for multi-valued attributes |

---

## 🔮 Future Enhancements
🔹 **Live Data Updates:** Integrate APIs to fetch real-time Netflix data.
🔹 **Advanced Predictive Models:** Use **Neural Networks & Ensemble Models** for better classification.
🔹 **Enhanced Clustering:** Explore **DBSCAN or Gaussian Mixture Models** for better grouping.

---

## 📌 Conclusion
This project successfully analyzed **Netflix’s content data** and created a **dashboard** for interactive insights. **Future iterations** will focus on **real-time data updates, improved predictive models, and advanced clustering techniques.**

---

## 📎 Repository Structure
```
📂 Netflix-Data-Analysis
 ├── 📁 data              # Dataset and cleaned data
 ├── 📁 notebooks         # Jupyter notebooks for EDA and ML models
 ├── 📁 scripts          # Python scripts for preprocessing & SQL handling
 ├── 📁 dashboard        # Streamlit/Power BI dashboard files
 ├── 📄 README.md        # Project documentation
 ├── 📄 requirements.txt  # Dependencies for running the project
```

---

## 🏗 Installation & Setup
### 🔹 Prerequisites
- Python 3.8+
- Jupyter Notebook / Google Colab
- SQLite (for database queries)
- Streamlit / Power BI (for dashboard visualization)

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run Jupyter Notebook
```bash
jupyter notebook
```

### 🔹 Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🤝 Contributing
Feel free to **fork** this repository and submit a **pull request**. Contributions are always welcome!

---

## 📜 License
This project is licensed under the **MIT License**.

---

## ✨ Acknowledgments
Special thanks to the **Netflix dataset providers** and open-source contributors.

---

**🚀 Happy Coding!**

