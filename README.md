# Mental Health Unsupervised Learning
 
##Overview
This project applies unsupervised machine learning techniques to analyze mental health survey data. The goal is to identify hidden patterns, groups similar respondents and gain insights into metal health trends without using labels outcomes.
Techniques used include:
Data preproccessing and cleaning
Dimensionality reduction (PCA, t-SNE)
Clustering (eg. K-Means or similiar algorithms)
Data visualization


#Project structure
mental-health-unsupervised-learning/
│
├── data/
│ └── mental_health_survey.csv # Raw dataset
│
├── notebooks/
│ └── analysis.ipynb # Exploratory analysis and experiments
│
├── visualizations/
│ ├── pca_plot.png # PCA visualization
│ ├── tsne_plot.png # t-SNE visualization
│ └── cluster_distribution.png # Cluster distribution plot
│
├── src/
│ ├── preprocessing.py # Data cleaning and preprocessing
│ ├── clustering.py # Clustering algorithms
│ └── visualization.py # Plotting functions
│
├── README.md
├── requirements.txt
└── report.pdf # Final analysis report

#Objectives
-Understand patterns in mental health survey responses
-Reduce high-dimensional data into lower dimensions for visualization
- Group similiar respondents uisng clusering techniques
- -Visualize and interpret cluster structures

#Technologies used
-python
-Pandas and NumPy
-Scikit-Learn
-Matplotlib and Seaborn
-Jupytter Notebook

##How to run the project
##1. Clone the repoditory
'''bash
git clone https://github.com/your-username/mental-health-unsupervised-learning.git
cd mental-health-unsupervised-learning

2. Install dependencies
   '''bash
   pip install -r requirements.txt

   3. Run the notebook
   Open Jupyter Notebook and run:
'''bash
notebooks/analysis.ipynb

Key steps in the analysis
1. Data preprocessing
   -handling missing values
   -encoding categorical variables
   -feature scaling

2. Dimensionality reduction
   -PCA for variance-based reduction
   -t-SNE for visualization of clusters

3. Clustering
   -applying clustering algorithms
   -determining optimal number of clusters

4. Visualization
   -PCA  and t-SNE plots
   -Cluster distribution analysis

Results
The model reveals distinct groupings in the dataset helping to uncover patterns in mental health responses that may not be visible in raw data 

Future improvements
-Try DBSCAN or hierarchical cluctering 
-improve feature engineering 
-test on larger datasets
-build an iteractive dashboard for explanation

Author
Phethakahle Mbokazi
