import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class DiseaseMedicineRecommender:
    def __init__(self, csv_path):
        # Load the CSV and drop duplicates
        self.df = pd.read_csv(csv_path)[['disease', 'drug']].drop_duplicates()
        
        # Vectorize disease names
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.df['disease'])
        
        # Fit Nearest Neighbors model
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.model.fit(self.X)

    def recommend(self, input_disease):
        # Convert input to vector and get top 5 similar entries
        input_vec = self.vectorizer.transform([input_disease])
        distances, indices = self.model.kneighbors(input_vec)
        
        # Return drug recommendations
        return self.df.iloc[indices[0]]['drug'].tolist()
