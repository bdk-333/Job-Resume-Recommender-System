import pandas as pd
import numpy as np
import ast
import re
import joblib  # Used for saving and loading sklearn models
import os

# Suppress warnings for a cleaner user experience
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def load_artifacts():
    """
    Loads all the pre-trained model components from disk.
    Returns a dictionary of all loaded components.
    """
    print("--- Loading all saved model components ---")
    artifacts = {}
    try:
        # Define paths
        model_path = '../data/final_stacking_model.pkl'
        preprocessor_path = '../data/preprocessor.pkl'
        le_path = '../data/label_encoder.pkl'
        uni_map_path = '../data/university_target_map.pkl'
        jobs_path = '../data/processed/all_jobs.csv'
        featured_data_path = '../data/processed/featured_data.csv'

        # Load core components
        artifacts['final_model'] = joblib.load(model_path)
        artifacts['preprocessor'] = joblib.load(preprocessor_path)
        artifacts['le'] = joblib.load(le_path)
        artifacts['university_map'] = joblib.load(uni_map_path)
        artifacts['all_jobs'] = pd.read_csv(jobs_path)

        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sentence_transformers import SentenceTransformer

        artifacts['embedding_model'] = SentenceTransformer('all-MiniLM-L6-v2')

        df_orig = pd.read_csv(featured_data_path)
        df_orig.columns = df_orig.columns.str.strip().str.replace(r'\ufeff', '', regex=True)
        resume_skills_c = df_orig['skills'].fillna('')
        resume_title_c = df_orig['positions'].fillna('')
        resume_exp_c = 'experience ' + df_orig['Resume_Years_Exp'].astype(str)
        clustering_text = resume_skills_c + ' ' + resume_title_c + ' ' + resume_exp_c
        tfidf_cluster = TfidfVectorizer(stop_words='english', max_features=5000, min_df=2).fit(clustering_text)

        # **FIX: Changed cluster count from 7 to 5**
        kmeans_model = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42).fit(
            tfidf_cluster.transform(clustering_text))
        artifacts['kmeans_model'] = kmeans_model
        artifacts['tfidf_cluster'] = tfidf_cluster

        print("All components loaded successfully.")
        return artifacts

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e.filename}")
        print("Please make sure all necessary files are in the correct directories.")
        return None
    except ImportError:
        print("\nPlease install the sentence-transformers library: !pip install -U sentence-transformers")
        return None


def process_degree_data(degree_string):
    """ Helper function to parse degree names into level and type. """
    master_keywords = ['master', 'm.sc', 'msc', 'm.a', 'mba', 'm.com']
    bachelor_keywords = ['bachelor', 'b.sc', 'bsc', 'b.tech', 'bba', 'b.a', 'b.com']
    doctorate_keywords = ['ph.d', 'phd', 'doctorate']
    stem_keywords = ['science', 'tech', 'eng', 'computer', 'math', 'statistic']
    business_keywords = ['business', 'bba', 'mba', 'account', 'financ', 'commerc']
    arts_keywords = ['arts', 'humanities']
    highest_level, degree_type = 'Other', 'Other'
    try:
        degrees = ast.literal_eval(str(degree_string))
        if not isinstance(degrees, list): degrees = [degrees]
        found_levels, found_types = [], []
        for degree in degrees:
            degree_lower = str(degree).lower()
            if any(k in degree_lower for k in doctorate_keywords):
                found_levels.append(3)
            elif any(k in degree_lower for k in master_keywords):
                found_levels.append(2)
            elif any(k in degree_lower for k in bachelor_keywords):
                found_levels.append(1)
            if any(k in degree_lower for k in stem_keywords):
                found_types.append('STEM')
            elif any(k in degree_lower for k in business_keywords):
                found_types.append('Business')
            elif any(k in degree_lower for k in arts_keywords):
                found_types.append('Arts')
        if found_levels:
            max_level = max(found_levels)
            if max_level == 3:
                highest_level = 'Doctorate'
            elif max_level == 2:
                highest_level = 'Masters'
            elif max_level == 1:
                highest_level = 'Bachelors'
        if 'STEM' in found_types:
            degree_type = 'STEM'
        elif 'Business' in found_types:
            degree_type = 'Business'
        elif 'Arts' in found_types:
            degree_type = 'Arts'
    except:
        pass
    return pd.Series([highest_level, degree_type])


def create_feature_matrix(new_resume_dict, artifacts):
    """ Generates the complete feature matrix needed for prediction. """
    print(f"\nProcessing new resume and preparing {len(artifacts['all_jobs'])} candidate pairs...")
    resume_df = pd.DataFrame([new_resume_dict])
    candidate_df = pd.merge(resume_df.assign(key=1), artifacts['all_jobs'].assign(key=1), on='key').drop('key', axis=1)

    candidate_df['Experience_Mismatch'] = abs(candidate_df['Resume_Years_Exp'] - candidate_df['experience_years_required'])

    def calculate_skill_overlap(row):
        try:
            r_skills = set(str(row['skills']).lower().split());
            j_skills = set(str(row['skills_required']).lower().split())
            intersection = r_skills.intersection(j_skills);
            union = r_skills.union(j_skills)
            return pd.Series([len(intersection), len(intersection) / len(union) if union else 0])
        except:
            return pd.Series([0, 0.0])

    candidate_df[['Skill_Overlap_Count', 'Skill_Jaccard_Score']] = candidate_df.apply(calculate_skill_overlap, axis=1)

    from sklearn.metrics.pairwise import cosine_similarity
    resume_text = candidate_df[
        ['career_objective', 'skills', 'major_field_of_studies', 'positions', 'responsibilities']].fillna('').astype(
        str).agg(' '.join, axis=1)
    job_text = candidate_df[
        ['job_position_name', 'educationaL_requirements', 'skills_required', 'responsibilities.1']].fillna('').astype(
        str).agg(' '.join, axis=1)
    resume_emb = artifacts['embedding_model'].encode(resume_text.tolist())
    job_emb = artifacts['embedding_model'].encode(job_text.tolist())
    candidate_df['Embedding_Cosine_Similarity'] = [cosine_similarity([resume_emb[i]], [job_emb[i]])[0][0] for i in
                                                   range(len(job_emb))]

    new_resume_cluster_text = f"{new_resume_dict.get('skills', '')} {new_resume_dict.get('positions', '')} experience {new_resume_dict.get('Resume_Years_Exp', 0)}"
    candidate_df['Resume_Cluster_KMeans'] = \
    artifacts['kmeans_model'].predict(artifacts['tfidf_cluster'].transform([new_resume_cluster_text]))[0]

    # **FIX: Engineer the new education features**
    candidate_df[['highest_education_level', 'degree_type']] = process_degree_data(
        new_resume_dict.get('degree_names', "['']"))
    candidate_df['gpa'] = new_resume_dict.get('gpa', 0.0)
    candidate_df['first_university'] = new_resume_dict.get('first_university', 'Unknown')
    candidate_df['university_encoded'] = candidate_df['first_university'].map(artifacts['university_map']).fillna(
        artifacts['university_map'].mean())

    print("Feature matrix created successfully.")
    return candidate_df


def recommend_top_n_jobs(resume_data, artifacts, top_n=7):
    """ Main recommendation function. """
    feature_df = create_feature_matrix(resume_data, artifacts)

    # **FIX: Update the final feature list to match the trained model**
    final_features_list = ['Job_Years_Exp', 'Skill_Overlap_Count', 'Skill_Jaccard_Score', 'Resume_Years_Exp',
                           'Experience_Mismatch', 'Embedding_Cosine_Similarity', 'gpa', 'university_encoded',
                           'Resume_Cluster_KMeans', 'highest_education_level', 'degree_type']
    X_predict_raw = feature_df[final_features_list]

    X_predict_processed = artifacts['preprocessor'].transform(X_predict_raw)
    predicted_probs = artifacts['final_model'].predict_proba(X_predict_processed)

    for i, class_label in enumerate(artifacts['le'].classes_):
        feature_df[f'P({class_label})'] = predicted_probs[:, i]

    recommendations = feature_df.sort_values(by='P(High)', ascending=False).head(top_n)
    return recommendations['job_position_name'].tolist()


def get_user_input():
    """ Interactively prompts the user to enter their resume details. """
    print("\n\nPlease enter the details for the new resume:")
    print("------------------------------------------")
    resume = {}
    resume['career_objective'] = input("Career Objective: ")
    resume['skills'] = input("Skills (comma-separated): ")
    resume['major_field_of_studies'] = input("Major Field of Study: ")
    resume['positions'] = input("Most Recent Position/Title: ")
    resume['responsibilities'] = input("Key Responsibilities: ")
    # **FIX: Added prompt for degree names**
    resume['degree_names'] = input("Degree(s) (e.g., ['B.Sc in Computer Science', 'Masters in AI']): ")
    try:
        resume['Resume_Years_Exp'] = int(input("Total Years of Professional Experience: "))
    except ValueError:
        resume['Resume_Years_Exp'] = 0
    try:
        resume['gpa'] = float(input("GPA (on a 4.0 scale): "))
    except ValueError:
        resume['gpa'] = 0.0
    resume['first_university'] = input("Primary University Name: ")
    return resume


def main():
    """ Main function to run the recommender application. """
    artifacts = load_artifacts()
    if artifacts:
        while True:
            user_resume = get_user_input()
            top_jobs_list = recommend_top_n_jobs(user_resume, artifacts, top_n=7)
            if top_jobs_list:
                print("\n\n===== Here are some recommended jobs for you =====")
                for job in top_jobs_list:
                    print(f"  - {job}")
                print("=" * 52)

            another = input("\nWould you like to try another resume? (yes/no): ").lower()
            if another != 'yes':
                print("Thank you for using the Job Recommender!")
                break


if __name__ == "__main__":
    main()
