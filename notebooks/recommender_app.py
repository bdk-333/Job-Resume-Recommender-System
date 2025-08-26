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
        le_path = '../data/label_encoder.pkl'
        uni_map_path = '../data/university_target_map.pkl'
        jobs_path = '../data/processed/all_jobs.csv'
        featured_data_path = '../data/processed/refined_df.csv'

        # Load components
        artifacts['final_model'] = joblib.load(model_path)
        artifacts['le'] = joblib.load(le_path)
        artifacts['university_map'] = joblib.load(uni_map_path)
        artifacts['all_jobs'] = pd.read_csv(jobs_path)

        # We need to re-train the helper models (KMeans, TF-IDF) as they were not saved.
        # This is a one-time setup cost when the app starts.
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        df_orig = pd.read_csv(featured_data_path)
        df_orig.columns = df_orig.columns.str.strip().str.replace(r'\ufeff', '', regex=True)

        # Text for clustering
        resume_skills_c = df_orig['skills'].fillna('')
        resume_title_c = df_orig['positions'].fillna('')
        resume_exp_c = 'experience ' + df_orig['Resume_Years_Exp'].astype(str)
        clustering_text = resume_skills_c + ' ' + resume_title_c + ' ' + resume_exp_c

        tfidf_cluster = TfidfVectorizer(stop_words='english', max_features=5000, min_df=2).fit(clustering_text)
        kmeans_model = KMeans(n_clusters=7, init='k-means++', n_init=10, random_state=42).fit(
            tfidf_cluster.transform(clustering_text))

        artifacts['kmeans_model'] = kmeans_model
        artifacts['tfidf_cluster'] = tfidf_cluster

        print("All components loaded successfully.")
        return artifacts

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e.filename}")
        print("Please make sure all necessary files (.pkl, .csv) are in the correct directories before running.")
        return None


def create_feature_matrix(new_resume_dict, all_jobs_df, uni_map, kmeans, tfidf_v):
    """
    Takes a new resume and all jobs, and generates the complete feature matrix needed for prediction.
    """
    print(f"\nProcessing new resume and preparing {len(all_jobs_df)} candidate pairs...")
    resume_df = pd.DataFrame([new_resume_dict])
    candidate_df = pd.merge(resume_df.assign(key=1), all_jobs_df.assign(key=1), on='key').drop('key', axis=1)

    # --- Apply all feature engineering steps ---
    candidate_df['Experience_Mismatch'] = abs(
        candidate_df['Resume_Years_Exp'] - candidate_df['experience_years_required'])

    def calculate_skill_overlap(row):
        try:
            r_skills = set(str(row['skills']).lower().split())
            j_skills = set(str(row['skills_required']).lower().split())
            intersection = r_skills.intersection(j_skills)
            union = r_skills.union(j_skills)
            return pd.Series([len(intersection), len(intersection) / len(union) if union else 0])
        except:
            return pd.Series([0, 0.0])

    candidate_df[['Skill_Overlap_Count', 'Skill_Jaccard_Score']] = candidate_df.apply(calculate_skill_overlap, axis=1)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances
    resume_text = candidate_df[
        ['career_objective', 'skills', 'major_field_of_studies', 'positions', 'responsibilities']].fillna('').astype(
        str).agg(' '.join, axis=1)
    job_text = candidate_df[
        ['job_position_name', 'educationaL_requirements', 'skills_required', 'responsibilities.1']].fillna('').astype(
        str).agg(' '.join, axis=1)
    sim_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000).fit(pd.concat([resume_text, job_text]))
    candidate_df['Cosine_Similarity'] = 1 - pairwise_distances(sim_vectorizer.transform(resume_text),
                                                               sim_vectorizer.transform(job_text),
                                                               metric='cosine').diagonal()

    new_resume_cluster_text = f"{new_resume_dict.get('skills', '')} {new_resume_dict.get('positions', '')} experience {new_resume_dict.get('Resume_Years_Exp', 0)}"
    new_resume_tfidf = tfidf_v.transform([new_resume_cluster_text])
    candidate_df['Resume_Cluster_KMeans'] = kmeans.predict(new_resume_tfidf)[0]

    candidate_df['gpa'] = new_resume_dict.get('gpa', 0.0)
    candidate_df['first_university'] = new_resume_dict.get('first_university', 'Unknown')
    candidate_df['university_encoded'] = candidate_df['first_university'].map(uni_map).fillna(uni_map.mean())

    print("Feature matrix created successfully.")
    return candidate_df


def recommend_top_n_jobs(resume_data, artifacts, top_n=7):
    """
    Main recommendation function.
    """
    feature_df = create_feature_matrix(
        resume_data,
        artifacts['all_jobs'],
        artifacts['university_map'],
        artifacts['kmeans_model'],
        artifacts['tfidf_cluster']
    )

    final_features = ['experience_years_required', 'Skill_Overlap_Count', 'Skill_Jaccard_Score',
                      'Resume_Years_Exp', 'Experience_Mismatch', 'Cosine_Similarity', 'gpa',
                      'Resume_Cluster_KMeans', 'university_encoded']
    X_predict = feature_df[final_features]

    predicted_probs = artifacts['final_model'].predict_proba(X_predict)

    # We still need the 'High' probability for ranking behind the scenes
    high_match_index = list(artifacts['le'].classes_).index('High')
    feature_df['P(High)'] = predicted_probs[:, high_match_index]

    # Rank to get the best ones, then just return the names
    recommendations = feature_df.sort_values(by='P(High)', ascending=False).head(top_n)

    return recommendations['job_position_name'].tolist()


def get_user_input():
    """
    Interactively prompts the user to enter their resume details.
    """
    print("\n\nPlease enter the details for the new resume:")
    print("------------------------------------------")
    resume = {}
    resume['career_objective'] = input("Career Objective: ")
    resume['skills'] = input("Skills (comma-separated, e.g., python, sql, machine learning): ")
    resume['major_field_of_studies'] = input("Major Field of Study (e.g., Computer Science): ")
    resume['positions'] = input("Most Recent Position/Title (e.g., Data Analyst): ")
    resume['responsibilities'] = input("Key Responsibilities in that role: ")
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
    """
    Main function to run the recommender application.
    """
    artifacts = load_artifacts()

    if artifacts:
        while True:
            # Get resume data from user
            user_resume = get_user_input()

            # Get recommendations
            # **MODIFICATION: Set top_n to 7**
            top_jobs_list = recommend_top_n_jobs(user_resume, artifacts, top_n=7)

            if top_jobs_list:
                # **MODIFICATION: Display results as an unranked list**
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
