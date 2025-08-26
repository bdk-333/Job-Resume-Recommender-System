import pandas as pd
import numpy as np
import ast
import re
import joblib
import os

# Suppress warnings for a cleaner user experience
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)


# --- STEP 2: ARTIFACT LOADING & HELPER FUNCTIONS ---

def load_artifacts():
    """ Loads all pre-trained components from disk. """
    print("--- Loading all saved model components ---")
    artifacts = {}
    try:
        # **MODIFICATION: Load the AdaBoost model**
        artifacts['model_pipeline'] = joblib.load('../data/final_adaboost_model.pkl')
        artifacts['le'] = joblib.load('../data/label_encoder.pkl')
        artifacts['university_map'] = joblib.load('../data/university_target_map.pkl')
        artifacts['all_jobs'] = pd.read_csv('../data/processed/all_jobs.csv')

        # In a full-scale app, other components like KMeans, SVD model, etc., would also be saved and loaded.
        # For this script, we will simulate or recalculate them for simplicity.

        print("All components loaded successfully.")
        return artifacts
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e.filename}")
        print("Please make sure you have run the saving code in your training notebook.")
        return None


def get_user_input():
    """ Interactively prompts the user for their resume details. """
    print("\n\nPlease enter the details for the new resume:")
    print("------------------------------------------")
    resume = {}
    resume['career_objective'] = input("Career Objective: ")
    resume['skills'] = input("Skills (comma-separated): ")
    resume['major_field_of_studies'] = input("Major Field of Study: ")
    resume['positions'] = input("Most Recent Position/Title: ")
    resume['degree_names'] = input("Degree(s) (e.g., ['B.Sc in Computer Science']): ")
    try:
        resume['Resume_Years_Exp'] = int(input("Total Years of Professional Experience: "))
    except ValueError:
        resume['Resume_Years_Exp'] = 0
    resume['first_university'] = input("Primary University Name: ")
    return resume


# --- STEP 3: FEATURE ENGINEERING PIPELINE ---

def create_feature_matrix(new_resume_dict, all_jobs_df, uni_map):
    """ Generates the complete feature matrix for a new resume against all jobs. """
    print(f"\nProcessing new resume and preparing {len(all_jobs_df)} candidate pairs...")
    resume_df = pd.DataFrame([new_resume_dict])
    candidate_df = pd.merge(resume_df.assign(key=1), all_jobs_df.assign(key=1), on='key').drop('key', axis=1)

    # --- Apply all feature engineering steps ---
    def process_degree_data(degree_string):
        master_keywords = ['master', 'm.sc', 'msc', 'm.a', 'mba', 'm.com'];
        bachelor_keywords = ['bachelor', 'b.sc', 'bsc', 'b.tech', 'bba', 'b.a', 'b.com'];
        doctorate_keywords = ['ph.d', 'phd', 'doctorate']
        stem_keywords = ['science', 'tech', 'eng', 'computer', 'math', 'statistic'];
        business_keywords = ['business', 'bba', 'mba', 'account', 'financ', 'commerc'];
        arts_keywords = ['arts', 'humanities']
        highest_level, degree_type = 'Other', 'Other'
        try:
            degrees = ast.literal_eval(str(degree_string));
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
                max_level = max(found_levels);
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

    def extract_job_experience(text):
        if isinstance(text, str):
            match = re.search(r'(\d+)', text.lower());
            return int(match.group(1)) if match else 0
        return 0

    candidate_df['Job_Years_Exp'] = candidate_df['experience_requirement'].apply(extract_job_experience)
    candidate_df['Experience_Mismatch'] = abs(candidate_df['Resume_Years_Exp'] - candidate_df['Job_Years_Exp'])
    candidate_df[['highest_education_level', 'degree_type']] = candidate_df['degree_names'].apply(process_degree_data)
    candidate_df['university_encoded'] = candidate_df['first_university'].map(uni_map).fillna(uni_map.mean())

    # Placeholders for complex features. A full app would load the trained models for these.
    candidate_df['Embedding_Cosine_Similarity'] = 0.5
    candidate_df['svd_predicted_score'] = 0.5
    candidate_df['Skill_Overlap_Count'] = 0
    candidate_df['Skill_Jaccard_Score'] = 0.0
    candidate_df['Resume_Cluster_KMeans'] = 0

    print("Feature matrix created successfully.")
    return candidate_df


# --- STEP 4: RECOMMENDATION LOGIC ---

def recommend_jobs(resume_data, artifacts):
    """
    Generates and displays recommendations from the classification model.
    """
    feature_df = create_feature_matrix(resume_data, artifacts['all_jobs'], artifacts['university_map'])

    # **CRITICAL:** The order and names of these columns must EXACTLY match what the model was trained on.
    final_features_list = ['Job_Years_Exp', 'Skill_Overlap_Count', 'Skill_Jaccard_Score', 'Resume_Years_Exp',
                           'Experience_Mismatch',
                           'Embedding_Cosine_Similarity', 'gpa', 'Resume_Cluster_KMeans', 'highest_education_level',
                           'degree_type', 'university_encoded', 'svd_predicted_score']

    # We must reorder the dataframe columns to ensure they match the training order
    X_predict = feature_df[final_features_list]

    # The loaded pipeline will automatically preprocess and predict
    predicted_probs = artifacts['model_pipeline'].predict_proba(X_predict)
    high_match_index = list(artifacts['le'].classes_).index('High')
    feature_df['high_match_probability'] = predicted_probs[:, high_match_index]

    cls_recs = feature_df.sort_values(by='high_match_probability', ascending=False).head(7)

    # --- Display Results ---
    print("\n\n" + "=" * 60)
    print("                     RECOMMENDATION RESULTS")
    print("=" * 60)
    print("\n--- Top 7 Recommendations (Ranked by 'High' Match Probability) ---")
    print(cls_recs[['job_position_name', 'high_match_probability']].to_string(index=False))
    print("\n" + "=" * 60)


# --- STEP 5: MAIN EXECUTION ---

def main():
    """ Main function to run the recommender application. """
    artifacts = load_artifacts()
    if artifacts:
        while True:
            user_resume = get_user_input()
            recommend_jobs(user_resume, artifacts)

            another = input("\nWould you like to try another resume? (yes/no): ").lower()
            if another != 'yes':
                print("Thank you for using the Job Recommender!")
                break


if __name__ == "__main__":
    main()