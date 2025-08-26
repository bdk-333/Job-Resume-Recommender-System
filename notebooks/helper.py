import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re


def calculate_resume_experience_from_lists(row):
    # These are the columns from your screenshot
    starts_str = row['start_dates']
    ends_str = row['end_dates']
    try:
        
        # Safely parse the strings into actual Python lists
        start_dates = ast.literal_eval(starts_str)
        end_dates = ast.literal_eval(ends_str)
    except Exception as e:
        # print(f"Error in row {starts_str}, {ends_str}, {e}")
        return 0
    
    # Ensure lists are of the same length to avoid errors
    if not (len(start_dates) == len(end_dates)):
        return 0

    total_exp = 0
    current_date = "2023-10-01"  # Use a fixed current date for consistency

    # Now loop through the lists, read start and end dates as dates. Then, subtract the start date from end date, get the difference in months. Get sum of all months of experiences. Then convert months to years
    for start, end in zip(start_dates, end_dates):
        try:
            if not start or start.lower() in ["n/a", "none", "na", "current","nan"]:
                continue
            start_date = pd.to_datetime(start, errors='coerce')
        except Exception as e:
            print(f"Error in checking start date: {start}, {e}")
        try:
            if not end or end.lower() in ["n/a", "none", "na", "current","nan", 'present', 'current', 'now', 'ongoing', "till date",]:
                end = current_date
            end_date = pd.to_datetime(end, errors='coerce')
        except Exception as e:
            print(f"error in checking end date: {end}: {e}")
        try:
            if pd.notnull(start_date) and pd.notnull(end_date):
                # Calculate the difference in months
                months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_exp += months_diff / 12
        except Exception as e:
            print(f"error in calculating experience for {start} to {end}: {e}")
    # Change the value to upper(ceiling) to get integer years
    total_exp = np.ceil(total_exp)
    return int(total_exp)


def extract_min_experience(experience_str):
    # Use regex to find all numbers in the string
    numbers = re.findall(r'\d+', experience_str)
    if not numbers:
        return 0  # Return 0 if no numbers found
    
    # Convert the found numbers to integers
    numbers = list(map(int, numbers))
    
    # Return the minimum number found
    return min(numbers)


def plot_wordcloud(text, title):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=50,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

# Function to get the ratio of matching skills in resume and job skills required
def get_matching_skills_ratio(resume_description, job_skills):
    if not resume_description or not job_skills or pd.isna(resume_description) or pd.isna(job_skills):
        return 0.0
    matched_skills = 0
    job_skills = job_skills.lower().split()
    job_skills = list(set(job_skills))  # Remove duplicates
    for skill in job_skills:
        skill = skill.strip()
        if skill in resume_description.lower():
            matched_skills += 1
    return matched_skills / len(job_skills) if job_skills else 0.0

def process_skills_text(series):
    # Drop NaNs and convert to a single string
    all_skills_text = series.dropna().str.cat(sep=' ')
    
    # Clean the text: remove list-like characters and extra quotes
    all_skills_text = re.sub(r"[\[\]',]", "", all_skills_text)
    
    # Convert to lowercase
    return all_skills_text.lower()

def plot_wordcloud(text, title):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=50,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()
    
# Function to get N gram frequencies from a dataframe column
def get_ngram_frequencies(column, n=1):
    text = ' '.join(column.dropna().astype(str).tolist())
    words = re.findall(r'\b\w+\b', text.lower())
    ngrams = zip(*[words[i:] for i in range(n)])
    ngram_freq = pd.Series(ngrams).value_counts()
    return ngram_freq

# Function to print all unique values in a column
def print_unique_values(column):
    # Iterate through each row in the column, read the data as a list. Append to a set to avoid duplicates
    unique_values = set()
    for value in column.dropna():
        try:
            # Convert string representation of list to actual list
            values_list = ast.literal_eval(value)
            if isinstance(values_list, list):
                if type(values_list[0]) is list:
                    for item in values_list:
                        unique_values.update(item)
                else:
                    unique_values.update(values_list)
            else:
                unique_values.add(values_list)
        except (ValueError, SyntaxError):
            # If conversion fails, just add the value as is
            unique_values.add(value)
        except Exception as e:
            pass
    # Print the unique values
    print(f"Unique values in column '{column.name}':")
    for val in unique_values:
        print(f"{val}")