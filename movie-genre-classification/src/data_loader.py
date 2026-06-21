import pandas as pd
import os

def load_data(filepath):
    """
    Loads the movie genre dataset from the specified text file.
    The expected format is: ID ::: TITLE ::: GENRE ::: DESCRIPTION
    If it's test data without solutions, it might be: ID ::: TITLE ::: DESCRIPTION
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}. Please ensure the dataset is downloaded and placed in the 'data/' directory.")

    print(f"Loading data from {filepath}...")
    
    ids = []
    titles = []
    genres = []
    descriptions = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                # Format: ID ::: TITLE ::: GENRE ::: DESCRIPTION
                ids.append(parts[0])
                titles.append(parts[1])
                genres.append(parts[2])
                descriptions.append(parts[3])
            elif len(parts) == 3:
                # Format: ID ::: TITLE ::: DESCRIPTION
                ids.append(parts[0])
                titles.append(parts[1])
                genres.append(None) # Missing genre for test set
                descriptions.append(parts[2])
            else:
                continue # Skip malformed lines

    df = pd.DataFrame({
        'id': ids,
        'title': titles,
        'genre': genres,
        'description': descriptions
    })
    
    print(f"Successfully loaded {len(df)} records.")
    return df

if __name__ == "__main__":
    # Quick test if run directly
    pass
