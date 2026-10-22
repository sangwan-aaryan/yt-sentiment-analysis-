# src/data/data_preprocessing.py

import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    # Convert to lowercase
    comment = comment.lower()

    # Remove trailing and leading whitespaces
    comment = comment.strip()

    # Remove newline characters
    comment = re.sub(r'\n', ' ', comment)

    # Remove non-alphanumeric characters, except punctuation
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    # Remove stopwords but retain important ones for sentiment analysis
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def ensure_nltk_resources() -> None:
    """Download and verify the NLTK corpora required by preprocessing."""
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    resources = (
        ("wordnet", ("corpora/wordnet", "corpora/wordnet.zip")),
        ("stopwords", ("corpora/stopwords", "corpora/stopwords.zip")),
    )
    for package, resource_paths in resources:
        if any(_nltk_resource_exists(path) for path in resource_paths):
            continue

        if not nltk.download(
            package,
            download_dir=nltk_data_dir,
            quiet=True,
        ):
            raise RuntimeError(f"Failed to download NLTK resource: {package}")
        if not any(_nltk_resource_exists(path) for path in resource_paths):
            raise LookupError(
                f"NLTK resource '{package}' was downloaded but is still unavailable "
                f"in {nltk_data_dir}"
            )


def _nltk_resource_exists(resource_path: str) -> bool:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        return False
    return True


def main():
    try:
        logger.debug("Starting data preprocessing...")
        ensure_nltk_resources()

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        # Fetch the data from data/raw
        train_data = pd.read_csv(os.path.join(root_dir, 'data/raw/train.csv'))
        test_data = pd.read_csv(os.path.join(root_dir, 'data/raw/test.csv'))
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(
            train_processed_data,
            test_processed_data,
            data_path=os.path.join(root_dir, 'data'),
        )
    except Exception:
        logger.exception('Failed to complete the data preprocessing process')
        raise

if __name__ == '__main__':
    main()
