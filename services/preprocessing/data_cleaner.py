import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Handles preprocessing and cleaning of tabular datasets (CSV, XLSX)
    prior to analysis or loading into the system.
    """
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans a pandas DataFrame by performing standard operations:
        1. Dropping completely empty rows or columns.
        2. Dropping duplicate rows.
        3. Filling or dropping missing values gracefully.
        
        Args:
            df (pd.DataFrame): The raw dataframe.
            
        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        if df is None or df.empty:
            logger.warning("Empty or None DataFrame passed to DataCleaner.")
            return df
        
        initial_shape = df.shape

        # Create a copy to avoid mutating the original prematurely
        cleaned_df = df.copy()

        # Step 1: Drop rows/columns that are 100% NaN
        cleaned_df.dropna(how='all', axis=0, inplace=True)
        cleaned_df.dropna(how='all', axis=1, inplace=True)

        # Step 2: Drop exact duplicates
        cleaned_df.drop_duplicates(inplace=True)

        # Step 3: Handle remaining missing values
        # For numeric columns, fill with median to avoid outlier skew
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if cleaned_df[col].isna().any():
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)

        # For categorical columns, fill with "Unknown" or the mode
        cat_cols = cleaned_df.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                
        final_shape = cleaned_df.shape
        logger.info(f"DataFrame cleaned. Shape changed from {initial_shape} to {final_shape}.")

        return cleaned_df
