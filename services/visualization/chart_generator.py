import logging
import pandas as pd
from typing import Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Generates rapid exploratory visualizations from DataFrames.
    """
    @staticmethod
    def generate_bar_chart(df: pd.DataFrame, max_categories: int = 10) -> Optional[Any]:
        """
        Returns a matplotlib figure of the most populated categorical column versus counts,
        or a generic numeric plot if only numerics exist.
        """
        try:
            import matplotlib.pyplot as plt
            
            if df.empty:
                return None

            fig, ax = plt.subplots(figsize=(8, 4))
            
            cat_cols = df.select_dtypes(exclude=['number']).columns
            if len(cat_cols) > 0:
                # Plot frequencies of the first categorical column
                target_col = cat_cols[0]
                counts = df[target_col].value_counts().head(max_categories)
                counts.plot(kind='bar', ax=ax, color='teal')
                ax.set_title(f"Distribution of {target_col}")
                ax.set_ylabel("Frequency")
            else:
                # Plot distribution of the first numeric column
                num_cols = df.select_dtypes(include=['number']).columns
                target_col = num_cols[0]
                df[target_col].plot(kind='hist', ax=ax, color='coral', bins=10)
                ax.set_title(f"Histogram of {target_col}")
                ax.set_ylabel("Count")

            plt.tight_layout()
            logger.info("Generated visualization figure successfully.")
            return fig

        except ImportError:
            logger.warning("matplotlib not installed. Cannot generate chart.")
            return None
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None
