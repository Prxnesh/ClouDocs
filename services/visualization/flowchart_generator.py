import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowchartGenerator:
    """
    Transforms text outlines or concepts into Markdown Mermaid.js schemas.
    """
    @staticmethod
    def generate_hierarchy(topics: List[str], title: str = "Document Flow") -> str:
        """
        Creates a top-down structural Mermaid flowchart.
        """
        if not topics:
            return "graph TD;\n  A[No Data Given];"

        mermaid_str = f"graph TD;\n  A[{title}]:::hero;\n"
        
        # Simple branching
        for i, topic in enumerate(topics):
            node_id = chr(66 + i)  # B, C, D...
            clean_topic = topic.replace('"', '').replace('[', '').replace(']', '')
            # Cap text length for nodes
            if len(clean_topic) > 30:
                clean_topic = clean_topic[:27] + "..."
                
            mermaid_str += f"  A --> {node_id}[{clean_topic}];\n"
            
        mermaid_str += "\n  classDef hero fill:#187d76,stroke:#333,stroke-width:2px,color:#fff;"
        logger.info("Mermaid JS script compiled.")
        return mermaid_str
