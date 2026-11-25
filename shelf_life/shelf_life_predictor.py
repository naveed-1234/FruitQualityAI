import json
import os

def get_shelf_life_info(fruit_name, quality_status):
    """Returns shelf-life duration and recommendations for a fruit and quality pair."""
    data_path = os.path.join(os.path.dirname(__file__), 'shelf_life_data.json')
    
    default_response = {
        "min_days": 0,
        "max_days": 0,
        "storage_condition": "Unknown",
        "recommendation": "Information unavailable for this selection."
    }
    
    if not os.path.exists(data_path):
        return default_response
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data.get(fruit_name, {}).get(quality_status, default_response)