def create_grade_bands(gpa):
    """
    Map GPA values to broader academic performance levels.
    Simplified to 3 levels to improve model accuracy and interpretation.
    """
    if gpa >= 3.4:
        return 'High'
    elif gpa >= 2.8:
        return 'Medium'
    else:
        return 'Low'

def get_grade_color_map():
    """
    Mapping performance levels to distinct colors.
    """
    return {
        'High': '#10B981',   # Green
        'Medium': '#F59E0B', # Orange
        'Low': '#EF4444'     # Red
    }

# Filtering constants
DEPARTMENTS_TO_REMOVE = ['DPT', 'BS Urdu', 'Radiology', 'BS Mathematics']

OUTLIER_COLUMNS = [
    'hours-study-per-day-average',
    'hours-per-day-use-mobile phone',
    'hours-sleep-per-night',
    'hours-per-day-use-social-media-apps',
    'GPA-current/previous-semester'
]

# Machine Learning Features
NUMERIC_FEATURES = [
    'current-semester', 
    'hours-study-per-day-average', 
    'hours-per-day-use-mobile phone', 
    'hours-sleep-per-night',
    'average-class-attendance', 
    'hours-per-day-use-social-media-apps',
    'academic-stress-level', 
    'motivation-level'
]

CLUSTERING_FEATURES = [
    'hours-study-per-day-average', 
    'hours-per-day-use-mobile phone',
    'hours-sleep-per-night',
    'average-class-attendance',
    'hours-per-day-use-social-media-apps',
    'academic-stress-level',
    'motivation-level'
]
