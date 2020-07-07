import pandas as pd
import numpy as np

def get_social_link_data(social):
	'''
	Returns a NumPy array of social data
	'''
	
	# Extracting Social Link Name
	social_name = social.iloc[:, 0].values

	# Extracting Social Link
	social_link = social.iloc[:, 1].values

	# Extracting Social Links Text
	social_text = social.iloc[:, 2].values

	# Extracting Social Logo path
	social_logo_path = social.iloc[:, 3].values

	return np.asarray([social_name, social_link, social_text, social_logo_path])

def get_ml_projects_data(ml_projects):
    # Sorting to bring recent projects at the beginning
    ml_projects.sort_values(by = 'Order', ascending = False, inplace = True)

    return ml_projects.T.values