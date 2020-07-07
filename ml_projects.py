# Importing Flask essentials
from flask import Flask, request, render_template, redirect, url_for

# Importing Pandas for Data Reading
import pandas as pd

# Importing own modules
from py_modules import data_extraction as de

# Flask App
ml_app = Flask(__name__)

# Root or Home page - About Me
@ml_app.route('/')
def all_projects():
	# Reading my social links data
	social = pd.read_csv('data/social-media-links.csv')

	# Extraction of social link data
	social_data = de.get_social_link_data(social)

	# Counting number of Data
	number_of_links = len(social_data[0])

	return render_template('all_projects.html', social_data=social_data, n = number_of_links)


# App Launcher
if __name__ == '__main__':
	ml_app.run(debug = True)