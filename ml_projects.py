# Importing Flask essentials
from flask import Flask, request, render_template, redirect, url_for

# Importing Pandas for Data Reading
import pandas as pd

# Importing own modules
from py_modules import data_extraction as de

# Import pickle module to load models
import pickle

# Flask App
ml_app = Flask(__name__)

# Reading my social links data
social = pd.read_csv('data/social-media-links.csv')

# Extraction of social link data
social_data = de.get_social_link_data(social)

# Counting number of Data
number_of_links = len(social_data[0])

# Root or Home page - About Me
@ml_app.route('/')
def all_projects():
	# Reading ML_Projects data
	ml_projects = pd.read_csv('data/ML_Projects.csv')
	# Calculating number of projects
	number_of_projects = ml_projects.shape[0]
	# Calculating rows for UI
	rows = (number_of_projects // 3) + 1
	# Extracting Machine Learning Projects Data
	ml_projects_data = de.get_ml_projects_data(ml_projects)

	# Extracting Libraries links
	libraries = ml_projects_data[9]
	libraries_used = [[str(library) for library in rows.split('|')] for rows in libraries]

	# Extracting Technologies
	technologies = ml_projects_data[11]
	technologies_used = [[str(technology) for technology in rows.split('|')] for rows in technologies]

	# Extracting Frame works
	frameworks = ml_projects_data[10]
	frameworks_used = [[str(framework) for framework in rows.split('|')] for rows in frameworks]

	# Extracting Tools / IDE
	tools = ml_projects_data[12]
	tools_used = [[str(tool) for tool in rows.split('|')] for rows in tools]

	return render_template('all_projects.html',
                        social_data=social_data,
                        n = number_of_links,
                        ml_projects_data = ml_projects_data,
                        rows = rows,
                        libraries_used = libraries_used,
                        technologies_used = technologies_used,
                        frameworks_used = frameworks_used,
                        tools_used = tools_used)


# Loan Status Predictor
@ml_app.route('/loan_status_predictor', methods = ['POST', 'GET'])
def loan_status_predictor():

	if request.method == "POST":
		# Getting Input Data from UI
		credit_history = float(request.form['credit_history'])
		loan_amount = float(request.form['loan_amount'])
		applicant_income = float(request.form['applicant_income'])
		coapplicant_income = float(request.form['coapplicant_income'])
		dependents = float(request.form['dependents'])

		# Forming an input array
		input_array = [[credit_history, loan_amount, applicant_income, coapplicant_income, dependents]]

		# Loading Loan Status Predictor Model
		loan_status_predictor_model = pickle.load(open('models/loan_status_predictor.pkl', 'rb'))

		# Prediction
		status_predicted = loan_status_predictor_model.predict(input_array)

		# Predicting Probability
		predict_proba = loan_status_predictor_model.predict_proba(input_array) * 100
		return render_template('loan_status_predictor.html',
        	                social_data = social_data,
            	            n = number_of_links,
                	        status_predicted = status_predicted,
                	        predict_proba = predict_proba)
	return render_template('loan_status_predictor.html',
                        social_data = social_data,
                        n = number_of_links)



# Iris Species Classifier
@ml_app.route('/iris_species_classifier', methods = ['POST', 'GET'])
def iris_species_classifier():
	if request.method == 'POST':
		# Getting Data from UI
		petal_length = float(request.form['petal_length'])
		petal_width = float(request.form['petal_width'])

		# Forming an Input Array
		input_array = [[petal_length, petal_width]]

		# Loading Iris Species Classifier Model
		iris_species_classifier_model = pickle.load(open('models/iris_species_classifier.pkl', 'rb'))

		# Prediction
		iris_predicted = iris_species_classifier_model.predict(input_array)

		# Predicting Probaility
		predict_proba = iris_species_classifier_model.predict_proba(input_array) * 100

		return render_template('iris_species_classifier.html',
        	                social_data = social_data,
            	            n = number_of_links,
            	            iris_predicted = iris_predicted,
            	            predict_proba = predict_proba)
	return render_template('iris_species_classifier.html',
        	                social_data = social_data,
            	            n = number_of_links)



# Gender Classifier
@ml_app.route('/gender_classifier', methods = ['POST', 'GET'])
def gender_classifier():
	if request.method == 'POST':
		# Getting Data from UI
		height = float(request.form['height'])
		weight = float(request.form['weight'])

		# Forming an Input Array
		input_array = [[height, weight]]

		# Loading Gender Classifier Model
		gender_classifier_model = pickle.load(open('models/gender_classifier.pkl', 'rb'))

		# Prediction
		gender_predicted = gender_classifier_model.predict(input_array)

		# Predicting Probaility
		predict_proba = gender_classifier_model.predict_proba(input_array) * 100

		return render_template('gender_classifier.html',
        	                social_data = social_data,
            	            n = number_of_links,
            	            gender_predicted = gender_predicted,
            	            predict_proba = predict_proba)
	return render_template('gender_classifier.html',
        	                social_data = social_data,
            	            n = number_of_links)



# Weight Predictor
@ml_app.route('/weight_predictor', methods = ['POST', 'GET'])
def weight_predictor():
	if request.method == 'POST':
		# Getting Data from UI
		gender = float(request.form['gender'])
		height = float(request.form['height'])

		# Forming an Input Array
		input_array = [[height, gender]]

		# Loading Weight Predictor Model
		weight_predictor_model = pickle.load(open('models/weight_predictor.pkl', 'rb'))

		# Prediction
		weight_predicted = round(weight_predictor_model.predict(input_array)[0], 2)

		# Predicting Probaility
		# predict_proba = weight_predictor_model.predict_proba(input_array) * 100

		return render_template('weight_predictor.html',
        	                social_data = social_data,
            	            n = number_of_links,
            	            weight_predicted = weight_predicted,
            	            input_array = input_array)
	return render_template('weight_predictor.html',
        	                social_data = social_data,
            	            n = number_of_links)


# App Launcher
if __name__ == '__main__':
	ml_app.run(debug = True, port = 3000)