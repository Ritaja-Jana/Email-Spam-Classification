import os
from flask import Flask, request, render_template
import joblib
import pandas as pd
import re

app = Flask(__name__)

# Determine the path to the models directory relative to current script
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_path = os.path.join(models_dir, 'spam_classifier_model.pkl')

# Load the trained model
model = joblib.load(model_path)

# Define a function to process the email input and extract features
def calculate_capital_run_lengths(email_content):
    capital_sequences = re.findall(r'[A-Z]+', email_content)
    
    if not capital_sequences:
        return 0, 0, 0
    
    capital_run_lengths = [len(seq) for seq in capital_sequences]
    
    capital_run_length_average = sum(capital_run_lengths) / len(capital_run_lengths)
    capital_run_length_longest = max(capital_run_lengths)
    capital_run_length_total = sum(capital_run_lengths)
    
    return capital_run_length_average, capital_run_length_longest, capital_run_length_total

def process_input(email_content):
    capital_run_length_average, capital_run_length_longest, capital_run_length_total = calculate_capital_run_lengths(email_content)
    
    features = {
        'word_freq_make': email_content.lower().count('make'),
        'word_freq_address': email_content.lower().count('address'),
        'word_freq_all': email_content.lower().count('all'),
        'word_freq_3d': email_content.lower().count('3d'),
        'word_freq_our': email_content.lower().count('our'),
        'word_freq_over': email_content.lower().count('over'),
        'word_freq_remove': email_content.lower().count('remove'),
        'word_freq_internet': email_content.lower().count('internet'),
        'word_freq_order': email_content.lower().count('order'),
        'word_freq_mail': email_content.lower().count('mail'),
        'word_freq_receive': email_content.lower().count('receive'),
        'word_freq_will': email_content.lower().count('will'),
        'word_freq_people': email_content.lower().count('people'),
        'word_freq_report': email_content.lower().count('report'),
        'word_freq_addresses': email_content.lower().count('addresses'),
        'word_freq_free': email_content.lower().count('free'),
        'word_freq_business': email_content.lower().count('business'),
        'word_freq_email': email_content.lower().count('email'),
        'word_freq_you': email_content.lower().count('you'),
        'word_freq_credit': email_content.lower().count('credit'),
        'word_freq_your': email_content.lower().count('your'),
        'word_freq_font': email_content.lower().count('font'),
        'word_freq_000': email_content.lower().count('000'),
        'word_freq_money': email_content.lower().count('money'),
        'word_freq_hp': email_content.lower().count('hp'),
        'word_freq_hpl': email_content.lower().count('hpl'),
        'word_freq_george': email_content.lower().count('george'),
        'word_freq_650': email_content.lower().count('650'),
        'word_freq_lab': email_content.lower().count('lab'),
        'word_freq_labs': email_content.lower().count('labs'),
        'word_freq_telnet': email_content.lower().count('telnet'),
        'word_freq_857': email_content.lower().count('857'),
        'word_freq_data': email_content.lower().count('data'),
        'word_freq_415': email_content.lower().count('415'),
        'word_freq_85': email_content.lower().count('85'),
        'word_freq_technology': email_content.lower().count('technology'),
        'word_freq_1999': email_content.lower().count('1999'),
        'word_freq_parts': email_content.lower().count('parts'),
        'word_freq_pm': email_content.lower().count('pm'),
        'word_freq_direct': email_content.lower().count('direct'),
        'word_freq_cs': email_content.lower().count('cs'),
        'word_freq_meeting': email_content.lower().count('meeting'),
        'word_freq_original': email_content.lower().count('original'),
        'word_freq_project': email_content.lower().count('project'),
        'word_freq_re': email_content.lower().count('re'),
        'word_freq_edu': email_content.lower().count('edu'),
        'word_freq_table': email_content.lower().count('table'),
        'word_freq_conference': email_content.lower().count('conference'),
        'char_freq_;': email_content.count(';'),
        'char_freq_(': email_content.count('('),
        'char_freq_[': email_content.count('['),
        'char_freq_!': email_content.count('!'),
        'char_freq_$': email_content.count('$'),
        'char_freq_#': email_content.count('#'),
        'capital_run_length_average': capital_run_length_average,
        'capital_run_length_longest': capital_run_length_longest,
        'capital_run_length_total': capital_run_length_total,
        'email_length': len(email_content)
    }

    return pd.DataFrame([features])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_content = request.form['email_content']

    input_data = process_input(email_content)
    prediction = model.predict(input_data)[0]
    prediction_text = 'Spam' if prediction == 1 else 'Not Spam'

    return render_template('index.html', prediction_text=f'The email is predicted to be: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
