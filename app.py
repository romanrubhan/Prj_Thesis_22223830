from flask import Flask, render_template, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_questions_endpoint')
def load_questions():
    # Run the qn_gen.py script and capture its output
    qn_gen_output = subprocess.check_output(["python", "C:/Users/Zeus/pythonProject/Iris_Tracking/Qn/qn_gen.py"], text=True)

    # Process the qn_gen_output to extract questions
    questions = []
    for line in qn_gen_output.strip().split('\n'):
        if line.startswith("Question"):
            questions.append(line.split(": ", 1)[1])

    return jsonify(questions)

@app.route('/run_iris_detection_endpoint')
def run_iris_detection():
    # Execute the iris detection script (c.py) and return the output as plain text
    iris_output = subprocess.check_output(["python", "C:/Users/Zeus/pythonProject/Iris_Tracking/c.py"], text=True)
    return iris_output


if __name__ == '__main__':
    app.run(debug=True)
