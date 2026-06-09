from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
try:
    model = pickle.load(open('crop_model.pkl', 'rb'))
except FileNotFoundError:
    model = None
    print("Error: crop_model.pkl not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model file not found. Run train_model.py first."
    try:
        features = [[
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity'])
        ]]
        prediction = model.predict(features)
        return render_template(
            'index.html',
            prediction_text=f"Recommended Crop: {prediction[0]}"
        )
    except Exception as e:
        return f"Error: {e}"
if __name__ == '__main__':
    app.run(debug=True)