# app.py
import joblib
from flask import Flask, request, jsonify
from kafka import KafkaConsumer
import json

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.joblib')

# Kafka consumer setup
consumer = KafkaConsumer(
    'your_kafka_topic',
    bootstrap_servers=['kafka:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

@app.route('/')
def home():
    return "Hello, this is the home page!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
