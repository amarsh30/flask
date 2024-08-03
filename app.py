from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Memuat model dari file .pkl
model = joblib.load('model.pkl')

# Mapping untuk label yang telah diencode
label_dict = {
    1: 'Bawang Merah',
    2: 'Buncis',
    3: 'Jagung',
    4: 'Jeruk',
    5: 'Kembang Kol',
    6: 'Padi',
    7: 'Pisang',
    8: 'Tomat'
}

# Fungsi untuk mengubah soiltype menjadi numerik
soil_type_dict = {
    'Lempung': 0.5,
    'Berpasir': 0,
    'Tanah Liat': 1,
    'Hitam': 0.25,
    'Merah': 0.75
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Menyiapkan input untuk model
        user_data = {
            'Nitrogen': data['Nitrogen'],
            'Phosporous': data['Phosporous'],
            'Potassium': data['Potassium'],
            'temperature': data['temperature'],
            'humidity': data['humidity'],
            'ph': data['ph'],
            'rainfall': data['rainfall'],
            'soil_type_encode': soil_type_dict[data['soil_type_encode']]
        }
        
        # Konversi input ke DataFrame
        user_input_df = pd.DataFrame([user_data])
        
        # Prediksi
        prediction = model.predict(user_input_df)
        prediction_categorical = label_dict.get(prediction[0], 'Unknown')
        
        return jsonify({'prediction': prediction_categorical})
    
    except KeyError as e:
        return jsonify({'error': f'Missing or invalid input: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
