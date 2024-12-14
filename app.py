from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import io
import base64

app = Flask(__name__)

# Memuat model yang sudah disimpan
model = joblib.load('cart_model.pkl')

# Halaman Utama - Klasifikasi Kanker Payudara
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Mengambil input dari formulir
            clump_thickness = int(request.form['clump_thickness'])
            uniformity_of_cell_size = int(request.form['uniformity_of_cell_size'])
            uniformity_of_cell_shape = int(request.form['uniformity_of_cell_shape'])
            marginal_adhesion = int(request.form['marginal_adhesion'])
            single_epithelial_cell_size = int(request.form['single_epithelial_cell_size'])
            bare_nuclei = int(request.form['bare_nuclei'])
            bland_chromatin = int(request.form['bland_chromatin'])
            normal_nucleoli = int(request.form['normal_nucleoli'])
            mitoses = int(request.form['mitoses'])

            # Convert input ke tipe data yang sesuai (angka)
            user_input = pd.DataFrame({
                'Clump_thickness': [clump_thickness],
                'Uniformity_of_cell_size': [uniformity_of_cell_size],
                'Uniformity_of_cell_shape': [uniformity_of_cell_shape],
                'Marginal_adhesion': [marginal_adhesion],
                'Single_epithelial_cell_size': [single_epithelial_cell_size],
                'Bare_nuclei': [bare_nuclei],
                'Bland_chromatin': [bland_chromatin],
                'Normal_nucleoli': [normal_nucleoli],
                'Mitoses': [mitoses]
            })

            # Melakukan prediksi
            prediction = model.predict(user_input)
            prediction_label = 'Jinak (2)' if prediction[0] == 2 else 'Ganas (4)'

            return render_template('index.html', result=prediction_label)

        except ValueError:
            return render_template('index.html', error="Pastikan semua input adalah angka yang valid!")

    return render_template('index.html')

# Halaman Visualisasi Tree
@app.route('/visualize_tree')
def visualize_tree():
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, filled=True, ax=ax)

    # Menyimpan gambar ke dalam buffer untuk ditampilkan
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode gambar ke base64 agar bisa ditampilkan di HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('visualize_tree.html', image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
