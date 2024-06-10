import skfuzzy as fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from flask import Flask, request, render_template
import io
import base64

app = Flask(__name__)
plt.switch_backend('Agg')

def select_evenly_spaced(data, num_points=11):
        # Tentukan posisi titik yang diinginkan
    indices = np.linspace(0, len(data) - 1, num_points).astype(int)
    selected_data = data[indices]
    return np.array(selected_data)

def fuzzy_logic(inputs):
    # Membuat semesta pembicaraan
    x_pregnancies = np.arange(0, 18, 1)
    x_glucose = np.arange(44, 233, 1)
    x_blood_pressure = np.arange(30, 123, 1)
    x_bmi = np.arange(18, 68, 1)
    x_dpf = np.arange(0.08, 2.30, 0.1)
    x_age = np.arange(0, 100, 1)
    x_diagnosis = np.arange(0, 100, 1)

    pregnant = select_evenly_spaced(x_pregnancies)
    bmi = select_evenly_spaced(x_bmi)
    dpf = select_evenly_spaced(x_dpf)
    glucose = select_evenly_spaced(x_glucose)
    bp = select_evenly_spaced(x_blood_pressure)

    # Membuat fungsi keanggotan
    # Membuat fungsi keanggotaan untuk Pregnancies
    pregnancies_normal = fuzz.trapmf(x_pregnancies, pregnant[[0, 0, 3, 4]])
    pregnancies_sedang = fuzz.trimf(x_pregnancies, pregnant[[3, 5, 7]])
    pregnancies_tidak_normal = fuzz.trapmf(x_pregnancies, pregnant[[6, 7, 10, 10]])

    # Membuat fungsi keanggotaan untuk BMI
    bmi_normal = fuzz.trapmf(x_bmi, bmi[[0, 0, 3, 4]])
    bmi_sedang = fuzz.trimf(x_bmi, bmi[[3, 5, 7]])
    bmi_tidak_normal = fuzz.trapmf(x_bmi, bmi[[6, 7, 10, 10]])

    # Membuat fungsi keanggotaan untuk Glucose
    glucose_normal = fuzz.trapmf(x_glucose, glucose[[0, 0, 3, 4]])
    glucose_sedang = fuzz.trimf(x_glucose, glucose[[3, 5, 7]])
    glucose_tidak_normal = fuzz.trapmf(x_glucose, glucose[[6, 7, 10, 10]])

    # Membuat fungsi keanggotaan untuk Diabetes Pedigree Function
    dpf_normal = fuzz.trapmf(x_dpf, dpf[[0, 0, 3, 4]])
    dpf_sedang = fuzz.trimf(x_dpf, dpf[[3, 5, 7]])
    dpf_tidak_normal = fuzz.trapmf(x_dpf, dpf[[6, 7, 10, 10]])

    # Membuat fungsi keanggotaan untuk Blood Pressure
    bp_normal = fuzz.trapmf(x_blood_pressure, bp[[0, 0, 3, 4]])
    bp_sedang = fuzz.trimf(x_blood_pressure, bp[[3, 5, 7]])
    bp_tidak_normal = fuzz.trapmf(x_blood_pressure, bp[[6, 7, 10, 10]])

    # Visualisasi fungsi keanggotaan untuk Pregnancies
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_pregnancies, pregnancies_tidak_normal, "r", linewidth=1.5, label="Tidak Normal"
    )
    plt.plot(x_pregnancies, pregnancies_normal, "g", linewidth=1.5, label="Normal")
    plt.plot(x_pregnancies, pregnancies_sedang, "b", linewidth=1.5, label="Sedang")
    plt.title("Fungsi Keanggotaan untuk Pregnancies")
    plt.xlabel("Jumlah Pregnancies")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualisasi fungsi keanggotaan untuk BMI
    plt.figure(figsize=(10, 6))
    plt.plot(x_bmi, bmi_tidak_normal, "r", linewidth=1.5, label="Tidak Normal")
    plt.plot(x_bmi, bmi_sedang, "g", linewidth=1.5, label="Sedang")
    plt.plot(x_bmi, bmi_normal, "b", linewidth=1.5, label="Normal")
    plt.title("Fungsi Keanggotaan untuk BMI")
    plt.xlabel("BMI")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualisasi fungsi keanggotaan untuk Glucose
    plt.figure(figsize=(10, 6))
    plt.plot(x_glucose, glucose_tidak_normal, "r", linewidth=1.5, label="Tidak Normal")
    plt.plot(x_glucose, glucose_sedang, "g", linewidth=1.5, label="Sedang")
    plt.plot(x_glucose, glucose_normal, "b", linewidth=1.5, label="Normal")
    plt.title("Fungsi Keanggotaan untuk Glukosa Plasma 2 Jam")
    plt.xlabel("Glucose")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualisasi fungsi keanggotaan untuk Diabetes Pedigree Function
    plt.figure(figsize=(10, 6))
    plt.plot(x_dpf, dpf_tidak_normal, "r", linewidth=1.5, label="Tidak Normal")
    plt.plot(x_dpf, dpf_sedang, "g", linewidth=1.5, label="Sedang")
    plt.plot(x_dpf, dpf_normal, "b", linewidth=1.5, label="Normal")
    plt.title("Fungsi Keanggotaan untuk Diabetes Pedigree Function")
    plt.xlabel("Diabetes Pedigree Function")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualisasi fungsi keanggotaan untuk Blood Pressure
    plt.figure(figsize=(10, 6))
    plt.plot(x_blood_pressure, bp_tidak_normal, "r", linewidth=1.5, label="Tidak Normal")
    plt.plot(x_blood_pressure, bp_sedang, "g", linewidth=1.5, label="Sedang")
    plt.plot(x_blood_pressure, bp_normal, "b", linewidth=1.5, label="Normal")
    plt.title("Fungsi Keanggotaan untuk Tekanan Darah Diastolik (mmHg)")
    plt.xlabel("Blood Pressure")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Membuat derajat keanggotaan
    def DerajatKeanggotaanBMI(x):
        normal = fuzz.interp_membership(x_bmi, bmi_normal, x)
        sedang = fuzz.interp_membership(x_bmi, bmi_sedang, x)
        tidak_normal = fuzz.interp_membership(x_bmi, bmi_tidak_normal, x)
        return normal, sedang, tidak_normal


    def DerajatKeanggotaanPregnant(x):
        normal = fuzz.interp_membership(x_pregnancies, pregnancies_normal, x)
        sedang = fuzz.interp_membership(x_pregnancies, pregnancies_sedang, x)
        tidak_normal = fuzz.interp_membership(x_pregnancies, pregnancies_tidak_normal, x)
        return normal, sedang, tidak_normal


    def DerajatKeanggotaanDPF(x):
        normal = fuzz.interp_membership(x_dpf, dpf_normal, x)
        sedang = fuzz.interp_membership(x_dpf, dpf_sedang, x)
        tidak_normal = fuzz.interp_membership(x_dpf, dpf_tidak_normal, x)
        return normal, sedang, tidak_normal


    def DerajatKeanggotaanBP(x):
        normal = fuzz.interp_membership(x_blood_pressure, bp_normal, x)
        sedang = fuzz.interp_membership(x_blood_pressure, bp_sedang, x)
        tidak_normal = fuzz.interp_membership(x_blood_pressure, bp_tidak_normal, x)
        return normal, sedang, tidak_normal


    def DerajatKeanggotaanGlucose(x):
        normal = fuzz.interp_membership(x_glucose, glucose_normal, x)
        sedang = fuzz.interp_membership(x_glucose, glucose_sedang, x)
        tidak_normal = fuzz.interp_membership(x_glucose, glucose_tidak_normal, x)
        return normal, sedang, tidak_normal



    DK = {
        "DK_BMI": DerajatKeanggotaanBMI(inputs["BMI"]),
        "DK_BP": DerajatKeanggotaanBP(inputs["BloodPressure"]),
        "DK_Glucose": DerajatKeanggotaanGlucose(inputs["Glucose"]),
        "DK_Pregnancies": DerajatKeanggotaanPregnant(inputs["Pregnancies"]),
        "DK_DPF": DerajatKeanggotaanDPF(inputs["DPF"]),
    }
    rules = []
    diagnosis_tinggi = []
    diagnosis_rendah = []
    for pregnant_val in ["normal", "sedang", "tidak_normal"]:
        for glucose_val in ["normal", "sedang", "tidak_normal"]:
            for bp_val in ["normal", "sedang", "tidak_normal"]:
                for bmi_val in ["normal", "sedang", "tidak_normal"]:
                    for dpf_val in ["normal", "sedang", "tidak_normal"]:
                        rule = f"If Hamil is {pregnant_val.capitalize()} and Glucose is {glucose_val.capitalize()} and Tekdarah is {bp_val.capitalize()} and BMI is {bmi_val.capitalize()} and Riwayat is {dpf_val.capitalize()}"
                        if pregnant_val == "normal":
                            i_pregnant = 0
                        elif pregnant_val == "sedang":
                            i_pregnant = 1
                        else:
                            i_pregnant = 2
                        
                        if glucose_val == "normal":
                            i_glucose = 0
                        elif glucose_val == "sedang":
                            i_glucose = 1
                        else:
                            i_glucose = 2
                        
                        if bp_val == "normal":
                            i_bp = 0
                        elif bp_val == "sedang":
                            i_bp = 1
                        else:
                            i_bp = 2
                        
                        if bmi_val == "normal":
                            i_bmi = 0
                        elif bmi_val == "sedang":
                            i_bmi = 1
                        else:
                            i_bmi = 2
                        
                        if dpf_val == "normal":
                            i_dpf = 0
                        elif dpf_val == "sedang":
                            i_dpf = 1
                        else:
                            i_dpf = 2
                        
                        if(i_dpf+i_bmi+i_bp+i_glucose+i_pregnant >= 6):
                            data = min(DK["DK_BMI"][i_bmi], DK["DK_BP"][i_bp], DK["DK_Glucose"][i_glucose], DK["DK_Pregnancies"][i_pregnant], DK["DK_DPF"][i_dpf])
                            diagnosis_tinggi.append(data)
                        else:
                            data = min(DK["DK_BMI"][i_bmi], DK["DK_BP"][i_bp], DK["DK_Glucose"][i_glucose], DK["DK_Pregnancies"][i_pregnant], DK["DK_DPF"][i_dpf])
                            diagnosis_rendah.append(data)
                        

    # print("Diagnosis Rendah: ", diagnosis_rendah)
    # print("Diagnosis Tinggi: ", diagnosis_tinggi)
    diagnosis_rendah = max(diagnosis_rendah)
    diagnosis_tinggi = min(diagnosis_tinggi)

    # Membuat fungsi keanggotaan untuk diagnosis rendah dan tinggi
    diagnosis_low = fuzz.trapmf(x_diagnosis, [0, 0, 35, 65])
    diagnosis_high = fuzz.trapmf(x_diagnosis, [35, 65, 100, 100])

    plt.plot(x_diagnosis, diagnosis_low, 'b', linewidth=1.5, label='Rendah')
    plt.plot(x_diagnosis, diagnosis_high, 'g', linewidth=1.5, label='Tinggi')
    plt.legend(loc='center right')
    plt.show()

    diagnosis_rendah = np.fmin(diagnosis_rendah, diagnosis_low)
    diagnosis_tinggi = np.fmin(diagnosis_tinggi, diagnosis_high)

    diagnosis_0 = np.zeros_like(x_diagnosis)
    diagnosis_r = np.zeros_like(diagnosis_low)
    diagnosis_t = np.zeros_like(diagnosis_high)




    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 6))

    ax0.fill_between(x_diagnosis, diagnosis_r, diagnosis_rendah, facecolor='b', alpha=0.7)
    ax0.plot(x_diagnosis, diagnosis_low, 'b', linewidth=1.5, linestyle='--', label='Rendah')
    ax0.set_title('Diagnosis')
    ax0.legend()
    # Fill Between dan Plot untuk nilai kelayakan tinggi
    ax1.fill_between(x_diagnosis,diagnosis_t, diagnosis_tinggi, facecolor='g', alpha=0.7)
    ax1.plot(x_diagnosis, diagnosis_high, 'g', linewidth=1.5, linestyle='--', label='Tinggi')
    ax1.set_title('Diagnosis')
    ax1.legend()

    ax2.fill_between(x_diagnosis, diagnosis_0, diagnosis_rendah, facecolor='b', alpha=0.7)
    ax2.plot(x_diagnosis, diagnosis_low, 'b', linewidth=1.5, linestyle='--', label='Rendah')
    ax2.fill_between(x_diagnosis, diagnosis_0, diagnosis_tinggi, facecolor='g', alpha=0.7)
    ax2.plot(x_diagnosis, diagnosis_high, 'g', linewidth=1.5, linestyle='--', label='Tinggi')
    ax2.set_title('Diagnosis')
    ax2.legend()

    # Matikan axes atas dan kanan
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Rapikan layout
    plt.tight_layout()
    plt.show()

    # Komposisi antara diagnosis rendah dan tinggi
    komposisi = np.fmax(diagnosis_rendah, diagnosis_tinggi)

    # Hasil Defuzifikasi
    diagnosis = fuzz.defuzz(x_diagnosis, komposisi, 'centroid')
    if(diagnosis <= 50):
        result = "Negatif"
    else:
        result = "Positif"

    # Nilai Fuzzy Untuk Membership Function
    diagnosis_defuz = fuzz.interp_membership(x_diagnosis, komposisi, diagnosis)

    # Visualisasi Hasil Defuzifikasi
    fig, ax0 = plt.subplots(figsize=(8, 3))
    ax0.plot(x_diagnosis, diagnosis_low, 'b', linewidth=1.5, linestyle='--', label='Rendah')
    ax0.plot(x_diagnosis, diagnosis_high, 'g', linewidth=1.5, linestyle='--', label='Tinggi')
    ax0.fill_between(x_diagnosis, diagnosis_0, komposisi, facecolor='Orange', alpha=0.7)
    ax0.plot([diagnosis, diagnosis], [0, diagnosis_defuz], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Diagnosis')
    ax0.legend()
    plt.show()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url, result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        inputs = {
            "Pregnancies": int(request.form['Pregnancies']),
            "Glucose": int(request.form['Glucose']),
            "BloodPressure": int(request.form['BloodPressure']),
            "BMI": float(request.form['BMI']),
            "DPF": float(request.form['DPF'])
        }
        plot_url, result = fuzzy_logic(inputs)
        return render_template('index.html', plot_url=plot_url, result=result)
    return render_template('index.html', plot_url=None, result=None)

if __name__ == '__main__':
    app.run(debug=True)
# print(fuzzy_logic({"Pregnancies": 0, "Glucose": 100, "BloodPressure": 100, "BMI": 30, "DPF": 1}))

