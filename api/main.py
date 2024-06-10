from flask import Flask, request, render_template
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
plt.switch_backend('Agg')

# Fungsi logika fuzzy dan visualisasi
def fuzzy_logic(inputs):
    # Membuat semesta pembicaraan
    x_pregnancies = np.arange(0, 18, 1)
    x_glucose = np.arange(44, 233, 1)
    x_blood_pressure = np.arange(30, 123, 1)
    x_bmi = np.arange(18, 68, 1)
    x_dpf = np.arange(0.08, 2.30, 0.1)
    x_diagnosis = np.arange(0, 100, 1)

    def select_evenly_spaced(data, num_points=11):
        indices = np.linspace(0, len(data) - 1, num_points).astype(int)
        selected_data = data[indices]
        return np.array(selected_data)

    pregnant = select_evenly_spaced(x_pregnancies)
    bmi = select_evenly_spaced(x_bmi)
    dpf = select_evenly_spaced(x_dpf)
    glucose = select_evenly_spaced(x_glucose)
    bp = select_evenly_spaced(x_blood_pressure)

    # Membuat fungsi keanggotaan
    pregnancies_normal = fuzz.trapmf(x_pregnancies, pregnant[[0, 0, 3, 4]])
    pregnancies_sedang = fuzz.trimf(x_pregnancies, pregnant[[3, 5, 7]])
    pregnancies_tidak_normal = fuzz.trapmf(x_pregnancies, pregnant[[6, 7, 10, 10]])

    bmi_normal = fuzz.trapmf(x_bmi, bmi[[0, 0, 3, 4]])
    bmi_sedang = fuzz.trimf(x_bmi, bmi[[3, 5, 7]])
    bmi_tidak_normal = fuzz.trapmf(x_bmi, bmi[[6, 7, 10, 10]])

    glucose_normal = fuzz.trapmf(x_glucose, glucose[[0, 0, 3, 4]])
    glucose_sedang = fuzz.trimf(x_glucose, glucose[[3, 5, 7]])
    glucose_tidak_normal = fuzz.trapmf(x_glucose, glucose[[6, 7, 10, 10]])

    dpf_normal = fuzz.trapmf(x_dpf, dpf[[0, 0, 3, 4]])
    dpf_sedang = fuzz.trimf(x_dpf, dpf[[3, 5, 7]])
    dpf_tidak_normal = fuzz.trapmf(x_dpf, dpf[[6, 7, 10, 10]])

    bp_normal = fuzz.trapmf(x_blood_pressure, bp[[0, 0, 3, 4]])
    bp_sedang = fuzz.trimf(x_blood_pressure, bp[[3, 5, 7]])
    bp_tidak_normal = fuzz.trapmf(x_blood_pressure, bp[[6, 7, 10, 10]])

    DK = {
        "DK_BMI": fuzz.interp_membership(x_bmi, bmi_normal, inputs["BMI"]),
        "DK_BP": fuzz.interp_membership(x_blood_pressure, bp_normal, inputs["BloodPressure"]),
        "DK_Glucose": fuzz.interp_membership(x_glucose, glucose_normal, inputs["Glucose"]),
        "DK_Pregnancies": fuzz.interp_membership(x_pregnancies, pregnancies_normal, inputs["Pregnancies"]),
        "DK_DPF": fuzz.interp_membership(x_dpf, dpf_normal, inputs["DPF"]),
    }

    diagnosis_rendah = []
    diagnosis_tinggi = []

    for pregnant_val in ["normal", "sedang", "tidak_normal"]:
        for glucose_val in ["normal", "sedang", "tidak_normal"]:
            for bp_val in ["normal", "sedang", "tidak_normal"]:
                for bmi_val in ["normal", "sedang", "tidak_normal"]:
                    for dpf_val in ["normal", "sedang", "tidak_normal"]:
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
                            data = min(DK["DK_BMI"], DK["DK_BP"], DK["DK_Glucose"], DK["DK_Pregnancies"], DK["DK_DPF"])
                            diagnosis_tinggi.append(data)
                        else:
                            data = min(DK["DK_BMI"], DK["DK_BP"], DK["DK_Glucose"], DK["DK_Pregnancies"], DK["DK_DPF"])
                            diagnosis_rendah.append(data)

    diagnosis_rendah = max(diagnosis_rendah)
    diagnosis_tinggi = min(diagnosis_tinggi)

    diagnosis_low = fuzz.trapmf(x_diagnosis, [0, 0, 35, 65])
    diagnosis_high = fuzz.trapmf(x_diagnosis, [35, 65, 100, 100])

    diagnosis_rendah = np.fmin(diagnosis_rendah, diagnosis_low)
    diagnosis_tinggi = np.fmin(diagnosis_tinggi, diagnosis_high)

    diagnosis_0 = np.zeros_like(x_diagnosis)
    diagnosis_r = np.zeros_like(diagnosis_low)
    diagnosis_t = np.zeros_like(diagnosis_high)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 6))

    ax0.fill_between(x_diagnosis, diagnosis_r, diagnosis_rendah, facecolor='b', alpha=0.7)
    ax0.plot(x_diagnosis, diagnosis_low, 'b', linewidth=1.5, linestyle='--', label='Rendah')
    ax0.set_title('Diagnosis Rendah')
    ax0.legend()

    ax1.fill_between(x_diagnosis, diagnosis_t, diagnosis_tinggi, facecolor='g', alpha=0.7)
    ax1.plot(x_diagnosis, diagnosis_high, 'g', linewidth=1.5, linestyle='--', label='Tinggi')
    ax1.set_title('Diagnosis Tinggi')
    ax1.legend()

    ax2.fill_between(x_diagnosis, diagnosis_0, diagnosis_rendah, facecolor='b', alpha=0.7)
    ax2.plot(x_diagnosis, diagnosis_low, 'b', linewidth=1.5, linestyle='--', label='Rendah')
    ax2.fill_between(x_diagnosis, diagnosis_0, diagnosis_tinggi, facecolor='g', alpha=0.7)
    ax2.plot(x_diagnosis, diagnosis_high, 'g', linewidth=1.5, linestyle='--', label='Tinggi')
    ax2.set_title('Gabungan Diagnosis')
    ax2.legend()

    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

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
        plot_url = fuzzy_logic(inputs)
        return render_template('index.html', plot_url=plot_url)
    return render_template('index.html', plot_url=None)

if __name__ == '__main__':
    app.run(debug=True)
