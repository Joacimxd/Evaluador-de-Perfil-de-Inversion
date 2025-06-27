import tkinter as tk
from tkinter import messagebox
import joblib 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

tree = joblib.load('decision_tree_model.joblib')

dataframe = pd.read_csv("perfil_inversion.csv")
#Preprocesamiento (Transformar valores categóricos a numéricos)
columnas = dataframe.columns[:-1].to_list()
columnas = [col for col in columnas if col != 'registro']
cat_cols = ['tiempo_inversion', 'conocimiento', 'experiencia', 'tolerancia_perdida', 'reaccion_ganancia'] #columnas no numericas

#Obtained from the k-means research
stock_clusters = [['BBBY', 'FV=F', 'NKLA', 'NVTA', 'PACB', 'PEAK', 'PXD', 'SQ', 'VMEX.MX', 'YNDX'], 
                 ['AMC'], 
                 ['BFLY', 'BIRD', 'FSLY', 'LCID', 'NVCR', 'SPCE', 'VERV', 'WBA', 'ZI'], 
                 ['BIDU', 'DGE.L', 'EDIT', 'EL', 'GIS', 'GOVZ', 'HRL', 'HSY', 'HUDI', 'MCD', 'NESN.SW', 'NTLA', 'SDGR', 'SJM', 'SPTL', 'TLT', 'U', 'UB=F', 'ZB=F', 'ZC=F', 'ZN=F'], 
                 ['AI', 'ALGN', 'BEAM', 'BIGC', 'CAG', 'CCI', 'CPB', 'CRSP', 'ETSY', 'GOVT', 'HIVE', 'IEF', 'JNJ', 'KHC', 'MAA', 'MBB', 'MDLZ', 'O', 'PATH', 'PEP', 'PLD', 'RBLX', 'RIOT', 'RIVN', 'SNOW', 'TSLA', 'TWLO', 'VGIT', 'VMBS'], 
                 ['AAPL', 'ADA-USD', 'AGG', 'AMT', 'ARE', 'BIV', 'BND', 'CVX', 'DDOG', 'DOCN', 'DVN', 'ENB', 'EQIX', 'EQR', 'FVRR', 'GME', 'HST', 'INTC', 'IQ', 'ISHG', 'KMB', 'KO', 'LOW', 'LQD', 'NDAQ', 'NET', 'OXY', 'PYPL', 'ROKU', 'SHOP', 'SLB', 'SOFI', 'SPAB', 'TIP', 'TOST', 'TROW', 'TRP', 'UDR', 'UNH', 'UPST', 'VCIT', 'VCLT', 'VNQ'], 
                 ['ABNB', 'AVB', 'BKR', 'BLK', 'BNDX', 'BSV', 'BXP', 'CHD', 'CME', 'COP', 'EOG', 'ESS', 'FRT', 'HAL', 'HD', 'HES', 'IJR', 'IWM', 'KR', 'MARA', 'MO', 'MS', 'MSFT', 'PG', 'PLTR', 'PM', 'TFC', 'TSN', 'USB', 'VEA', 'VTR', 'VWO', 'W', 'XLE', 'XLU', 'XLV', 'XLY', 'XOM', 'ZS'], 
                 ['AFRM', 'AMZN', 'BAC', 'C', 'CL', 'DIA', 'DLR', 'DOGE-USD', 'EFA', 'ETH-USD', 'GOOGL', 'HYG', 'ICE', 'IGSB', 'IJH', 'KMI', 'MA', 'META', 'PNC', 'QQQ', 'SCHW', 'SHY', 'V', 'VCSH', 'VLO', 'VT', 'WELL', 'WMT', 'XLK'], 
                 ['AIG', 'ALL', 'AXP', 'BK', 'BTC-USD', 'COIN', 'COST', 'CRWD', 'ET', 'FANG', 'GS', 'IVV', 'JPM', 'MET', 'MPC', 'NVDA', 'PRU', 'PSX', 'SPG', 'SPSB', 'SPY', 'VOO', 'VTI', 'WFC', 'WMB', 'XLF', 'XLI'], 
                 ['SOL-USD']]


encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    dataframe[col] = le.fit_transform(dataframe[col])
    encoders[col] = le  # Guardas el encoder para usarlo después

#definicion de las asociaciones entre preguntas y columnas
preguntas = [
    {
        "columna": "edad",
        "pregunta": "¿Cuál es tu edad?",
        "tipo": "numerico"
    },
    {
        "columna": "ingreso",
        "pregunta": "¿Cuál es tu ingreso mensual aproximado (en pesos)?",
        "tipo": "numerico"
    },
    {
        "columna": "dependientes",
        "pregunta": "¿Cuántas personas dependen económicamente de ti?",
        "tipo": "numerico"
    },
    {
        "columna": "fuentes_ingreso",
        "pregunta": "¿Cuántas fuentes de ingreso diferentes tienes actualmente? (máximo 3)",
        "tipo": "numerico"
    },
    {
        "columna": "tiempo_inversion",
        "pregunta": "¿Por cuánto tiempo planeas mantener una inversión? (años)",
        "tipo": "categorico",
        "opciones": ["Menos de 1", "1 a 3", "4 a 7", "Mas de 8"]
    },
    {
        "columna": "conocimiento",
        "pregunta": "¿Qué tanto sabes sobre temas de inversión y finanzas?",
        "tipo": "categorico",
        "opciones": ["Nada", "Poco", "Intermedio", "Avanzado"]
    },
    {
        "columna": "experiencia",
        "pregunta": "¿Has invertido alguna vez antes?",
        "tipo": "categorico",
        "opciones": ["Si", "No"]
    },
    {
        "columna": "tolerancia_perdida",
        "pregunta": "Si tu inversión pierde temporalmente un 20%, ¿cómo te sentirías?",
        "tipo": "categorico",
        "opciones": [
            "Muy incomodo",
            "Incomodo",
            "Neutral",
            "Tranquilo"
        ]
    },
    {
        "columna": "reaccion_ganancia",
        "pregunta": "Si obtuvieras una gran ganancia inicial, ¿qué harías?",
        "tipo": "categorico",
        "opciones": [
            "Retiraria",
            "Mantendria",
            "Incrementaria"
        ]
    }
]

columnas = [q["columna"] for q in preguntas]

# Identificar columnas categóricas
cat_cols = [q["columna"] for q in preguntas if q["tipo"] == "categorico"]
num_cols = [q["columna"] for q in preguntas if q["tipo"] == "numerico"]

#INTERFAZ

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluador de Perfil de Inversión")

        self.main_window()

    def main_window(self):
        self.clear_window()

        label = tk.Label(self.root, text="Bienvenido al Evaluador de Perfil", font=("Arial", 18))
        label.pack(pady=20)

        button = tk.Button(self.root, text="Ver mi perfil", font=("Arial", 14), command=self.open_questionnaire)
        button.pack(pady=10)

    def open_questionnaire(self):
        self.clear_window()

        self.answers = {}
        self.current_question = 0
        self.show_question()

    def show_question(self):
        if self.current_question >= len(preguntas):
            self.make_prediction()
            return

        self.clear_window()

        pregunta = preguntas[self.current_question]
        texto_pregunta = pregunta["pregunta"]

        label = tk.Label(self.root, text=texto_pregunta, font=("Arial", 16), wraplength=450, justify="center")
        label.pack(pady=20)

        if pregunta["tipo"] == "categorico":
            self.selected_option = tk.StringVar(master=self.root)
            for opt in pregunta["opciones"]:
                tk.Radiobutton(self.root, text=opt, variable=self.selected_option, value=opt, font=("Arial", 12)).pack(anchor='w', padx=40)
        else:
            self.entry = tk.Entry(self.root, font=("Arial", 14))
            self.entry.pack(pady=10)

        next_button = tk.Button(self.root, text="Siguiente", command=self.next_question, font=("Arial", 14))
        next_button.pack(pady=20)

    def next_question(self):
        pregunta = preguntas[self.current_question]
        col = pregunta["columna"]

        if pregunta["tipo"] == "categorico":
            respuesta = self.selected_option.get()
            if respuesta == "":
                messagebox.showwarning("Advertencia", "Por favor selecciona una opción")
                return
        else:
            respuesta = self.entry.get()
            if respuesta.strip() == "":
                messagebox.showwarning("Advertencia", "Por favor ingresa un valor")
                return
            try:
                respuesta = float(respuesta)
            except:
                messagebox.showerror("Error", "Por favor ingresa un número válido")
                return

        self.answers[col] = respuesta

        self.current_question += 1
        self.show_question()

    def make_prediction(self):
        prod_features = []

        for feat in columnas:
            if feat in cat_cols:
                try:
                    value = encoders[feat].transform([self.answers[feat]])[0]
                except Exception as e:
                    messagebox.showerror("Error", f"Error en la codificación de {feat}: {e}")
                    return
            else:
                value = self.answers[feat]
            prod_features.append(value)

        # Crear un dataframe con los datos capturados
        df_prod = pd.DataFrame([prod_features], columns=columnas)
        prediccion = tree.predict(df_prod)
        resultado = prediccion[0]
        self.show_result(resultado)

    def show_result(self, resultado):
        self.clear_window()

        label2 = tk.Label(self.root, text=f"Tu nivel de riesgo es: {resultado-1}", font=("Arial", 18))
        label = tk.Label(self.root, text=f"Las acciones recomendadas son:\n{', '.join(stock_clusters[resultado-1])}", font=("Arial", 18), wraplength=500, justify="left")
        label.pack(pady=20)
        label2.pack(pady=20)

        restart_button = tk.Button(self.root, text="Volver al inicio", command=self.main_window, font=("Arial", 14))
        restart_button.pack(pady=10)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("550x500")
    app = App(root)
    root.mainloop()
