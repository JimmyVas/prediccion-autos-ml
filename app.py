from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)

# ===============================
# CARGAR MODELO
# ===============================

modelo = pickle.load(open("modelo_autos.pkl", "rb"))
columnas = pickle.load(open("columnas_modelo.pkl", "rb"))

# ===============================
# OBTENER MARCAS
# ===============================

marcas = [col.replace("Make_", "") for col in columnas if col.startswith("Make_")]
marcas.sort()

# ===============================
# IMPORTANCIA DEL MODELO
# ===============================

importancias = modelo.feature_importances_

df_importancia = pd.DataFrame({
    "feature": columnas,
    "importance": importancias
})

df_importancia = df_importancia.sort_values(by="importance", ascending=False).head(10)

plt.figure(figsize=(8,5))
plt.barh(df_importancia["feature"], df_importancia["importance"])
plt.xlabel("Importancia")
plt.title("Variables más importantes del modelo")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("static/importance.png")
plt.close()

# ===============================
# HOME
# ===============================

@app.route("/")
def home():
    return render_template("index.html", marcas=marcas)

# ===============================
# PREDICCION
# ===============================

@app.route("/predecir", methods=["POST"])
def predecir():

    make = request.form["make"]
    hp = float(request.form["hp"])
    cylinders = int(request.form["cylinders"])
    mpg = float(request.form["mpg"])
    age = int(request.form["age"])
    popularity = float(request.form["popularity"])
    doors = int(request.form["doors"])

    datos = {
        "Engine HP": hp,
        "Engine Cylinders": cylinders,
        "Number of Doors": doors,
        "highway MPG": mpg,
        "Popularity": popularity,
        "Vehicle_Age": age
    }

    df = pd.DataFrame([datos])

    df["Make_" + make] = 1

    df = df.reindex(columns=columnas, fill_value=0)

    prediccion = modelo.predict(df)[0]

    precio = "{:,.0f}".format(prediccion)

    return render_template(
        "index.html",
        prediction_text=f"${precio}",
        make=make,
        hp=hp,
        cylinders=cylinders,
        doors=doors,
        mpg=mpg,
        popularity=popularity,
        age=age,
        marcas=marcas
    )


if __name__ == "__main__":
    app.run(debug=True)