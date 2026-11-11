
import csv
import os
import subprocess
import time

BASE = "/sys/bus/i2c/devices/6-0040/iio:device0/"
CARPETA = "results/energia"
PRECISION = "fp32"
N = 10
REPS = 7000000
MEDICIONES = 5
SEMILLA = 12345
INTERVALO = 0.1

if not os.path.isdir(CARPETA):
    os.makedirs(CARPETA)


def leer_potencias():
    valores = []

    for canal in range(3):
        ruta = BASE + "in_power{}_input".format(canal)

        with open(ruta) as sensor:
            valores.append(float(sensor.read().strip()))

    return valores


def calcular_energia(muestras, canal):
    energia = 0.0

    for i in range(1, len(muestras)):
        anterior = muestras[i - 1]
        actual = muestras[i]

        dt = actual[0] - anterior[0]

        energia += (
            (anterior[canal] + actual[canal]) / 2.0
        ) * dt

    return energia


def comando_softmax(reps):
    return [
        "./softmax_cpu",
        "--precision", PRECISION,
        "--n", str(N),
        "--reps", str(reps),
        "--trials", "1",
        "--seed", str(SEMILLA),
        "--csv", ""
    ]


# Calentamiento previo
subprocess.run(
    comando_softmax(1000),
    check=True,
    stdout=subprocess.DEVNULL
)

resumen = os.path.join(
    CARPETA,
    "resumen_{}_N{}.csv".format(PRECISION, N)
)

with open(resumen, "w", newline="") as archivo_resumen:
    writer_resumen = csv.writer(archivo_resumen)

    writer_resumen.writerow([
        "precision",
        "n",
        "medicion",
        "reps",
        "duracion_s",
        "muestras",
        "intervalo_medido_s",
        "potencia_media_modulo_W",
        "potencia_media_cpu_W",
        "energia_modulo_mJ",
        "energia_cpu_mJ",
        "energia_cpu_por_llamada_uJ",
        "energia_cpu_por_elemento_uJ"
    ])

    for medicion in range(1, MEDICIONES + 1):

        nombre = os.path.join(
            CARPETA,
            "potencia_{}_N{}_m{}.csv".format(
                PRECISION, N, medicion
            )
        )

        muestras = []

        inicio = time.monotonic()

        proceso = subprocess.Popen(
            comando_softmax(REPS),
            stdout=subprocess.DEVNULL
        )

        while proceso.poll() is None:
            valores = leer_potencias()
            tiempo = time.monotonic() - inicio

            muestras.append([tiempo] + valores)

            time.sleep(INTERVALO)

        codigo = proceso.wait()
        duracion = time.monotonic() - inicio

        if codigo != 0:
            raise RuntimeError(
                "Fallo softmax en medicion {}".format(
                    medicion
                )
            )

        if len(muestras) < 2:
            raise RuntimeError(
                "Muestras insuficientes"
            )

        with open(nombre, "w", newline="") as archivo:
            writer = csv.writer(archivo)

            writer.writerow([
                "tiempo_s",
                "potencia_modulo_mW",
                "potencia_gpu_mW",
                "potencia_cpu_mW"
            ])

            writer.writerows(muestras)

        intervalo = muestras[-1][0] - muestras[0][0]

        energia_modulo = calcular_energia(muestras, 1)
        energia_cpu = calcular_energia(muestras, 3)

        potencia_modulo = energia_modulo / intervalo / 1000.0
        potencia_cpu = energia_cpu / intervalo / 1000.0

        energia_llamada = energia_cpu * 1000.0 / REPS
        energia_elemento = energia_llamada / N

        writer_resumen.writerow([
            PRECISION, N, medicion, REPS,
            duracion, len(muestras), intervalo,
            potencia_modulo, potencia_cpu,
            energia_modulo, energia_cpu,
            energia_llamada, energia_elemento
        ])

        archivo_resumen.flush()

        print(
            "Medicion {} | duracion={:.3f}s | "
            "muestras={} | CPU={:.3f}W | "
            "energia_CPU={:.3f}mJ".format(
                medicion,
                duracion,
                len(muestras),
                potencia_cpu,
                energia_cpu
            ),
            flush=True
        )

print("\nResultados guardados en:", CARPETA)
