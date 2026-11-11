# Ejercicio A: Softmax en CPU

## Descripción

Implementación de softmax en CPU para la plataforma NVIDIA Jetson Nano, utilizando dos variantes de precisión:

- FP16: almacenamiento de entrada y salida mediante `__fp16`. La exponencial y la reduccion se realizan en FP32.
- FP32: implementación con datos de tipo `float`.

Se utiliza una referencia en FP64 (`double`) para evaluar los errores numéricos.

## Requisitos

- NVIDIA Jetson Nano (arquitectura AArch64).
- Compilador g++ compatible con C++17.
- GNU Make.
- Python 3 para el procesamiento y la adquisicion energética.
- Sensor INA3221 accesible para las mediciones de potencia.

Las pruebas se realizaron con GCC 7.5.0 y una semilla fija de 12345.

## Compilación

Desde el directorio `CPU/`:

```bash
make

```

Comando equivalente:

```bash
g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm
```

Para eliminar el ejecutable:

```bash
make clean
```

## Ejecución

Ejemplo con FP16:

```bash
./softmax_cpu --precision fp16 --n 100 --reps 10000
```

Ejemplo con FP32:

```bash
./softmax_cpu --precision fp32 --n 100 --reps 10000
```

Para especificar cinco lotes y guardar resultados:

```bash
./softmax_cpu --precision fp32 --n 1000 --reps 10000 --trials 5 --seed 12345 --csv results/medicion_ejemplo.csv
```

Parametros utilizados:

- `--precision`: fp16 o fp32.
- `--n`: cantidad de elementos del vector.
- `--reps`: número de llamadas por lote.
- `--trials`: cantidad de lotes medidos.
- `--seed`: semilla para generar la entrada.
- `--csv`: archivo de salida.

## Metodología experimental

Se evaluaron FP16 y FP32 con N = 10, 100, 500 y 1000.

Para el perfilado de la función se utilizaron 10000 repeticiones por lote y cinco lotes medidos por configuración.

Los resultados se validaron frente a una referencia FP64, evaluando errores numéricos, suma de probabilidades, rango y posición del máximo.

Las mediciones del programa completo se registraron por separado.

## Medición de energía

La potencia se adquirió mediante el sensor INA3221 de la Jetson Nano.

Canales utilizados:

- `POM_5V_IN`: alimentación monitorizada del módulo.
- `POM_5V_GPU`: alimentación monitorizada de la GPU.
- `POM_5V_CPU`: alimentación monitorizada de la CPU.

Para ejecutar el script:

```bash
sudo python3 medir_energia.py
```

Antes de ejecutarlo, configurar `PRECISION`, `N` y `REPS` dentro del archivo.

| N | Repeticiones por medición |
|---|---:|
| 10 | 7000000 |
| 100 | 1000000 |
| 500 | 200000 |
| 1000 | 100000 |

Se realizaron cinco mediciones energéticas independientes por configuración, para un total de 40.

La potencia se muestreó aproximadamente cada 100 ms, más el tiempo de lectura del sensor.

La energía se calculó mediante integración trapezoidal de las muestras. Posteriormente, para estimar la energía de la duración completa del proceso, se utilizó:

E_estimada = P_media_muestreada * T_proceso

La energía por llamada se calculó dividiendo por el número de repeticiones.

### Limitaciones

- Las mediciones incluyen el consumo base del circuito monitorizado.
- La extrapolación supone que la potencia media medida representa el intervalo completo.
- Los canales de CPU y módulo no deben sumarse.
- El script requiere permisos suficientes para leer los sensores.
- Los resultados representan estimaciones experimentales y no la energía exclusiva de las operaciones de softmax.

## Resultados

Los resultados experimentales se encuentran en `results/`.

Archivos principales:

- `results/mediciones_cpu.csv`: tiempos de función y validación.
- `results/tiempos_programa_completo.csv`: tiempos del proceso completo.
- `results/hardware_config.txt`: información de hardware.
- `results/tegrastats_cpu_fp32_N1000.log`: monitoreo de CPU.
- `results/potencia_reposo.csv`: referencia de potencia en reposo.
- `results/energia/potencia_*.csv`: mediciones de potencia.
- `results/energia/resumen_fp16_N*.csv`: resultados FP16.
- `results/energia/resumen_fp32_N*.csv`: resultados FP32.
- `results/energia/resumen_final.csv`: resumen energético consolidado.

### Energía estimada del canal CPU por llamada

| N | FP16 (uJ) | FP32 (uJ) |
|---|---:|---:|
| 10 | 0.48492 | 0.37089 |
| 100 | 3.15523 | 2.83465 |
| 500 | 15.74315 | 14.08724 |
| 1000 | 30.21519 | 27.56144 |

En esta implementacion, FP32 obtuvo menores tiempos de ejecución y menores valores de energía estimada por llamada en los cuatro tamaños evaluados.
