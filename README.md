# Laboratorio Capítulo 4 - HLS

## Integrantes, curso y profesor

### Integrantes

1. Fiorela Chavarría Castrillo
2. Fabián Parreaguirre Hidalgo
3. Brayan Rodríguez Villalobos

### Curso

Introducción a la Computación Heterogénea - EL-5859

### Profesor

Dr. Luis G. León-Vega


## Descripción

Se implementa softmax en FP16 y FP32 para vectores de N = 10, 100, 500 y 1000 en la CPU y la GPU de una Jetson Nano y en la FPGA de una KV260, y se compara tiempo, energia y error contra una referencia en FP64. Todas las plataformas usan los mismos vectores de entrada (semilla 12345). El codigo de cada parte esta en su carpeta: `CPU/`, `GPU/` y `FPGA/`.

## Ejercicio A

### Descripción

Implementación de softmax en CPU para la plataforma NVIDIA Jetson Nano, utilizando dos variantes de precisión:

- FP16: almacenamiento de entrada y salida mediante `__fp16`. La exponencial y la reduccion se realizan en FP32.
- FP32: implementación con datos de tipo `float`.

Se utiliza una referencia en FP64 (`double`) para evaluar los errores numéricos.

### Requisitos

- NVIDIA Jetson Nano (arquitectura AArch64).
- Compilador g++ compatible con C++17.
- GNU Make.
- Python 3 para el procesamiento y la adquisicion energética.
- Sensor INA3221 accesible para las mediciones de potencia.

Las pruebas se realizaron con GCC 7.5.0 y una semilla fija de 12345.

### Compilación

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

### Ejecución

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

### Metodología experimental

Se evaluaron FP16 y FP32 con N = 10, 100, 500 y 1000.

Para el perfilado de la función se utilizaron 10000 repeticiones por lote y cinco lotes medidos por configuración.

Los resultados se validaron frente a una referencia FP64, evaluando errores numéricos, suma de probabilidades, rango y posición del máximo.

Las mediciones del programa completo se registraron por separado.

### Archivos de resultados

Los resultados experimentales se encuentran en `CPU/results/`.

- `CPU/results/mediciones_cpu.csv`: tiempos de función y validación.
- `CPU/results/tiempos_programa_completo.csv`: tiempos del proceso completo.
- `CPU/results/hardware_config.txt`: información de hardware.
- `CPU/results/tegrastats_cpu_fp32_N1000.log`: monitoreo de CPU.
- `CPU/results/potencia_reposo.csv`: referencia de potencia en reposo.
- `CPU/results/energia/potencia_*.csv`: mediciones de potencia.
- `CPU/results/energia/resumen_fp16_N*.csv`: resultados FP16.
- `CPU/results/energia/resumen_fp32_N*.csv`: resultados FP32.
- `CPU/results/energia/resumen_final.csv`: resumen energético consolidado.

## Ejercicio B

### Que hace el programa

`softmax_cuda.cu` calcula softmax en la GPU de la Jetson Nano, en FP16 (`__half`) y FP32 (`float`). En FP16 los datos se guardan en `__half` pero la exponencial y las sumas se hacen en float, igual que en la CPU.

El kernel hace las tres etapas: el maximo, la suma de `exp(x - max)` y la normalizacion. Se usa un solo bloque de 128 hilos (4 warps) y cada hilo recorre el vector de 128 en 128. Para el maximo y la suma se reduce primero dentro de cada warp con `__shfl_down_sync` y despues entre los warps con un arreglo en memoria compartida. La entrada y la salida estan en memoria global.

### Compilar y correr

Desde la carpeta `GPU/`. Se corrio en la Jetson Nano con CUDA 10.2 en modo MAXN (el mismo que se uso en la CPU).

```bash
make
# o
nvcc -O3 -std=c++14 -lineinfo -arch=sm_53 softmax_cuda.cu -o softmax_cuda

./softmax_cuda --precision fp16 --input inputs/vector_N100.csv --reps 10000 --measure kernel --measurements 5
./softmax_cuda --precision fp16 --input inputs/vector_N100.csv --reps 10000 --measure end-to-end --measurements 5
```

El N lo toma del csv de entrada. Primero hace un calentamiento, despues las 5 mediciones, y guarda los tiempos en `results_N<n>_<precision>_<modo>.csv`.

### Como se midio

Se usan los mismos vectores que en la CPU (semilla 12345). En modo `kernel` la entrada se copia una vez y se mide solo el kernel con eventos de CUDA. En `end-to-end` cada llamada copia la entrada, lanza el kernel, espera y copia la salida, y se mide con el reloj de la CPU. Para validar se compara contra una version en double y se revisa que la suma de 1.

Los tiempos end-to-end cambian bastante entre corridas. En la primera (`GPU/results/tiempos/`) dieron entre 243 y 360 us. La GPU de la Nano sube y baja la frecuencia sola (en tegrastats se ve pasar de 768 a 230 MHz), entonces en los resultados pusimos los tiempos de las mismas corridas de la energia. Los del kernel si dan casi iguales.

### Perfilador

Se corrio nvprof en las 16 combinaciones con 1000 repeticiones, porque con el perfilador va mas lento:

```bash
sudo /usr/local/cuda/bin/nvprof -u us --log-file nvprof_fp32_N1000_end-to-end.txt ./softmax_cuda --precision fp32 --input inputs/vector_N1000.csv --reps 1000 --measure end-to-end --measurements 5
```

Promedio por llamada en end-to-end, en us. Kernel y copias es tiempo de la GPU, lo demas son las llamadas en la CPU (cudaMemcpy se llama 2 veces por llamada):

| Precisión | N | Kernel | Copia a GPU | Copia de GPU | cudaMemcpy | Lanzar kernel | Sincronizar |
|---|---|---|---|---|---|---|---|
| FP32 | 10 | 24.8 | 1.2 | 2.3 | 70.4 | 48.0 | 58.0 |
| FP32 | 100 | 25.0 | 1.5 | 2.6 | 72.1 | 54.4 | 52.0 |
| FP32 | 500 | 28.1 | 1.8 | 2.3 | 66.1 | 53.0 | 53.8 |
| FP32 | 1000 | 24.0 | 1.6 | 1.7 | 58.7 | 54.9 | 24.7 |
| FP16 | 10 | 25.1 | 1.2 | 2.3 | 65.9 | 42.1 | 63.0 |
| FP16 | 100 | 16.9 | 0.9 | 1.7 | 63.9 | 54.6 | 39.1 |
| FP16 | 500 | 21.5 | 1.0 | 1.5 | 62.2 | 59.0 | 26.2 |
| FP16 | 1000 | 26.5 | 1.1 | 1.4 | 54.0 | 46.5 | 31.3 |

O sea la GPU trabaja unos 25 us en el kernel y menos de 3 us en las copias, y el resto es la CPU llamando a CUDA. En modo kernel el promedio dio entre 20 y 35 us, menos en FP16 con N=1000 que dio 69 us porque hubo kernels de hasta 142 us (otra vez por la frecuencia).

### Comentarios

- El kernel tarda casi lo mismo para todos los N (unos 23 us), con vectores tan chiquitos la GPU casi no tiene trabajo y lo que pesa es lanzar el kernel.
- End-to-end es como 7 veces mas lento que el kernel, y por el perfilador se ve que es por las llamadas de CUDA (copias, lanzar y esperar), no por la GPU.
- FP16 no salio mas rapido que FP32 porque las cuentas igual son en float, y tiene mas error.

Todo lo que salio esta en `GPU/results/`: los tiempos, la energia (resumen, muestras del sensor y logs de tegrastats) y las salidas de nvprof.

## Ejercicio C

## Ejercicio D

### CPU

La potencia se adquirió mediante el sensor INA3221 de la Jetson Nano.

Canales utilizados:

- `POM_5V_IN`: alimentación monitorizada del módulo.
- `POM_5V_GPU`: alimentación monitorizada de la GPU.
- `POM_5V_CPU`: alimentación monitorizada de la CPU.

Para ejecutar el script, desde `CPU/`:

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

Limitaciones:

- Las mediciones incluyen el consumo base del circuito monitorizado.
- La extrapolación supone que la potencia media medida representa el intervalo completo.
- Los canales de CPU y módulo no deben sumarse.
- El script requiere permisos suficientes para leer los sensores.
- Los resultados representan estimaciones experimentales y no la energía exclusiva de las operaciones de softmax.

En esta implementacion, FP32 obtuvo menores tiempos de ejecución y menores valores de energía estimada por llamada en los cuatro tamaños evaluados.

### GPU

Para la energia de la GPU usamos el mismo sensor que en la CPU, el INA3221. `GPU/medir_energia_gpu.py` corre el programa, lee el sensor cada 0.1 s y solo toma en cuenta las muestras de las 5 mediciones (no el arranque ni el calentamiento). La energia por llamada es la potencia promedio por el tiempo de la medicion entre las repeticiones, y la de cada elemento es esa entre N.

```bash
sudo python3 medir_energia_gpu.py --precision fp32 --n 1000 --measure end-to-end --reps 10000
```

En `kernel` se usaron 100000 repeticiones y en `end-to-end` 10000, para que cada medicion durara un par de segundos. Tambien se dejo corriendo tegrastats para tener las temperaturas (la GPU anduvo entre 30 y 38.5 C).

En los resultados la potencia y la energia de la GPU son del canal de la GPU, por llamada end-to-end. Si se cuenta todo el modulo la potencia fue de 3.2 a 3.3 W, unos 550-610 uJ por llamada. La energia por elemento baja bastante con N porque lo que cuesta cada llamada se reparte en mas elementos.

### FPGA

## Resultados

Mediana de 5 mediciones. En la CPU el tiempo es el de la funcion y en la GPU el del kernel, el end-to-end solo aplica a la GPU. La potencia y la energia son del canal de la CPU para la CPU y del canal de la GPU para la GPU (por llamada end-to-end).

FP16:

| Plataforma | N | Tiempo (us) | End-to-end (us) | Potencia (W) | Energía (uJ) | uJ/elem. | Error máx. |
|---|---|---|---|---|---|---|---|
| CPU | 10 | 0.495 | - | 0.975 | 0.485 | 0.0485 | 8.35e-5 |
| CPU | 100 | 3.416 | - | 0.905 | 3.155 | 0.0316 | 5.83e-5 |
| CPU | 500 | 16.357 | - | 0.929 | 15.743 | 0.0315 | 1.61e-5 |
| CPU | 1000 | 32.605 | - | 0.923 | 30.215 | 0.0302 | 9.99e-6 |
| GPU | 10 | 22.7 | 175.8 | 0.164 | 29.3 | 2.927 | 8.35e-5 |
| GPU | 100 | 22.7 | 176.9 | 0.164 | 29.0 | 0.290 | 5.83e-5 |
| GPU | 500 | 22.8 | 183.9 | 0.166 | 31.1 | 0.062 | 1.61e-5 |
| GPU | 1000 | 22.7 | 173.6 | 0.205 | 36.3 | 0.036 | 9.99e-6 |
| KV260 | 10 | - | - | - | - | - | - |
| KV260 | 100 | - | - | - | - | - | - |
| KV260 | 500 | - | - | - | - | - | - |
| KV260 | 1000 | - | - | - | - | - | - |

FP32:

| Plataforma | N | Tiempo (us) | End-to-end (us) | Potencia (W) | Energía (uJ) | uJ/elem. | Error máx. |
|---|---|---|---|---|---|---|---|
| CPU | 10 | 0.413 | - | 0.946 | 0.371 | 0.0371 | 2.12e-8 |
| CPU | 100 | 3.078 | - | 0.894 | 2.835 | 0.0283 | 2.52e-8 |
| CPU | 500 | 15.045 | - | 0.921 | 14.087 | 0.0282 | 8.73e-9 |
| CPU | 1000 | 30.050 | - | 0.900 | 27.561 | 0.0276 | 3.26e-9 |
| GPU | 10 | 22.8 | 174.5 | 0.154 | 24.5 | 2.447 | 2.12e-8 |
| GPU | 100 | 22.7 | 168.7 | 0.164 | 23.7 | 0.237 | 9.25e-9 |
| GPU | 500 | 22.6 | 178.8 | 0.164 | 31.9 | 0.064 | 1.64e-9 |
| GPU | 1000 | 30.2 | 173.5 | 0.205 | 35.6 | 0.036 | 1.39e-9 |
| KV260 | 10 | - | - | - | - | - | - |
| KV260 | 100 | - | - | - | - | - | - |
| KV260 | 500 | - | - | - | - | - | - |
| KV260 | 1000 | - | - | - | - | - | - |

## Especificaciones de los Sistemas

- Jetson Nano (`jetson-183`), AArch64, 4 nucleos de CPU a 1479 MHz y GPU Maxwell (`sm_53`) de hasta 921.6 MHz, en modo de potencia MAXN.
- GCC 7.5.0 y CUDA 10.2.

## Transparencia respecto al uso de IA
[ChatGPT](https://chatgpt.com/share/6af7d1f8-3984-83e8-9784-2eaa69c2c8fa)
[ChatGPT](https://chatgpt.com/share/6af7d218-ccd8-83e8-9211-068f93968d49)