#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <getopt.h>
#include <random>
#include <chrono>
#include <cmath>
#include <climits>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>


#include <cuda_runtime.h>
#include <cuda_fp16.h>


enum class Precision    { FP16, FP32 };
enum class Measure      { Kernel, EndToEnd};

struct Config {
    Precision precision = Precision::FP16;  // valores por defecto
    int n    = 100;
    int reps = 10000;
    Measure measure = Measure::Kernel;
    std::string input_path;
    int measurements = 5;
};


// Tomado de laboratorio de Semana 7
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

// Convierte texto a entero positivo; termina el programa si es inválido.
static int parse_positive_int(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' ||
    value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "Valor invalido para --%s: '%s'\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;  // empieza con los defaults

    static struct option long_options[] = {
        {"precision", required_argument, 0, 'p'},
        {"n",         required_argument, 0, 'n'},
        {"reps",      required_argument, 0, 'r'},
        {"measure",   required_argument, 0, 'm'},
        {"input",     required_argument, 0, 'i'},
        {"measurements",required_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:n:r:m:i:t: ", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p':
                if (std::strcmp(optarg, "fp16") == 0)      cfg.precision = Precision::FP16;
                else if (std::strcmp(optarg, "fp32") == 0) cfg.precision = Precision::FP32;
                else {
                    std::fprintf(stderr, "Precision desconocida: '%s' (use fp16 o fp32)\n", optarg);
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 'n':
                cfg.n = parse_positive_int(optarg, "n");
                break;
            case 'r':
                cfg.reps = parse_positive_int(optarg, "reps");
                break;
            case 'm':
                if (std::strcmp(optarg, "kernel") == 0)         cfg.measure = Measure::Kernel;
                else if (std::strcmp(optarg, "end-to-end") == 0)        cfg.measure = Measure::EndToEnd;
                else {
                        std::fprintf(stderr,"Medición desconocida: '%s'\n",optarg);
                        std::exit(EXIT_FAILURE);
                }
                break;
            case 'i':
                cfg.input_path = optarg;
                break;
            case 't':
                cfg.measurements = parse_positive_int(optarg, "measurements");
                break;
            default:  // '?' : getopt_long ya imprimió el error
                std::exit(EXIT_FAILURE);
        }
    }
    if (cfg.input_path.empty()) {
    std::fprintf(stderr, "Debe especificar --input archivo.csv\n");
    std::exit(EXIT_FAILURE);
}
    return cfg;
}
std::vector<double> load_csv(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "No se pudo abrir el archivo: " + filename
        );
    }

    std::string line;

    // Validar encabezado.
    if (!std::getline(file, line) || line != "index,value") {
        throw std::runtime_error(
            "Encabezado CSV invalido en: " + filename
        );
    }

    std::vector<double> input;
    int expected_index = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream row(line);

        int index;
        char separator;
        double value;

        if (!(row >> index >> separator >> value) ||
            separator != ',') {
            throw std::runtime_error(
                "Fila CSV invalida: " + line
            );
        }

        row >> std::ws;

        if (!row.eof()) {
            throw std::runtime_error(
                "Columnas adicionales en: " + line
            );
        }

        if (index != expected_index) {
            throw std::runtime_error(
                "Indice incorrecto en: " + filename
            );
        }

        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Valor no finito en: " + filename
            );
        }

        input.push_back(value);
        expected_index++;
    }

    if (input.empty()) {
        throw std::runtime_error(
            "El archivo no contiene datos: " + filename
        );
    }

    return input;
}
std::vector<double> generate_input(int n) {
    std::vector<double> input(n);

    std::mt19937 gen(42);                                  // motor con semilla fija
    std::uniform_real_distribution<double> dist(-4.0, 4.0);  // completa los límites

    for (int i = 0; i < n ; ++i) {
        input[i] = dist(gen);
    }
    return input;
}
std::vector<float> softmax_fp32(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    // TODO
    return out;
}

template <typename T>
std::vector<double> run_softmax(
    const std::vector<double>& input,
    const Config& cfg,
    double& elapsed_ms
);
// Despachador: elige variante segun cfg y devuelve el resultado en double
// para poder compararlo con la referencia.
std::vector<double> softmax(
    const Config& cfg,
    const std::vector<double>& input,
    double& elapsed_ms
) {
    switch (cfg.precision) {
        case Precision::FP32:
            return run_softmax<float>(input, cfg, elapsed_ms);

        case Precision::FP16:
            return run_softmax<__half>(input, cfg,elapsed_ms);
    }

    return {};
}
constexpr int THREADS = 128;
constexpr unsigned FULL_MASK = 0xffffffffu;

// --- Conversiones: unico punto donde difiere el tipo de dato ---
__device__ __host__ inline float to_float(float v)  { return v; }
__device__ __host__ inline float to_float(__half v) { return __half2float(v); }

template <typename T> __device__ __host__ inline T from_float(float v);
template <> __device__ __host__ inline float  from_float<float>(float v)   { return v; }
template <> __device__ __host__ inline __half from_float<__half>(float v)  { return __float2half(v); }

// --- Reducciones por bloque (devuelven el resultado a TODOS los hilos) ---
__device__ float block_reduce_max(float val) {
    __shared__ float scratch[32];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    // 1. Reducción dentro de cada warp.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }

    // 2. El primer hilo de cada warp guarda su resultado.
    if (lane == 0) {
        scratch[warp_id] = val;
    }
    __syncthreads();

    // 3. El primer warp reduce los resultados parciales.
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        val = (lane < num_warps) ? scratch[lane] : -INFINITY;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
        }

        if (lane == 0) {
            scratch[0] = val;
        }
    }

    // 4. Todos los hilos reciben el máximo final.
    __syncthreads();
    return scratch[0];
}

__device__ float block_reduce_sum(float val) {
    __shared__ float scratch[32];

    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    // 1. Reducción dentro de cada warp.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }

    // 2. Guardar resultados parciales.
    if (lane == 0) {
        scratch[warp_id] = val;
    }
    __syncthreads();

    // 3. Reducción entre warps.
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        val = (lane < num_warps) ? scratch[lane] : 0.0f;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }

        if (lane == 0) {
            scratch[0] = val;
        }
    }

    // 4. Todos los hilos reciben la suma final.
    __syncthreads();
    return scratch[0];
}

template <typename T>
__global__ void softmax_kernel(const T* in, T* out, int n) {
    const int tid = threadIdx.x;

    // ETAPA 1: calcular el máximo local.
    float local_max = -INFINITY;

    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, to_float(in[i]));
    }

    const float m = block_reduce_max(local_max);

    // ETAPA 2: calcular la suma de exponenciales.
    float local_sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += expf(to_float(in[i]) - m);
    }

    const float s = block_reduce_sum(local_sum);

    // ETAPA 3: normalizar y almacenar.
    for (int i = tid; i < n; i += blockDim.x) {
        const float value = expf(to_float(in[i]) - m) / s;
        out[i] = from_float<T>(value);
    }
}

template <typename T>
std::vector<double> run_softmax(
    const std::vector<double>& input,
    const Config& cfg,
    double& elapsed_ms
) {
    const size_t bytes = static_cast<size_t>(cfg.n) * sizeof(T);

    // 1. Preparar los datos en el host.
    // Esta conversión queda fuera de las mediciones.
    std::vector<T> h_in(cfg.n);
    std::vector<T> h_out(cfg.n);

    for (int i = 0; i < cfg.n; ++i) {
        h_in[i] = from_float<T>(static_cast<float>(input[i]));
    }

    // 2. Reservar memoria en el dispositivo.
    T* d_in = nullptr;
    T* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // 3. Crear eventos CUDA.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const dim3 block(THREADS);
    const dim3 grid(1);

    double wall_time_ms = 0.0;
    float gpu_time_ms = 0.0f;

    if (cfg.measure == Measure::Kernel) {

        // La transferencia inicial no se incluye en la medición.
        CUDA_CHECK(cudaMemcpy(
            d_in, h_in.data(), bytes, cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Medición con CUDA Events.
        CUDA_CHECK(cudaEventRecord(start));

        const auto wall_start = std::chrono::steady_clock::now();

        for (int rep = 0; rep < cfg.reps; ++rep) {
            softmax_kernel<T><<<grid, block>>>(d_in, d_out, cfg.n);
        }

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        const auto wall_stop = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

        // Tiempo promedio por llamada al kernel.
        elapsed_ms = static_cast<double>(gpu_time_ms) / cfg.reps;

        wall_time_ms =
            std::chrono::duration<double, std::milli>(
                wall_stop - wall_start
            ).count();

        // Recuperar el resultado de la última repetición.
        CUDA_CHECK(cudaMemcpy(
            h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost
        ));


    } else {

        // Cada repetición incluye H2D + kernel + sincronización + D2H.
        const auto wall_start = std::chrono::steady_clock::now();

        for (int rep = 0; rep < cfg.reps; ++rep) {

            // Host -> Device.
            CUDA_CHECK(cudaMemcpy(
                d_in, h_in.data(), bytes, cudaMemcpyHostToDevice
            ));

            // Lanzamiento del kernel.
            softmax_kernel<T><<<grid, block>>>(d_in, d_out, cfg.n);
            CUDA_CHECK(cudaGetLastError());

            // Esperar a que termine el kernel.
            CUDA_CHECK(cudaDeviceSynchronize());

            // Device -> Host.
            CUDA_CHECK(cudaMemcpy(
                h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost
            ));
        }

        const auto wall_stop = std::chrono::steady_clock::now();

        wall_time_ms =
            std::chrono::duration<double, std::milli>(
                wall_stop - wall_start
            ).count();

        // Tiempo promedio por ejecución end-to-end.
        elapsed_ms = wall_time_ms / cfg.reps;


        std::printf("\nMedicion: end-to-end\n");
        std::printf("Tiempo total: %.6f ms\n", wall_time_ms);
        std::printf("Tiempo promedio por repeticion: %.6f ms\n",
                    wall_time_ms / cfg.reps);
    }

    // 4. Convertir el resultado al formato double para compararlo.
    std::vector<double> result(cfg.n);

    for (int i = 0; i < cfg.n; ++i) {
        result[i] = static_cast<double>(to_float(h_out[i]));
    }

    // 5. Liberar recursos.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return result;
}

std::vector<double> softmax_reference(
    const std::vector<double>& x
) {
    std::vector<double> out(x.size());

    const double max_value =
        *std::max_element(x.begin(), x.end());

    double sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - max_value);
        sum += out[i];
    }

    for (double& value : out) {
        value /= sum;
    }

    return out;
}

struct Statistics {
    double median;
    double minimum;
    double maximum;
};

Statistics calculate_statistics(std::vector<double> times) {
    std::sort(times.begin(), times.end());

    const size_t n = times.size();

    double median;

    if (n % 2 == 0) {
        median = (times[n / 2 - 1] + times[n / 2]) / 2.0;
    } else {
        median = times[n / 2];
    }

    return {
        median,
        times.front(),
        times.back()
    };
}

void write_raw_csv(
    const Config& cfg,
    const std::vector<double>& times
) {
    const std::string precision =
        cfg.precision == Precision::FP16 ? "fp16" : "fp32";

    const std::string measure =
        cfg.measure == Measure::Kernel ? "kernel" : "end-to-end";

    const std::string filename =
        "results_N" + std::to_string(cfg.n) +
        "_" + precision +
        "_" + measure + ".csv";

    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("No se pudo crear " + filename);
    }

    file << "file,n,precision,measure,measurement,reps,time_ms\n";

    for (size_t i = 0; i < times.size(); ++i) {
        file << cfg.input_path << ","
             << cfg.n << ","
             << precision << ","
             << measure << ","
             << (i + 1) << ","
             << cfg.reps << ","
             << std::fixed << std::setprecision(9)
             << times[i] << "\n";
    }
std::printf("Datos guardados en %s\n", filename.c_str());
}

int main(int argc, char* argv[]) {
    try {
        Config cfg = parse_args(argc, argv);

        // 1. Cargar vector desde CSV.
        const std::vector<double> input = load_csv(cfg.input_path);
        cfg.n = static_cast<int>(input.size());

        std::printf("Archivo: %s\n", cfg.input_path.c_str());
        std::printf("N: %d\n", cfg.n);
        std::printf("Precision: %s\n",
            cfg.precision == Precision::FP16 ? "FP16" : "FP32");
        std::printf("Medicion: %s\n",
            cfg.measure == Measure::Kernel ? "Kernel" : "End-to-end");
        std::printf("Llamadas por medicion: %d\n", cfg.reps);
        std::printf("Cantidad de mediciones: %d\n", cfg.measurements);

        // 2. Identificar GPU.
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::printf("GPU: %s (sm_%d%d)\n",
            prop.name, prop.major, prop.minor);

        // 3. Calentamiento.
        std::printf("\nEjecutando calentamiento...\n");

        double warmup_time = 0.0;
        std::vector<double> output =
            softmax(cfg, input, warmup_time);

        // 4. Mediciones independientes.
        std::vector<double> times;
        times.reserve(cfg.measurements);

        std::printf("\nIniciando mediciones...\n");

        for (int i = 0; i < cfg.measurements; ++i) {
            double elapsed_ms = 0.0;

            output = softmax(cfg, input, elapsed_ms);

            times.push_back(elapsed_ms);

            std::printf("Medicion %d: %.6f ms\n",
                i + 1, elapsed_ms);
        }

        // 5. Estadisticas.
        const Statistics stats = calculate_statistics(times);

        std::printf("\nEstadisticas:\n");
        std::printf("Mediana: %.6f ms\n", stats.median);
        std::printf("Minimo:  %.6f ms\n", stats.minimum);
        std::printf("Maximo:  %.6f ms\n", stats.maximum);

        // 6. Guardar mediciones crudas.
        write_raw_csv(cfg, times);

        std::printf("\nDatos guardados en raw_results.csv\n");

        // 7. Validacion.
        const std::vector<double> reference =
            softmax_reference(input);

        double max_error = 0.0;
        double output_sum = 0.0;

        for (int i = 0; i < cfg.n; ++i) {
            const double error =
                std::abs(output[i] - reference[i]);

            max_error = std::max(max_error, error);
            output_sum += output[i];
        }

        std::printf("\nValidacion:\n");
        std::printf("Error absoluto maximo: %.10e\n", max_error);
        std::printf("Suma de salidas: %.10f\n", output_sum);

    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
