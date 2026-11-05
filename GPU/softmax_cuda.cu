#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <getopt.h>


#include <cuda_runtime.h>
#include <cuda_fp16.h>


enum class Precision 	{ FP16, FP32 };
enum class Measure 	{ Kernel, EndToEnd};

struct Config {
    Precision precision = Precision::FP16;  // valores por defecto
    int n    = 100;
    int reps = 10000;
    Measure measure = Measure::Kernel;
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
    if (end == text || *end != '\0' || value <= 0) {
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
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:n:r:m:", long_options, &option_index)) != -1) {
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
		if (std::strcmp(optarg, "kernel"))		cfg.measure = Measure::Kernel;
		else if (std::strcmp(optarg, "end-to-end"))	cfg.measure = Measure::EndToEnd;
	        else {
			std::fprintf(stderr,"Medición desconocida: '%s'\n",optarg);
			std::exit(EXIT_FAILURE);
		}	

            default:  // '?' : getopt_long ya imprimió el error
                std::exit(EXIT_FAILURE);
        }
    }
    return cfg;
}

std::vector<double> generate_input(int n) {
    std::vector<double> input(n);   // constructor con tamaño, no "= []"
    // TODO: llenar con valores en [-4, 4] usando <random>
    return input;
}


std::vector<float> softmax_fp32(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    // TODO
    return out;
}

//std::vector<_Float16> softmax_fp16(const std::vector<_Float16>& x) {
  //  std::vector<_Float16> out(x.size());
    // TODO: documentar aqui la decision de usar FP32 para exp y la suma
    //return out;
//}

// Despachador: elige variante segun cfg y devuelve el resultado en double
// para poder compararlo con la referencia.
std::vector<double> softmax(const Config& cfg, const std::vector<double>& input) {
    std::vector<double> result;
    switch (cfg.precision) {
        case Precision::FP32:
            // TODO: convertir input a float, repetir cfg.reps veces, convertir salida a double
            break;
        case Precision::FP16:
            // TODO: idem con _Float16
            break;
    }
    return result;
}

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::printf("precision=%s n=%d reps=%d meaure=%s\n",
                cfg.precision == Precision::FP16 ? "fp16" : "fp32",
                cfg.n, cfg.reps,
		cfg.measure == Measure::Kernel ? "kernel": "end-to-end");

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    std::vector<double> input = generate_input(cfg.n);
    (void)input; // TODO: ref, kernels, medicion, error
    return 0;
}
