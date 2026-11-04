#include <algorithm>
#include <chrono>
#include <exception>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Las entradas y salidas FP16 se almacenan con __fp16. La exponencial
// y la suma se calculan en float porque std::exp no proporciona una
// sobrecarga portable para __fp16 en este entorno.
enum class Precision { FP16, FP32 };

struct Config {
    Precision precision = Precision::FP16;
    int n = 100;
    int reps = 10000;
    int trials = 5;
    uint32_t seed = 12345;
    std::string csv = "results/function_times.csv";
    std::string input_csv;
    std::string export_input;
};

static int parse_positive_int(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > INT32_MAX) {
        std::fprintf(stderr, "Valor invalido para --%s: '%s'\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

static uint32_t parse_seed(const char* text) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || value > UINT32_MAX) {
        std::fprintf(stderr, "Semilla invalida: '%s'\n", text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint32_t>(value);
}

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    static struct option long_options[] = {
        {"precision", 	required_argument, nullptr, 'p'},
        {"n", 		required_argument, nullptr, 'n'},
        {"reps", 	required_argument, nullptr, 'r'},
        {"trials", 	required_argument, nullptr, 't'},
        {"seed", 	required_argument, nullptr, 's'},
        {"csv", 	required_argument, nullptr, 'c'},
        {"input-csv", 	required_argument, nullptr, 'i'},
        {"export-input",required_argument, nullptr, 'e'},
        {"help", 	no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "p:n:r:t:s:c:i:e:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'p':
                if (std::strcmp(optarg, "fp16") == 0) cfg.precision = Precision::FP16;
                else if (std::strcmp(optarg, "fp32") == 0) cfg.precision = Precision::FP32;
                else { std::fprintf(stderr, "Precision desconocida: '%s'\n", optarg); std::exit(EXIT_FAILURE); }
                break;
            case 'n':
		cfg.n = parse_positive_int(optarg, "n");
		break;
            case 'r':
		cfg.reps = parse_positive_int(optarg, "reps");
		break;
            case 't':
		cfg.trials = parse_positive_int(optarg, "trials");
		break;
            case 's':
		cfg.seed = parse_seed(optarg);
		break;
            case 'c':
		cfg.csv = optarg;
		break;
            case 'i':
		cfg.input_csv = optarg;
		break;
            case 'e':
		cfg.export_input = optarg;
		break;
            case 'h':
                std::printf("Uso: %s --precision fp16|fp32 --n N --reps R [--trials 5] [--seed 12345] [--csv archivo] [--input-csv archivo] [--export-input archivo]\n", argv[0]);
                std::exit(EXIT_SUCCESS);
            default: std::exit(EXIT_FAILURE);
        }
    }
    if (optind < argc) { std::fprintf(stderr, "Argumento inesperado: %s\n", argv[optind]); std::exit(EXIT_FAILURE); }
    return cfg;
}

// Escalado explicito de la salida de mt19937 para compartir entradas por CSV.
std::vector<double> generate_input(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<double> input(static_cast<size_t>(n));
    constexpr double max_rng = static_cast<double>(UINT32_MAX);
    for (double& value : input) value = -4.0 + 8.0 * (static_cast<double>(rng()) / max_rng);
    return input;
}

void save_input(const std::string& path, const std::vector<double>& input) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("No se pudo escribir: " + path);
    file << "index,value\n" << std::setprecision(17);
    for (size_t i = 0; i < input.size(); ++i) file << i << ',' << input[i] << '\n';
}

std::vector<double> read_input(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir: " + path);
    std::vector<double> input;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line == "index,value") continue;
        const size_t comma = line.find(',');
        if (comma == std::string::npos) throw std::runtime_error("CSV de entrada invalido");
        const size_t index = static_cast<size_t>(std::stoull(line.substr(0, comma)));
        const double value = std::stod(line.substr(comma + 1));
        if (index != input.size() || !std::isfinite(value) || value < -4.0 || value > 4.0)
            throw std::runtime_error("Indice o valor invalido en CSV de entrada");
        input.push_back(value);
    }
    if (input.empty()) throw std::runtime_error("CSV de entrada vacio");
    return input;
}

// Referencia FP64 con la misma formula numericamente estable.
std::vector<double> softmax_ref(const std::vector<double>& x) {
    const double maximum = *std::max_element(x.begin(), x.end());
    std::vector<double> out(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - maximum);
        sum += out[i];
    }
    for (double& value : out) value /= sum;
    return out;
}

std::vector<float> softmax_fp32(const std::vector<float>& x) {
    const float maximum = *std::max_element(x.begin(), x.end());
    std::vector<float> out(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - maximum);
        sum += out[i];
    }
    for (float& value : out) value /= sum;
    return out;
}

std::vector<__fp16> softmax_fp16(const std::vector<__fp16>& x) {
    float maximum = static_cast<float>(x[0]);
    for (size_t i = 1; i < x.size(); ++i)
        maximum = std::max(maximum, static_cast<float>(x[i]));

    std::vector<float> tmp(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        tmp[i] = std::exp(static_cast<float>(x[i]) - maximum);
        sum += tmp[i];
    }
    std::vector<__fp16> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = static_cast<__fp16>(tmp[i] / sum);
    return out;
}

template <typename T>
std::vector<double> to_double(const std::vector<T>& x) {
    std::vector<double> out;
    out.reserve(x.size());
    for (T value : x) out.push_back(static_cast<double>(value));
    return out;
}

struct Validation {
    double sum = 0.0, max_abs = 0.0, mean_abs = 0.0, max_rel = 0.0;
    bool range_ok = true, argmax_ok = true, sum_ok = true;
};

Validation validate(const std::vector<double>& actual, const std::vector<double>& reference, Precision p) {
    Validation v;
    const size_t reference_max = static_cast<size_t>(std::max_element(reference.begin(), reference.end()) - reference.begin());
    const size_t actual_max = static_cast<size_t>(std::max_element(actual.begin(), actual.end()) - actual.begin());
    v.argmax_ok = reference_max == actual_max;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double val = actual[i];
        v.range_ok = v.range_ok && std::isfinite(val) && val >= 0.0 && val <= 1.0;
        v.sum += val;
        const double diff = std::abs(val - reference[i]);
        v.max_abs = std::max(v.max_abs, diff);
        v.mean_abs += diff;
        v.max_rel = std::max(v.max_rel, diff / std::abs(reference[i]));
    }
    v.mean_abs /= static_cast<double>(actual.size());
    v.sum_ok = std::abs(v.sum - 1.0) <= (p == Precision::FP16 ? 0.01 : 1e-4);
    return v;
}

static std::string precision_name(Precision p) { return p == Precision::FP16 ? "fp16" : "fp32"; }

void append_csv(const Config& cfg, int trial, double elapsed_us, const Validation& v) {
    if (cfg.csv.empty()) return;
    bool header = true;
    { std::ifstream check(cfg.csv, std::ios::binary | std::ios::ate); header = !check || check.tellg() == 0; }
    std::ofstream file(cfg.csv, std::ios::app);
    if (!file) throw std::runtime_error("No se pudo abrir CSV de resultados: " + cfg.csv);
    if (header) file << "precision,n,reps,trials,seed,trial,elapsed_us,us_per_call,sum,max_abs_error,mean_abs_error,max_rel_error,range_ok,argmax_ok,sum_ok\n";
    file << std::setprecision(12) << precision_name(cfg.precision) << ',' << cfg.n << ',' << cfg.reps
         << ',' << cfg.trials << ',' << cfg.seed << ',' << trial << ',' << elapsed_us << ','
         << elapsed_us / cfg.reps << ',' << v.sum << ',' << v.max_abs << ',' << v.mean_abs
         << ',' << v.max_rel << ',' << v.range_ok << ',' << v.argmax_ok << ',' << v.sum_ok << '\n';
}

// Cinco lotes independientes, con calentamiento y mismo vector para ambas variantes.
// No se incluyen en la medicion: generacion, conversiones de entrada y validacion del cronometraje.
volatile double timing_sink = 0.0;

template <typename T, typename Function>
std::vector<double> measure(const std::vector<T>& input, Function operation, const Config& cfg,
                            const std::vector<double>& reference) {
    auto warmup = operation(input);
    timing_sink = timing_sink + static_cast<double>(warmup[0]);
    const auto result = to_double(warmup);
    const Validation v = validate(result, reference, cfg.precision);
    std::printf("sum=%.10f rango=%s argmax=%s suma=%s error_abs_max=%.9g error_rel_max=%.9g\n",
                v.sum, v.range_ok ? "OK" : "ERROR", v.argmax_ok ? "OK" : "ERROR",
                v.sum_ok ? "OK" : "ERROR", v.max_abs, v.max_rel);
    if (!(v.range_ok && v.argmax_ok && v.sum_ok))
        throw std::runtime_error("La salida no paso las verificaciones");

    std::vector<double> times;
    for (int trial = 1; trial <= cfg.trials; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < cfg.reps; ++rep) {
            auto y = operation(input);
            timing_sink = timing_sink + static_cast<double>(y[rep % y.size()]);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(stop - start).count();
        times.push_back(us / cfg.reps);
        append_csv(cfg, trial, us, v);
        std::printf("medicion=%d tiempo_lote_us=%.3f tiempo_por_llamada_us=%.6f\n", trial, us, us / cfg.reps);
    }
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    std::printf("tiempo_por_llamada_us: minimo=%.6f mediana=%.6f maximo=%.6f\n",
                times.front(), median, times.back());
    return result;
}

int main(int argc, char* argv[]) {
    try {
        Config cfg = parse_args(argc, argv);
        std::vector<double> input = cfg.input_csv.empty() ? generate_input(cfg.n, cfg.seed) : read_input(cfg.input_csv);
        if (input.size() != static_cast<size_t>(cfg.n)) throw std::runtime_error("N no coincide con el CSV de entrada");
        if (!cfg.export_input.empty()) save_input(cfg.export_input, input);
        const std::vector<double> reference = softmax_ref(input);
        std::printf("precision=%s n=%d reps=%d trials=%d seed=%u\n",
                    precision_name(cfg.precision).c_str(), cfg.n, cfg.reps, cfg.trials, cfg.seed);
        if (cfg.precision == Precision::FP32) {
            std::vector<float> x(input.begin(), input.end());
            measure(x, softmax_fp32, cfg, reference);
        } else {
            std::vector<__fp16> x;
            x.reserve(input.size());
            for (double value : input) x.push_back(static_cast<__fp16>(value));
            measure(x, softmax_fp16, cfg, reference);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
