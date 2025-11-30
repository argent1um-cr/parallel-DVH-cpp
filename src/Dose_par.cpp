#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <omp.h> // OpenMP
#include <cstdio> // Для работы с popen

// --- Кроссплатформенная реализация пайпов для Gnuplot ---
#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

const int MAX_GRID_INDEX = 200; // Порог для проверки "выбросов" индексов

struct GraphPoint {
    double doseVal;
    double volumePercent; // 0..100
};

// Функция чтения из файла (последовательная, т.к. диск - узкое место)
std::vector<double> readDataFromFile(const std::string& filename) {
    std::vector<double> doses;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "!!! Ошибка: Не удалось открыть файл " << filename << std::endl;
        return doses;
    }

    std::string line;
    int lineNum = 0;
    std::cout << "Чтение файла..." << std::endl;

    while (std::getline(file, line)) {
        lineNum++;
        if (line.empty()) continue;

        // Заменяем запятые на точки, если вдруг формат дробных чисел русский
        std::replace(line.begin(), line.end(), ',', '.');

        std::stringstream ss(line);
        int x, y, z;
        double dose = 0.0;

        // Читаем 3 координаты
        if (!(ss >> x >> y >> z)) {
            // Если формат совсем битый
            continue;
        }

        // Проверка на адекватность индексов
        if (x < 0 || y < 0 || z < 0) {
            std::cerr << "Warning (строка " << lineNum << "): Отрицательные индексы [" << x << " " << y << " " << z << "]. Игнорируем.\n";
            continue;
        }
        if (x > MAX_GRID_INDEX || y > MAX_GRID_INDEX || z > MAX_GRID_INDEX) {
            std::cerr << "Warning (строка " << lineNum << "): Индекс-выброс [" << x << " " << y << " " << z << "]. Игнорируем.\n";
            continue;
        }

        // Читаем дозу. Если её нет или ошибка чтения — она останется 0.0
        ss >> dose;
        if (dose < 0) dose = 0.0; // По условию доза > 0, иначе 0

        doses.push_back(dose);
    }

    file.close();
    return doses;
}

// Параллельная генерация случайных данных
std::vector<double> generateRandomData(size_t count, double maxDose) {
    std::vector<double> doses(count);

    std::cout << "Генерация данных (" << count << " элементов) в потоках..." << std::endl;

    // Распараллеливаем цикл заполнения
#pragma omp parallel
    {
        // У каждого потока свой seed, зависящий от номера потока и времени
        unsigned int seed = static_cast<unsigned int>(1234 + omp_get_thread_num() * 54321);
        std::mt19937 gen(seed);

        // Используем нормальное распределение для интереса (Гаусс), 
        // чтобы график был красивее (S-образный DVH), или равномерное.
        // Пусть будет равномерное, как просили "от 0 до...".
        std::uniform_real_distribution<> dis(0.0, maxDose);

#pragma omp for
        for (long long i = 0; i < count; ++i) {
            doses[i] = dis(gen);
        }
    }
    return doses;
}

// Основной расчет DVH (Гистограмма доза-объем)
std::vector<GraphPoint> calculateDVH(const std::vector<double>& doses, double step) {
    if (doses.empty()) return {};

    // 1. Находим максимум (Parallel Reduction)
    double maxDoseVal = 0.0;
#pragma omp parallel for reduction(max: maxDoseVal)
    for (size_t i = 0; i < doses.size(); ++i) {
        if (doses[i] > maxDoseVal) maxDoseVal = doses[i];
    }

    // Количество корзин (bins)
    int numBins = static_cast<int>(std::ceil(maxDoseVal / step)) + 2;
    std::vector<long long> global_frequency(numBins, 0);

    // 2. Считаем частоты (Parallel Histogram)
    // Чтобы не блокировать потоки atomic-операциями, каждый считает свой кусочек,
    // а потом сливаем.
#pragma omp parallel
    {
        std::vector<long long> local_freq(numBins, 0);

#pragma omp for nowait
        for (size_t i = 0; i < doses.size(); ++i) {
            int bin = static_cast<int>(doses[i] / step);
            if (bin < numBins) {
                local_freq[bin]++;
            }
        }

#pragma omp critical
        {
            for (int j = 0; j < numBins; ++j) {
                global_frequency[j] += local_freq[j];
            }
        }
    }

    // 3. Считаем кумулятивную гистограмму (Sequential)
    // Идем от большой дозы к маленькой.
    std::vector<GraphPoint> result;
    result.reserve(numBins);

    long long currentSum = 0;
    double totalCount = static_cast<double>(doses.size());

    // Временный вектор для реверса
    std::vector<GraphPoint> tempPoints;

    for (int i = numBins - 1; i >= 0; --i) {
        currentSum += global_frequency[i];

        // Если накопили какие-то значения, записываем точку
        // (Или записываем все, включая нули, чтобы график дошел до конца оси X)
        double d = i * step;
        double volPerc = (currentSum / totalCount) * 100.0;

        tempPoints.push_back({ d, volPerc });
    }

    // tempPoints сейчас от MaxDose -> 0. Нам для Gnuplot удобнее от 0 -> MaxDose,
    // но DVH обычно рисуется как есть. Gnuplot сам расставит точки.
    // Перевернем для порядка (сортировка по X по возрастанию)
    std::reverse(tempPoints.begin(), tempPoints.end());

    return tempPoints;
}

// Функция вызова Gnuplot
void plotWithGnuplot(const std::vector<GraphPoint>& points) {
    std::cout << "Запуск Gnuplot..." << std::endl;

    // Открываем пайп к процессу gnuplot
    // Ключ -persist оставляет окно открытым после завершения программы
    FILE* gp = POPEN("gnuplot -persist", "w");

    if (gp == NULL) {
        std::cerr << "Ошибка: Не найден gnuplot! Убедитесь, что он установлен и добавлен в PATH.\n";
#ifdef _WIN32
        std::cerr << "Для Windows: скачайте с gnuplot.info и при установке поставьте галочку 'Add to PATH'.\n";
#else
        std::cerr << "Для Linux: sudo apt install gnuplot\n";
#endif
        return;
    }

    // Отправляем команды настройки графика
    fprintf(gp, "set title 'Dose-Volume Histogram (DVH)' font ',14'\n");
    fprintf(gp, "set xlabel 'Dose (units)'\n");
    fprintf(gp, "set ylabel 'Volume (%)'\n");
    fprintf(gp, "set grid\n");

    // Настройка диапазонов
    fprintf(gp, "set yrange [0:105]\n"); // Чуть больше 100, чтобы было красиво
    fprintf(gp, "set xrange [0:*]\n");   // Автоматически по X

    // Стиль линий
    fprintf(gp, "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5\n"); // Синяя линия

    // Команда рисования. '-' означает, что данные пойдут далее в потоке
    fprintf(gp, "plot '-' with lines ls 1 title 'Cumulative DVH'\n");

    // Передача данных
    for (const auto& p : points) {
        fprintf(gp, "%f %f\n", p.doseVal, p.volumePercent);
    }

    // Признак конца данных для Gnuplot
    fprintf(gp, "e\n");

    fflush(gp); // Сбрасываем буфер
    PCLOSE(gp); // Закрываем пайп
}

int main() {
    // Настройка локали для корректного вывода русского текста
    setlocale(LC_ALL, "");

    std::vector<double> doseData;
    int mode = 0;

    std::cout << "=== DVH Calculator (OpenMP + Gnuplot) ===\n";
    std::cout << "1. Загрузить из файла (x y z dose)\n";
    std::cout << "2. Сгенерировать случайно\n";
    std::cout << "Выбор: ";
    std::cin >> mode;

    if (mode == 1) {
        std::string filename;
        std::cout << "Введите имя файла (напр. data.txt): ";
        std::cin >> filename;
        doseData = readDataFromFile(filename);
    }
    else if (mode == 2) {
        size_t n;
        double maxD;
        std::cout << "Количество элементов (напр. 1000000): ";
        std::cin >> n;
        std::cout << "Макс. доза (напр. 50.0): ";
        std::cin >> maxD;
        doseData = generateRandomData(n, maxD);
    }
    else {
        std::cout << "Неверный режим.\n";
        return 1;
    }

    if (doseData.empty()) {
        std::cout << "Нет данных для построения.\n";
        return 0;
    }

    double step;
    std::cout << "Введите шаг графика по дозе (напр. 0.1): ";
    std::cin >> step;
    if (step <= 0) step = 0.1;

    // Замер времени расчета
    double t_start = omp_get_wtime();

    // Расчет данных для графика
    std::vector<GraphPoint> graphData = calculateDVH(doseData, step);

    double t_end = omp_get_wtime();
    std::cout << "Расчет выполнен за " << std::fixed << std::setprecision(4)
        << (t_end - t_start) << " сек." << std::endl;

    // Построение графика
    plotWithGnuplot(graphData);

    #ifdef _WIN32
        std::cout << std::endl;
        system("pause");
    #else
        std::cout << "\nНажмите Enter, чтобы выйти...";
        // Очистка нужна, потому что предыдущие std::cin оставили символ переноса строки в буфере
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
    #endif

    return 0;
}
