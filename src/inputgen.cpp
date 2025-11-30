#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <omp.h>
#include <iomanip>
#include <cmath>       // Для sqrt, exp, pow
#include <algorithm>   // Для std::max, std::min
#include <limits>      // Для паузы

// Структура одной записи
struct VoxelData {
    int x, y, z;
    double dose;
};

// Вспомогательная функция для ограничения диапазона (clamp)
// (Есть в C++17 std::clamp, но напишем свою для совместимости со старыми стандартами)
double clampDose(double val, double minV, double maxV) {
    if (val < minV) return minV;
    if (val > maxV) return maxV;
    return val;
}

int main() {
    setlocale(LC_ALL, ""); // Русский язык

    int nx, ny, nz;
    double maxDose;
    std::string filename;
    int distMode;

    std::cout << "=== Продвинутый Генератор 3D Доз ===\n";
    std::cout << "Введите размеры (X Y Z): ";
    if (!(std::cin >> nx >> ny >> nz)) return 0;

    std::cout << "Введите максимальную дозу: ";
    std::cin >> maxDose;

    std::cout << "\nВыберите тип распределения:\n";
    std::cout << "1 - Равномерный шум (Uniform)\n    -> График DVH будет почти прямой линией.\n";
    std::cout << "2 - Нормальное распределение (Gaussian)\n    -> Дозы кучкуются вокруг середины. График S-образный.\n";
    std::cout << "3 - Имитация пучка (Hotspot)\n    -> В центре массива - максимум, к краям спад.\n";
    std::cout << "Ваш выбор: ";
    std::cin >> distMode;

    std::cout << "Введите имя файла: ";
    std::cin >> filename;

    long long totalElements = (long long)nx * ny * nz;

    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Ошибка открытия файла!\n";
        system("pause");
        return 1;
    }
    outFile << std::fixed << std::setprecision(2);

    std::cout << "Генерация " << totalElements << " элементов..." << std::endl;

    double startTime = omp_get_wtime();
    const int CHUNK_SIZE = 1000000;
    std::vector<VoxelData> buffer(CHUNK_SIZE);
    long long processedCount = 0;

    // Параметры для режимов
    // Центр массива для режима 3
    double cx = nx / 2.0;
    double cy = ny / 2.0;
    double cz = nz / 2.0;
    // Максимальный радиус (от центра до угла)
    double maxRadius = std::sqrt(cx * cx + cy * cy + cz * cz);

    for (long long offset = 0; offset < totalElements; offset += CHUNK_SIZE) {
        int currentBatchSize = std::min((long long)CHUNK_SIZE, totalElements - offset);

#pragma omp parallel
        {
            // Уникальный seed для потока
            unsigned int seed = 12345 + omp_get_thread_num() + (unsigned int)offset;
            std::mt19937 gen(seed);

            // Распределения инициализируем здесь
            std::uniform_real_distribution<> distUniform(0.0, maxDose);
            // Для Гаусса: среднее = половина дозы, разброс = 1/6 от макс (чтобы почти всё влезло в 0..max)
            std::normal_distribution<> distNormal(maxDose * 0.5, maxDose * 0.15);
            // Для шума в режиме 3
            std::normal_distribution<> distNoise(0.0, maxDose * 0.05);

#pragma omp for
            for (int i = 0; i < currentBatchSize; ++i) {
                long long globalIndex = offset + i;
                long long temp = globalIndex;
                int z = temp % nz;
                temp /= nz;
                int y = temp % ny;
                int x = temp / ny;

                buffer[i].x = x;
                buffer[i].y = y;
                buffer[i].z = z;

                double val = 0.0;

                if (distMode == 1) {
                    // === РАВНОМЕРНОЕ ===
                    val = distUniform(gen);
                }
                else if (distMode == 2) {
                    // === НОРМАЛЬНОЕ (GAUSSIAN) ===
                    val = distNormal(gen);
                }
                else if (distMode == 3) {
                    // === ПРОСТРАНСТВЕННОЕ (HOTSPOT) ===
                    // Считаем расстояние от центра массива до текущей точки
                    double dx = x - cx;
                    double dy = y - cy;
                    double dz = z - cz;
                    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                    // Формула: чем дальше от центра, тем меньше доза.
                    // Используем косинусоидальный спад или линейный.
                    // Пусть будет параболический спад: Dose = Max * (1 - (dist/R)^2)
                    double ratio = dist / maxRadius;
                    if (ratio > 1.0) ratio = 1.0;

                    // Основная форма
                    val = maxDose * (1.0 - ratio * ratio); // Парабола
                    // val = maxDose * std::exp(- (ratio * ratio) * 5.0); // Или экспонента (Гауссов пучок)

                    // Добавляем немного случайного шума, чтобы не было идеально гладко
                    val += distNoise(gen);
                }

                // Гарантируем, что доза не уйдет в минус и не превысит (слишком сильно) макс
                buffer[i].dose = clampDose(val, 0.0, maxDose * 1.5);
            }
        }

        // Запись на диск (последовательно)
        for (int i = 0; i < currentBatchSize; ++i) {
            outFile << buffer[i].x << " "
                << buffer[i].y << " "
                << buffer[i].z << " "
                << buffer[i].dose << "\n";
        }

        processedCount += currentBatchSize;

        // Прогресс
        if (processedCount % (CHUNK_SIZE * 5) == 0 || processedCount == totalElements) {
            double percent = (double)processedCount / totalElements * 100.0;
            std::cout << "\rПрогресс: " << (long long)percent << "%" << std::flush;
        }
    }

    outFile.close();
    std::cout << "\nФайл " << filename << " создан.\n";

    // === ПАУЗА ===
#ifdef _WIN32
    std::cout << std::endl;
    system("pause");
#else
    std::cout << "\nНажмите Enter...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
#endif

    return 0;
}