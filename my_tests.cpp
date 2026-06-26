#include "hashing.cpp"

#include <assert.h>


void hash_test(){
    uint32_t hash1 = 1, hash2 = 1, hash3 = 1, hash4 = 0;

    // Проверка детерминированности
    MurmurHash3("test", 4, 42, &hash1);
    MurmurHash3("test", 4, 42, &hash2);

    assert(hash1 == hash2);

    MurmurHash3("nottest", 7, 42, &hash3);

    assert(hash1 != hash3);

    // Проверка "лавинного эффекта"
    MurmurHash3("tesс", 4, 42, &hash4);

    assert(hash1 != hash4);

    std::cout << "Hash function tests passed" << std::endl;
}

void Log_reg_test()
{
    // Иницилзация тестового датасета
    Sample samp1, samp2;
    samp1.features = text_to_features("spams");
    samp1.label = 1;
    samp2.features = text_to_features("hams");
    samp2.label = 0;

    // Иницилзация модели и её обучение
    std::vector<Sample> trainData = {samp1, samp2};
    LogisticRegression model;

    model.train(trainData, trainData);

    //Предсказания модели для данных из тестового датасета
    int pred1 = model.predict(samp1.features);
    int pred2 = model.predict(samp2.features);

    //Проверка, что модель правильно обучилась на очень простом сете
    assert(pred1 == 1);
    assert(pred2 == 0);

    std::cout << "Logistic regression tests passed" << std::endl;
}

void normalization_test()
{
    // Переводим фичи из текста в вектор с L2-нормализацией
    std::vector<double> feats = text_to_features("normalization");

    // Вычисляем квадрат нормы вектора
    double norm = 0.0;
    for (double val : feats) {
        norm += val * val;
    }

    // Так как вектор уже нормализован(приведён к единичному), то его норма должна быть равна 1
    assert(std::abs(sqrt(norm) - 1.0) < 1e-5);

    std::cout << "Normalization test passed" << std::endl;
}

int main(){
    hash_test();
    Log_reg_test();
    normalization_test();
}