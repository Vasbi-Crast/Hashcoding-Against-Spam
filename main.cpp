#include "hashing.cpp"

int main() {
    std::vector<Sample> allData;
    std::vector<Sample> valData;

    if (!read_csv("entrypoint/data_train.csv", allData)) {
        std::cout<<"Ошибка data_train_files";
        return 1;
    }

    if (!read_csv("entrypoint/dataset.csv", valData)) {
        std::cout<<"Ошибка dataset_files";
        return 1;
    }

    // Обучение модели

    size_t spam_count = std::count_if(allData.begin(), allData.end(), 
        [](const Sample& s) { return s.label == 1; });
    size_t ham_count = allData.size() - spam_count;

    LogisticRegression model;
    model.class_weight_0 = static_cast<double>(allData.size()) / (2.0 * ham_count);
    model.class_weight_1 = static_cast<double>(allData.size()) / (spam_count * 2.0);
    model.train(allData, valData);


    std::vector<double> metrics = model.evaluate(valData);
    std::cout << metrics[0] << ' '  <<  metrics[1] << ' ' <<  metrics[2] << ' ' <<  metrics[3]<< ' '<< std::endl;
    std::cout << "Accuracy " << ( metrics[0]+ metrics[1]) / (metrics[0] +  metrics[1] + metrics[2] +  metrics[3])  << std::endl;
    std::cout << "Recall " << (metrics[0]) / (metrics[0]+metrics[3])  << std::endl;

    return 0;
}
