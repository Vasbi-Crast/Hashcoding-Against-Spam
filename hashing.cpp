#include "MurmurHash3.cpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <random>
#include <iomanip>

/**
 * Type definitions:
 *   - FEATURE: vector of doubles for storing features.
 *   - LABEL: integer type for class label (0 or 1).
 *   - HASH_DIM: dimensionality of the hashed feature space.
 */
typedef std::vector<double> FEATURE;
typedef int LABEL;
const int HASH_DIM = pow(2, 5) + 15;

struct Sample {
    FEATURE features;
    LABEL   label;
};

/**
 * Convert raw text into a fixed-size feature vector using the Feature Hashing trick.
 *
 * Pipeline:
 *   1. Tokenize the input string by whitespace.
 *   2. Lowercase each token.
 *   3. Hash the token with MurmurHash3 (seed = 8) and map to an index via modulo HASH_DIM.
 *   4. Increment the corresponding bin (count-based representation).
 *   5. Apply L2 normalization to the resulting vector.
 *
 * @param text  Raw input text.
 * @return      A dense vector of size HASH_DIM, L2-normalized.
 */
std::vector<double> text_to_features(const std::string& text) {
    // Инициализируем вектор признаков нулями
    FEATURE feats(HASH_DIM, 0.0);
    
    // Токенизация текста
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        // Приводим токен к нижнему регистру
        for (char &c : token) {
            c = std::tolower(c);
        }
        
        // Вычисляем хеш от токена
        uint32_t hash_value;
        MurmurHash3(token.c_str(), token.length(), 8, &hash_value);
        
        // Вычисляем индекс в векторном представлении
        size_t index = hash_value % HASH_DIM;
        
        // Увеличиваем значение в соответствующей ячейке
        feats[index] += 1.0;
    }
    
    // Выполняем L2-нормализацию
    double norm = 0.0;
    for (double val : feats) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0) {
        for (double &val : feats) {
            val /= norm;
        }
    }
    
    return feats;
}

class LogisticRegression {
public:
    size_t dim = HASH_DIM;
    double lr = 0.01;
    double reg_lambda = 0.1;
    double class_weight_0 = 1.0;  // weight for class 0 (ham)
    double class_weight_1 = 1.0;  // weight for class 1 (spam)
    
    LogisticRegression()
    : weights(dim),
      learning_rate(lr),
      lambda(reg_lambda),
      threshold(0.5),
      epochs(10000) {

        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.0, 0.01);
    
        for(auto& weight : weights) {
            weight = dist(gen);
        }
    }

    /**
     * Train the logistic regression model using full-batch gradient descent.
     *
     * Training dynamics:
     *   - Cross-entropy loss with L2 regularization.
     *   - Learning rate schedule: 0.01 -> 0.005 (epoch 400) -> 0.001 (epoch 4000).
     *   - Early stopping: every 10 epochs, validation metrics are evaluated;
     *     weights with the best (accuracy, precision) pair are preserved.
     *   - After training, the optimal classification threshold is selected
     *     by sweeping [0.1, 1.0) with step 0.01, subject to Recall >= 0.6.
     *
     * @param trainData  Training samples (features + labels).
     * @param validData  Validation samples used for early stopping and threshold tuning.
     */
    void train(const std::vector<Sample>& trainData, const std::vector<Sample>& validData) { 
        bool quiet_mode = false; // Silent training mode
        
        double epsilon = pow(10, -10);
        std::vector<double> L(dim, 0.0);
        std::vector<double> error_pred;
        auto best_weights = weights;

        double prediction;
        double loss = 0;
        double class_weight;

        double best_acc = 0;
        double best_precision = 0;
        int best_epoch;     
        
        // Цикл по эпохам
        for (int epoch = 0; epoch < epochs; epoch++) {
            
            loss = 0;
            error_pred.clear();
            std::fill(L.begin(), L.end(), 0.0);

            // Вычисление ошибки для каждого сэмпла
            for (const auto& sample : trainData) {
                prediction = sigmoid(dot_product(weights, sample.features));
                loss += sample.label * std::log(prediction + epsilon) + (1 - sample.label) * std::log(1 - prediction + epsilon); 
                error_pred.push_back(prediction - sample.label);
            }
            
            // Вычисление градиента
            for (int idx_sample = 0; idx_sample < trainData.size(); idx_sample++) {
                for(int idx_w = 0; idx_w < weights.size(); ++idx_w){
                    L[idx_w] += error_pred[idx_sample] * trainData[idx_sample].features[idx_w];
                }
            }

            // Нормализация градиента
            for(int idx_w = 0; idx_w < weights.size(); ++idx_w){
                L[idx_w] /= trainData.size();
            }

            // Град спуск
            for(int idx_w = 0; idx_w < weights.size(); ++idx_w){
                weights[idx_w] -= lr * (L[idx_w] + lambda * weights[idx_w]); 
            }
            
            if (epoch == 400){
                lr = 0.005;
            }
            else if (epoch == 4000){
                lr = 0.001;
            }
            
            if ((epoch + 1) % 10 == 0) {
                auto val_metrics = evaluate(validData);
                auto train_metrics = evaluate(trainData);
                if ((0.5 <= val_metrics[0]/(val_metrics[0] + val_metrics[3]))
                    || (best_precision < val_metrics[0]/(val_metrics[0] + val_metrics[2]))){
                    best_acc = (val_metrics[0] + val_metrics[1])/ (val_metrics[0] + val_metrics[1] + val_metrics[2] + val_metrics[3]);
                    best_precision = val_metrics[0]/(val_metrics[0] + val_metrics[2]);
                    best_epoch = epoch + 1;
                    best_weights = weights;
                }
            }

            // Вывод метрик
            if (((epoch + 1) % 200 == 0) && (!quiet_mode)) {
               auto val_metrics = evaluate(validData);
               auto train_metrics = evaluate(trainData);               
               std::cout << "\nEpoch: " << epoch + 1 << " loss: " << -loss/trainData.size() << ' ' << lr << std::endl;
              
               std::cout<<"____________________________________________________\n|\t|   Accuracy   |   Precision   |   Recall   |\n---------------------------------------------------\n";
               std::cout<< std::fixed << std::setprecision(6) <<
               "| Train |    " <<  (train_metrics[0] + train_metrics[1])/ (train_metrics[0] + train_metrics[1] + train_metrics[2] + train_metrics[3])
                << "  |   " << train_metrics[0]/(train_metrics[0] + train_metrics[2])
                << "   |   " << train_metrics[0]/(train_metrics[0] + train_metrics[3]) << " |\n"

                << "|  Val  |    " << (val_metrics[0] + val_metrics[1])/ (val_metrics[0] + val_metrics[1] + val_metrics[2] + val_metrics[3])
                << "  |   " << val_metrics[0]/(val_metrics[0] + val_metrics[2])
                << "   |   " << val_metrics[0]/(val_metrics[0] + val_metrics[3]) << " |\n";
            }
        }
        
        if (!quiet_mode)
            std::cout<< best_epoch << "\t" << best_acc << "\t" << best_precision << std::endl;
        
        weights = best_weights;
        find_best_threshold(validData);

        if (!quiet_mode){
            auto val_metrics = evaluate(validData);
            std::cout<<"After find_best_threshold (" << threshold << "):\n";
            std::cout<<"____________________________________________________\n|\t|   Accuracy   |   Precision   |   Recall   |\n---------------------------------------------------\n";
            std::cout<< std::fixed << std::setprecision(6) << "|  Val  |    " << (val_metrics[0] + val_metrics[1])/ (val_metrics[0] + val_metrics[1] + val_metrics[2] + val_metrics[3])
                << "  |   " << val_metrics[0]/(val_metrics[0] + val_metrics[2])
                << "   |   " << val_metrics[0]/(val_metrics[0] + val_metrics[3]) << " |\n";
        }
    }

    /**
     * Evaluate the model on a labeled dataset.
     *
     * @param data  Validation samples.
     * @return      A vector of four counts: {TP, TN, FP, FN}.
     */
    std::vector<double> evaluate(const std::vector<Sample>& data) const {
        // Инициализируем массив для метрик
        std::vector<double> metrics(4, 0.0); // TP, TN, FP, FN
        
        // Проходим по всем примерам
        for (const auto& sample : data) {
            int prediction = predict(sample.features);
            
            // Считаем метрики
            if (prediction == 1 && sample.label == 1) metrics[0] += 1; // TP
            if (prediction == 0 && sample.label == 0) metrics[1] += 1; // TN
            if (prediction == 1 && sample.label == 0) metrics[2] += 1; // FP
            if (prediction == 0 && sample.label == 1) metrics[3] += 1; // FN
        }
        
        return metrics;
    }

    int predict(const FEATURE& feats) const {
        return (sigmoid(dot_product(weights, feats)) >= threshold) ? 1 : 0;
    }

    void find_best_threshold(const std::vector<Sample>& Data){
        double best_acc = 0.0;
        double best_prec = 0.0;
        double best_threshold;
        double acc;
        double prec;
        double rec;

        for (double test_threshold = 0.1; test_threshold < 1.0; test_threshold += 0.01){
            threshold = test_threshold;
            auto metrics = evaluate(Data);
            acc = (metrics[0] + metrics[1])/ (metrics[0] + metrics[1] + metrics[2] + metrics[3]);
            prec = metrics[0]/(metrics[0] + metrics[2]);
            rec = metrics[0]/(metrics[0] + metrics[3]);
            if ((best_acc < acc) && (0.6 <= rec)){
                best_threshold = test_threshold;
                best_acc = acc;
                best_prec = prec;
            }
        }
        threshold = best_threshold;   
    }

private:
    std::vector<double> weights;
    double learning_rate;
    double lambda;
    int epochs;
    double threshold;

    static double dot_product(const FEATURE& w, const FEATURE& x) {
        double res = 0.0;
        for (size_t i = 0; i < w.size(); ++i) {
            res += w[i] * x[i];
        }
        return res;
    }
    
    static double sigmoid(double z) {
        return 1.0 / (1.0 + exp(-z));
    }
};

/**
 * Parse a CSV file with the schema "class,text" and populate the dataset.
 *
 * Each row is tokenized and converted to a feature vector via text_to_features.
 * The first row is treated as a header and skipped.
 *
 * @param filename  Path to the CSV file (e.g. "entrypoint/data_train.csv").
 * @param data      Output vector to append samples to.
 * @return          true if the file was opened and parsed successfully.
 */
bool read_csv(const std::string& filename, std::vector<Sample>& data) { 
    std::ifstream file(filename);

    if (!file.is_open())
        return false;
    
    std::string line;
    std::string label;
    std::string text;
    bool is_header = true;
    Sample sample;
    
    while (getline(file, line)) {
        size_t pos = line.find(",");

        if (pos != std::string::npos){
            label = line.substr(0, pos);
            text = line.substr(pos+1);
        }
        else
            continue;

        if (is_header){
            is_header = false;
            continue;
        }
    
        sample.features = text_to_features(text);
        sample.label = label == "spam" ? 1 : 0;  // spam - 1, ham - 0     
        data.push_back(sample);
    }
    file.close();
    return true;
}