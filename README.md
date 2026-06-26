# Hashcoding Against Spam

Imagine: you naively posted your phone number in a huge Telegram channel called "Cats and Crypto," thinking the world was full of kindness. But now your messenger is a hellish mix of:

- "Congratulations! You've won 10 kg of buckwheat! Send us your card number!"
- "Your account has been hacked. Urgently send a photo of your passport and your favorite cat!"
- "Exclusive: how to make money breeding snails in the garage. Webinar starts in 5 minutes!"

Instead of running off to the wilderness, you decide to teach a machine to tell spam (crypto scams) from ham (normal messages like "Mom, buy bread"). That's exactly what this project does.

## What's Inside

A binary text classifier written in pure C++. No external ML libraries. Input: raw text. Output: `spam` or `ham` label.

## How It Works

1. **Text vectorization.** Each message is split into tokens and lowercased. Every token is hashed via MurmurHash3; the hash is used as an index in a fixed-size vector (Feature Hashing / Hashing Trick). The vector is L2-normalized.

2. **Training.** Logistic regression is trained with full-batch gradient descent, L2 regularization, and Cross-Entropy Loss. The learning rate decays on a schedule: `0.01 → 0.005` (at epoch 400) `→ 0.001` (at epoch 4000). Every 10 epochs the model is evaluated on validation, and the weights with the best metrics are saved (early stopping).

3. **Class weighting.** Before training, class weights are computed based on the distribution in the training set — so the model doesn't collapse into predicting the majority class.

4. **Threshold tuning.** After training, thresholds from `0.1` to `0.99` are swept with a step of `0.01`. The threshold with the highest Accuracy is chosen, subject to Recall >= 0.6.

## Tech Stack

- C++11, STL
- MurmurHash3 (custom implementation)
- Logistic regression, gradient descent, L2 regularization
- Feature Hashing

## Project Structure

- `MurmurHash3.cpp` — MurmurHash3 implementation (32-bit, bitwise operations, rotations, final mixing).
- `hashing.cpp` — text vectorization (`text_to_features`), `LogisticRegression` class (training, prediction, evaluation, threshold tuning), CSV parsing.
- `main.cpp` — data loading, class weight computation, training launch, metrics output.
- `my_tests.cpp` — tests: hash determinism and avalanche effect, L2 normalization, model convergence on synthetic data.

## Input Data

CSV format `class,text`, first line is the header:

```
class,text
spam,Free money now!!!
ham,Hi, how are you?
```

- `entrypoint/data_train.csv` — training dataset.
- `entrypoint/dataset.csv` — validation dataset.

## Metrics

Target metrics on validation:

- **Accuracy** >= 0.6
- **Recall** >= 0.6

## Build and Run

```bash
g++ -O3 -std=c++11 main.cpp -o spam_filter
./spam_filter
```

## Tests

```bash
g++ -O3 -std=c++11 my_tests.cpp -o tests
./tests
```

Three test groups:

- `hash_test` — identical inputs produce identical hashes, different inputs produce different hashes.
- `normalization_test` — L2 norm of `text_to_features` output equals 1.
- `Log_reg_test` — the model separates two synthetic classes.

---

P.S. The author is not responsible if, after studying the code, you start seeing hash vectors in your nightmares.