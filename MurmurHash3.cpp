#include <cstdint>
#include <cstring>


/**
 * MurmurHash3
 * 
 * @param key   - указатель на данные для хеширования
 * @param len   - длина данных в байтах
 * @param seed  - начальное значение (seed) хеша
 * @param out   - указатель на переменную, куда будет записан 32-битный хеш
 * 
 * @solve
 */
void MurmurHash3(const void *key, int len, uint32_t seed, void *out) {
    const uint8_t *data = (const uint8_t*)key;
    const uint8_t *end = data + len;
    uint32_t h1 = seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    // Основной цикл обработки данных
    const uint32_t *data32 = reinterpret_cast<const uint32_t*>(data);
    const uint32_t *end32 = reinterpret_cast<const uint32_t*>(end);

    while (data32 < end32) {
        uint32_t k = *data32++;

        k *= c1;
        k = (k << 15) | (k >> 17);  // Ротация вправо на 17 бит
        k *= c2;

        h1 ^= k;
        h1 = (h1 << 13) | (h1 >> 19);  // Ротация вправо на 19 бит
        h1 = h1 * 5 + 0xe6546b64;
    }

    // Обработка оставшихся байтов
    const uint8_t *tail = (const uint8_t*)(data32);

    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= (uint32_t)tail[2] << 16;
        case 2: k1 ^= (uint32_t)tail[1] << 8;
        case 1: k1 ^= (uint32_t)tail[0];
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);
            k1 *= c2;
            h1 ^= k1;
    }

    // Финальное перемешивание
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    // Копирование результата
    std::memcpy(out, &h1, sizeof(h1));
}
