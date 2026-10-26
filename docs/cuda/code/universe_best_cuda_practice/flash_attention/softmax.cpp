#include <vector>
#include <iostream>
#include <cmath>

std::vector<float> navieSoftmax(std::vector<float> &src)
{
    std::vector<float> dst;
    dst.resize(src.size());
    // 1. get sum
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++)
    {
        sum += std::exp(src[i]);
    }

    // 2. caculate output
    for (int i = 0; i < src.size(); i++)
    {
        dst[i] = std::exp(src[i]) / sum;
    }

    return dst;
}

std::vector<float> safeSoftmax(std::vector<float> &src)
{
    std::vector<float> dst;
    dst.resize(src.size());
    // 1. get max
    float max_value = -99999.f; // set it to MIN_FLOAT
    for (int i = 0; i < src.size(); i++)
    {
        if (src[i] > max_value)
        {
            max_value = src[i];
        }
    }

    // 2. get sum
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++)
    {
        sum += std::exp(src[i] - max_value);
    }

    // 3. caculate output
    for (int i = 0; i < src.size(); i++)
    {
        dst[i] = std::exp(src[i] - max_value) / sum;
    }

    return dst;
}

std::vector<float> onlineSoftmax(std::vector<float> &src)
{
    std::vector<float> dst;
    dst.resize(src.size());
    // 1. get max and get sum
    float max_value = -99999.f; // set it to MIN_FLOAT
    float pre_max_value = 0.f;
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++)
    {
        max_value = std::max(max_value, src[i]);
        sum = sum * std::exp(pre_max_value - max_value) + std::exp(src[i] - max_value);
        pre_max_value = max_value;
    }

    // 2. caculate output
    for (int i = 0; i < src.size(); i++)
    {
        dst[i] = std::exp(src[i] - max_value) / sum;
    }

    return dst;
}

float onlineSoftmaxWithDotProduct(std::vector<float> &src, std::vector<float> &value)
{
    float dst = 0.f;
    // 1. get max and get sum
    float max_value = -99999.f; // set it to MIN_FLOAT
    float pre_max_value = 0.f;
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++)
    {
        max_value = std::max(max_value, src[i]);
        sum = sum * std::exp(pre_max_value - max_value) + std::exp(src[i] - max_value);
        pre_max_value = max_value;
    }

    // 2. caculate output
    for (int i = 0; i < src.size(); i++)
    {
        dst += std::exp(src[i] - max_value) / sum * value[i];
    }

    return dst;
}

float onlineSoftmaxWithDotProductPerfect(std::vector<float> &src, std::vector<float> &value)
{
    float dst = 0.f;
    float max_value = -99999.f; // set it to MIN_FLOAT
    float pre_max_value = 0.f;
    float pre_sum = 0.f;
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++)
    {
        max_value = std::max(max_value, src[i]);
        sum = sum * std::exp(pre_max_value - max_value) + std::exp(src[i] - max_value);
        dst = dst * (pre_sum * std::exp(pre_max_value - max_value) / sum) + std::exp(src[i] - max_value) / sum * value[i];
        pre_max_value = max_value;
        pre_sum = sum;
    }

    return dst;
}

int main()
{
    std::vector<float> src = {1.0f, 2.55f, -9.2f, 4.3f, 10.25f};
    std::vector<float> dst = navieSoftmax(src);
    for (const float element : dst)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    std::vector<float> dst1 = safeSoftmax(src);
    for (const float element : dst1)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    std::vector<float> dst2 = onlineSoftmax(src);
    for (const float element : dst2)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    std::vector<float> value = {1.57f, 0.55f, 7.2f, 2.3f, 3.25f};
    float result = onlineSoftmaxWithDotProduct(src, value);
    std::cout << result << std::endl;
    float result1 = onlineSoftmaxWithDotProductPerfect(src, value);
    std::cout << result1 << std::endl;
}