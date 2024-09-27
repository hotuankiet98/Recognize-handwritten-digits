#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
using namespace std;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t uint8;
#define REPORT_ERROR_WHILE_TRAINING() 1 

const size_t c_numInputNeurons = 784;
const size_t c_numHiddenNeurons = 30;
const size_t c_numOutputNeurons = 10;

const size_t c_trainingEpochs = 5;
const size_t c_miniBatchSize = 10;
const float c_learningRate = 3.0f;

// ============================================================================================
//                                     SBlockTimer
// ============================================================================================
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
struct SBlockTimer
{
    SBlockTimer(const char* label)
    {
        m_start = chrono::high_resolution_clock::now();
        m_label = label;
    }

    ~SBlockTimer()
    {
        chrono::duration<float> seconds = chrono::high_resolution_clock::now() - m_start;
        printf("%s%0.2f seconds\n", m_label, seconds.count());
    }

    chrono::high_resolution_clock::time_point m_start;
    const char* m_label;
};

// ============================================================================================
//                                    MNIST DATA LOADER
// ============================================================================================

inline uint32 EndianSwap(uint32 a)
{
    return (a << 24) | ((a << 8) & 0x00ff0000) |
        ((a >> 8) & 0x0000ff00) | (a >> 24);
}
// Kernel CUDA để chuyển đổi từ uint8 sang float
__global__ void convertUint8ToFloatCUDA(const uint8_t* pixels, float* pixelsFloat, size_t numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        pixelsFloat[idx] = static_cast<float>(pixels[idx]) / 255.0f;
    }
}

// Hàm thực hiện chuyển đổi từ uint8 sang float trên GPU
void convertToFloatCUDA(const uint8_t* pixels, float* pixelsFloat, size_t numPixels) {
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    uint8_t* d_pixels;
    float* d_pixelsFloat;

    CHECK(cudaMalloc(&d_pixels, numPixels * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_pixelsFloat, numPixels * sizeof(float)));

    CHECK(cudaMemcpy(d_pixels, pixels, numPixels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    convertUint8ToFloatCUDA << <numBlocks, blockSize >> > (d_pixels, d_pixelsFloat, numPixels);

    CHECK(cudaMemcpy(pixelsFloat, d_pixelsFloat, numPixels * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_pixels));
    CHECK(cudaFree(d_pixelsFloat));
}
// Kernel CUDA để chuyển đổi từ uint8 sang float sử dụng shared memory
__global__ void convertUint8ToFloatCUDAKernel2(const uint8_t* pixels, float* pixelsFloat, size_t numPixels) {
    extern __shared__ float sharedPixels[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        sharedPixels[threadIdx.x] = static_cast<float>(pixels[idx]) / 255.0f;
        __syncthreads(); // Đồng bộ hóa tất cả các luồng trong block để đảm bảo dữ liệu đã được sao chép vào shared memory
        pixelsFloat[idx] = sharedPixels[threadIdx.x];
    }
}

// Hàm thực hiện chuyển đổi từ uint8 sang float trên GPU
void convertToFloatCUDAKernel2(const uint8_t* pixels, float* pixelsFloat, size_t numPixels) {
     const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    uint8_t* d_pixels;
    float* d_pixelsFloat;

    cudaMalloc(&d_pixels, numPixels * sizeof(uint8_t));
    cudaMalloc(&d_pixelsFloat, numPixels * sizeof(float));

    cudaMemcpy(d_pixels, pixels, numPixels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    convertUint8ToFloatCUDA << <numBlocks, blockSize, blockSize * sizeof(float) >> > (d_pixels, d_pixelsFloat, numPixels);

    cudaMemcpy(pixelsFloat, d_pixelsFloat, numPixels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_pixelsFloat);
}

__constant__ float kConversionFactor = 1.0f / 255.0f;

__global__ void convertUint8ToFloatCUDAKernel3(const uint8_t* pixels, float* pixelsFloat, size_t numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        pixelsFloat[idx] = static_cast<float>(pixels[idx]) * kConversionFactor;
    }
}

// Hàm thực hiện chuyển đổi từ uint8 sang float trên GPU
void convertToFloatCUDAKernel3(const uint8_t* pixels, float* pixelsFloat, size_t numPixels) {
    const int blockSize = 256;
    const int numBlocks = (numPixels + blockSize - 1) / blockSize;

    uint8_t* d_pixels;
    float* d_pixelsFloat;

    cudaMalloc(&d_pixels, numPixels * sizeof(uint8_t));
    cudaMalloc(&d_pixelsFloat, numPixels * sizeof(float));

    cudaMemcpy(d_pixels, pixels, numPixels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    convertUint8ToFloatCUDAKernel3 << <numBlocks, blockSize >> > (d_pixels, d_pixelsFloat, numPixels);

    cudaMemcpy(pixelsFloat, d_pixelsFloat, numPixels * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_pixelsFloat);
}
// MNIST data and file format description is from http://yann.lecun.com/exdb/mnist/
class CMNISTData
{
public:
    CMNISTData()
    {
        m_labelData = nullptr;
        m_imageData = nullptr;

        m_imageCount = 0;
        m_labels = nullptr;
        m_pixels = nullptr;
    }
    bool Load(bool training)
    {
        // set the expected image count
        m_imageCount = training ? 60000 : 10000;

        // read labels
        const char* labelsFileName = training ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
        FILE* file = fopen(labelsFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", labelsFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_labelData = new uint8[fileSize];
        fread(m_labelData, fileSize, 1, file);
        fclose(file);

        // read images
        const char* imagesFileName = training ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
        file = fopen(imagesFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", imagesFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_imageData = new uint8[fileSize];
        fread(m_imageData, fileSize, 1, file);
        fclose(file);

        // endian swap label file if needed, just first two uint32's.  The rest is uint8's.
        uint32* data = (uint32*)m_labelData;
        if (data[0] == 0x01080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
        }

        // verify that the label file has the right header
        if (data[0] != 2049 || data[1] != m_imageCount)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        m_labels = (uint8*)&(data[2]);

        // endian swap the image file if needed, just first 4 uint32's. The rest is uint8's.
        data = (uint32*)m_imageData;
        if (data[0] == 0x03080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
            data[2] = EndianSwap(data[2]);
            data[3] = EndianSwap(data[3]);
        }

        // verify that the image file has the right header
        if (data[0] != 2051 || data[1] != m_imageCount || data[2] != 28 || data[3] != 28)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        m_pixels = (uint8*)&(data[4]);

        // convert the pixels from uint8 to float
        m_pixelsFloat.resize(28 * 28 * m_imageCount);
        convertToFloatCUDA(m_pixels, m_pixelsFloat.data(), 28 * 28 * m_imageCount);

        // success!
        return true;
    }
    bool Loadkernel1(bool training)
    {
        // set the expected image count
        m_imageCount = training ? 60000 : 10000;

        // read labels
        const char* labelsFileName = training ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
        FILE* file = fopen(labelsFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", labelsFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_labelData = new uint8[fileSize];
        fread(m_labelData, fileSize, 1, file);
        fclose(file);

        // read images
        const char* imagesFileName = training ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
        file = fopen(imagesFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", imagesFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_imageData = new uint8[fileSize];
        fread(m_imageData, fileSize, 1, file);
        fclose(file);

        // endian swap label file if needed, just first two uint32's.  The rest is uint8's.
        uint32* data = (uint32*)m_labelData;
        if (data[0] == 0x01080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
        }

        // verify that the label file has the right header
        if (data[0] != 2049 || data[1] != m_imageCount)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        m_labels = (uint8*)&(data[2]);

        // endian swap the image file if needed, just first 4 uint32's. The rest is uint8's.
        data = (uint32*)m_imageData;
        if (data[0] == 0x03080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
            data[2] = EndianSwap(data[2]);
            data[3] = EndianSwap(data[3]);
        }

        // verify that the image file has the right header
        if (data[0] != 2051 || data[1] != m_imageCount || data[2] != 28 || data[3] != 28)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        int labelCounts[10] = { 0 };
        const int desiredCountPerLabel = 1000;
        m_pixels = (uint8*)&(data[4]);

        // convert the pixels from uint8 to float
        m_pixelsFloat.resize(28 * 28 * m_imageCount);
        convertToFloatCUDA(m_pixels, m_pixelsFloat.data(), 28 * 28 * m_imageCount);
       
        int selectedCount = 0;
        for (size_t i = 0; i < m_imageCount; ++i) {
            uint8 label = m_labels[i];
            if (labelCounts[label] < desiredCountPerLabel) {
                ++labelCounts[label];
                ++selectedCount;
            }
        }
        // Xáo trộn các mẫu đã chọn
        random_shuffle(m_pixelsFloat.begin(), m_pixelsFloat.begin() + selectedCount);

        // Đặt số lượng hình ảnh cho số lượng mẫu đã chọn
        m_imageCount = selectedCount;
        /*
        for (int i = 0; i < 10; ++i) {
            printf("Label %d: %d images\n", i, labelCounts[i]);
        }
        */
        // success!
        return true;
    }
    bool Loadkernel2(bool training)
    {
        // set the expected image count
        m_imageCount = training ? 60000 : 10000;

        // read labels
        const char* labelsFileName = training ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
        FILE* file = fopen(labelsFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", labelsFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_labelData = new uint8[fileSize];
        fread(m_labelData, fileSize, 1, file);
        fclose(file);

        // read images
        const char* imagesFileName = training ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
        file = fopen(imagesFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", imagesFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_imageData = new uint8[fileSize];
        fread(m_imageData, fileSize, 1, file);
        fclose(file);

        // endian swap label file if needed, just first two uint32's.  The rest is uint8's.
        uint32* data = (uint32*)m_labelData;
        if (data[0] == 0x01080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
        }

        // verify that the label file has the right header
        if (data[0] != 2049 || data[1] != m_imageCount)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        m_labels = (uint8*)&(data[2]);

        // endian swap the image file if needed, just first 4 uint32's. The rest is uint8's.
        data = (uint32*)m_imageData;
        if (data[0] == 0x03080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
            data[2] = EndianSwap(data[2]);
            data[3] = EndianSwap(data[3]);
        }

        // verify that the image file has the right header
        if (data[0] != 2051 || data[1] != m_imageCount || data[2] != 28 || data[3] != 28)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        int labelCounts[10] = { 0 };
        const int desiredCountPerLabel = 1000;
        m_pixels = (uint8*)&(data[4]);

        // convert the pixels from uint8 to float
        m_pixelsFloat.resize(28 * 28 * m_imageCount);
        convertToFloatCUDAKernel2(m_pixels, m_pixelsFloat.data(), 28 * 28 * m_imageCount);

        int selectedCount = 0;
        for (size_t i = 0; i < m_imageCount; ++i) {
            uint8 label = m_labels[i];
            if (labelCounts[label] < desiredCountPerLabel) {
                ++labelCounts[label];
                ++selectedCount;
            }
        }
        // Xáo trộn các mẫu đã chọn
        random_shuffle(m_pixelsFloat.begin(), m_pixelsFloat.begin() + selectedCount);

        // Đặt số lượng hình ảnh cho số lượng mẫu đã chọn
        m_imageCount = selectedCount;
        /*
        for (int i = 0; i < 10; ++i) {
            printf("Label %d: %d images\n", i, labelCounts[i]);
        }
        */
        // success!
        return true;
    }
    bool Loadkernel3(bool training)
    {
        // set the expected image count
        m_imageCount = training ? 60000 : 10000;

        // read labels
        const char* labelsFileName = training ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
        FILE* file = fopen(labelsFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", labelsFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_labelData = new uint8[fileSize];
        fread(m_labelData, fileSize, 1, file);
        fclose(file);

        // read images
        const char* imagesFileName = training ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
        file = fopen(imagesFileName, "rb");
        if (!file)
        {
            printf("could not open %s for reading.\n", imagesFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_imageData = new uint8[fileSize];
        fread(m_imageData, fileSize, 1, file);
        fclose(file);

        // endian swap label file if needed, just first two uint32's.  The rest is uint8's.
        uint32* data = (uint32*)m_labelData;
        if (data[0] == 0x01080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
        }

        // verify that the label file has the right header
        if (data[0] != 2049 || data[1] != m_imageCount)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        m_labels = (uint8*)&(data[2]);

        // endian swap the image file if needed, just first 4 uint32's. The rest is uint8's.
        data = (uint32*)m_imageData;
        if (data[0] == 0x03080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
            data[2] = EndianSwap(data[2]);
            data[3] = EndianSwap(data[3]);
        }

        // verify that the image file has the right header
        if (data[0] != 2051 || data[1] != m_imageCount || data[2] != 28 || data[3] != 28)
        {
            printf("Label data had unexpected header values.\n");
            return false;
        }
        int labelCounts[10] = { 0 };
        const int desiredCountPerLabel = 1000;
        m_pixels = (uint8*)&(data[4]);

        // convert the pixels from uint8 to float
        m_pixelsFloat.resize(28 * 28 * m_imageCount);
        convertToFloatCUDAKernel3(m_pixels, m_pixelsFloat.data(), 28 * 28 * m_imageCount);

        int selectedCount = 0;
        for (size_t i = 0; i < m_imageCount; ++i) {
            uint8 label = m_labels[i];
            if (labelCounts[label] < desiredCountPerLabel) {
                ++labelCounts[label];
                ++selectedCount;
            }
        }
        // Xáo trộn các mẫu đã chọn
        random_shuffle(m_pixelsFloat.begin(), m_pixelsFloat.begin() + selectedCount);

        // Đặt số lượng hình ảnh cho số lượng mẫu đã chọn
        m_imageCount = selectedCount;
        /*
        for (int i = 0; i < 10; ++i) {
            printf("Label %d: %d images\n", i, labelCounts[i]);
        }
        */
        // success!
        return true;
    }
    ~CMNISTData()
    {
        delete[] static_cast<int*>(m_labelData);
        delete[] static_cast<float*>(m_imageData);
    }

    size_t NumImages() const { return m_imageCount; }

    const float* GetImage(size_t index, uint8& label) const
    {
        label = m_labels[index];
        return &m_pixelsFloat[index * 28 * 28];
    }

private:
    void* m_labelData;
    void* m_imageData;

    size_t m_imageCount;
    uint8* m_labels;
    uint8* m_pixels;

    std::vector<float> m_pixelsFloat;
};
// ============================================================================================
//                                    NEURAL NETWORK
// ============================================================================================

template <size_t INPUTS, size_t HIDDEN_NEURONS, size_t OUTPUT_NEURONS>
class CNeuralNetwork
{
public:
    CNeuralNetwork()
    {
        // khởi tạo trọng số và độ lệch cho số ngẫu nhiên phân phối gaussian với giá trị trung bình 0, stddev 1.0
        random_device rd;
        mt19937 e2(rd());
        normal_distribution<float> dist(0, 1);

        for (float& f : m_hiddenLayerBiases)
            f = dist(e2);

        for (float& f : m_outputLayerBiases)
            f = dist(e2);

        for (float& f : m_hiddenLayerWeights)
            f = dist(e2);

        for (float& f : m_outputLayerWeights)
            f = dist(e2);
    }

    void Train(const CMNISTData& trainingData, size_t miniBatchSize, float learningRate)
    {
        // xáo trộn thứ tự dữ liệu huấn luyện cho các lô nhỏ

        if (m_trainingOrder.size() != trainingData.NumImages())
        {
            
            m_trainingOrder.resize(trainingData.NumImages());
            size_t index = 0;
            for (size_t& v : m_trainingOrder)
            {
                v = index;
                ++index;
            }
            
        }
        static  random_device rd;
        static  mt19937 e2(rd());
        shuffle(m_trainingOrder.begin(), m_trainingOrder.end(), e2);

        // xử lý tất cả các minibatch cho đến khi hết mẫu huấn luyện
        size_t trainingIndex = 0;
        while (trainingIndex < trainingData.NumImages())
        {
            // Xóa các dẫn xuất minibatch. Chúng tôi tổng hợp lại rồi chia khi kết thúc trận đấu nhỏ
            fill(m_miniBatchHiddenLayerBiasesDeltaCost.begin(), m_miniBatchHiddenLayerBiasesDeltaCost.end(), 0.0f);
            fill(m_miniBatchOutputLayerBiasesDeltaCost.begin(), m_miniBatchOutputLayerBiasesDeltaCost.end(), 0.0f);
            fill(m_miniBatchHiddenLayerWeightsDeltaCost.begin(), m_miniBatchHiddenLayerWeightsDeltaCost.end(), 0.0f);
            fill(m_miniBatchOutputLayerWeightsDeltaCost.begin(), m_miniBatchOutputLayerWeightsDeltaCost.end(), 0.0f);

            // xử lý minibatch
            size_t miniBatchIndex = 0;
            while (miniBatchIndex < miniBatchSize && trainingIndex < trainingData.NumImages())
            {
                // lấy vật phẩm huấn luyện
                uint8 imageLabel = 0;
                const float* pixels = trainingData.GetImage(m_trainingOrder[trainingIndex], imageLabel);

                // chạy chuyển tiếp mạng
                uint8 labelDetected = ForwardPass(pixels, imageLabel);

                // chạy ngược lại để lấy đạo hàm của hàm chi phí
                BackwardPass(pixels, imageLabel);

                // cộng các đạo hàm hiện tại vào mảng đạo hàm minibatch để có thể tính trung bình cộng của chúng ở cuối minibatch thông qua phép chia.
                for (size_t i = 0; i < m_hiddenLayerBiasesDeltaCost.size(); ++i)
                    m_miniBatchHiddenLayerBiasesDeltaCost[i] += m_hiddenLayerBiasesDeltaCost[i];
                for (size_t i = 0; i < m_outputLayerBiasesDeltaCost.size(); ++i)
                    m_miniBatchOutputLayerBiasesDeltaCost[i] += m_outputLayerBiasesDeltaCost[i];
                for (size_t i = 0; i < m_hiddenLayerWeightsDeltaCost.size(); ++i)
                    m_miniBatchHiddenLayerWeightsDeltaCost[i] += m_hiddenLayerWeightsDeltaCost[i];
                for (size_t i = 0; i < m_outputLayerWeightsDeltaCost.size(); ++i)
                    m_miniBatchOutputLayerWeightsDeltaCost[i] += m_outputLayerWeightsDeltaCost[i];

                ++trainingIndex;
                ++miniBatchIndex;
            }


            float miniBatchLearningRate = learningRate / float(miniBatchIndex);

            // áp dụng huấn luyện cho độ lệch và trọng số
            for (size_t i = 0; i < m_hiddenLayerBiases.size(); ++i)
                m_hiddenLayerBiases[i] -= m_miniBatchHiddenLayerBiasesDeltaCost[i] * miniBatchLearningRate;
            for (size_t i = 0; i < m_outputLayerBiases.size(); ++i)
                m_outputLayerBiases[i] -= m_miniBatchOutputLayerBiasesDeltaCost[i] * miniBatchLearningRate;
            for (size_t i = 0; i < m_hiddenLayerWeights.size(); ++i)
                m_hiddenLayerWeights[i] -= m_miniBatchHiddenLayerWeightsDeltaCost[i] * miniBatchLearningRate;
            for (size_t i = 0; i < m_outputLayerWeights.size(); ++i)
                m_outputLayerWeights[i] -= m_miniBatchOutputLayerWeightsDeltaCost[i] * miniBatchLearningRate;
        }
    }

    // Hàm này đánh giá mạng cho các pixel đầu vào đã cho và trả về nhãn mà nó cho là từ 0-9
    uint8 ForwardPass(const float* pixels, uint8 correctLabel)
    {
        // đầu tiên làm lớp ẩn
        for (size_t neuronIndex = 0; neuronIndex < HIDDEN_NEURONS; ++neuronIndex)
        {
            float Z = m_hiddenLayerBiases[neuronIndex];

            for (size_t inputIndex = 0; inputIndex < INPUTS; ++inputIndex)
                Z += pixels[inputIndex] * m_hiddenLayerWeights[HiddenLayerWeightIndex(inputIndex, neuronIndex)];

            m_hiddenLayerOutputs[neuronIndex] = 1.0f / (1.0f + exp(-Z));
        }

        // sau đó thực hiện lớp đầu ra
        for (size_t neuronIndex = 0; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
        {
            float Z = m_outputLayerBiases[neuronIndex];

            for (size_t inputIndex = 0; inputIndex < HIDDEN_NEURONS; ++inputIndex)
                Z += m_hiddenLayerOutputs[inputIndex] * m_outputLayerWeights[OutputLayerWeightIndex(inputIndex, neuronIndex)];

            m_outputLayerOutputs[neuronIndex] = 1.0f / (1.0f + exp(-Z));
        }

        // tính toán lỗi.
        // Đây là độ lớn của vectơ Mong muốn - Thực tế.
        // Không cần thiết.
        /*
        {
            error = 0.0f;
            for (size_t neuronIndex = 0; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
            {
                float desiredOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;
                float diff = (desiredOutput - m_outputLayerOutputs[neuronIndex]);
                error += diff * diff;
            }
            error =  sqrt(error);
        }
        */

        // tìm giá trị lớn nhất của lớp đầu ra và trả về chỉ mục đó làm nhãn
        float maxOutput = m_outputLayerOutputs[0];
        uint8 maxLabel = 0;
        for (uint8 neuronIndex = 1; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
        {
            if (m_outputLayerOutputs[neuronIndex] > maxOutput)
            {
                maxOutput = m_outputLayerOutputs[neuronIndex];
                maxLabel = neuronIndex;
            }
        }
        return maxLabel;
    }

    // Hàm lấy giá trị trọng số/độ lệch. Được sử dụng để tạo tệp JSON.
    const  array<float, HIDDEN_NEURONS>& GetHiddenLayerBiases() const { return m_hiddenLayerBiases; }
    const  array<float, OUTPUT_NEURONS>& GetOutputLayerBiases() const { return m_outputLayerBiases; }
    const  array<float, INPUTS* HIDDEN_NEURONS>& GetHiddenLayerWeights() const { return m_hiddenLayerWeights; }
    const  array<float, HIDDEN_NEURONS* OUTPUT_NEURONS>& GetOutputLayerWeights() const { return m_outputLayerWeights; }

private:

    static size_t HiddenLayerWeightIndex(size_t inputIndex, size_t hiddenLayerNeuronIndex)
    {
        return hiddenLayerNeuronIndex * INPUTS + inputIndex;
    }

    static size_t OutputLayerWeightIndex(size_t hiddenLayerNeuronIndex, size_t outputLayerNeuronIndex)
    {
        return outputLayerNeuronIndex * HIDDEN_NEURONS + hiddenLayerNeuronIndex;
    }

    // hàm này sử dụng các giá trị đầu ra nơ-ron từ chuyển tiếp để truyền ngược lỗi
    // của mạng để tính toán độ dốc cần thiết cho việc huấn luyện. Nó tìm ra lỗi gì
    // bằng cách so sánh nhãn mà nó nghĩ ra với nhãn mà lẽ ra nó phải nghĩ ra ( CorrectLabel).
    void BackwardPass(const float* pixels, uint8 correctLabel)
    {
        // vì đang quay ngược lại nên hãy thực hiện lớp đầu ra trước
        for (size_t neuronIndex = 0; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
        {
            // tính deltaCost/deltaBias cho mỗi nơ ron đầu ra.
            // Đây cũng là lỗi của nơ-ron và có cùng giá trị với deltaCost/deltaZ.
            //
            // deltaCost/deltaZ = deltaCost/deltaO * deltaO/deltaZ
            //
            // deltaCost/deltaO = O - desiredOutput
            // deltaO/deltaZ = O * (1 - O)
            //
            float desiredOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;

            float deltaCost_deltaO = m_outputLayerOutputs[neuronIndex] - desiredOutput;
            float deltaO_deltaZ = m_outputLayerOutputs[neuronIndex] * (1.0f - m_outputLayerOutputs[neuronIndex]);

            m_outputLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

            // tính deltaCost/deltaWeight cho mỗi trọng số đi vào nơ-ron
            //
            // deltaCost/deltaWeight = deltaCost/deltaZ * deltaCost/deltaWeight
            // deltaCost/deltaWeight = deltaCost/deltaBias * input
            //
            for (size_t inputIndex = 0; inputIndex < HIDDEN_NEURONS; ++inputIndex)
                m_outputLayerWeightsDeltaCost[OutputLayerWeightIndex(inputIndex, neuronIndex)] = m_outputLayerBiasesDeltaCost[neuronIndex] * m_hiddenLayerOutputs[inputIndex];
        }

        // sau đó thực hiện lớp ẩn
        for (size_t neuronIndex = 0; neuronIndex < HIDDEN_NEURONS; ++neuronIndex)
        {
            // tính deltaCost/deltaBias cho mỗi nơ-ron ẩn.
            // Đây cũng là lỗi của nơ-ron và có cùng giá trị với deltaCost/deltaZ.
            //
            // deltaCost/deltaO =
            //   Sum for each output of this neuron:
            //     deltaCost/deltaDestinationZ * deltaDestinationZ/deltaSourceO
            //
            // deltaCost/deltaDestinationZ đã được tính toán và tồn tại trong m_outputLayerBiasesDeltaCost[destinationNeuronIndex].
            // deltaTargetZ/deltaSourceO là giá trị trọng số kết nối nơron nguồn và nơron đích.
            //
            // deltaCost/deltaZ = deltaCost/deltaO * deltaO/deltaZ
            // deltaO/deltaZ = O * (1 - O)
            //
            float deltaCost_deltaO = 0.0f;
            for (size_t destinationNeuronIndex = 0; destinationNeuronIndex < OUTPUT_NEURONS; ++destinationNeuronIndex)
                deltaCost_deltaO += m_outputLayerBiasesDeltaCost[destinationNeuronIndex] * m_outputLayerWeights[OutputLayerWeightIndex(neuronIndex, destinationNeuronIndex)];
            float deltaO_deltaZ = m_hiddenLayerOutputs[neuronIndex] * (1.0f - m_hiddenLayerOutputs[neuronIndex]);
            m_hiddenLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

            // tính deltaCost/deltaWeight cho mỗi trọng số đi vào nơ-ron
            //
            // deltaCost/deltaWeight = deltaCost/deltaZ * deltaCost/deltaWeight
            // deltaCost/deltaWeight = deltaCost/deltaBias * input
            //
            for (size_t inputIndex = 0; inputIndex < INPUTS; ++inputIndex)
                m_hiddenLayerWeightsDeltaCost[HiddenLayerWeightIndex(inputIndex, neuronIndex)] = m_hiddenLayerBiasesDeltaCost[neuronIndex] * pixels[inputIndex];
        }
    }

private:

    // độ lệch và trọng số
    array<float, HIDDEN_NEURONS>                 m_hiddenLayerBiases;
    array<float, OUTPUT_NEURONS>                 m_outputLayerBiases;

    array<float, INPUTS* HIDDEN_NEURONS>            m_hiddenLayerWeights;
    array<float, HIDDEN_NEURONS* OUTPUT_NEURONS>    m_outputLayerWeights;

    // giá trị kích hoạt nơ-ron hay còn gọi là giá trị "O"
    array<float, HIDDEN_NEURONS>                 m_hiddenLayerOutputs;
    array<float, OUTPUT_NEURONS>                 m_outputLayerOutputs;

    // dẫn xuất của độ lệch và trọng số cho một ví dụ huấn luyện
    array<float, HIDDEN_NEURONS>                 m_hiddenLayerBiasesDeltaCost;
    array<float, OUTPUT_NEURONS>                 m_outputLayerBiasesDeltaCost;

    array<float, INPUTS* HIDDEN_NEURONS>            m_hiddenLayerWeightsDeltaCost;
    array<float, HIDDEN_NEURONS* OUTPUT_NEURONS>    m_outputLayerWeightsDeltaCost;

    // dẫn xuất của độ lệch và trọng số cho minibatch. Trung bình của tất cả các mặt hàng trong minibatch.
    array<float, HIDDEN_NEURONS>                 m_miniBatchHiddenLayerBiasesDeltaCost;
    array<float, OUTPUT_NEURONS>                 m_miniBatchOutputLayerBiasesDeltaCost;

    array<float, INPUTS* HIDDEN_NEURONS>            m_miniBatchHiddenLayerWeightsDeltaCost;
    array<float, HIDDEN_NEURONS* OUTPUT_NEURONS>    m_miniBatchOutputLayerWeightsDeltaCost;

    // được sử dụng để tạo minibatch
    vector<size_t>                                   m_trainingOrder;
};

// ============================================================================================
//                                   DRIVER PROGRAM
// ============================================================================================

// dữ liệu huấn luyện và kiểm tra
CMNISTData g_trainingData;
CMNISTData g_testData;

// neural network
CNeuralNetwork<c_numInputNeurons, c_numHiddenNeurons, c_numOutputNeurons> g_neuralNetwork;

float GetDataAccuracy(const CMNISTData& data)
{
    size_t correctItems = 0;
    for (size_t i = 0, c = data.NumImages(); i < c; ++i)
    {
        uint8 label;
        const float* pixels = data.GetImage(i, label);
        uint8 detectedLabel = g_neuralNetwork.ForwardPass(pixels, label);

        if (detectedLabel == label)
            ++correctItems;
    }
    return float(correctItems) / float(data.NumImages());
}

void ShowImage(const CMNISTData& data, size_t imageIndex)
{
    uint8 label = 0;
    const float* pixels = data.GetImage(imageIndex, label);
    printf("showing a %i\n", label);
    for (int iy = 0; iy < 28; ++iy)
    {
        for (int ix = 0; ix < 28; ++ix)
        {
            if (*pixels < 0.125)
                printf(" ");
            else
                printf("+");
            ++pixels;
        }
        printf("\n");
    }
}
// Hàm tiền xử lý ảnh, bạn cần thay đổi cho phù hợp với dữ liệu đầu vào của mô hình
vector<float> PreprocessImage(const  string& imagePath) {
    // Đọc ảnh từ đường dẫn
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Không thể đọc ảnh từ đường dẫn: " << imagePath << endl;
        return {};
    }

    // Resize ảnh về kích thước mong muốn (vd: 28x28)
    cv::resize(image, image, cv::Size(28, 28));

    // Chuyển đổi ma trận ảnh thành vector float
    vector<float> input;
    input.reserve(image.rows * image.cols);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            input.push_back(static_cast<float>(image.at<uchar>(i, j)) / 255.0f); // Chuẩn hóa giá trị pixel về [0, 1]
        }
    }

    return input;
}
// Hàm dự đoán số từ ảnh đầu vào sử dụng mô hình neural network
uint8_t Predict(const  vector<float>& input) {
    // Sử dụng mô hình neural network đã được huấn luyện để dự đoán
    uint8_t predictedLabel = g_neuralNetwork.ForwardPass(input.data(), 0); // 0 là nhãn tùy ý vì đây là dự đoán
    return predictedLabel;
}
int main(int argc, char** argv)
{
    int i;
    printf("Kernel: \n");
    printf("Kernel 0: Load mnist by host \n");
    printf("Kernel 1: Load mnist by device \n");
    printf("Kernel 2: Load mnist by SMEM \n");
    printf("Kernel 3: Load mnist by CMEM \n");
    printf("Kernel 4: Compare host and kernel \n");
    scanf("Click on sceen: %d" ,&i);
    GpuTimer timer;
    float time, time1, time2, time3;
    switch (i) {
    case 0:
        timer.Start();
        printf("Load MNIST on Host \n");
        if (!g_trainingData.Load(true) || !g_testData.Load(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time = timer.Elapsed();
        printf("Host time: %f ms\n", time);
        break;
    case 1:
        timer.Start();
        printf("Kernel 1: Load MNIST on Device \n");
        if (!g_trainingData.Loadkernel1(true) || !g_testData.Loadkernel1(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time = timer.Elapsed();
        printf("Kernel 1 time: %f ms\n", time);
        break;
    case 2:
        timer.Start();
        printf("Kernel 2: Load MNIST on Device by SMEM \n");
        if (!g_trainingData.Loadkernel2(true) || !g_testData.Loadkernel2(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time = timer.Elapsed();
        printf("Kernel 2 time: %f ms\n", time);
        break;
    case 3:
        timer.Start();
        printf("Kernel 3: Load MNIST on Device by CMEM \n");
        if (!g_trainingData.Loadkernel3(true) || !g_testData.Loadkernel3(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time = timer.Elapsed();
        printf("Kernel 3 time: %f ms\n", time);
        break;
    case 4:
        timer.Start();
        printf("Load MNIST on Host\n");
        if (!g_trainingData.Load(true) || !g_testData.Load(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time = timer.Elapsed();
        timer.Start();
        printf("Kernel 1: Load MNIST on Device\n");
        if (!g_trainingData.Loadkernel1(true) || !g_testData.Loadkernel1(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time1 = timer.Elapsed();
       
        //
        timer.Start();
        printf("Kernel 2: Load MNIST on Device by SMEM \n");
        if (!g_trainingData.Loadkernel2(true) || !g_testData.Loadkernel2(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time2 = timer.Elapsed();
      
        //
        timer.Start();
        printf("Kernel 3: Load MNIST on Device by CMEM \n");
        if (!g_trainingData.Loadkernel3(true) || !g_testData.Loadkernel3(false)) {
            printf("Could not load mnist data, aborting!\n");
            system("pause");
            return 1;
        }
        timer.Stop();
        time3 = timer.Elapsed();
       
        printf("Host time: %f ms\n", time);
        printf("Kernel 1 time: %f ms\n", time1);
        printf("Kernel 2 time: %f ms\n", time2);
        printf("Kernel 3 time: %f ms\n", time3);
        break;
    default:
        printf("Invalid case!\n");
        break;
    }

    printf("\n \n CNeuralNetwork is training!!!\n");
    // huấn luyện mạng, báo lỗi trước mỗi lần huấn luyện
    for (size_t epoch = 0; epoch < c_trainingEpochs; ++epoch)
    {
        printf("Training epoch %zu / %zu...\n", epoch + 1, c_trainingEpochs);
        g_neuralNetwork.Train(g_trainingData, c_miniBatchSize, c_learningRate);
        float accuracyTraining = GetDataAccuracy(g_trainingData);
        float accuracyTest = GetDataAccuracy(g_testData);
        printf("Training Data Accuracy: %0.2f%%\n", 100.0f * accuracyTraining);
        printf("Test Data Accuracy: %0.2f%%\n\n", 100.0f * accuracyTest);
    }


    // Kết quả báo lỗi cuối cùng
    float accuracyTraining = GetDataAccuracy(g_trainingData);
    float accuracyTest = GetDataAccuracy(g_testData);
    printf("\nFinal Training Data Accuracy: %0.2f%%\n", 100.0f * accuracyTraining);
    printf("Final Test Data Accuracy: %0.2f%%\n\n", 100.0f * accuracyTest);

    // Sử dụng đoạn mã như dưới đây để trực quan hóa hình ảnh nếu muốn.
    //ShowImage(g_testData, 9);

    vector< string> imagePaths = { "a.png","b.png","c.png","d.png","e.png" };

    // Dự đoán trên từng bức ảnh
    for (const auto& imagePath : imagePaths) {
        // Tiền xử lý ảnh
        vector<float> input = PreprocessImage(imagePath);

        // Dự đoán số từ ảnh đầu vào
        uint8_t predictedLabel = Predict(input);

        // In ra kết quả dự đoán
        cout << "Predict image " << imagePath << " is: " << static_cast<int>(predictedLabel) << endl;
    }
    return 0;
}