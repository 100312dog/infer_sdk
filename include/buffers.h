#ifndef BUFFERS_H
#define BUFFERS_H

#ifdef TENSORRT_BACKEND
#include <cuda_runtime_api.h>
#endif
#include <cstdlib>
#include <memory>
#include "common.h"
#include <cassert>
#include <spdlog/spdlog.h>


namespace infer_sdk{

    class Buffer
    {
    public:
        static int cal_offset(const std::vector<int>& shape, const std::vector<int>& index);
        template<typename T>
        T at(const std::vector<int>& index){
            return *(data_ptr<T>() + cal_offset(shape_, index));
        }
        Buffer(const std::vector<int>& shape, DataType type, DeviceType device=DeviceType::HOST);
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        int ndims() {return shape_.size();}
        int dim(int i);
        int size() {return size_;}
        int byte_size() { return size_ * element_size(type_);};
        DataType type() { return type_;};
        std::vector<int> shape() const {return shape_;};
        void resize(const std::vector<int>& shape);

//        template<typename T>
//        T get(const std::vector<int>& index){
//            assert(index.size() == shape_.size());
//            int sum = 0;
//            for(int i = 0; i < index.size(); i++){
//                if(index[i] > 0 && index[i] < shape_[i]){
//                    sum += index[i] * std::accumulate(shape_.begin() + i + 1, shape_.end(), 1, std::multiplies<>());
//                }else{
//                    SPDLOG_ERROR("index out of range");
//                }
//            }
//            return *(data_ptr<T>() + sum);
//        }

        template<typename T=void>
        T* data_ptr(){
            return static_cast<T*>(buffer_);
        }
        template<typename T>
        void print_buffer_info(int n = 10){
            int nElement = size();
            auto pArray = data_ptr<T>();
            double sum      = double(pArray[0]);
            double absSum   = double(fabs(double(pArray[0])));
            double sum2     = double(pArray[0]) * double(pArray[0]);
            double diff     = 0.0;
            double maxValue = double(pArray[0]);
            double minValue = double(pArray[0]);
            for (int i = 1; i < nElement; ++i)
            {
                sum += double(pArray[i]);
                absSum += double(fabs(double(pArray[i])));
                sum2 += double(pArray[i]) * double(pArray[i]);
                maxValue = double(pArray[i]) > maxValue ? double(pArray[i]) : maxValue;
                minValue = double(pArray[i]) < minValue ? double(pArray[i]) : minValue;
                diff += abs(double(pArray[i]) - double(pArray[i - 1]));
            }
            double mean = sum / nElement;
            double var  = sum2 / nElement - mean * mean;

            std::cout << "absSum=" << std::fixed << std::setprecision(4) << std::setw(7) << absSum << ",";
            std::cout << "mean=" << std::fixed << std::setprecision(4) << std::setw(7) << mean << ",";
            std::cout << "var=" << std::fixed << std::setprecision(4) << std::setw(7) << var << ",";
            std::cout << "max=" << std::fixed << std::setprecision(4) << std::setw(7) << maxValue << ",";
            std::cout << "min=" << std::fixed << std::setprecision(4) << std::setw(7) << minValue << ",";
            std::cout << "diff=" << std::fixed << std::setprecision(4) << std::setw(7) << diff << ",";
            std::cout << std::endl;

            // print first n element and last n element
            for (int i = 0; i < n; ++i)
            {
                std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
            }
            std::cout << std::endl;
            for (int i = nElement - n; i < nElement; ++i)
            {
                std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
            }
            std::cout << std::endl;
        };
        ~Buffer(){free_func_(buffer_);};

    private:
        size_t size_, capacity_;
        std::vector<int> shape_;
        DeviceType device_;
        DataType type_;
        void *buffer_;
        std::function<bool(void**, size_t)> alloc_func_;
        std::function<void(void*)> free_func_;
    };

    class DeviceAllocator
    {
    public:
        bool operator()(void **ptr, size_t size) const
        {
            return CUDA_RUNTIME_CHECK(cudaMalloc(ptr, size));
        }
    };

    class DeviceFree
    {
    public:
        void operator()(void *ptr) const
        {
            cudaFree(ptr);
        }
    };

    class HostAllocator
    {
    public:
        bool operator()(void **ptr, size_t size) const
        {
#ifdef TENSORRT_BACKEND
            return CUDA_RUNTIME_CHECK(cudaMallocHost(ptr, size));
#else
            *ptr = malloc(size);
            return *ptr != nullptr;
#endif
        }
    };

    class HostFree
    {
    public:
        void operator()(void *ptr) const
        {
#ifdef TENSORRT_BACKEND
            CUDA_RUNTIME_CHECK(cudaFreeHost(ptr));
#else
            free(ptr);
#endif
        }
    };

    int get_size(const std::vector<int> &shape);
}


#endif