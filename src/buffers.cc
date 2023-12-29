#include "buffers.h"


namespace infer_sdk{
    using namespace std;
    using namespace nvinfer1;

    int get_size(const vector<int> &shape)
    {
        if (shape.empty())
        {
            return 0;
        }
        auto size = std::accumulate(begin(shape), end(shape), 1, std::multiplies<>());
        return max(0, size);
    }

    int Buffer::cal_offset(const std::vector<int>& shape, const std::vector<int>& index){
        assert(shape.size() == index.size());
        int sum = 0;
        for(int i = 0; i < index.size(); i++){
            sum += index[i] * std::accumulate(begin(shape) + i + 1, end(shape), 1, std::multiplies<>());
        }
        return sum;
    }


    Buffer::Buffer(const vector<int> &shape, DataType type, DeviceType device) 
    : shape_(shape), type_(type), device_(device), size_(get_size(shape)), capacity_(size_){
        switch(device_){
            case DeviceType::HOST:
                alloc_func_ = HostAllocator();
                free_func_ = HostFree();
                break;
            case DeviceType::DEVICE:
                alloc_func_ = DeviceAllocator();
                free_func_ = DeviceFree();
                break;
        }
        if(!alloc_func_(&buffer_, byte_size())){
            throw std::bad_alloc();
        }
    }

    int Buffer::dim(int i){
        if(i >= 0 && i < shape_.size()){
            return shape_[i];
        }
        if(i < 0 && i >= -shape_.size()){
            return shape_[shape_.size() + i];
        }
        SPDLOG_ERROR("index out of range");
    }

    void Buffer::resize(const vector<int>& shape){
        auto new_size = get_size(shape);
        size_ = new_size;
        if(new_size > capacity_){
            free_func_(buffer_);
            if (!alloc_func_(&buffer_, byte_size()))
            {
                throw std::bad_alloc();
            }
            capacity_ = new_size;
        }
    }

}