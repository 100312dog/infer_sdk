#ifndef BACKEND_H
#define BACKEND_H

#include <memory>
#include <spdlog/spdlog.h>
#include <fstream>
#include "buffers.h"
#include "attr.h"

namespace infer_sdk{

    class Backend{
    public:
        std::vector<char> load_model(const std::string& model_path);
        int num_inputs(){return input_names_.size();};
        int num_outputs(){return output_names_.size();};
        std::shared_ptr<Buffer> inputs(int i);
        std::shared_ptr<Buffer> inputs(const std::string& name);
        std::shared_ptr<Buffer> outputs(int i);
        std::shared_ptr<Buffer> outputs(const std::string& name);
#ifdef TENSORRT_BACKEND
        virtual void forward(cudaStream_t stream) = 0;
        virtual void copy_inputs_from_host_to_device(cudaStream_t stream) = 0;
        virtual void copy_outputs_from_device_to_host(cudaStream_t stream) = 0;
#else
        virtual void forward() = 0;
#endif
        virtual ~Backend() = default;
    protected:
        std::vector<std::shared_ptr<Buffer>> inputs_;
        std::vector<std::shared_ptr<Buffer>> outputs_;
#ifdef TENSORRT_BACKEND
        std::vector<std::shared_ptr<Buffer>> device_inputs_;
        std::vector<std::shared_ptr<Buffer>> device_outputs_;
#endif
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
    };

    struct NvObjDeleter
    {
        template <typename T>
        void operator()(T *obj) const{
            delete obj;
        }
    };

    template <typename T>
    using nv_unique_ptr = std::unique_ptr<T, NvObjDeleter>;
    
    class TensorRTBackend:public Backend{
    public:
        explicit TensorRTBackend(const std::string& model_path, int max_batch_size=1);
        void forward(cudaStream_t stream=nullptr) override;
        void copy_inputs_from_host_to_device(cudaStream_t stream=nullptr) override;
        void copy_outputs_from_device_to_host(cudaStream_t stream=nullptr) override;
    private:
        void print();
        bool setup_env(const std::vector<char>& engine_data);
        void alloc_input_output_buffers();
//        void set_device_buffers_address();
        // todo deal with multi batch, set batch before inference
        // void set_batch_size(int i = 0);
        nv_unique_ptr<nvinfer1::IRuntime> runtime_;
        nv_unique_ptr<nvinfer1::ICudaEngine> engine_;
        nv_unique_ptr<nvinfer1::IExecutionContext> context_;
        int max_batch_size_;
    };

}


#endif