#include "backend.h"


namespace infer_sdk{
    using namespace nvinfer1;
    using namespace std;
    // class Logger : public ILogger           
    // {
    //     void log(Severity severity, const char* msg) noexcept override
    //     {
    //         // suppress info-level messages
    //         if (severity <= Severity::kWARNING)
    //             SPDLOG_INFO("{}", msg);
    //     }
    // } glogger;

    class Logger : public ILogger
    {
    public:
        Severity reportableSeverity;

        Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

        void log(Severity severity, const char *msg) noexcept override
        {
            if (severity > reportableSeverity)
            {
                return;
            }
            switch (severity)
            {
            case Severity::kINTERNAL_ERROR:
                SPDLOG_ERROR("INTERNAL_ERROR: {}", msg);
                break;
            case Severity::kERROR:
                SPDLOG_ERROR("ERROR: {}", msg);
                break;
            case Severity::kWARNING:
                SPDLOG_INFO("WARNING: {}", msg);
                break;
            case Severity::kINFO:
                SPDLOG_INFO("INFO: {}", msg);
                break;
            default:
                SPDLOG_INFO("VERBOSE: {}", msg);
                break;
            }
        }
    };

    static Logger gLogger(ILogger::Severity::kERROR);

    vector<char> Backend::load_model(const string& model_path){
        ifstream model_file(model_path, ios::in | ios::binary);
        if (!model_file.is_open()){
            SPDLOG_ERROR("Failed opening file: {}", model_path.c_str());
            return {};
        }

        model_file.seekg(0, ios::end);
        size_t fsize = model_file.tellg();
        if (fsize == 0)
        {
            SPDLOG_ERROR("Empty file: {}", model_path.c_str());
            model_file.close();
            return {};
        }

        std::vector<char> model_data(fsize);
        model_file.seekg(0, ios::beg);
        model_file.read(model_data.data(), fsize);
        model_file.close();
        return model_data;
    }

    shared_ptr<Buffer> Backend::inputs(int i){
        if(num_inputs() > 0 && i < num_inputs()){
            return inputs_[i];
        }else{
            SPDLOG_ERROR("input index {} out of range", i);
            return {};
        }
    }

    shared_ptr<Buffer> Backend::inputs(const string& name){
        auto iter = find(input_names_.begin(), input_names_.end(), name);
        if(iter != input_names_.end()){
            return inputs_[distance(input_names_.begin(), iter)];
        }else{
            SPDLOG_ERROR("input {} not found", name);
            return {};
        }
    }

    shared_ptr<Buffer> Backend::outputs(int i){
        if(num_outputs() > 0 && i < num_outputs()){
            return outputs_[i];
        }else{
            SPDLOG_ERROR("host output index {} out of range", i);
            return {};
        }
    }

    shared_ptr<Buffer> Backend::outputs(const string& name){
        auto iter = find(output_names_.begin(), output_names_.end(), name);
        if(iter != output_names_.end()){
            return outputs_[distance(output_names_.begin(), iter)];
        }else{
            SPDLOG_ERROR("host output {} not found", name);
            return {};
        }
    }

    TensorRTBackend::TensorRTBackend(const string& model_path, int max_batch_size):
    max_batch_size_(max_batch_size)
    {
        auto model_data = load_model(model_path);
        if(!setup_env(model_data)){
            SPDLOG_ERROR("Failed to setup env");
        }
        alloc_input_output_buffers();
//        set_device_buffers_address();
        print();
    }

//    void TensorRTBackend::forward(cudaStream_t stream){
//        context_->enqueueV3(stream);
//    }

    void TensorRTBackend::forward(cudaStream_t stream){
        vector<void *> device_buffers;
        for(int i = 0; i < num_inputs(); i++) {
            device_buffers.emplace_back(device_inputs_[i]->data_ptr());
        }
        for(int i = 0; i < num_outputs(); i++) {
            device_buffers.emplace_back(device_outputs_[i]->data_ptr());
        }
        context_->enqueueV2(device_buffers.data(), stream, nullptr);
    }

    void TensorRTBackend::print(){
        SPDLOG_INFO("\tInputs: {}", num_inputs());
        for(int i = 0; i < num_inputs(); i++){
            auto name = input_names_[i];
            auto shape = shape_to_string(inputs(i)->shape());
            auto type = data_type_to_string(inputs(i)->type());
            SPDLOG_INFO("\t\t{}.{} : shape {}, {}", i, name, shape, type);
        }
        SPDLOG_INFO("\tOutputs: {}", num_outputs());
        for(int i = 0; i < num_outputs(); i++){
            auto name = output_names_[i];
            auto shape = shape_to_string(outputs(i)->shape());
            auto type = data_type_to_string(outputs(i)->type());
            SPDLOG_INFO("\t\t{}.{} : shape {}, {}", i, name, shape, type);
        }
    }

    bool TensorRTBackend::setup_env(const vector<char>& engine_data)
    {
        // todo only one global runtime?
        runtime_ = nv_unique_ptr<IRuntime>(createInferRuntime(gLogger));
        if(runtime_ == nullptr){
            SPDLOG_ERROR("Failed to create runtime");
            return false;
        }
        engine_ = nv_unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()), NvObjDeleter());
        if(engine_ == nullptr){
            SPDLOG_ERROR("Failed to create engine");
            return false;
        }
        context_ = nv_unique_ptr<IExecutionContext>(engine_->createExecutionContext());
        if(context_ == nullptr){
            SPDLOG_ERROR("Failed to create context");
            return false;
        }
        return true;
    }

//    void TensorRTBackend::alloc_input_output_buffers(){
//        int nIO = static_cast<int>(engine_->getNbIOTensors());
//        for (int i = 0; i < nIO; ++i)
//        {
//            auto name = engine_->getIOTensorName(i);
//            auto shape = engine_->getTensorShape(name);
//            if(shape.d[0] == -1){
//                shape.d[0] = max_batch_size_;
//            }
//            vector<int> shape_vec;
//            for(int i = 0; i < shape.nbDims; i++)
//                shape_vec.emplace_back(shape.d[i]);
//
//            auto type = engine_->getTensorDataType(name);
//            auto mode = engine_->getTensorIOMode(name);
//            switch(mode){
//                case TensorIOMode::kINPUT:
//                    context_->setInputShape(name, shape);
//                    input_names_.emplace_back(name);
//                    inputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type)));
//                    device_inputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type), DeviceType::DEVICE));
//                    break;
//                case TensorIOMode::kOUTPUT:
//                    output_names_.emplace_back(name);
//                    outputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type)));
//                    device_outputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type), DeviceType::DEVICE));
//                    break;
//            }
//        }
//    }

    void TensorRTBackend::alloc_input_output_buffers() {
        int nIO = static_cast<int>(engine_->getNbBindings());
        for (int i = 0; i < nIO; ++i)
        {
            auto name = engine_->getBindingName(i);
            auto shape = engine_->getBindingDimensions(i);
            if(shape.d[0] == -1){
                shape.d[0] = max_batch_size_;
            }
            vector<int> shape_vec;
            for(int i = 0; i < shape.nbDims; i++)
                shape_vec.emplace_back(shape.d[i]);

            auto type = engine_->getBindingDataType(i);
            if(engine_->bindingIsInput(i)){
                context_->setBindingDimensions(i, shape);
                input_names_.emplace_back(name);
                inputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type)));
                device_inputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type), DeviceType::DEVICE));
            }else{
                output_names_.emplace_back(name);
                outputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type)));
                device_outputs_.emplace_back(make_shared<Buffer>(shape_vec, data_type(type), DeviceType::DEVICE));
            }
        }
    }

//    void TensorRTBackend::set_device_buffers_address(){
//        for(int i = 0; i < num_inputs(); i++){
//            context_->setTensorAddress(input_names_[i].c_str(), device_inputs_[i]->data_ptr());
//        }
//        for(int i = 0; i < num_outputs(); i++){
//            context_->setTensorAddress(output_names_[i].c_str(), device_outputs_[i]->data_ptr());
//        }
//    }

    void TensorRTBackend::copy_inputs_from_host_to_device(cudaStream_t stream){
        for(int i = 0; i < num_inputs(); i++){
            auto host_input = inputs_[i];
            auto device_input = device_inputs_[i];
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(device_input->data_ptr(), host_input->data_ptr(), host_input->byte_size(), cudaMemcpyHostToDevice, stream));
        }
    }

    void TensorRTBackend::copy_outputs_from_device_to_host(cudaStream_t stream){
        for(int i = 0; i < num_outputs(); i++){
            auto host_output = outputs_[i];
            auto device_output = device_outputs_[i];
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(host_output->data_ptr(), device_output->data_ptr(), device_output->byte_size(), cudaMemcpyDeviceToHost, stream));
        }
        cudaStreamSynchronize(stream);
    }
    
}