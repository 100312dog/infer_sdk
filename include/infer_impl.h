#ifndef INFER_IMPL_H
#define INFER_IMPL_H

#include "backend.h"

namespace infer_sdk {
    class FeatInfer;
    class DetInfer;

    template<typename Input, typename Output>
    class InferImpl {
    public:
#ifdef TENSORRT_BACKEND

        Output infer(const Input &input, cudaStream_t stream = nullptr) {
            preprocess(input);
            forward(stream);
            postprocess();
            return output_;
        };
#endif
        // todo multi batch
//        Output batch_infer(const std::vector<Input>& input){
//            多线程前处理 后处理
//        }
    private:
        virtual void preprocess(const Input &input) = 0;

#ifdef TENSORRT_BACKEND

        void forward(cudaStream_t stream = nullptr) {
            backend_->copy_inputs_from_host_to_device(stream);
            backend_->forward(stream);
            backend_->copy_outputs_from_device_to_host(stream);
#endif
        }

        virtual void postprocess() = 0;

    protected:
        std::shared_ptr<Backend> backend_;
        Output output_;
        friend class FeatInfer;
    };
}
#endif
