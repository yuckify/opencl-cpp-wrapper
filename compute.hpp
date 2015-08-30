#ifndef COMPUTE_HPP
#define COMPUTE_HPP

#include "os.hpp"
#include "math.hpp"


#include <CL/cl.h>
#include <string>
#include <iostream>
#include <vector>
#include <typeinfo>

void CL_CALLBACK OclErrorCallback(const char *error_info, 
                             const void *private_info, size_t cb, 
                             void *user_data);

namespace compute {

void OclCheckError(cl_int error_code, std::string message);
void OclCheckError(cl_int error_code, const char *message);

struct Dim {
    Dim(size_t X) : dimensions(1), x(X), y(1), z(1) {}
    Dim(size_t X, size_t Y) : dimensions(2), x(X), y(Y), z(1) {}
    Dim(size_t X, size_t Y, size_t Z) : dimensions(3), x(X), y(Y), z(Z) {}
    Dim(size_t dim, size_t X, size_t Y, size_t Z) :
        dimensions(dim), x(X), y(Y), z(Z) {}
    
    size_t dimensions;
    union {
        struct {
            size_t x;
            size_t y;
            size_t z;
        };
        size_t array[3];
    };
};

class Device {
    friend class Kernel;
    template< typename T > friend class Buffer;
public:
    Device(Os::WindowId window_id = NULL);
    
    void ErrorCallback(const char *error_info, const void *private_info, 
                       size_t cb);
    
    void Wait();
    
    cl_context get_context() const {
        return context_;
    }
    
    cl_command_queue get_command_queue() const {
        return command_queue_;
    }
    
    cl_device_id get_device_id() const {
        return device_id_;
    }
    
protected:
    void AddEvent(cl_event event);
    
private:
    
    cl_device_id device_id_;
    cl_context context_;
    cl_command_queue command_queue_;
    
    // track the events so they may be waited on in the future
    std::vector< cl_event > recent_events_;
};

template< typename T >
class LocalBuffer {
public:
    LocalBuffer(size_t element_count) 
        : element_count_(element_count), buffer_size_(element_count*sizeof(T)) 
    {}
    
    size_t Size() const {
        return element_count_;
    }
    
    size_t SizeBytes() const {
        return buffer_size_;
    }
    
private:
    size_t element_count_;
    size_t buffer_size_;
};

template< typename T >
class Buffer {
public:
    Buffer(Device *device, size_t element_count) {
        device_ = device;
        element_count_ = element_count;
        buffer_size_ = sizeof(T)*element_count;
        host_buffer_ = Os::Memory::Malloc(buffer_size_);
        cl_int cl_status = CL_SUCCESS;
        
        device_buffer_ = clCreateBuffer(device->get_context(), 
                                        CL_MEM_READ_WRITE,
                                        buffer_size_, NULL, &cl_status);
        OclCheckError(cl_status, "clCreateBuffer");
        
    }
    
    ~Buffer() {
        clReleaseMemObject(device_buffer_);
        Os::Memory::Free(host_buffer_);
    }
    
    void CopyToHost() {
        cl_event event = NULL;
        cl_int cl_status = clEnqueueReadBuffer(device_->get_command_queue(), 
                                               device_buffer_, CL_FALSE, 0, 
                                               buffer_size_, host_buffer_, 
                                               0, NULL, &event);
        OclCheckError(cl_status, "read buffer");
        device_->AddEvent(event);
    }
    
    void CopyToDevice() {
        cl_event event = NULL;
        cl_int cl_status = clEnqueueWriteBuffer(device_->get_command_queue(), 
                                                device_buffer_, CL_FALSE, 0, 
                                                buffer_size_, host_buffer_, 
                                                0, NULL, &event);
        OclCheckError(cl_status, "write buffer");
        device_->AddEvent(event);
    }
    
    T &operator[](size_t i) {
        return reinterpret_cast< T * >(host_buffer_)[i];
    }
    
    void Clear() {
        ::memset(host_buffer_, 0, buffer_size_);
    }
    
    size_t Size() const {
        return element_count_;
    }
    
    size_t SizeBytes() const {
        return buffer_size_;
    }
    
    cl_mem *get_mem() {
        return &device_buffer_;
    }
    
private:
    // the device this buffer is located on
    Device *device_;
    
    // reference to the device buffer
    cl_mem device_buffer_;
    // the buffer allocated on the host
    void *host_buffer_;
    
    // buffer size metrics
    // size of the buffer in bytes
    size_t buffer_size_;
    // number of elements of T in the buffer
    size_t element_count_;
};

class Program {
    friend class Kernel;
public:
    Program(Device *device, std::string source);
    
    Device *get_device() const { return device_; }
    
protected:
    cl_program get_program() const { return program_; }
    
private:
    Device *device_;
    cl_program program_;
};

class Kernel {
    template< typename T >
    struct Arg {
        static cl_int Set(cl_kernel, cl_uint, T &) {
            std::cout << "unimplemented argument for type \"" 
                      << typeid(T).name() << "\"" << std::endl;
            abort();
            return -1;
        }
    };
    template< typename T >
    struct Arg< Buffer< T > > {
        static cl_int Set(cl_kernel kernel, cl_uint index, Buffer< T > &arg) {
            return clSetKernelArg(kernel, index, sizeof(cl_mem), arg.get_mem());
        }
    };
    template< typename T >
    struct Arg< LocalBuffer< T > > {
        static cl_int Set(cl_kernel kernel, cl_uint index, LocalBuffer< T > &arg) {
            return clSetKernelArg(kernel, index, arg.SizeBytes(), NULL);
        }
    };
    
public:
    Kernel(Program *program, std::string kernel_name);
    
    template< typename A, typename B, typename C, typename D, typename E,
              typename F, typename G, typename H, typename I, typename J >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2, D &arg_3,
                    E &arg_4, F &arg_5, G &arg_6, H &arg_7, I &arg_8, 
                    J &arg_9) {
        cl_int cl_status = Arg< J >::Set(kernel_, 9, arg_9);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,
                arg_7, arg_8);
    }
    
    template< typename A, typename B, typename C, typename D, typename E,
              typename F, typename G, typename H, typename I >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2, D &arg_3,
                    E &arg_4, F &arg_5, G &arg_6, H &arg_7, I &arg_8) {
        cl_int cl_status = Arg< I >::Set(kernel_, 8, arg_8);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,
                arg_7);
    }
    
    template< typename A, typename B, typename C, typename D, typename E,
              typename F, typename G, typename H >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2, D &arg_3,
                    E &arg_4, F &arg_5, G &arg_6, H &arg_7) {
        cl_int cl_status = Arg< H >::Set(kernel_, 7, arg_7);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6);
    }
    
    template< typename A, typename B, typename C, typename D, typename E,
              typename F, typename G >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2, D &arg_3,
                    E &arg_4, F &arg_5, G &arg_6) {
        cl_int cl_status = Arg< G >::Set(kernel_, 6, arg_6);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5);
    }
    
    template< typename A, typename B, typename C, typename D, typename E,
              typename F >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2, D &arg_3,
                    E &arg_4, F &arg_5) {
        cl_int cl_status = Arg< F >::Set(kernel_, 5, arg_5);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1, arg_2, arg_3, arg_4);
    }
    
    template< typename A, typename B, typename C, typename D, typename E >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2, D &arg_3,
                    E &arg_4) {
        cl_int cl_status = Arg< E >::Set(kernel_, 4, arg_4);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1, arg_2, arg_3);
    }
    
    template< typename A, typename B, typename C, typename D >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2, D &arg_3) {
        cl_int cl_status = Arg< D >::Set(kernel_, 3, arg_3);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1, arg_2);
    }
    
    template< typename A, typename B, typename C >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1, C &arg_2) {
        cl_int cl_status = Arg< C >::Set(kernel_, 2, arg_2);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0, arg_1);
    }
    
    template< typename A, typename B >
    void operator()(Dim local_size, Dim global_size, A &arg_0, B &arg_1) {
        cl_int cl_status = Arg< B >::Set(kernel_, 1, arg_1);
        OclCheckError(cl_status, "set kernel argument");
        (*this)(local_size, global_size, arg_0);
    }
    
    template< typename A >
    void operator()(Dim local_size, Dim global_size, A& arg_0) {
        cl_int cl_status = Arg< A >::Set(kernel_, 0, arg_0);
        
        cl_event event = NULL;
        cl_status = clEnqueueNDRangeKernel(program_->get_device()->get_command_queue(), kernel_, 
                                           1, NULL, global_size.array, 
                                           local_size.array, 0, NULL, &event);
        OclCheckError(cl_status, "enqueue kernel");
        
        program_->get_device()->AddEvent(event);
    }
    
private:
    Program *program_;
    cl_kernel kernel_;
}; // class Kernel

template<>
struct Kernel::Arg< cl_int > {
	static cl_int Set(cl_kernel kernel, cl_uint index, cl_int &arg) {
		return clSetKernelArg(kernel, index, sizeof(cl_int), &arg);
	}
};
template<>
struct Kernel::Arg< cl_uint > {
	static cl_int Set(cl_kernel kernel, cl_uint index, cl_uint &arg) {
		return clSetKernelArg(kernel, index, sizeof(cl_uint), &arg);
	}
};
template<>
struct Kernel::Arg< cl_long > {
	static cl_int Set(cl_kernel kernel, cl_uint index, cl_long &arg) {
		return clSetKernelArg(kernel, index, sizeof(cl_long), &arg);
	}
};
template<>
struct Kernel::Arg< cl_ulong > {
	static cl_int Set(cl_kernel kernel, cl_uint index, cl_ulong &arg) {
		return clSetKernelArg(kernel, index, sizeof(cl_ulong), &arg);
	}
};
template<>
struct Kernel::Arg< cl_float > {
	static cl_int Set(cl_kernel kernel, cl_uint index, cl_float &arg) {
		return clSetKernelArg(kernel, index, sizeof(cl_float), &arg);
	}
};
template<>
struct Kernel::Arg< cl_double > {
	static cl_int Set(cl_kernel kernel, cl_uint index, cl_double &arg) {
		return clSetKernelArg(kernel, index, sizeof(cl_double), &arg);
	}
};

}  // namespace compute

#endif // COMPUTE_HPP
