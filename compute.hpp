#ifndef COMPUTE_HPP
#define COMPUTE_HPP

#include "math.hpp"


#include <CL/cl.h>
#include <string>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <assert.h>
#include <algorithm>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/thread/mutex.hpp>

void CL_CALLBACK OclErrorCallback(const char *error_info, 
                             const void *private_info, size_t cb, 
                             void *user_data);

namespace compute {

#define OclCheckError(error_code, message)										\
	if (error_code != CL_SUCCESS) {												\
		std::cerr <<__FILE__ <<":" <<__LINE__									\
				<< " error code \"" << error_code << "\" error message \""		\
				<< message << "\"" << std::endl;								\
		abort();																\
	}

struct Dim {
	Dim() : x(0), y(0), z(0) {}
	Dim(size_t X) : x(X), y(1), z(1) {}
	Dim(size_t X, size_t Y) : x(X), y(Y), z(1) {}
	Dim(size_t X, size_t Y, size_t Z) : x(X), y(Y), z(Z) {}

	Dim GetMin(Dim other) {
		Dim ret;

		ret.x = std::min(x, other.x);
		ret.y = std::min(y, other.y);
		ret.z = std::min(z, other.z);

		return ret;
	}

	Dim GetMax(Dim other) {
		Dim ret;

		ret.x = std::max(x, other.x);
		ret.y = std::max(y, other.y);
		ret.z = std::max(z, other.z);

		return ret;
	}

	size_t GetDimensions() const {
		return size_t(x > 1) + size_t(y > 1) + size_t(z > 1);
	}

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
public:
#ifdef _WIN32
typedef HDC WindowId;
#else
typedef void * WindowId;
#endif

	Device(WindowId window_id = NULL);
    
	void ErrorCallback(const char *error_info, const void *, size_t);
    
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

	Dim GetMaxLocalWorkItems() {
		Dim ret;

		cl_int status = clGetDeviceInfo(device_id_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
										sizeof(ret.array), &ret.array, NULL);
		OclCheckError(status, "clGetDeviceInfo()");

		return ret;
	}

	cl_ulong GetLocalMemorySize() {
		cl_ulong ret;

		cl_int status = clGetDeviceInfo(device_id_, CL_DEVICE_LOCAL_MEM_SIZE,
										sizeof(ret), &ret, NULL);
		OclCheckError(status, "clGetDeviceInfo()");

		return ret;
	}

	cl_uint GetMaxFrequency() {
		cl_ulong ret;

		cl_int status = clGetDeviceInfo(device_id_, CL_DEVICE_MAX_CLOCK_FREQUENCY,
										sizeof(ret), &ret, NULL);
		OclCheckError(status, "clGetDeviceInfo()");

		return ret;
	}

	cl_uint GetMaxComputeUnits() {
		cl_ulong ret;

		cl_int status = clGetDeviceInfo(device_id_, CL_DEVICE_MAX_COMPUTE_UNITS,
										sizeof(ret), &ret, NULL);
		OclCheckError(status, "clGetDeviceInfo()");

		return ret;
	}

private:
	cl_device_id device_id_;
	cl_context context_;
	cl_command_queue command_queue_;

	struct DeviceSelector {
		static boost::mutex lock_;
		static std::vector<cl_device_id> ids_;
		static unsigned next_;
	};
	DeviceSelector sel_;
};

template< typename T >
class LocalBuffer {
public:
    LocalBuffer(size_t element_count) 
		: element_count_(element_count)
    {}
    
    size_t Size() const {
        return element_count_;
    }
    
    size_t SizeBytes() const {
		return element_count_*sizeof(T);
    }
    
private:
    size_t element_count_;
};

template< typename T >
class Buffer : public std::vector<T> {
	/* only POD types are supported*/
	BOOST_STATIC_ASSERT((boost::is_same<double,  T>::value ||
						 boost::is_same<float, T>::value ||
						 boost::is_same<short, T>::value ||
						 boost::is_same<unsigned short, T>::value ||
						 boost::is_same<int, T>::value ||
						 boost::is_same<unsigned int, T>::value ||
						 boost::is_same<long, T>::value ||
						 boost::is_same<unsigned long, T>::value));
public:

	Buffer(Device *device, size_t size = 0)
		: device_(device), device_buffer_(NULL)
	{
		if (size)
			std::vector<T>::resize(size);
	}

	Buffer(const Buffer &other)
		: std::vector<T>(other), device_(other.device_), device_buffer_(NULL) {
	}

	Buffer &operator=(const Buffer &other) {
		std::vector<T>::operator =(other);
		return *this;
	}

	Buffer(Device *device, const std::vector< T > &other)
		: std::vector< T >(other), device_(device), device_buffer_(NULL) {
	}
    
	~Buffer() {
		if (device_buffer_) {
			clReleaseMemObject(device_buffer_);
			device_buffer_ = NULL;
		}
    }

    void CopyToHost() {
		cl_int status = clEnqueueReadBuffer(device_->get_command_queue(),
											device_buffer_, CL_FALSE, 0,
											SizeBytes(), std::vector<T>::data(),
											0, NULL, NULL);
		OclCheckError(status, "clEnqueueReadBuffer");
    }
    
	void CopyToDevice() {
		SyncGPUBuffer();

		cl_int status = clEnqueueWriteBuffer(device_->get_command_queue(),
											 device_buffer_, CL_FALSE, 0,
											 SizeBytes(), std::vector<T>::data(),
											 0, NULL, NULL);
		OclCheckError(status, "clEnqueueWriteBuffer");
    }

	cl_uint GetReferenceCount() {
		if (!device_buffer_)
			return 0;

		cl_uint ret;
		cl_int status = clGetMemObjectInfo(device_buffer_, CL_MEM_REFERENCE_COUNT,
										   sizeof(ret), &ret, NULL);
		OclCheckError(status, "clGetMemObjectInfo");

		return ret;
	}

	size_t SizeBytes() {
		return std::vector<T>::size()*sizeof(T);
	}

	size_t DeviceBufferBytes() {
		if (!device_buffer_)
			return 0;

		size_t ret = 0;

		cl_int status = clGetMemObjectInfo(device_buffer_, CL_MEM_SIZE,
										   sizeof(ret), &ret, NULL);
		OclCheckError(status, "clGetMemObjectInfo");

		return ret;
	}

	cl_mem *GetDeviceMem() {
        return &device_buffer_;
    }
    
	void SyncGPUBuffer() {
		if (DeviceBufferBytes() < SizeBytes() || !device_buffer_) {
			if (device_buffer_) {
				clReleaseMemObject(device_buffer_);
			}

			cl_int cl_status = CL_SUCCESS;
			device_buffer_ = clCreateBuffer(device_->get_context(),
											CL_MEM_READ_WRITE,
											SizeBytes(), NULL, &cl_status);
			OclCheckError(cl_status, "clCreateBuffer");
		}
	}

	void CopyToDeviceBuffer(Buffer< T > &dst, size_t dst_pos, size_t src_pos, size_t len) {
		assert(device_buffer_ && dst.device_buffer_);
		assert(dst_pos + len <= dst.DeviceBufferBytes()/sizeof(T));
		assert(src_pos + len <= DeviceBufferBytes()/sizeof(T));

		cl_int status = clEnqueueCopyBuffer(device_->get_command_queue(),
											device_buffer_, dst.device_buffer_,
											src_pos, dst_pos, len*sizeof(T), 0, NULL,
											NULL);
		OclCheckError(status, "clEnqueueCopyBuffer");
	}

	void FillDeviceBuffer(T value, size_t count, size_t offset = 0) {
		assert(device_buffer_);
		assert(count);

		cl_int status = clEnqueueFillBuffer(device_->get_command_queue(),
											device_buffer_, &value, sizeof(T),
											sizeof(T)*offset, sizeof(T)*count,
											0, NULL, NULL);
		OclCheckError(status, "clEnqueueFillBuffer");
	}

private:

    // the device this buffer is located on
    Device *device_;
    
    // reference to the device buffer
	cl_mem device_buffer_;
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
			return clSetKernelArg(kernel, index, sizeof(cl_mem), arg.GetDeviceMem());
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
	~Kernel();
    
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

		assert(local_size.GetDimensions() == global_size.GetDimensions());
        cl_status = clEnqueueNDRangeKernel(program_->get_device()->get_command_queue(), kernel_, 
										   local_size.GetDimensions(), NULL, global_size.array,
										   local_size.array, 0, NULL, NULL);
        OclCheckError(cl_status, "enqueue kernel");
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

std::ostream &operator<<(std::ostream &out, const compute::Dim &dim);

#endif // COMPUTE_HPP
