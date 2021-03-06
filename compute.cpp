#include "compute.hpp"
#include <CL/cl_gl.h>

#include <algorithm>
#include <iostream>

#if defined(__windows__)
#include <Windows.h>
#endif

#if defined(DEBUG)
#define DEBUG_STATEMENT(x) x
#else
#define DEBUG_STATEMENT(x)
#endif

boost::mutex compute::Device::DeviceSelector::lock_;
std::vector<cl_device_id> compute::Device::DeviceSelector::ids_;
unsigned compute::Device::DeviceSelector::next_ = 0;

void CL_CALLBACK OclErrorCallback(const char *error_info, 
                             const void *private_info, size_t cb, 
                             void *user_data) {
	compute::Device *this_device = (compute::Device *)user_data;
    this_device->ErrorCallback(error_info, private_info, cb);
}

namespace compute {

Device::Device(WindowId window_id) {
    device_id_ = NULL;
    
    cl_int cl_status = CL_SUCCESS;
    
	DEBUG_STATEMENT(std::cout << "Initializing compute::Device" << std::endl);
    
    // get the number of platforms
    cl_uint platform_count = 0;
    cl_status = clGetPlatformIDs(0, NULL, &platform_count);
    OclCheckError(cl_status, "clGetPlatformIDs, get platform count");
    DEBUG_STATEMENT(std::cout << "found platform(s) " << platform_count << std::endl);
    if (platform_count == 0) {
        std::cerr << "could not find any OpenCL platforms" << std::endl;
        abort();
    }
    
    // get the list of platforms
    cl_platform_id *platform_list = new cl_platform_id[platform_count];
    cl_status = clGetPlatformIDs(platform_count, platform_list, NULL);
    OclCheckError(cl_status, "clGetPlatformIDs, get platform list");
	cl_platform_id platform_use = NULL;
    
	for (cl_uint i = 0; i < platform_count; i++) {
		char str[128];
		size_t ret_size = 0;
		cl_int status = 0;

		status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_VENDOR,
								   128, str, &ret_size);
		OclCheckError(status, "clGetPlatformInfo");
		assert(128 >= ret_size);

		if (!strcmp("Advanced Micro Devices, Inc.", str) ||
				!strcmp("NVIDIA Corporation", str)) {
			platform_use = platform_list[i];
		}
	}
	assert(platform_use);

	DEBUG_STATEMENT(
    for (cl_uint i = 0; i < platform_count; i++) {
		char str[128];
		size_t ret_size = 0;
		cl_int status = 0;

		status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_PROFILE,
								   128, str, &ret_size);
		OclCheckError(status, "clGetPlatformInfo");
		assert(128 >= ret_size);
		std::cout<<"CL_PLATFORM_PROFILE    " <<str <<std::endl;

		status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_VERSION,
								   128, str, &ret_size);
		OclCheckError(status, "clGetPlatformInfo");
		assert(128 >= ret_size);
		std::cout<<"CL_PLATFORM_VERSION    " <<str <<std::endl;

		status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_NAME,
								   128, str, &ret_size);
		OclCheckError(status, "clGetPlatformInfo");
		assert(128 >= ret_size);
		std::cout<<"CL_PLATFORM_NAME       " <<str <<std::endl;

		status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_VENDOR,
								   128, str, &ret_size);
		OclCheckError(status, "clGetPlatformInfo");
		assert(128 >= ret_size);
		std::cout<<"CL_PLATFORM_VENDOR     " <<str <<std::endl;

		status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_EXTENSIONS,
								   128, str, &ret_size);
		OclCheckError(status, "clGetPlatformInfo");
		assert(128 >= ret_size);
		std::cout<<"CL_PLATFORM_EXTENSIONS " <<str <<std::endl;
    }
	);
    delete[] platform_list;
    
    
    // get the number of gpu devices
    cl_uint device_count = 0;
    cl_status = clGetDeviceIDs(platform_use, CL_DEVICE_TYPE_GPU, 0, NULL, &device_count);
    OclCheckError(cl_status, "get gpu device count");
    if (device_count == 0) {
        std::cerr << "could not find any OpenCL GPU devices" << std::endl;
    }
    DEBUG_STATEMENT(std::cout << "found devices(s) " << device_count << std::endl);
    
    // get the list of gpu devices
    cl_device_id *device_list = new cl_device_id[device_count];
    cl_status = clGetDeviceIDs(platform_use, CL_DEVICE_TYPE_GPU, device_count, device_list, NULL);
    OclCheckError(cl_status, "get gpu device list");
    
    // select the most powerful device
	sel_.lock_.lock();
	if (!sel_.ids_.size()) {
		cl_uint power = 0;

		// find the most powerful device
		for (cl_uint i = 0; i < device_count; i++) {
			device_id_ = device_list[i];
			cl_uint tmp = GetMaxComputeUnits()*GetMaxFrequency();
			if (tmp > power)
				power = tmp;
		}

		// check if there are multiple powerful devices
		for (cl_uint i = 0; i < device_count; i++) {
			device_id_ = device_list[i];
			if (power != GetMaxComputeUnits()*GetMaxFrequency())
				continue;

			sel_.ids_.push_back(device_id_);
		}
	}

	sel_.next_ = sel_.next_ >= sel_.ids_.size() ? 0 : sel_.next_;
	device_id_ = sel_.ids_[sel_.next_++];
	sel_.lock_.unlock();
    
    DEBUG_STATEMENT(
    for (cl_uint i = 0; i < device_count; i++) {
        const size_t kBufferSize = 1024;
        char buffer[kBufferSize] = {0};
        cl_device_id device_id = device_list[i];
        
        
        cl_status = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, kBufferSize,
                                    buffer, NULL);
        OclCheckError(cl_status, "get device name");
        std::cout << "device(" << i << ") vendor: " <<buffer << std::endl;
        
        cl_status = clGetDeviceInfo(device_id, CL_DEVICE_NAME, kBufferSize,
                                    buffer, NULL);
        OclCheckError(cl_status, "get device name");
        std::cout << "device(" << i << ") name: " << buffer << std::endl;
        
        cl_status = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, kBufferSize,
                                    buffer, NULL);
        OclCheckError(cl_status, "get device version");
        std::cout << "device(" << i << ") version: " << buffer << std::endl;
        
        cl_status = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, kBufferSize,
                                    buffer, NULL);
        OclCheckError(cl_status, "get driver version");
        std::cout << "device(" << i << ") driver version: " << buffer << std::endl;
        
        
    }
                );
    delete[] device_list;
    
    // create a cl_context
    std::vector< cl_context_properties > properties_use;
#if defined(__windows__)
    if (window_id != NULL) {
        HGLRC gl_context = wglGetCurrentContext();
        if (!gl_context) {
            std::cout << "no gl context available" << std::endl;
            abort();
        }
        
        properties_use.push_back(CL_GL_CONTEXT_KHR);
        properties_use.push_back((cl_context_properties)gl_context);
        properties_use.push_back(CL_WGL_HDC_KHR);
        properties_use.push_back((cl_context_properties)window_id);
    }
#else
	// TODO file this in
#endif
    
    properties_use.push_back(CL_CONTEXT_PLATFORM);
    properties_use.push_back((cl_context_properties)platform_use);
    properties_use.push_back(0);
    context_ = clCreateContext(&properties_use.front(), 1, &device_id_,
                               OclErrorCallback, this, &cl_status);
    OclCheckError(cl_status, "create opencl context");
    
    // make a command queue
    command_queue_ = clCreateCommandQueue(context_, device_id_, 0, &cl_status);
    OclCheckError(cl_status, "create command queue");
}

void Device::ErrorCallback(const char *error_info, const void *, size_t) {
    std::cout << "ErrorCallback: " << error_info << std::endl;
}

void Device::Wait() {
	cl_int status;

	status = clFlush(command_queue_);
	OclCheckError(status, "wait for command queue to finish");
	status = clFinish(command_queue_);
	OclCheckError(status, "wait for command queue to finish");
}

Program::Program(Device *device, std::string source) {
    device_ = device;
    const char *kernel_source = source.c_str();
    cl_int cl_status = CL_SUCCESS;
    program_ = clCreateProgramWithSource(device_->get_context(), 
                                         1, &kernel_source, NULL, &cl_status);
    OclCheckError(cl_status, "create program");
    
    cl_status = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
    if (cl_status == CL_BUILD_PROGRAM_FAILURE) {
        // print the build log
        size_t build_log_size = 0;
        cl_status = clGetProgramBuildInfo(program_, device_->get_device_id(),
                                          CL_PROGRAM_BUILD_LOG, 0, NULL,
                                          &build_log_size);
        OclCheckError(cl_status, "get size of the build log");
        
        char *build_log = new char[build_log_size];
        cl_status = clGetProgramBuildInfo(program_, device->get_device_id(),
                                          CL_PROGRAM_BUILD_LOG, build_log_size,
                                          build_log, NULL);
        OclCheckError(cl_status, "get the build log");
        
        std::cout << "**************************************************"
                  << std::endl << "BUILD LOG:" << std::endl
                  << build_log
                  << "**************************************************"
                  << std::endl;
        delete[] build_log;
        abort();
    } else {
        OclCheckError(cl_status, "build program");
    }
}

Kernel::Kernel(Program *program, std::string kernel_name) {
    program_ = program;
    cl_int cl_status;
    kernel_ = clCreateKernel(program_->get_program(), kernel_name.c_str(), &cl_status);
	OclCheckError(cl_status, "create kernel");
}

Kernel::~Kernel()
{
	clReleaseKernel(kernel_);
}

}  // namespace compute


std::ostream &operator<<(std::ostream &out, const compute::Dim &dim)
{
	out<<"{ x = " <<dim.x <<", y = " <<dim.y <<", z = " <<dim.z <<" }";
	return out;
}
