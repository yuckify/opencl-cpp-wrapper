#include "os.hpp"

#ifdef __linux__
#include <stdlib.h>
#endif

#if defined(__windows__)
#include <Windows.h>
#endif

void *Os::Memory::AlignedMalloc(size_t size, size_t alignment) {
#if defined(__windows__)
    return _aligned_malloc(size, alignment);
#elif defined(__linux__)
	return aligned_alloc(alignment, size);
#else
#error AlignedMalloc undefined for mac os
#endif
}

void Os::Memory::AlignedFree(void *pointer) {
#if defined(__windows__)
	_aligned_free(pointer);
#elif defined(__linux__)
	free(pointer);
#else
#error AlignedMalloc undefined for mac os
#endif
}

void *Os::Atomic::CompareAndSwap(void **destination, void *compare, void *new_value) {
#if defined(__windows__)
    return InterlockedCompareExchangePointer(destination, new_value, compare);
#elif defined(__linux__)
	return __sync_val_compare_and_swap(destination, compare, new_value);
#else
#error Atomic::CompareAndSwap undefined for mac os
#endif
}
