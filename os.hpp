#ifndef OS_HPP
#define OS_HPP

#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(_WIN64)
#define __windows__
#include <Windows.h>
#endif

#include <stdint.h>

#ifdef __linux__
#include <stddef.h>
#include <unistd.h>
#endif

namespace Os {

#ifdef _WIN32
typedef HDC WindowId;
#else
typedef void * WindowId;
#endif

namespace Memory {
void *AlignedMalloc(size_t size, size_t alignment);

void AlignedFree(void *pointer);

inline void *Malloc(size_t size) { return malloc(size); }
void *Realloc(void *ptr, size_t size);

inline void Free(void *pointer) { AlignedFree(pointer); }


}  // namespace Memory


inline void SpinPause() {
#ifdef _WIN32
    _mm_pause();
#else
	usleep(1);
#endif
}

namespace Atomic {
void *CompareAndSwap(void **destination, void *compare, void *new_value);
}  // namespace Atomic

}  // namespace Os

#endif // OS_HPP
