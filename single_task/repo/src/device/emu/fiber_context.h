// Tessera - emulated device backend: minimal user-space context switching.
//
// glibc's swapcontext() saves and restores the signal mask with a system call
// on every switch, which dominates the cost of barrier-heavy kernels. On
// x86-64 and AArch64 we switch stacks with a few lines of assembly that only
// spill the callee-saved registers (and the FP control state on x86-64).
// Other targets fall back to ucontext.
#pragma once

#include <cstddef>

#if !defined(TSR_EMU_FORCE_UCONTEXT) && (defined(__x86_64__) || defined(__aarch64__))
#    define TSR_EMU_ASM_CONTEXT 1
#else
#    include <ucontext.h>
#endif

namespace tsr::emu
{
struct FiberContext
{
#if defined(TSR_EMU_ASM_CONTEXT)
    void *sp = nullptr; // saved stack pointer of a suspended context
#else
    ucontext_t uc;
#endif
};

// Prepare `ctx` so that the first switch into it calls entry() on the given
// stack. entry() must never return.
void context_init(FiberContext &ctx, void *stack_base, std::size_t stack_size, void (*entry)());

// Save the current context into `from` and resume `to`.
void context_switch(FiberContext &from, FiberContext &to);

} // namespace tsr::emu
