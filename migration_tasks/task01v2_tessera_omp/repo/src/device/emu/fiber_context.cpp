// Tessera - emulated device backend: minimal user-space context switching.
#include "fiber_context.h"

#include <cstdint>
#include <cstring>

#if defined(TSR_EMU_ASM_CONTEXT)

extern "C" void tsr_emu_switch(void **save_sp, void *load_sp);

// tsr_emu_switch(save_sp, load_sp):
//   push the callee-saved state on the current stack, store the stack pointer
//   in *save_sp, load load_sp and pop the state saved there, then return into
//   the resumed context.
#    if defined(__x86_64__)
// Frame (low to high): [mxcsr:4|x87cw:2|pad:2] r15 r14 r13 r12 rbx rbp <ret>
asm(R"(
    .pushsection .text
    .globl tsr_emu_switch
    .type tsr_emu_switch,@function
    .p2align 4
tsr_emu_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq $8, %rsp
    stmxcsr (%rsp)
    fnstcw 4(%rsp)
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    ldmxcsr (%rsp)
    fldcw 4(%rsp)
    addq $8, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
    .size tsr_emu_switch,.-tsr_emu_switch
    .popsection
)");
#    elif defined(__aarch64__)
// Frame (low to high): x19..x28, x29, x30, d8..d15  (176 bytes)
asm(R"(
    .pushsection .text
    .globl tsr_emu_switch
    .type tsr_emu_switch,%function
    .p2align 4
tsr_emu_switch:
    sub sp, sp, #176
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    mov x9, sp
    str x9, [x0]
    mov sp, x1
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    ldp x29, x30, [sp, #80]
    ldp d8, d9, [sp, #96]
    ldp d10, d11, [sp, #112]
    ldp d12, d13, [sp, #128]
    ldp d14, d15, [sp, #144]
    add sp, sp, #176
    ret
    .size tsr_emu_switch,.-tsr_emu_switch
    .popsection
)");
#    endif

namespace tsr::emu
{
void context_init(FiberContext &ctx, void *stack_base, std::size_t stack_size, void (*entry)())
{
    auto top = reinterpret_cast<std::uintptr_t>(stack_base) + stack_size;
    top &= ~static_cast<std::uintptr_t>(15);
#    if defined(__x86_64__)
    // After `ret` pops the entry address, rsp must be 8 mod 16 as if entry had
    // been reached by a call. Lay out: [ctrl][r15..rbp = 0][entry][pad].
    auto *slots = reinterpret_cast<std::uint64_t *>(top - 8 * 9);
    std::uint32_t mxcsr;
    std::uint16_t fpucw;
    asm volatile("stmxcsr %0" : "=m"(mxcsr));
    asm volatile("fnstcw %0" : "=m"(fpucw));
    std::memset(slots, 0, 8 * 9);
    std::memcpy(reinterpret_cast<char *>(&slots[0]), &mxcsr, sizeof(mxcsr));
    std::memcpy(reinterpret_cast<char *>(&slots[0]) + 4, &fpucw, sizeof(fpucw));
    slots[7] = reinterpret_cast<std::uint64_t>(entry); // return address
    ctx.sp   = slots;
#    else
    // x30 (link register) holds the entry point; sp lands 16-byte aligned.
    auto *frame = reinterpret_cast<std::uint64_t *>(top - 176);
    std::memset(frame, 0, 176);
    frame[11] = reinterpret_cast<std::uint64_t>(entry); // x30 slot at offset 88
    ctx.sp    = frame;
#    endif
}

void context_switch(FiberContext &from, FiberContext &to) { tsr_emu_switch(&from.sp, to.sp); }
} // namespace tsr::emu

#else // ucontext fallback

namespace tsr::emu
{
void context_init(FiberContext &ctx, void *stack_base, std::size_t stack_size, void (*entry)())
{
    getcontext(&ctx.uc);
    ctx.uc.uc_stack.ss_sp   = stack_base;
    ctx.uc.uc_stack.ss_size = stack_size;
    ctx.uc.uc_link          = nullptr;
    makecontext(&ctx.uc, entry, 0);
}

void context_switch(FiberContext &from, FiberContext &to) { swapcontext(&from.uc, &to.uc); }
} // namespace tsr::emu

#endif
