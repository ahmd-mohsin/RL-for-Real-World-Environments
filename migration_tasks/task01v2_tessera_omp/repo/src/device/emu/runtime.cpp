// Tessera - emulated device backend: fiber scheduler and grid executor.
#include "tsr/device/emu/runtime.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "fiber_context.h"
#include "tsr/sched/thread_pool.h"

namespace tsr::emu
{
namespace
{
constexpr unsigned kWarpSize = 32;

// cudaError_t values used by the launcher (kept numerically identical to the
// CUDA runtime so error strings and checks line up across backends).
constexpr int kErrInvalidValue         = 1;
constexpr int kErrInvalidConfiguration = 9;
constexpr int kErrNotSupported         = 801;

[[noreturn]] void fatal(const char *what)
{
    std::fprintf(stderr, "tsr::emu fatal: %s\n", what);
    std::fflush(stderr);
    std::abort();
}

std::size_t env_size(const char *name, std::size_t fallback)
{
    const char *v = std::getenv(name);
    if(v == nullptr || *v == '\0')
    {
        return fallback;
    }
    char *end            = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    return (end != v && parsed > 0) ? static_cast<std::size_t>(parsed) : fallback;
}

// ---------------------------------------------------------------------------
// Fiber stacks: mmap'd with a PROT_NONE guard page below the stack so that an
// overflow faults immediately instead of corrupting a neighbouring fiber.
// ---------------------------------------------------------------------------
struct Stack
{
    void *mapping       = nullptr;
    std::size_t mapped  = 0;
    void *base          = nullptr;
    std::size_t size    = 0;

    Stack() = default;
    Stack(const Stack &) = delete;
    Stack &operator=(const Stack &) = delete;
    Stack(Stack &&o) noexcept { *this = std::move(o); }
    Stack &operator=(Stack &&o) noexcept
    {
        std::swap(mapping, o.mapping);
        std::swap(mapped, o.mapped);
        std::swap(base, o.base);
        std::swap(size, o.size);
        return *this;
    }
    ~Stack()
    {
        if(mapping != nullptr)
        {
            munmap(mapping, mapped);
        }
    }

    static Stack allocate(std::size_t bytes)
    {
        const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        bytes                  = (bytes + page - 1) / page * page;
        Stack s;
        s.mapped  = bytes + page;
        s.mapping = mmap(nullptr, s.mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if(s.mapping == MAP_FAILED)
        {
            fatal("could not map a fiber stack");
        }
        mprotect(s.mapping, page, PROT_NONE);
        s.base = static_cast<char *>(s.mapping) + page;
        s.size = bytes;
        return s;
    }
};

enum class FiberState : std::uint8_t
{
    Ready,
    AtBlockBarrier,
    AtWarpBarrier,
    Done,
};

struct Fiber
{
    FiberContext ctx;
    ThreadContext tc;
    FiberState state = FiberState::Ready;
    std::uint64_t inbox[kWarpSize]; // values delivered by the last warp round
    unsigned inbox_lanes = 0;       // lanes that took part in that round
};

// One in-flight warp-synchronous operation for a given participation mask.
// Sub-warp tiles (e.g. 16-lane partitions) run independent rounds side by side.
struct WarpRound
{
    unsigned mask    = 0;
    unsigned arrived = 0;
    std::uint64_t slots[kWarpSize];
};

struct WarpState
{
    unsigned dead = 0; // lanes that finished or do not exist
    std::vector<WarpRound> rounds;
};

struct BlockState
{
    const GridShape *shape = nullptr;
    InvokeFn invoke        = nullptr;
    const void *closure    = nullptr;
    unsigned nthreads      = 0;
    Fiber *fibers          = nullptr;
    FiberContext sched_ctx;
    std::vector<WarpState> warps;
    void *dynamic_smem = nullptr;

    // block barrier reductions
    int acc_count = 0;
    bool acc_and  = true;
    bool acc_or   = false;
    int res_count = 0;
    bool res_and  = true;
    bool res_or   = false;
};

struct WorkerResources
{
    std::vector<Stack> stacks;
    std::unique_ptr<Fiber[]> fibers;
    std::size_t fiber_capacity = 0;
    std::vector<std::max_align_t> smem; // 16-byte aligned dynamic shared memory
    ThreadContext direct_ctx{};
};

thread_local WorkerResources tl_res;
thread_local ThreadContext *tl_ctx = nullptr;
thread_local BlockState *tl_block  = nullptr;
thread_local Fiber *tl_fiber       = nullptr;

std::size_t stack_bytes()
{
    static const std::size_t bytes = env_size("TSR_EMU_STACK_KB", 128) * 1024;
    return bytes;
}

void ensure_fibers(unsigned n)
{
    WorkerResources &r = tl_res;
    if(r.fiber_capacity < n)
    {
        r.fibers.reset(new Fiber[n]);
        r.fiber_capacity = n;
    }
    while(r.stacks.size() < n)
    {
        r.stacks.push_back(Stack::allocate(stack_bytes()));
    }
}

void *ensure_smem(std::size_t bytes)
{
    if(bytes == 0)
    {
        return nullptr;
    }
    WorkerResources &r      = tl_res;
    const std::size_t elems = (bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
    if(r.smem.size() < elems)
    {
        r.smem.resize(elems);
    }
    return r.smem.data();
}

void fill_context(ThreadContext &tc, const GridShape &shape, const uint3 &bidx, unsigned linear, BlockState *block, Mode mode)
{
    const dim3 &bd = shape.block;
    tc.thread_idx  = uint3{linear % bd.x, (linear / bd.x) % bd.y, linear / (bd.x * bd.y)};
    tc.block_idx   = bidx;
    tc.block_dim   = shape.block;
    tc.grid_dim    = shape.grid;
    tc.linear_tid  = linear;
    tc.lane        = linear % kWarpSize;
    tc.warp        = linear / kWarpSize;
    tc.block       = block;
    tc.mode        = mode;
}

// ---------------------------------------------------------------------------
// Warp rounds
// ---------------------------------------------------------------------------
unsigned lanes_of_warp(const BlockState &b, unsigned warp)
{
    const unsigned first = warp * kWarpSize;
    const unsigned count = std::min(kWarpSize, b.nthreads - first);
    return count == kWarpSize ? 0xffffffffu : ((1u << count) - 1u);
}

// Deliver a completed round to every participant and wake the ones waiting.
void complete_round(BlockState &b, unsigned warp, std::size_t round_idx)
{
    WarpState &w        = b.warps[warp];
    const WarpRound rnd = w.rounds[round_idx];
    w.rounds.erase(w.rounds.begin() + static_cast<std::ptrdiff_t>(round_idx));
    for(unsigned lane = 0; lane < kWarpSize; ++lane)
    {
        if((rnd.arrived >> lane) & 1u)
        {
            Fiber &f = b.fibers[warp * kWarpSize + lane];
            std::memcpy(f.inbox, rnd.slots, sizeof(rnd.slots));
            f.inbox_lanes = rnd.arrived;
            if(f.state == FiberState::AtWarpBarrier)
            {
                f.state = FiberState::Ready;
            }
        }
    }
}

// Re-check every pending round of a warp, e.g. after one of its lanes exited.
void check_rounds(BlockState &b, unsigned warp)
{
    WarpState &w = b.warps[warp];
    for(std::size_t i = 0; i < w.rounds.size();)
    {
        const unsigned need = w.rounds[i].mask & ~w.dead;
        if(w.rounds[i].arrived != 0 && (w.rounds[i].arrived & need) == need)
        {
            complete_round(b, warp, i);
            continue; // complete_round erased index i
        }
        ++i;
    }
}

void yield_to_scheduler()
{
    Fiber *f      = tl_fiber;
    BlockState *b = tl_block;
    context_switch(f->ctx, b->sched_ctx);
    // Resumed by the scheduler; restore this fiber's view of the world.
    tl_fiber = f;
    tl_block = b;
    tl_ctx   = &f->tc;
}

void fiber_entry()
{
    BlockState *b = tl_block;
    Fiber *f      = tl_fiber;
    b->invoke(b->closure);
    f->state                 = FiberState::Done;
    const unsigned warp      = f->tc.warp;
    b->warps[warp].dead     |= 1u << f->tc.lane;
    check_rounds(*b, warp);
    context_switch(f->ctx, b->sched_ctx);
    fatal("finished fiber was resumed");
}

void describe_deadlock(const BlockState &b)
{
    std::fprintf(stderr, "tsr::emu: deadlock in block (%u,%u,%u):\n", b.fibers[0].tc.block_idx.x, b.fibers[0].tc.block_idx.y, b.fibers[0].tc.block_idx.z);
    unsigned shown = 0;
    for(unsigned i = 0; i < b.nthreads && shown < 16; ++i)
    {
        const Fiber &f = b.fibers[i];
        if(f.state == FiberState::Done)
        {
            continue;
        }
        std::fprintf(stderr, "  thread %u (warp %u lane %u) waits at %s\n", i, f.tc.warp, f.tc.lane,
                     f.state == FiberState::AtBlockBarrier ? "__syncthreads" : "a warp-synchronous operation");
        ++shown;
    }
}

// Kept out of line: with the ucontext fallback getcontext() behaves like
// setjmp, and inlining it into the scheduler loop would make the compiler
// treat the loop's locals as clobbered.
__attribute__((noinline)) void init_fiber(Fiber &f, const Stack &stack)
{
    context_init(f.ctx, stack.base, stack.size, fiber_entry);
}

void run_block_cooperative(const GridShape &shape, const uint3 &bidx, InvokeFn invoke, const void *closure)
{
    const unsigned n = shape.block.x * shape.block.y * shape.block.z;
    ensure_fibers(n);

    BlockState b;
    b.shape        = &shape;
    b.invoke       = invoke;
    b.closure      = closure;
    b.nthreads     = n;
    b.fibers       = tl_res.fibers.get();
    b.dynamic_smem = ensure_smem(shape.dynamic_smem);
    const unsigned nwarps = (n + kWarpSize - 1) / kWarpSize;
    b.warps.resize(nwarps);
    for(unsigned w = 0; w < nwarps; ++w)
    {
        b.warps[w].dead = ~lanes_of_warp(b, w);
    }

    for(unsigned i = 0; i < n; ++i)
    {
        Fiber &f = b.fibers[i];
        fill_context(f.tc, shape, bidx, i, &b, Mode::Cooperative);
        f.state = FiberState::Ready;
        init_fiber(f, tl_res.stacks[i]);
    }

    tl_block = &b;
    while(true)
    {
        bool ran = false;
        for(unsigned i = 0; i < n; ++i)
        {
            Fiber &f = b.fibers[i];
            if(f.state != FiberState::Ready)
            {
                continue;
            }
            tl_fiber = &f;
            tl_ctx   = &f.tc;
            context_switch(b.sched_ctx, f.ctx);
            ran = true;
        }

        unsigned done = 0, at_block = 0, at_warp = 0;
        for(unsigned i = 0; i < n; ++i)
        {
            switch(b.fibers[i].state)
            {
            case FiberState::Done: ++done; break;
            case FiberState::AtBlockBarrier: ++at_block; break;
            case FiberState::AtWarpBarrier: ++at_warp; break;
            case FiberState::Ready: break;
            }
        }
        if(done == n)
        {
            break;
        }
        if(at_warp == 0 && at_block == n - done)
        {
            // Every live thread reached __syncthreads: publish the reductions
            // and let the whole block continue.
            b.res_count = b.acc_count;
            b.res_and   = b.acc_and;
            b.res_or    = b.acc_or;
            b.acc_count = 0;
            b.acc_and   = true;
            b.acc_or    = false;
            for(unsigned i = 0; i < n; ++i)
            {
                if(b.fibers[i].state == FiberState::AtBlockBarrier)
                {
                    b.fibers[i].state = FiberState::Ready;
                }
            }
            continue;
        }
        if(!ran)
        {
            describe_deadlock(b);
            fatal("block can make no progress (divergent __syncthreads or warp operation)");
        }
    }
    tl_block = nullptr;
    tl_fiber = nullptr;
    tl_ctx   = nullptr;
}

void run_block_direct(const GridShape &shape, const uint3 &bidx, InvokeFn invoke, const void *closure)
{
    const unsigned n  = shape.block.x * shape.block.y * shape.block.z;
    ThreadContext &tc = tl_res.direct_ctx;
    void *smem        = ensure_smem(shape.dynamic_smem);
    (void)smem;
    for(unsigned i = 0; i < n; ++i)
    {
        fill_context(tc, shape, bidx, i, nullptr, Mode::Direct);
        tl_ctx = &tc;
        invoke(closure);
    }
    tl_ctx = nullptr;
}

void run_block(const GridShape &shape, Mode mode, std::uint64_t linear_block, InvokeFn invoke, const void *closure)
{
    const dim3 &g = shape.grid;
    const uint3 bidx{
        static_cast<unsigned>(linear_block % g.x),
        static_cast<unsigned>((linear_block / g.x) % g.y),
        static_cast<unsigned>(linear_block / (static_cast<std::uint64_t>(g.x) * g.y)),
    };
    if(mode == Mode::Cooperative)
    {
        run_block_cooperative(shape, bidx, invoke, closure);
    }
    else
    {
        run_block_direct(shape, bidx, invoke, closure);
    }
}

sched::ThreadPool &worker_pool()
{
    // The launching thread always executes blocks too, so the pool holds one
    // thread fewer than the configured worker count.
    static sched::ThreadPool pool(static_cast<std::size_t>(std::max(1, worker_count() - 1)));
    return pool;
}

std::mutex g_attr_mutex;
std::unordered_map<const void *, int> g_max_dynamic_smem;

BlockState &current_block(const char *op)
{
    ThreadContext &tc = current_thread();
    if(tc.mode != Mode::Cooperative || tl_block == nullptr)
    {
        std::string msg = std::string(op) + " used in a kernel launched in direct mode (launch_linear); use launch()";
        fatal(msg.c_str());
    }
    return *tl_block;
}

int barrier_impl(int pred)
{
    BlockState &b = current_block("__syncthreads");
    const bool p  = pred != 0;
    b.acc_count  += p ? 1 : 0;
    b.acc_and     = b.acc_and && p;
    b.acc_or      = b.acc_or || p;
    tl_fiber->state = FiberState::AtBlockBarrier;
    yield_to_scheduler();
    return 0;
}
} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
const DeviceLimits &limits()
{
    static const DeviceLimits l{};
    return l;
}

int worker_count()
{
    static const int n = []
    {
        const unsigned hw     = std::thread::hardware_concurrency();
        const std::size_t def = std::min<std::size_t>(8, hw == 0 ? 1 : hw);
        return static_cast<int>(env_size("TSR_EMU_WORKERS", def));
    }();
    return n;
}

bool in_kernel() noexcept { return tl_ctx != nullptr; }

ThreadContext &current_thread()
{
    if(tl_ctx == nullptr)
    {
        fatal("device intrinsic used outside of a kernel");
    }
    return *tl_ctx;
}

void *dynamic_shared_memory()
{
    current_thread();
    return tl_res.smem.empty() ? nullptr : tl_res.smem.data();
}

void block_barrier() { barrier_impl(1); }

int block_barrier_count(int pred)
{
    barrier_impl(pred);
    return tl_block->res_count;
}

int block_barrier_and(int pred)
{
    barrier_impl(pred);
    return tl_block->res_and ? 1 : 0;
}

int block_barrier_or(int pred)
{
    barrier_impl(pred);
    return tl_block->res_or ? 1 : 0;
}

const std::uint64_t *warp_exchange(unsigned mask, std::uint64_t value, unsigned *participants)
{
    BlockState &b     = current_block("warp-synchronous intrinsic");
    ThreadContext &tc = *tl_ctx;
    const unsigned me = 1u << tc.lane;
    if((mask & me) == 0)
    {
        fatal("warp intrinsic called by a lane that is not in its own mask");
    }
    WarpState &w = b.warps[tc.warp];

    std::size_t idx = w.rounds.size();
    for(std::size_t i = 0; i < w.rounds.size(); ++i)
    {
        if(w.rounds[i].mask == mask && (w.rounds[i].arrived & me) == 0)
        {
            idx = i;
            break;
        }
    }
    if(idx == w.rounds.size())
    {
        WarpRound r;
        r.mask = mask;
        w.rounds.push_back(r);
    }
    WarpRound &round      = w.rounds[idx];
    round.slots[tc.lane]  = value;
    round.arrived        |= me;

    const unsigned need = mask & ~w.dead;
    Fiber *self         = tl_fiber;
    if((round.arrived & need) == need)
    {
        complete_round(b, tc.warp, idx);
    }
    else
    {
        self->state = FiberState::AtWarpBarrier;
        yield_to_scheduler();
    }
    if(participants != nullptr)
    {
        *participants = self->inbox_lanes;
    }
    return self->inbox;
}

void warp_barrier(unsigned mask) { warp_exchange(mask, 0); }

unsigned warp_live_mask()
{
    ThreadContext &tc = current_thread();
    if(tc.mode != Mode::Cooperative || tl_block == nullptr)
    {
        return 1u << tc.lane;
    }
    return ~tl_block->warps[tc.warp].dead;
}

void set_max_dynamic_smem(const void *kernel_id, int bytes)
{
    const std::scoped_lock lock(g_attr_mutex);
    g_max_dynamic_smem[kernel_id] = bytes;
}

int max_dynamic_smem(const void *kernel_id)
{
    const std::scoped_lock lock(g_attr_mutex);
    const auto it = g_max_dynamic_smem.find(kernel_id);
    return it == g_max_dynamic_smem.end() ? limits().shared_mem_per_block : it->second;
}

int run_grid(const GridShape &shape, Mode mode, InvokeFn invoke, const void *closure, const void *kernel_id)
{
    const DeviceLimits &lim = limits();
    const dim3 &g           = shape.grid;
    const dim3 &bd          = shape.block;
    const std::uint64_t threads = static_cast<std::uint64_t>(bd.x) * bd.y * bd.z;
    if(threads == 0 || threads > static_cast<std::uint64_t>(lim.max_threads_per_block) || bd.z > 64)
    {
        return kErrInvalidConfiguration;
    }
    if(g.x == 0 || g.y == 0 || g.z == 0 || g.x > static_cast<unsigned>(lim.max_grid_x)
       || g.y > static_cast<unsigned>(lim.max_grid_yz) || g.z > static_cast<unsigned>(lim.max_grid_yz))
    {
        return kErrInvalidConfiguration;
    }
    if(shape.dynamic_smem > static_cast<std::size_t>(std::min(max_dynamic_smem(kernel_id), lim.shared_mem_per_block_optin)))
    {
        return kErrInvalidValue;
    }
    if(in_kernel())
    {
        return kErrNotSupported; // no dynamic parallelism
    }

    const std::uint64_t total = static_cast<std::uint64_t>(g.x) * g.y * g.z;
    const int workers         = static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(worker_count()), total));
    if(workers <= 1)
    {
        for(std::uint64_t i = 0; i < total; ++i)
        {
            run_block(shape, mode, i, invoke, closure);
        }
        return 0;
    }

    std::atomic<std::uint64_t> next{0};
    auto drain = [&]
    {
        for(std::uint64_t i = next.fetch_add(1); i < total; i = next.fetch_add(1))
        {
            run_block(shape, mode, i, invoke, closure);
        }
    };
    sched::multi_future<void> helpers;
    for(int w = 1; w < workers; ++w)
    {
        helpers.push_back(worker_pool().submit_task(drain, sched::pr::high));
    }
    drain();
    helpers.get();
    return 0;
}

} // namespace tsr::emu
