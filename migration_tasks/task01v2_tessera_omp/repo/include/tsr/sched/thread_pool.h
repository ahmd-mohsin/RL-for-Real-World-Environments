// Tessera - host task scheduling
//
// Portions derived from BS::thread_pool by Barak Shoshany (MIT License),
// https://github.com/bshoshany/thread-pool. See third_party/LICENSES/thread-pool.txt.
//
// A fixed-size pool of worker threads with an optional priority queue, block
// partitioning helpers and futures. The emulated device backend executes thread
// blocks on one of these pools, and the frame scheduler fans views out on another.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace tsr::sched
{
using priority_t = std::int8_t;

namespace pr
{
constexpr priority_t lowest  = -128;
constexpr priority_t low     = -64;
constexpr priority_t normal  = 0;
constexpr priority_t high    = +64;
constexpr priority_t highest = +127;
} // namespace pr

// A type-erased, move-only callable. std::function requires copyable targets,
// which would rule out tasks that own a std::promise.
template<typename>
class move_only_function;

template<typename R, typename... Args>
class move_only_function<R(Args...)>
{
public:
    move_only_function() = default;
    move_only_function(move_only_function &&) noexcept = default;
    move_only_function &operator=(move_only_function &&) noexcept = default;
    move_only_function(const move_only_function &) = delete;
    move_only_function &operator=(const move_only_function &) = delete;

    template<typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, move_only_function>>>
    move_only_function(F &&func) // NOLINT(hicpp-explicit-conversions)
        : ptr_(std::make_unique<model<std::decay_t<F>>>(std::forward<F>(func)))
    {
    }

    R operator()(Args... args) { return ptr_->invoke(std::forward<Args>(args)...); }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    struct concept_t
    {
        virtual ~concept_t()           = default;
        virtual R invoke(Args... args) = 0;
    };
    template<typename F>
    struct model final : concept_t
    {
        explicit model(F &&f) : func(std::move(f)) {}
        explicit model(const F &f) : func(f) {}
        R invoke(Args... args) override { return func(std::forward<Args>(args)...); }
        F func;
    };
    std::unique_ptr<concept_t> ptr_;
};

using task_t = move_only_function<void()>;

struct pr_task
{
    explicit pr_task(task_t &&t, const priority_t p = 0) : task(std::move(t)), priority(p) {}
    friend bool operator<(const pr_task &lhs, const pr_task &rhs) noexcept { return lhs.priority < rhs.priority; }
    task_t task;
    priority_t priority = 0;
};

// A vector of futures with collective wait/get.
template<typename T>
class multi_future : public std::vector<std::future<T>>
{
public:
    using std::vector<std::future<T>>::vector;

    [[nodiscard]] std::conditional_t<std::is_void_v<T>, void, std::vector<T>> get()
    {
        if constexpr(std::is_void_v<T>)
        {
            for(std::future<T> &f : *this)
            {
                f.get();
            }
            return;
        }
        else
        {
            std::vector<T> results;
            results.reserve(this->size());
            for(std::future<T> &f : *this)
            {
                results.push_back(f.get());
            }
            return results;
        }
    }

    [[nodiscard]] std::size_t ready_count() const
    {
        std::size_t count = 0;
        for(const std::future<T> &f : *this)
        {
            if(f.wait_for(std::chrono::duration<double>::zero()) == std::future_status::ready)
            {
                ++count;
            }
        }
        return count;
    }

    void wait() const
    {
        for(const std::future<T> &f : *this)
        {
            f.wait();
        }
    }
};

// Splits [first, last) into num_blocks contiguous ranges whose sizes differ by
// at most one. The first `remainder` blocks get the extra element.
template<typename T>
class blocks
{
public:
    blocks(const T first_index, const T index_after_last, const std::size_t num_blocks)
        : num_blocks_(num_blocks), first_index_(first_index), index_after_last_(index_after_last)
    {
        if(index_after_last_ > first_index_)
        {
            const std::size_t total_size = static_cast<std::size_t>(index_after_last_ - first_index_);
            num_blocks_                  = std::min(num_blocks_, total_size);
            if(num_blocks_ == 0)
            {
                num_blocks_ = 1;
            }
            block_size_ = total_size / num_blocks_;
            remainder_  = total_size % num_blocks_;
            if(block_size_ == 0)
            {
                block_size_ = 1;
                num_blocks_ = (total_size > 1) ? total_size : 1;
            }
        }
        else
        {
            num_blocks_ = 0;
        }
    }

    [[nodiscard]] T start(const std::size_t block) const noexcept
    {
        return first_index_ + static_cast<T>(block * block_size_)
             + static_cast<T>(block < remainder_ ? block : remainder_);
    }

    [[nodiscard]] T end(const std::size_t block) const noexcept
    {
        return (block == num_blocks_ - 1) ? index_after_last_ : start(block + 1);
    }

    [[nodiscard]] std::size_t num_blocks() const noexcept { return num_blocks_; }

private:
    std::size_t block_size_ = 0;
    std::size_t num_blocks_ = 0;
    std::size_t remainder_  = 0;
    T first_index_          = 0;
    T index_after_last_     = 0;
};

struct wait_deadlock : public std::runtime_error
{
    wait_deadlock() : std::runtime_error("tsr::sched::wait_deadlock: a pool task waited on its own pool") {}
};

class ThreadPool;

// Per-thread information about the pool the calling thread belongs to.
class this_thread
{
public:
    [[nodiscard]] static std::optional<std::size_t> get_index() noexcept { return index_; }
    [[nodiscard]] static std::optional<ThreadPool *> get_pool() noexcept { return pool_; }

private:
    inline static thread_local std::optional<std::size_t> index_ = std::nullopt;
    inline static thread_local std::optional<ThreadPool *> pool_  = std::nullopt;
    friend class ThreadPool;
};

class ThreadPool
{
public:
    explicit ThreadPool(const std::size_t num_threads = 0) { create_threads(num_threads); }

    ThreadPool(const ThreadPool &)            = delete;
    ThreadPool(ThreadPool &&)                 = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool &operator=(ThreadPool &&)      = delete;

    ~ThreadPool()
    {
        try
        {
            wait();
        }
        catch(...)
        {
        }
        destroy_threads();
    }

    [[nodiscard]] std::size_t get_thread_count() const noexcept { return thread_count_; }

    [[nodiscard]] std::size_t get_tasks_queued() const
    {
        const std::scoped_lock lock(tasks_mutex_);
        return tasks_.size();
    }

    [[nodiscard]] std::size_t get_tasks_running() const
    {
        const std::scoped_lock lock(tasks_mutex_);
        return tasks_running_;
    }

    [[nodiscard]] std::size_t get_tasks_total() const
    {
        const std::scoped_lock lock(tasks_mutex_);
        return tasks_running_ + tasks_.size();
    }

    template<typename F>
    void detach_task(F &&task, const priority_t priority = pr::normal)
    {
        {
            const std::scoped_lock lock(tasks_mutex_);
            tasks_.emplace(task_t(std::forward<F>(task)), priority);
        }
        task_available_cv_.notify_one();
    }

    template<typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
    [[nodiscard]] std::future<R> submit_task(F &&task, const priority_t priority = pr::normal)
    {
        auto promise         = std::make_shared<std::promise<R>>();
        std::future<R> fut   = promise->get_future();
        detach_task(
            [task = std::forward<F>(task), promise]() mutable
            {
                try
                {
                    if constexpr(std::is_void_v<R>)
                    {
                        task();
                        promise->set_value();
                    }
                    else
                    {
                        promise->set_value(task());
                    }
                }
                catch(...)
                {
                    try
                    {
                        promise->set_exception(std::current_exception());
                    }
                    catch(...)
                    {
                    }
                }
            },
            priority
        );
        return fut;
    }

    // Run block(start, end) over [first_index, index_after_last) split into
    // num_blocks ranges (0 = one per thread).
    template<typename T, typename F, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
    [[nodiscard]] multi_future<R> submit_blocks(
        const T first_index,
        const T index_after_last,
        F &&block,
        const std::size_t num_blocks = 0,
        const priority_t priority    = pr::normal
    )
    {
        if(index_after_last <= first_index)
        {
            return {};
        }
        const blocks<T> blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count_);
        auto block_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(block));
        multi_future<R> futures;
        futures.reserve(blks.num_blocks());
        for(std::size_t b = 0; b < blks.num_blocks(); ++b)
        {
            futures.push_back(submit_task(
                [block_ptr, start = blks.start(b), end = blks.end(b)] { return (*block_ptr)(start, end); },
                priority
            ));
        }
        return futures;
    }

    template<typename T, typename F>
    [[nodiscard]] multi_future<void> submit_loop(
        const T first_index,
        const T index_after_last,
        F &&loop,
        const std::size_t num_blocks = 0,
        const priority_t priority    = pr::normal
    )
    {
        auto loop_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(loop));
        return submit_blocks(
            first_index,
            index_after_last,
            [loop_ptr](const T start, const T end)
            {
                for(T i = start; i < end; ++i)
                {
                    (*loop_ptr)(i);
                }
            },
            num_blocks,
            priority
        );
    }

    template<typename T, typename F>
    void detach_loop(
        const T first_index,
        const T index_after_last,
        F &&loop,
        const std::size_t num_blocks = 0,
        const priority_t priority    = pr::normal
    )
    {
        if(index_after_last <= first_index)
        {
            return;
        }
        const blocks<T> blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count_);
        auto loop_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(loop));
        for(std::size_t b = 0; b < blks.num_blocks(); ++b)
        {
            detach_task(
                [loop_ptr, start = blks.start(b), end = blks.end(b)]
                {
                    for(T i = start; i < end; ++i)
                    {
                        (*loop_ptr)(i);
                    }
                },
                priority
            );
        }
    }

    // Block until every queued and running task has finished. Calling this
    // from one of the pool's own workers would deadlock, so it throws instead.
    void wait()
    {
        if(this_thread::get_pool() == this)
        {
            throw wait_deadlock();
        }
        std::unique_lock lock(tasks_mutex_);
        waiting_ = true;
        tasks_done_cv_.wait(lock, [this] { return tasks_running_ == 0 && tasks_.empty(); });
        waiting_ = false;
    }

    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period> &duration)
    {
        if(this_thread::get_pool() == this)
        {
            throw wait_deadlock();
        }
        std::unique_lock lock(tasks_mutex_);
        waiting_            = true;
        const bool finished = tasks_done_cv_.wait_for(lock, duration, [this] { return tasks_running_ == 0 && tasks_.empty(); });
        waiting_            = false;
        return finished;
    }

    // Drop every task that has not started yet.
    void purge()
    {
        const std::scoped_lock lock(tasks_mutex_);
        tasks_ = {};
    }

    void reset(const std::size_t num_threads = 0)
    {
        wait();
        destroy_threads();
        create_threads(num_threads);
    }

private:
    static std::size_t determine_thread_count(const std::size_t num_threads)
    {
        if(num_threads > 0)
        {
            return num_threads;
        }
        const unsigned hw = std::thread::hardware_concurrency();
        return hw > 0 ? hw : 1;
    }

    void create_threads(const std::size_t num_threads)
    {
        thread_count_ = determine_thread_count(num_threads);
        {
            const std::scoped_lock lock(tasks_mutex_);
            tasks_running_   = thread_count_; // workers count as running until they first block
            workers_running_ = true;
        }
        threads_.clear();
        threads_.reserve(thread_count_);
        for(std::size_t i = 0; i < thread_count_; ++i)
        {
            threads_.emplace_back([this, i] { worker(i); });
        }
    }

    void destroy_threads()
    {
        {
            const std::scoped_lock lock(tasks_mutex_);
            workers_running_ = false;
        }
        task_available_cv_.notify_all();
        for(std::thread &t : threads_)
        {
            if(t.joinable())
            {
                t.join();
            }
        }
        threads_.clear();
    }

    void worker(const std::size_t idx)
    {
        this_thread::index_ = idx;
        this_thread::pool_  = this;
        std::unique_lock lock(tasks_mutex_);
        while(true)
        {
            --tasks_running_;
            if(waiting_ && tasks_running_ == 0 && tasks_.empty())
            {
                tasks_done_cv_.notify_all();
            }
            task_available_cv_.wait(lock, [this] { return !tasks_.empty() || !workers_running_; });
            if(!workers_running_ && tasks_.empty())
            {
                break;
            }
            // priority_queue::top() is const; the task is moved out before pop().
            task_t task = std::move(const_cast<pr_task &>(tasks_.top()).task);
            tasks_.pop();
            ++tasks_running_;
            lock.unlock();
            task();
            lock.lock();
        }
        this_thread::index_ = std::nullopt;
        this_thread::pool_  = std::nullopt;
    }

    std::priority_queue<pr_task> tasks_;
    mutable std::mutex tasks_mutex_;
    std::condition_variable task_available_cv_;
    std::condition_variable tasks_done_cv_;
    std::size_t tasks_running_ = 0;
    std::size_t thread_count_  = 0;
    bool waiting_              = false;
    bool workers_running_      = false;
    std::vector<std::thread> threads_;
};

} // namespace tsr::sched
