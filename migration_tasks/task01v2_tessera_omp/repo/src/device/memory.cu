// Tessera - device memory ownership.
#include "tsr/device/memory.h"

namespace tsr
{
void Arena::reserve(std::size_t bytes)
{
    bytes = align_up(bytes);
    if(bytes > main_.size())
    {
        TSR_CHECK(offset_ == 0 && overflow_.empty(), "Arena::reserve called while allocations are live");
        main_.resize(bytes);
    }
}

void Arena::reset()
{
    if(!overflow_.empty())
    {
        // Grow the main block so the next frame fits without overflowing.
        const std::size_t want = align_up(high_water_);
        overflow_.clear();
        main_.release();
        main_.resize(want);
    }
    offset_ = 0;
    used_   = 0;
}

void *Arena::allocate_bytes(std::size_t bytes)
{
    const std::size_t aligned = align_up(bytes == 0 ? 1 : bytes);
    used_                    += aligned;
    if(used_ > high_water_)
    {
        high_water_ = used_;
    }
    if(offset_ + aligned <= main_.size())
    {
        void *p  = main_.data() + offset_;
        offset_ += aligned;
        return p;
    }
    overflow_.emplace_back(aligned);
    return overflow_.back().data();
}

std::size_t Arena::capacity() const noexcept
{
    std::size_t total = main_.size();
    for(const auto &b : overflow_)
    {
        total += b.size();
    }
    return total;
}
} // namespace tsr
