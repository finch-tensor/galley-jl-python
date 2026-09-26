# Fixes for bugs in the pinned Finch.jl, evaluated into Finch.Galley at import.
# Delete each fix once the pinned Finch includes it.

# Finch <= 1.6.0 builds `OrderedDict{DCKey}()`, which has no constructor, so
# Galley throws a MethodError when it re-checks a cached plan for tensors with
# 50 or more degree constraints (e.g. einsums over 4-d tensors).
# Upstream fix: https://github.com/finch-tensor/Finch.jl/pull/822
function issimilar(stat1::DCStats, stat2::DCStats, rel_granularity)
    if length(stat1.dcs) < 50
        for dc1 in stat1.dcs
            for dc2 in stat2.dcs
                if dc1.X == dc2.X && dc1.Y == dc2.Y &&
                    abs(log(rel_granularity, dc1.d) - log(rel_granularity, dc2.d)) > 1
                    return false
                end
            end
        end
    else
        dc_dict = OrderedDict{DCKey,Float64}()
        for dc1 in stat1.dcs
            dc_dict[get_dc_key(dc1)] = dc1.d
        end
        for dc2 in stat2.dcs
            # like the loop above, only compare DCs present in both stats
            d1 = get(dc_dict, get_dc_key(dc2), nothing)
            if !isnothing(d1) &&
                abs(log(rel_granularity, d1) - log(rel_granularity, dc2.d)) > 1
                return false
            end
        end
    end
    return true
end
