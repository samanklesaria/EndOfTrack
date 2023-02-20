
mutable struct FoldMap{A, F, G, T} <: Functors.AbstractWalk
    map_fn::A
    agg_fn::F
    exclude::G
    output::T
end

function (walk::FoldMap)(recurse, x)
    if walk.exclude(x)
        walk.output = walk.agg_fn(walk.output, walk.map_fn(x))
    else
        foreach(recurse, Functors.children(x))
    end
    return walk.output
end

function foldmap(f, agg, init, args...; exclude=Functors.isleaf)
    w = FoldMap(f, agg, exclude, init)
    fmap(w, identity, args...)
end

na = [CartesianIndex()]
