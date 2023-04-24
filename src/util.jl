
mutable struct FoldMap{A, F, G, T} <: Functors.AbstractWalk
    map_fn::A
    agg_fn::F
    exclude::G
    output::T
end

broadcast_to(x, y) = fill(y, length(x))

function children_like(y, x)
    if Functors.isleaf(y)
        broadcast_to(x, y)
    else
        Functors.functor(typeof(x), y)[1]
    end
end

function (walk::FoldMap)(recurse, x, ys...)
    if walk.exclude(x)
        walk.output = walk.agg_fn(walk.output, walk.map_fn(x, ys...))
    else
        func, re = Functors.functor(x)
        yfuncs = map(y -> children_like(y, x), ys)
        foreach(recurse, func, yfuncs...)
    end
    return walk.output
end

function foldmap(f, agg, init, args...; exclude=Functors.isleaf)
    w = FoldMap(f, agg, exclude, init)
    fmap(w, nothing, args...)
end

newEnd(x::AbstractArray) = reshape(x, (size(x)...,1))
new1(x::AbstractArray) = reshape(x, (1, size(x)...))

const na = @SVector [CartesianIndex()]

const indent_level = Ref(0)

indent!() = indent_level[] += 1

function dedent!()
  @assert indent_level[] > 0
  indent_level[] -= 1
end

function printindent(a)
  print(String(fill(' ', 4 * indent_level[])))
  print(a)
end

function log_action(st, action_val::ValuedAction; bound=nothing)
  action = action_val.action 
  if isnothing(bound)
    val = "$(action_val.value)"
  else
    val = "$(action_val.value), $(bound)"
  end
  if action[1] == 0
    println("[$val] (fake action)")
  elseif action[1] < 6
    old_pos = st.positions[st.player].pieces[:, action[1]]
    println("[$val] $(st.player) moves from $old_pos to $(action[2])")
  else
    old_pos = st.positions[st.player].ball
    println("[$val] $(st.player) kicks from $old_pos to $(action[2])")
  end
end

flip(f) = (a,b)-> f(b,a)

fleaves(a; f=identity, t=Float32, leaf=Functors.isleaf) = foldmap(f, append!, Vector{t}(), a; exclude=leaf)

# using Plots
# function plot_state(st::State)
#   scatter(st.positions[1].pieces[1, :],
#     st.positions[1].pieces[2, :])
#   scatter!(st.positions[1].ball[1:1], st.positions[1].ball[2:2])
#   scatter!(st.positions[2].pieces[1,:],
#     st.positions[2].pieces[2, :])
#   scatter!(st.positions[2].ball[1:1], st.positions[2].ball[2:2])
# end
