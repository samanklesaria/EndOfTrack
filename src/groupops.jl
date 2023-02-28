
abstract type GroupElt end

struct TokenPerm <: GroupElt
  perm::SVector{5, UInt8}
end

struct FlipHor <: GroupElt end

struct FlipVert <: GroupElt end

const mean_vert = SVector{2, Int8}([4, 0])
const flip_vert = SVector{2, Int8}([-1, 1])

const mean_hor = SVector{2}([0, 4.5])
const flip_hor = SVector{2}([1, -1])

flip_pos_vert(a::Pos) = flip_vert .* (a .- mean_vert) .+ mean_vert
flip_pos_vert(a::SMatrix) = flip_vert[:, na] .* (a .- mean_vert[:, na]) .+ mean_vert[:, na]

flip_pos_hor(a::Pos) = Pos(flip_hor .* (a .- mean_hor) .+ mean_hor)
flip_pos_hor(a::SMatrix) = SMatrix{2,5, Int8}(flip_hor[:, na] .* (a .- mean_hor[:, na]) .+ mean_hor[:, na])

function group_op(::FlipVert, a::Action)
  @set a[2] = flip_pos_vert(a[2])
end

function group_op(::FlipHor, a::Action)
  @set a[2] = flip_pos_hor(a[2])
end

function group_op(p::TokenPerm, a::Action) 
  if a[1] < 6
    @set a[1] = p.perm[a[1]]
  else
    a
  end
end

ismat(::SMatrix) = true
ismat(_) = false

struct Transformation
  action_map::Vector{GroupElt}
  value_map::Int
end

function normalized(st)
  action_map = GroupElt[]
  value_map = 1
  
  # Swap players so that the current player is 1.
  if st.player == 2
    st = fmap(flip_pos_hor, st)
    st = State(1, reverse(st.positions))
    value_map = -1
    push!(action_map, FlipHor())
  end
  
  # # Flip the board left or right
  st2 = fmap(flip_pos_vert, st)
  if hash(st2) > hash(st)
    st = st2
    push!(action_map, FlipVert())
  end
  
  # Sort tokens of each player
  # ixs = fmapstructure(a->sortperm(nestedview(a)), st; exclude=ismat)
  # st = fmap(st, ixs) do a, ix
  #   ix === (()) ? a : a[:, ix]
  # end
  # push!(action_map, TokenPerm(invperm(ixs.positions[1].pieces)))
  
  # Maybe easier: sort tokens of the current player?
  # ixs = sortperm(nestedview(st.positions[1].pieces))
  # st2 = @set st.positions[1].pieces = st.positions[1].pieces[:, ixs]
  # @assert all(st.positions[1].pieces .== st2.positions[1].pieces[:, invperm(ixs)])
  # st = st2
  # push!(action_map, TokenPerm(invperm(ixs)))
  
  if VALIDATE
    assert_valid_state(st)
  end
  Transformation(action_map, value_map), st
end


function (t::Transformation)(a::ValuedAction)
  ValuedAction(t(a.action), t.value_map * a.value)
end

(t::Transformation)(a::Action) = foldr(group_op, t.action_map; init=a)
(t::Transformation)(::Nothing) = nothing

