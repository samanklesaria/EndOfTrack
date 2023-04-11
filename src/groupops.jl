
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
  if a[1] == 0 # fake action
    a
  elseif a[1] < 6
    @set a[1] = p.perm[a[1]]
  else
    a
  end
end

# Just for packaging games for the nn
function normalize_player(st)
  if st.player == 2
    st = map_state(flip_pos_hor, st)
    st = State(1, reverse(st.positions))
    (-1, st)
  else
    (1, st)
  end
end

# For data augmentation with the nn
function flip_players(st) 
    st = map_state(flip_pos_hor, st)
    State(next_player(st.player), reverse(st.positions))
end

struct Transformation
  action_map::Vector{GroupElt}
  value_map::Int
end

function normalized(st::State)
  action_map = GroupElt[]
  value_map = 1
  
  # Swap players so that the current player is 1.
  if st.player == 2
    st = map_state(flip_pos_hor, st)
    st = State(1, reverse(st.positions))
    value_map = -1
    push!(action_map, FlipHor())
  end
  
  # # Flip the board left or right
  st2 = map_state(flip_pos_vert, st)
  if hash(st2) > hash(st)
    st = st2
    push!(action_map, FlipVert())
  end
  
  # Sort tokens of the current player
  # pieces = st.positions[st.player].pieces
  # ixs = sortperm(encode(pieces))
  # st = @set st.positions[st.player].pieces = pieces[:, ixs]
  # push!(action_map, TokenPerm(my_invperm(ixs)))
   
  if VALIDATE
    assert_valid_state(st)
  end
  Transformation(action_map, value_map), st
end

function (t::Transformation)(a::ValuedAction)
  ValuedAction(t(a.action), t.value_map * a.value)
end

Base.@propagate_inbounds function my_invperm(p::StaticVector)
     ip = similar(p)
     ip[p] = 1:length(p)
     similar_type(p)(ip)
end

(t::Transformation)(a::Action) = foldr(group_op, t.action_map; init=a)

