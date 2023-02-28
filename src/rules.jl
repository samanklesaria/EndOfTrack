const limits = @SVector [7, 8]

const Pos = SVector{2, Int8}

struct PlayerState
  ball::Pos
  pieces::SMatrix{2,5, Int8}
end
@functor PlayerState

struct State
  player::Int
  positions::SVector{2, PlayerState}
end
@functor State (positions,)

const start_state = State(1, 
  SVector{2}([
    PlayerState(
      SVector{2}([4,1]),
      SMatrix{2,5}(Int8[collect(2:6) fill(1, 5)]')),
    PlayerState(
      SVector{2}([4,8]),
      SMatrix{2,5}(Int8[collect(2:6) fill(8, 5)]'))
      ]))
      
      
function is_terminal(state)
  state.positions[1].ball[2] == limits[2] || state.positions[2].ball[2] == 1
end

function occupied(state, y)
  inner(p::SMatrix) = any(all(p .== y[:, na]; dims=1))
  inner(_) = false
  foldmap(inner, Base.:|, false, state)
end

struct PotentialMove
  y::Pos
  norm::Int8
end

function lub(a::PotentialMove, b::PotentialMove)
  if a.norm < b.norm
    return a
  end
  if a.norm == b.norm
    error("Two pieces at the same position: ", a.y, ", ", b.y)
  end
  return b
end

const PieceIx = UInt8
const Action = Tuple{PieceIx, Pos}

function piece_actions(st::State)
  actions = Vector{Action}()
  ball_pos = st.positions[st.player].ball
  for i in 1:5
    x = st.positions[st.player].pieces[:, i]
    if any(x .!= ball_pos)
      for move in (SVector{2}([1,2]), SVector{2}([2,1]))
        for d1 in (-1, 1)
          for d2 in (-1, 1)
            pos = x .+ move .* (@SVector [d1, d2])
            if all(pos .>= 1) && all(pos .<= limits) && !occupied(st, pos)
              push!(actions, (i, pos))
            end
          end
        end
      end
    end
  end
  actions
end


function ball_actions(st::State)
  y = st.positions[st.player].ball
  passes = Set{Pos}([y])
  balls = [y]
  while length(balls) > 0
    x = pop!(balls)
    for move in pass_actions(st, x)
      yy = move.y
      if yy ∉ passes 
        push!(passes, yy)
        push!(balls, yy)
      end
    end
  end
  pop!(passes, y)
  [(UInt8(6), p) for p in passes]
end

next_player(player::Int) = (1 ⊻ (player - 1)) + 1

const Dir = Pos

function encode(pieces::AbstractMatrix)
  (pieces[2,:] .- 1) .* limits[1] .+ pieces[1,:]
end

encode(piece::AbstractVector) = (piece[2] - 1) * limits[1] + piece[1]

function assert_valid_state(st::State)
  for i in (1,2)
    encoded = encode(st.positions[i].pieces)
    @assert length(encoded) == length(unique(encoded))
    ball_pos = encode(st.positions[i].ball[:, na])
    @assert ball_pos[1] in encoded
    @assert all(st.positions[i].pieces .>= 1)
    @assert all(st.positions[i].pieces .<= limits[:, na])
  end
end

function pass_actions(st::State, x::Pos)
  if VALIDATE
    assert_valid_state(st)
  end
  moves = Dict{Dir, PotentialMove}()
  for player in (st.player, next_player(st.player))
    vecs = st.positions[player].pieces .- x[:, na]
    absvecs = abs.(vecs)
    diagonals = absvecs[1, :] .== absvecs[2, :]
    verticals = vecs[1, :] .== 0
    horizontals = vecs[1, :] .== 0
    norms = maximum(absvecs; dims=1)[1, :]
    nz = norms .> 0
    valid = (diagonals .| verticals .| horizontals) .& nz
    norms = norms[valid]
    units = vecs[:, valid] .÷ norms[na, :]
    y = st.positions[player].pieces[:, valid]
    if player == st.player
      for i in 1:size(units, 2)
        u = SVector{2}(units[:, i])
        if haskey(moves, u)
          moves[u] = lub(moves[u], PotentialMove(y[:,i], norms[i]))
        else
          moves[u] = PotentialMove(y[:,i], norms[i])
        end
      end
    else
      for i in 1:size(units, 2)
        u = SVector{2}(units[:, i])
        if haskey(moves, u) && norms[i] < moves[u].norm
          pop!(moves, u)
        end
      end
    end
  end
  values(moves)
end

struct ValuedAction
  action::Union{Action, Nothing}
  value::Float32
end

Base.:*(a::Number, b::ValuedAction) = ValuedAction(b.action, a * b.value)
