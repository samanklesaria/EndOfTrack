# 2x8 matrix of knights moves
const moves = cu(reduce((x,y)->cat(x, y; dims=2),
  Int8[move .* [d1, d2] for move in
    ([1,2], [2,1]) for d1 in (-1, 1) for d2 in (-1, 1)]))
    
const limits = cu(Int8[7, 8])

# 2x8 matrix of passes
const passes = cu(reduce((x,y)->cat(x, y; dims=2),
  Int8[[d1, d2] for d1 in (-1f0, 1f0) for d2 in (-1f0, 1f0)]))

function is_terminal(x, player_ix, ball_ix)
  x[:, 2, ball_ix] .== limits[player_ix]
end

# x is a 2x5 positions patrix
# ixs is a (pieces x 8) matrix, with a 1 where moves are possible
function piece_actions(x, ball_ix)
  ball_pos = x[:, ball_ix] # 2
  y = x[:,:,na] .+ moves[:,na,:] # 2 x 5 x 8
  ixs = all(y .>= 1; dims=1) .& all(y .<= limits[:, na]; dims=1) .&
    .!occupied.(x, encode(y)) .& (encode(x) .!= encode(ball_pos))[:,na]
  moves[:, ixs] # 2 x valid
end

next_player(player::Integer) = (1 ⊻ (player - 1)) + 1

function encode(pieces::AbstractMatrix)
  (pieces[2,:] .- 1) .* limits[1] .+ pieces[1,:]
end

encode(piece::AbstractVector) = (piece[2] - 1) * limits[1] + piece[1]

# Returns an 8-vector mask for how much in each direction you can pass from x
function match(vecs)
  absvecs = abs.(vecs)
  norms = maximum(absvecs; dims=1)[1, :] # 5
  units = vecs .÷ norms # 2 x 5
  mat = all(units[:,:,na] .== passes[:, na,:]; dims=1) # 5 x 8
  mat, minimum(mat .* norms[:, na]; dims= 1) # 5, 8
end

function adj_row(pos, player_ix, x)
  mat, mine = match(pos[player_ix] .- x[:, na])
  _, yours = match(pos[next_player(player_ix)] .- x[:, na])
  mask = mine .< yours
  sum(mat[:, mask]; dims=2)
end

function adjacencies(pos, player_ix)
  reduce((x,y)->cat(x,y; dims=2),
    mapslices(x-> adj_row(pos, player_ix, x), pos[player_ix]; dims=1))
end
  
function ball_actions(pos, player_ix, ball_ix)
  adj = adjacencies(st)
  connected = (adj + I)^5 - I
  ixs = connected[ball_ix]
  pos[player_ix, ixs]
end

function actions(pos, player_ix, ball_ix)
  [piece_actions(pos, player_ix, ball_ix); ball_actions(pos, player_ix)]
end
