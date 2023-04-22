const mean_vert = SVector{2, Float32}([0, (limits[2] + 1)/2])
const flip_vert = SVector{2, Int8}([1, -1])

const mean_hor = SVector{2}([(limits[1] + 1)/2, 0])
const flip_hor = SVector{2}([-1, 1])

flip_pos_vert(a::Pos) = flip_vert .* (a .- mean_vert) .+ mean_vert
flip_pos_vert(a::SMatrix) = flip_vert[:, na] .* (a .- mean_vert[:, na]) .+ mean_vert[:, na]

flip_pos_hor(a::Pos) = Pos(flip_hor .* (a .- mean_hor) .+ mean_hor)
flip_pos_hor(a::SMatrix) = SMatrix{2,5, Int8}(flip_hor[:, na] .* (a .- mean_hor[:, na]) .+ mean_hor[:, na])

function normalize_player(st)
  if st.player == 2
    st = map_state(flip_pos_vert, st)
    st = State(1, reverse(st.positions))
    (-1, st)
  else
    (1, st)
  end
end

flip_board(st) = map_state(flip_pos_hor, st)

function flip_players(st) 
    st = map_state(flip_pos_vert, st)
    State(next_player(st.player), reverse(st.positions))
end
