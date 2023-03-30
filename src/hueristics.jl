function ball_yval_hueristic(st::State)
  y = st.positions[st.player].ball[2]
  if st.player == 1
    y / limits[2]
  else
    (limits[2] - y + 1) / limits[2]
  end
end

function piece_yval_hueristic(st::State)
  if st.player == 1
    y = maximum(st.positions[1].pieces[2,:])
    return 0.5f0 * (y / limits[2])
  else
    y = minimum(st.positions[2].pieces[2,:])
    return 0.5f0 * (limits[2] - y + 1) / limits[2]
  end
end

# fast_hueristic(st) = piece_yval_hueristic(st)
fast_hueristic(st) = 0f0
