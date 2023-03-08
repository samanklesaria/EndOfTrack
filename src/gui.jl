using GLMakie

struct GuiPlayer{T}
  from::Observable{Vector{Float32}}
  to::Observable{Vector{Float32}}
  chan::Channel{Tuple{Pos, Pos}}
  ax::T
end
  
function GuiPlayer(from, to, ax)
  chan = Channel{Tuple{Pos, Pos}}()
  on(to) do to
    put!(chan, (SVector{2}(from[]), SVector{2}(to)))
  end
  GuiPlayer(from, to, chan, ax)
end 

function (gui::GuiPlayer)(st::State)
  plot_state(gui.ax, st)
  from, to = take!(gui.chan)
  positions = st.positions[st.player]
  encpos = encode(from)
  if encpos == encode(positions.ball)
    relpos = 6
  else
    relpos = findfirst(encode(positions.pieces) .== encpos)
  end
  result = ValuedAction((relpos, to), 0f0)
  plot_state(gui.ax, apply_action(st, result))
  result
end

function gui()
  f = Figure()
  ax = Axis(f[1,1]; xticks=1:10, yticks=1:10)
  from = Observable(zeros(Float32, 2))
  to = Observable(zeros(Float32, 2))
  deregister_interaction!(ax, :rectanglezoom)
  register_interaction!(ax, :play) do e::MouseEvent, ax
    if e.type == MouseEventTypes.leftdown
      from[] = round.(e.data)
    elseif e.type == MouseEventTypes.leftup
      to[] = round.(e.data)
    end
  end
  display(f)
  players = (GuiPlayer(from, to, ax), AlphaBeta(depth=5))
  simulate(start_state, players)
end

# ALSO: would be nice to show a ghost piece as you're moving it.
# We can do this by just sleeping a lot

# NOTE: instead of clearing the frame, we can also make observables

function plot_state(ax, st::State)
  empty!(ax)
  scatter!(ax, Vector(st.positions[1].pieces[1, :]),
    Vector(st.positions[1].pieces[2, :]), marker=:circle, color=:green, markersize=17)
  scatter!(ax, Vector(st.positions[1].ball[1:1]),
    Vector(st.positions[1].ball[2:2]), marker=:cross, color=:green, markersize=50)
  scatter!(ax, Vector(st.positions[2].pieces[1,:]),
    Vector(st.positions[2].pieces[2, :]), marker=:circle, color=:red, markersize=17)
  scatter!(ax, Vector(st.positions[2].ball[1:1]),
    Vector(st.positions[2].ball[2:2]), marker=:cross, color=:red, markersize=50)
  limits!(ax, 0.5, 7.5, 0.5, 8.5)
  ax.aspect = DataAspect()
end
