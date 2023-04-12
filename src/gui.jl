using GLMakie, ArraysOfArrays

const FPos = SVector{2, Float32}

struct GuiPlayer
  from::Observable{Vector{Float32}}
  to::Observable{Vector{Float32}}
  chan::Channel{Tuple{Pos, Pos}}
  p1::Observable{Vector{FPos}}
  b1::Observable{FPos}
  p2::Observable{Vector{FPos}}
  b2::Observable{FPos}
  ghost::Observable{Vector{FPos}}
end

scatterable(pieces) = SVector.(nestedview(Float32.(pieces)))

function GuiPlayer()
  chan = Channel{Tuple{Pos, Pos}}()
  from = Observable(zeros(Float32, 2))
  to = Observable(zeros(Float32, 2))
  on(to) do to
    put!(chan, (SVector{2}(from[]), SVector{2}(to)))
  end
  p1 = scatterable(start_state.positions[1].pieces)
  b1 = start_state.positions[1].ball
  p2 = scatterable(start_state.positions[2].pieces)
  b2 = start_state.positions[2].ball
  ghost = SVector{2, Float32}[]
  GuiPlayer(from, to, chan, p1, b1, p2, b2, ghost)
end 

function update_state!(gui::GuiPlayer, st::State)
  gui.p1[] = scatterable(st.positions[1].pieces)
  gui.p2[] = scatterable(st.positions[2].pieces)
  gui.b1[] = st.positions[1].ball
  gui.b2[] = st.positions[2].ball
end

function animate!(a, b, o)
  if !all(a .== b)
    T = 30
    for t in 1:T
      o[] = (1 - t / T) * a .+ (t / T) * b
      sleep(1/60)
    end
    o[] = b
  end
end

function animate_state!(gui::GuiPlayer, st::State)
  animate!(gui.p1[], scatterable(st.positions[1].pieces), gui.p1)
  animate!(gui.p2[], scatterable(st.positions[2].pieces), gui.p2)
  animate!(gui.b1[], st.positions[1].ball, gui.b1)
  animate!(gui.b2[], st.positions[2].ball, gui.b2)
end

function (gui::GuiPlayer)(st::State)
  animate_state!(gui, st)
  from, to = take!(gui.chan)
  positions = st.positions[st.player]
  encpos = encode(from)
  if encpos == encode(positions.ball)
    relpos = 6
  else
    relpos = findfirst(encode(positions.pieces) .== encpos)
  end
  result = ValuedAction((relpos, to), 0f0)
  update_state!(gui, apply_action(st, result))
  result
end

function gui()
  f = Figure()
  ax = Axis(f[1,1]; xticks=1:10, yticks=1:10)
  deregister_interaction!(ax, :rectanglezoom)
  gp = GuiPlayer()
  register_interaction!(ax, :play) do e::MouseEvent, ax
    if e.type == MouseEventTypes.leftdown
      gp.ghost[] = push!(gp.ghost[], e.data)
      gp.from[] = round.(e.data)
    elseif e.type == MouseEventTypes.leftup
      ghosts = gp.ghost[]
      pop!(ghosts)
      gp.ghost[] = ghosts
      gp.to[] = round.(e.data)
    elseif e.type == MouseEventTypes.leftdrag
      ghosts = gp.ghost[]
      if length(ghosts) > 0
        ghosts[1] = e.data
        gp.ghost[] = ghosts
      end
    end
  end
  players = (gp, AlphaBeta(6))
  scatter!(ax, gp.p1, marker=:circle, color=:green, markersize=17)
  scatter!(ax, gp.p2, marker=:circle, color=:red, markersize=17)
  scatter!(ax, gp.b1, marker=:cross, color=:green, markersize=50)
  scatter!(ax, gp.b2, marker=:cross, color=:red, markersize=50)
  scatter!(ax, gp.ghost, marker=:circle, color=:grey, markersize=17)
  limits!(ax, 0.5, 7.5, 0.5, 8.5)
  ax.aspect = DataAspect()
  display(f)
  simulate(start_state, players)
end
