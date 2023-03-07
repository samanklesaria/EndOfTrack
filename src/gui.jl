struct GuiPlayer
  from::Observable{Vector{Float32}}
  to::Observable{Vector{Float32}}
  
  function GuiPlayer(from, to)
    on(to) do to
      println("Got $from -> $to")
    end
    new(from, to)
  end
  end
end


function gui()
  f = Figure()
  ax = Axis(f[1,1]; xticks=1:10, yticks=1:10)
  from = Observeable(zeros(Float32, 2))
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
  players = (GuiPlayer(from, to), AlphaBeta(depth=2))
  simulate(start_state, players)
end
