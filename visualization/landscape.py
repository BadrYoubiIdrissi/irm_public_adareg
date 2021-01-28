import plotly.express as px

def get_percent_range(arr, percent=0.1):
  min_val, max_val  = arr.flatten().min(), arr.flatten().max()
  return min_val+(1-percent)*(max_val-min_val), max_val

def make_surface_plots(X, Y, error, penalty, scaling):
  fig = make_subplots(
      rows=1, cols=2,
      specs=[[{'is_3d': True}, {'is_3d': True}]],
      subplot_titles=("ERM loss term", "IRM Penalization loss term"))

  lighting_effects = dict(diffuse=1, roughness = 0.4, specular=0.2)

  colors = ["white", "blue"]
  names = ["ERM Max", "IRM Max"]
  for i, err in enumerate([error, penalty]):
    err_range = get_percent_range(scaling(err))
    fig.add_trace(go.Surface(x=X,y=Y,z=scaling(err), 
                            lighting=lighting_effects, 
                            reversescale=False, 
                            colorscale="Viridis", 
                            showscale=False, 
                            contours = {
                                  "z": {"show": True, "start": err_range[0], "end": err_range[1], "size": (err_range[1]-err_range[0])/20}
                              }), row=1, col=1+i)
    
    x_opt, y_opt = np.unravel_index(np.argmin(err, axis=None), err.shape)
    fig.add_trace(go.Scatter3d(x=[X[x_opt,y_opt]], y=[Y[x_opt,y_opt]], z=[scaling(err[x_opt,y_opt])],name=names[i], mode="markers", marker=dict(line_color="midnightblue", color=colors[i], 
                                                                              line_width=2, size=6)), row=1, col=1+i)
  fig.layout.update(width=1300, height=700)
  fig.update_layout(scene = dict(
                      xaxis_title='Phi_0',
                      yaxis_title='Phi_1'))
  fig.update_layout(scene2 = dict(
                      xaxis_title='Phi_0',
                      yaxis_title='Phi_1'))
  return fig

def make_penalty_plot(fig, X, Y, penalty, scaling, name):

  lighting_effects = dict(diffuse=1, roughness = 0.4, specular=0.2)

  fig.add_trace(go.Surface(x=X,y=Y,z=scaling(penalty), 
                            lighting=lighting_effects, 
                            reversescale=False, 
                            colorscale="Viridis", 
                            showscale=False,
                            opacity=0.5))
    
  x_opt, y_opt = np.unravel_index(np.argmin(penalty, axis=None), penalty.shape)
  fig.add_trace(go.Scatter3d(x=[X[x_opt,y_opt]], y=[Y[x_opt,y_opt]], z=[scaling(penalty[x_opt,y_opt])],name=name+' max', mode="markers", marker=dict(line_color="midnightblue", 
                                                                            line_width=2, size=6)))
  fig.layout.update(width=1000, height=700)
  fig.update_layout(scene = dict(
                      xaxis_title='Phi_0',
                      yaxis_title='Phi_1'))
  return fig

def add_phi_evolution_lines(history, scaling):
  colors = px.colors.qualitative.Dark24
  for i, name in enumerate(history):
    for j, trace in enumerate(history[name]):
      fig.add_trace(go.Scatter3d(
          x=trace["x"][0:-1:10], y=trace["y"][0:-1:10], z=scaling(trace["error"][0:-1:10]),
          mode="lines",
          line=dict(width=6, color=colors[i]),
          showlegend=False
      ), row=1, col=1)

      fig.add_trace(go.Scatter3d(
          x=[trace["x"][-1]], y=[trace["y"][-1]], z=[scaling(trace["error"][-1])],
          mode="markers",
          marker=dict(size=6,line_color="midnightblue",line_width=2,color=colors[i]),
          name=name+"_endpoint",
          showlegend=(j==0)
      ), row=1, col=1)

      fig.add_trace(go.Scatter3d(
          x=trace["x"][0:-1:10], y=trace["y"][0:-1:10], z=scaling(trace["penalty"][0:-1:10]),
          mode="lines",
          line=dict(width=6, color=colors[i]),
          name=name,
          showlegend=(j==0)
      ), row=1, col=2)

      fig.add_trace(go.Scatter3d(
          x=[trace["x"][-1]], y=[trace["y"][-1]], z=[scaling(trace["penalty"][-1])],
          mode="markers",
          marker=dict(size=6,line_color="midnightblue",line_width=2,color=colors[i]),
          showlegend=False
      ), row=1, col=2)
  return fig

def make_heatmap(phi_x,phi_y, error, penalty, history, scaling):

  fig = make_subplots(
      rows=1, cols=2,
      subplot_titles=("ERM loss term", "IRM Penalization loss term"))
      # specs=[[{'is_3d': True}, {'is_3d': True}]])

  for i, err in enumerate([error, penalty]):
    # err_min, err_max = get_percent_range(scaling(err), 0.2)
    fig.add_trace(go.Heatmap(x=phi_x, y=phi_y, z=scaling(err.T) ,colorscale="Viridis", showscale=False), row=1, col=1+i)

  colors = px.colors.qualitative.Dark24

  for i, name in enumerate(history):
    for j, trace in enumerate(history[name]):
      fig.add_trace(go.Scatter(
          x=trace["x"][0:-1:10], y=trace["y"][0:-1:10],
          mode="lines",
          line=dict(width=2, color=colors[i]),
          showlegend=False
      ), row=1, col=1)

      fig.add_trace(go.Scatter(
          x=trace["x"][0:-1:10], y=trace["y"][0:-1:10], 
          mode="lines",
          line=dict(width=2, color=colors[i]),
          name=name,
          showlegend=(j==0)
      ), row=1, col=2)

  fig.add_trace(go.Scatter(x=[1.0], y=[0.0], name="Optimal Point", mode="markers", marker=dict(line_color="midnightblue", color="lightskyblue", 
                                                                              line_width=2, size=8)), row=1, col=1)
  fig.add_trace(go.Scatter(x=[1.0], y=[0.0], showlegend=False,  marker=dict(line_color="midnightblue", color="lightskyblue", 
                                                                              line_width=2, size=8)), row=1, col=2)

  fig.add_trace(go.Scatter(x=[0.0], y=[0.0], name="Zero", mode="markers", marker=dict(line_color="midnightblue", color="white", 
                                                                              line_width=2, size=8)), row=1, col=1)
  fig.add_trace(go.Scatter(x=[0.0], y=[0.0], showlegend=False,  marker=dict(line_color="midnightblue", color="white", 
                                                                              line_width=2, size=8)), row=1, col=2)

  fig.layout.update(width=1300, height=700)
  fig.update_yaxes(title_text="Phi_1", row=1, col=1)
  fig.update_yaxes(title_text="Phi_1", row=1, col=2)
  fig.update_xaxes(title_text="Phi_0", row=1, col=1)
  fig.update_xaxes(title_text="Phi_0", row=1, col=2)
  fig.show()