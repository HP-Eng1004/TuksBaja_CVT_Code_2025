import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

def plot_torque_transfer(result, q, w, e, r, t,shim,legend_columns=2):
    ys = result['ys']
    T1_plt = result['T1_plt']
    T2_plt = result['T2_plt']
    Tmax1_plt = result['Tmax1_plt']
    Tmax2_plt = result['Tmax2_plt']
    slip = result['slip']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ys, y=T1_plt, mode='lines', name='Primary Torque (T1)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=ys, y=Tmax1_plt, mode='lines', name='Max Primary Torque (Tmax1)', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=ys, y=T2_plt, mode='lines', name='Secondary Torque (T2)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=ys, y=Tmax2_plt, mode='lines', name='Max Secondary Torque (Tmax2)', line=dict(color='orange', dash='dash')))
    
    # Calculate itemwidth based on desired number of columns and plot width
    plot_width = 1000  # Fixed plot width in pixels
    item_width = plot_width /(4*legend_columns)  # Approximate width per legend item

    fig.update_layout(
        template='plotly_white',
        title=f"Torque Transfer Capacity - Setup: {[q, w, e, r, t]} - {shim:.0f}",
        title_x=0.5,
        margin=dict(t=40, b=10,r=0),
        xaxis_title="Shift Percentage",
        yaxis_title="Torque (Nm)",
        width=plot_width,
        height=500,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            font=dict(size=14),
            tracegroupgap=5,  # Small gap between legend items
            # itemsizing="constant",
            itemwidth=item_width  # Set item width to control number of columns
        ),

    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def plot_radial_force(result, q, w, e, r, t, shim, legend_columns=2):
    ys = result['ys']
    Fc1_plt = result.get('Fc1_plt', [])
    Fc2_plt = result.get('Fc2_plt', [])
    Rs1_plt = result.get('Rs1_plt', [])
    Rs2_plt = result.get('Rs2_plt', [])
    R1_plt = result.get('R1_plt', [])
    R2_plt = result.get('R2_plt', [])

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatter(x=ys, y=Fc1_plt, mode='lines', line=dict(dash='dash', color='orange'), name='Primary Centrifugal Force'))
    fig_rad.add_trace(go.Scatter(x=ys, y=Fc2_plt, mode='lines', line=dict(dash='dash', color='blue'), name='Secondary Centrifugal Force'))
    fig_rad.add_trace(go.Scatter(x=ys, y=Rs1_plt, mode='lines', line=dict(dash='dot', color='orange'), name='Primary Side to Radial Force'))
    fig_rad.add_trace(go.Scatter(x=ys, y=Rs2_plt, mode='lines', line=dict(dash='dot', color='blue'), name='Secondary Side to Radial Force'))
    fig_rad.add_trace(go.Scatter(x=ys, y=R1_plt, mode='lines', line=dict(color='orange'), name='Primary Total Radial Force'))
    fig_rad.add_trace(go.Scatter(x=ys, y=R2_plt, mode='lines', line=dict(color='blue'), name='Secondary Total Radial Force'))

    # Calculate itemwidth based on desired number of columns and plot width
    plot_width = 1000  # Fixed plot width in pixels
    item_width = plot_width /(4*legend_columns)  # Approximate width per legend item

    fig_rad.update_layout(
        title=f"Radial Force along Shift % - Setup: {[q, w, e, r, t]} - {shim:.0f}",
        title_x=0.5,
        margin=dict(t=40, b=10,r=0),
        xaxis_title="Shift Percentage",
        yaxis_title="Radial Force [N]",
        template="plotly_white",
        width=plot_width,
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            font=dict(size=14),
            tracegroupgap=5,  # Small gap between legend items
            # itemsizing="constant",
            itemwidth=item_width  # Set item width to control number of columns
        ),
        hovermode='closest'
    )
    fig_rad.update_xaxes(showgrid=True)
    fig_rad.update_yaxes(showgrid=True)

    return fig_rad

import plotly.graph_objects as go

def plot_error(result, q, w, e, r, t, shim, legend_columns=2):
    """
    Plot CVT simulation errors before and after iteration using Plotly.

    Parameters:
    result: dict, output from cvt_simulation containing 'ys', 'F2_err_plt', 'Terr_plt',
            'T_err_history', 'F2_err_history'
    q, w, e, r, t: CVT setup parameters (flyweights, primary spring, ramp angle, secondary pretension, secondary spring)
    shim: Spring shim displacement
    legend_columns: Number of columns in the legend (default=2)

    Returns:
    Plotly figure object
    """
    # Extract data
    ys = result['ys']
    F2_err_plt = result.get('F2_err_plt', [])
    Terr_plt = result.get('Terr_plt', [])
    T_err_history = result.get('T_err_history', [])
    F2_err_history = result.get('F2_err_history', [])

    # Extract initial errors from history (first iteration)
    T_err_initial = [errors[0] if errors else 0 for errors in T_err_history]
    F2_err_initial = [errors[0] if errors else 0 for errors in F2_err_history]

    # Create figure
    fig_err = go.Figure()

    # Add traces for final errors (after iteration)
    fig_err.add_trace(
        go.Scatter(
            x=ys,
            y=F2_err_plt,
            mode='lines+markers',
            name='Secondary Force Error (After)',
            line=dict(color='orange')
        )
    )
    fig_err.add_trace(
        go.Scatter(
            x=ys,
            y=Terr_plt,
            mode='lines+markers',
            name='Torque Error (After)',
            line=dict(color='blue')
        )
    )

    # Add traces for initial errors (before iteration)
    fig_err.add_trace(
        go.Scatter(
            x=ys,
            y=F2_err_initial,
            mode='lines+markers',
            name='Secondary Force Error (Before)',
            line=dict(color='orange', dash='dash')
        )
    )
    fig_err.add_trace(
        go.Scatter(
            x=ys,
            y=T_err_initial,
            mode='lines+markers',
            name='Torque Error (Before)',
            line=dict(color='blue', dash='dash')
        )
    )

    # Calculate itemwidth based on desired number of columns and plot width
    plot_width = 1000  # Fixed plot width in pixels
    item_width = plot_width / (4 * legend_columns)  # Approximate width per legend item

    # Update layout
    fig_err.update_layout(
        title=f"Percentage Difference of Torque and Secondary Force Errors (Before vs After Iteration) - Setup: {[q, w, e, r, t]} - {shim:.0f}",
        title_x=0.5,
        margin=dict(t=40, b=10, r=0),
        xaxis_title="Shift Percentage",
        yaxis_title="% Difference Error",
        template="plotly_white",
        yaxis=dict(range=[0, max(max(T_err_initial + F2_err_initial + Terr_plt + F2_err_plt, default=30), 30)]),
        width=plot_width,
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            orientation="h",
            font=dict(size=14),
            tracegroupgap=5,
            itemwidth=item_width
        ),
        hovermode='closest'
    )

    # Update axes
    fig_err.update_xaxes(showgrid=True)
    fig_err.update_yaxes(showgrid=True)

    return fig_err

def plot_force_balance(results, q_range, w_range):
    F1_values = np.zeros((len(q_range), len(w_range)))
    F2_values = np.zeros((len(q_range), len(w_range)))
    
    for result in results:
        q, w = int(result['params'][0]), int(result['params'][1])
        if q in q_range and w in w_range:
            i, j = list(q_range).index(q), list(w_range).index(w)
            F1_values[i, j] = np.mean(result['F1_plt'])
            F2_values[i, j] = np.mean(result['F2_plt'])
    
    balance = np.abs(F1_values - F2_values)
    
    fig = go.Figure(data=go.Heatmap(
        z=balance,
        x=[f"Spring {w}" for w in w_range],
        y=[f"Flyweight {q}" for q in q_range],
        colorscale='Viridis',
        colorbar=dict(title='|F1 - F2| (N)')
    ))
    
    fig.update_layout(
        template='plotly_white',
        title="Force Balance Across Flyweight and Primary Spring Combinations",
        title_x=0.5,
        xaxis_title="Primary Spring Index",
        yaxis_title="Flyweight Index",
        showlegend=False
    )
    return fig

def plot_slip_risk(results, goal, tolerance):
    q_vals, w_vals, slips, deviations = [], [], [], []
    
    for result in results:
        engine_rpms = np.array(result['engine_rpms'])
        slip = result['slip']
        q, w = result['params'][0], result['params'][1]
        straight_rpms = engine_rpms[4:9]
        if np.all(np.abs(straight_rpms - goal) <= tolerance):
            q_vals.append(q)
            w_vals.append(w)
            slips.append(slip)
            deviations.append(np.mean(np.abs(straight_rpms - goal)))
    
    fig = go.Figure(data=go.Scatter(
        x=q_vals,
        y=w_vals,
        mode='markers',
        marker=dict(
            size=[10 + 5 * d for d in deviations],
            color=slips,
            colorscale='RdYlGn',
            colorbar=dict(title='Slip (1=Yes, 0=No)'),
            showscale=True
        ),
        text=[f"Dev: {d:.0f} RPM" for d in deviations],
        hoverinfo='x+y+text'
    ))
    
    fig.update_layout(
        template='plotly_white',
        title=f"Slip Risk and RPM Deviation (Goal: {goal} RPM, Tolerance: {tolerance} RPM)",
        title_x=0.5,
        xaxis_title="Flyweight Index",
        yaxis_title="Primary Spring Index",
        showlegend=False
    )
    return fig

def plot_rpm_surface(results, q_range, w_range, shift_idx=4):
    rpm_values = np.zeros((len(q_range), len(w_range)))
    
    for result in results:
        q, w = int(result['params'][0]), int(result['params'][1])
        if q in q_range and w in w_range:
            i, j = list(q_range).index(q), list(w_range).index(w)
            rpm_values[i, j] = result['engine_rpms'][shift_idx]
    
    fig = go.Figure(data=[go.Surface(
        z=rpm_values,
        x=[f"Spring {w}" for w in w_range],
        y=[f"Flyweight {q}" for q in q_range],
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        template='plotly_white',
        title=f"Engine RPM at Shift {result['ys'][shift_idx]*100:.0f}%",
        title_x=0.5,
        scene=dict(
            xaxis_title="Primary Spring Index",
            yaxis_title="Flyweight Index",
            zaxis_title="Engine RPM"
        ),
        showlegend=False
    )
    return fig

def plot_engagement_rpms(results):
    labels, idle_rpms, max_idle_rpms, eng_rpms, clu_rpms = [], [], [], [], []
    
    for result in results:
        params = result['params']
        engine_rpms = result['engine_rpms']
        idle_rpms.append(engine_rpms[0])
        max_idle_rpms.append(engine_rpms[1])
        eng_rpms.append(engine_rpms[2])
        clu_rpms.append(engine_rpms[3])
        labels.append(str(params))
    
    fig = go.Figure(data=[
        go.Bar(name='Idle RPM', x=labels, y=idle_rpms, marker_color='grey'),
        go.Bar(name='Max Idle RPM', x=labels, y=max_idle_rpms, marker_color='blue'),
        go.Bar(name='Engagement RPM', x=labels, y=eng_rpms, marker_color='orange'),
        go.Bar(name='Clutching RPM', x=labels, y=clu_rpms, marker_color='green')
    ])
    
    fig.update_layout(
        template='plotly_white',
        title="Engagement and Clutching RPMs Across Setups",
        title_x=0.5,
        xaxis_title="Setup [q,w,e,r,t]",
        yaxis_title="RPM",
        barmode='group',
        showlegend=True,
        hovermode='closest'
    )
    fig.update_xaxes(tickangle=45)
    return fig

def plot_cvt_error_convergence(simulation_data):
    """
    Visualize CVT simulation error convergence over iterations using Plotly.

    Parameters:
    simulation_data: dict, output from cvt_simulation function containing
                     'ys', 'T_err_history', 'F2_err_history'

    Returns:
    Plotly figure object
    """
    # Extract data
    ys = simulation_data['ys']
    T_err_history = simulation_data['T_err_history']
    F2_err_history = simulation_data['F2_err_history']

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Torque Error Convergence",
            "Force Error Convergence"
        ),
        horizontal_spacing=0.15
    )

    # Plot 1: Torque Error Convergence
    for i, errors in enumerate(T_err_history):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(errors))), 
                y=errors, 
                mode='lines+markers',
                name=f'Torque Err y={ys[i]:.2f}',
            ),
            row=1, col=1
        )

    # Plot 2: Force Error Convergence
    for i, errors in enumerate(F2_err_history):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(errors))), 
                y=errors, 
                mode='lines+markers',
                name=f'Force Err y={ys[i]:.2f}',
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title="CVT Error Convergence",
        height=400,
        width=1000,
        showlegend=True,
        template="plotly_white"
    )

    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Torque Error (%)", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Force Error (%)", row=1, col=2)

    return fig

def plot_exhaustive_search(results, goal, Vsmin, Vsmax, ErpmMax, ErpmMin):
    fig = go.Figure()
    Low = [0, Vsmin]
    High = [0, Vsmax]
    PeakP = [goal, goal]
    Govspeed = [ErpmMax, ErpmMax]
    Idle = [ErpmMin, ErpmMin]
    RPM = [0, ErpmMax]
    dash = "dash"

    fig.add_trace(go.Scatter(
        x=[0, Vsmin], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Low gear ratio',
        hovertemplate='name'
    ))
    fig.add_trace(go.Scatter(
        x=[Vsmin, Vsmax], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='High gear ratio',
        hovertemplate='name'
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=PeakP,
        mode='lines', line=dict(dash='dash', color='green'),
        name='Ideal shift',
        hovertemplate='name'
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=Govspeed,
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
        hovertemplate='name'
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=Idle,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
        hovertemplate='name'
    ))

    for result in results:
        lame = str(result['params'])
        if result['slip'] == 1:
            lame += " - SLIP"
            fig.add_trace(go.Scatter(
                x=result['veh_speed'], y=result['engine_rpms'],
                name=lame, hoverinfo='name', line=dict(dash=dash)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=result['veh_speed'], y=result['engine_rpms'],
                name=lame, hoverinfo='name'
            ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text="Engine RPM along Vehicle Speed (All Setups)", x=0.5, xanchor='center'),
        xaxis_title="Vehicle Speed in km/h",
        yaxis_title="Engine Speed in RPM",
        yaxis=dict(range=[1000, 4500]),
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=8)
        ),
        margin=dict(b=150),
        hovermode='closest',
        width=1000,
        height=600
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig