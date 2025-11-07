import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import numpy as np
    import plotly.graph_objs as go
    from matplotlib import pyplot as plt
    from numpy import pi as pi
    from numpy import tan as tan

    from CVT_Parts_2025 import FW, PS, SS, Helix, SPT, FW_names, PS_names, SPT_names

    from CVT_Plotting_2025 import plot_torque_transfer, plot_error, plot_radial_force, plot_cvt_error_convergence


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CVT Model Simulation
    ### Interactive Simulation with Sliders
    #### JL (Hannes) Pretorius - (2025)
    - Simulates CVT performance for given parameters
    - Includes sliders for interactive parameter adjustment
    - Plots Engine RPM vs. Vehicle Speed, Radial Forces, and Errors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    q_slider = mo.ui.slider(start=0, stop=8, step=1, value=7,)
    w_slider = mo.ui.slider(start=0, stop=6, step=1, value=3,)
    e_slider = mo.ui.slider(start=0, stop=9, step=1, value=8,)
    r_slider = mo.ui.slider(start=0, stop=4, step=1, value=0,)
    t_slider = mo.ui.slider(start=0, stop=2, step=1, value=0,)
    goal_slider = mo.ui.slider(start=3000, stop=3800, step=50, value=3600)
    shim_slider = mo.ui.slider(start=0, stop=15, step=1, value=0)

    mu_d =  mo.ui.slider(
        start=0, 
        stop=1, 
        step=0.05, 
        value=0.3, 
        label="Dynamic Coeff. of Friction"
    )

    T_start =  mo.ui.slider(
        start=0, 
        stop=25, 
        step=1, 
        value=17.4, 
        label="Take-off Torque [Nm]"
    )
    return (
        T_start,
        e_slider,
        goal_slider,
        mu_d,
        q_slider,
        r_slider,
        shim_slider,
        t_slider,
        w_slider,
    )


@app.cell
def _(
    T_start,
    e_slider,
    goal_slider,
    mu_d,
    q_slider,
    r_slider,
    shim_slider,
    t_slider,
    w_slider,
):
    # General CVT parameters for plotting
    GR = 11.3
    CVTH = 0.75
    CVTL = 3.8
    TRL = GR * CVTL
    TRH = GR * CVTH
    ErpmMax = 3700
    ErpmMin = 1800
    Wdia = 23 * 0.0254
    Wcirc = 1.74
    Vsmax = ErpmMax / TRH / 60 * Wcirc * 3.6
    Vsmin = ErpmMax / TRL / 60 * Wcirc * 3.6

    # Run simulation
    result = cvt_simulation(
        q=q_slider.value,
        w=w_slider.value,
        e=e_slider.value,
        r=r_slider.value,
        t=t_slider.value,
        goal=goal_slider.value,
        shim=shim_slider.value,
        cf_dyn = mu_d.value, #Dynamic Coeff. of Friction
        T_takeoff = T_start.value # Takeoff Torque
    )

    q=q_slider.value
    w=w_slider.value
    e=e_slider.value
    r=r_slider.value
    t=t_slider.value
    shim=shim_slider.value

    cf_dyn = mu_d.value #Dynamic Coeff. of Friction
    T_takeoff = T_start.value # Takeoff Torque

    # Part names
    FWn = FW_names()
    PSn = PS_names()
    SSn = np.array(["Red", "Black", "Silver"])
    SPTn = SPT_names()

    rpm_idle_max = result['Idle_rpm']
    rpm_eng = result['Engage_rpm']
    rpm_clu = result['Clutch_rpm']
    engine_rpms = result['engine_rpms']
    veh_speed = result['veh_speed']
    return (
        ErpmMax,
        ErpmMin,
        FWn,
        GR,
        PSn,
        SPTn,
        SSn,
        Vsmax,
        Vsmin,
        Wdia,
        e,
        engine_rpms,
        q,
        r,
        result,
        rpm_clu,
        rpm_eng,
        rpm_idle_max,
        shim,
        t,
        veh_speed,
        w,
    )


@app.cell(hide_code=True)
def _(
    T_start,
    e_slider,
    goal_slider,
    mo,
    mu_d,
    q_slider,
    r_slider,
    shim_slider,
    t_slider,
    w_slider,
):
    mo.vstack([
        mo.hstack([T_start,T_start.value,"[Nm] ; ", mu_d,mu_d.value], justify="start"),
        mo.md(f"**Goal RPM**: {goal_slider.value} {goal_slider}"),
        mo.hstack([
            mo.md(f"**Flyweights**: {q_slider.value} {q_slider}"),
            mo.md(f"**Primary Spring**: {w_slider.value} {w_slider}"),

            mo.md(f"**Shim**: {shim_slider.value} mm {shim_slider}")
        ], justify="start"),
        mo.hstack([
            mo.md(f"**Ramp Angle**: {e_slider.value} {e_slider}"),
            mo.md(f"**Pretension**: {r_slider.value} {r_slider}"),
            mo.md(f"**Secondary Spring**: {t_slider.value} {t_slider}"),

        ], justify="start")
    ])
    return


@app.cell
def _(
    ErpmMax,
    ErpmMin,
    FWn,
    PSn,
    SPTn,
    SSn,
    Vsmax,
    Vsmin,
    e,
    e_slider,
    engine_rpms,
    goal_slider,
    mo,
    q,
    q_slider,
    r,
    r_slider,
    result,
    rpm_clu,
    rpm_eng,
    rpm_idle_max,
    shim,
    shim_slider,
    t,
    t_slider,
    veh_speed,
    w,
    w_slider,
):
    # Display plots and simulation results
    info = True
    vert = False
    # Generate plots
    plots = plot_simulation(
        result,
        q=q_slider.value,
        w=w_slider.value,
        e=e_slider.value,
        r=r_slider.value,
        t=t_slider.value,
        shim=shim_slider.value,
        goal=goal_slider.value,
        Vsmin=Vsmin,
        Vsmax=Vsmax,
        ErpmMax=ErpmMax,
        ErpmMin=ErpmMin,
    )


    if info == True and vert == False:
        print("Hallo World")
        mo.output.append(
            mo.vstack([
                mo.hstack([
                    mo.ui.plotly(plots['main_plot']),
                    mo.md(
                       f"""      
                       **CVT Setup:** {[q,w,e,r,t]} - {shim:.0f}\n
                       **Flyweight [{q}]:** {FWn[q]}\n
                       **Spring 1 [{w}]:** {PSn[w]}\n
                       **Shim:** {shim:.0f}mm\n
                       **Ramp Angle [{e}]:** {e*2 +32}\n
                       **Pretension [{r}]:** {SPTn[r]}\n
                       **Spring 2 [{t}]:** {SSn[t]}\n
                       \n

                       **Useful Output Parameters:** \n
                        - Max Idle: {rpm_idle_max:.0f} rpm \n
                        - Engagement: {rpm_eng:.0f} rpm\n
                        - Low Gear: {rpm_clu:.0f} rpm\n
                        - Ave. Shift: {np.average(engine_rpms[4:]):.0f} rpm\n
                        - Shift Bandwidth : {np.max(engine_rpms[4:])-np.min(engine_rpms[4:]):.0f} rpm\n
                        - Max. Speed: {veh_speed[-1]:.0f}km/h
                        """),


                   ],widths=[4,1])]))

    if info == True and vert == True:
        mo.output.append(
            mo.vstack([
                mo.ui.plotly(plots['main_plot']),
                mo.hstack([
                    mo.md(
                       f"""
                       **Primary Parts:**\n
                       **CVT Setup:** {[q,w,e,r,t]} - {shim:.0f}\n
                       **Flyweight [{q}]:** {FWn[q]}\n
                       **Spring 1 [{w}]:** {PSn[w]}\n
                       **Shim:** {shim:.0f}mm\n
                        """),
                    mo.md(
                        f"""
                        **Secondary Parts**\n
                        **Ramp Angle [{e}]:** {e*2 +32}\n
                        **Pretension [{r}]:** {SPTn[r]}\n
                        **Spring 2 [{t}]:** {SSn[t]}\n
                        """
                    ),
                    mo.md(
                        f"""
                        **Useful Output Parameters:** \n
                        - Max Idle: {rpm_idle_max:.0f} rpm \n
                        - Engagement: {rpm_eng:.0f} rpm\n
                        - Low Gear: {rpm_clu:.0f} rpm\n
                        - Ave. Shift: {np.average(engine_rpms[4:]):.0f} rpm\n
                        - Shift Bandwidth : {np.max(engine_rpms[4:])-np.min(engine_rpms[4:]):.0f} rpm\n
                        - Max. Speed: {veh_speed[-1]:.0f}km/h"""
                    )

                   ],widths=[1,1,1])]))

    if info == False:
        mo.output.append( 
            mo.vstack([mo.ui.plotly(plots['main_plot'])]))
    return


@app.cell
def _(e, q, r, result, shim, t, w):
    plot_radial_force(result, q, w, e, r, t, shim)
    return


@app.cell
def _(e, q, r, result, shim, t, w):
    plot_torque_transfer(result, q, w, e, r, t, shim)
    return


@app.cell
def _(e, q, r, result, shim, t, w):
    plot_error(result, q, w, e, r, t,shim)
    return


@app.cell
def _(result):
    plot_cvt_error_convergence(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mathematical Model
    The function below defines al the different stages the shifting curve and calculate all the applicable parameters for plotting or further calculations
    """)
    return


@app.function
def cvt_simulation(q=7, w=3, e=8, r=1, t=1, goal=3400, shim=0, no_T2=False, cf_dyn = 0.3, T_takeoff = 17.4):
    """
    Simulate CVT based on parameters.

    Parameters:
    q: Flyweights index (0-8)
    w: Primary spring index (0-6)
    e: Ramp angle index (0-9)
    r: Secondary pretension index (0-4)
    t: Secondary spring index (0-2)
    goal: Peak power RPM
    shim: Spring shim displacement
    """
    # Initialize part parameters
    FWs = FW()
    PSs = PS()
    SSs = SS()
    Ramps = Helix()
    SPTs = SPT()

    # Belt parameters
    L = 1.0414
    M = 0.365
    m_prime = M / L
    beta = np.radians(12)

    # General CVT parameters
    cf = 0.625  # Effective Coeff. of Friction
    GR = 11.3  # Gear Reduction Ratio
    CVTH = 0.76  # CVT High Ratio
    CVTL = 3.8  # CVT Low Ratio
    TRL = GR * CVTL  # Torque Ratio Low
    TRH = GR * CVTH  # Torque Ratio High
    ErpmMax = 3700  # Max engine RPM
    ErpmMin = 1800  # Min engine RPM
    Wdia = 23 * 0.0254  # wheel diameter in m
    Wcirc = 1.74  # Wdia * np.pi ### needs calibration
    Vsmax = ErpmMax / TRH / 60 * Wcirc * 3.6  # Max vehicle speed in km/h
    Vsmin = ErpmMax / TRL / 60 * Wcirc * 3.6  # Min vehicle speed in km/h

    eff = 1  # Efficiency of torque transfer (Not used 2025)
    floss = 1 # Friction Losses (Not used 2025)

    r_h = 0.045  # Torque Feedback Ramp Radius
    P0 = 63.5  # static spring displacement
    Peng = 59  # Spring displacement when belt starts to engage
    Pmax = 30.226  # Spring displacement when clutch is fully shifted

    w_goal = goal* np.pi/30
    T1 = engTorq(w_goal)

    # Part names (for reference, optional output)
    FWn = FW_names()
    PSn = PS_names()
    SSn = np.array(["Red", "Black", "Silver"])
    SPTn = SPT_names()

    # Initialize lists
    ys = []  # Shift percentage
    F1_plt = []  # Primary Side Force
    F2_plt = []  # Secondary Side Force
    F2_err_plt = []
    Fc1_plt = []  # Primary Centrifugal Force
    Fc2_plt = []  # Secondary Centrifugal Force
    Rs1_plt = []  # Primary Side to Radial Force
    Rs2_plt = []  # Secondary Side to Radial Force
    R1_plt = []  # Total Primary Radial Force
    R2_plt = []  # Total Secondary Radial Force
    T1_plt = []
    T2_plt = []
    Terr_plt = []
    Tmax1_plt = []  # Required Primary Force
    Tmax2_plt = []  # Required Primary Force

    veh_speed = []
    engine_rpms = []

    T_err_history = []
    F2_err_history = []

    slip = 0

    # Flyweights
    mfw = FWs[q, 1]  # Flyweight Mass
    MFW = 3 * mfw / 1000  # 3 flyweights, kg

    # Primary spring
    k1 = PSs[w, 2]
    Ls = PSs[w, 1]
    f0 = k1 * (Ls - (P0 - shim))  # Spring Force at no primary movement
    feng = k1 * (Ls - (Peng - shim))  # Spring Force at belt engagement
    fs1x = k1 * (Ls - (Peng - shim))  # Spring Force at zero shift
    Fs1x = k1 * (Ls - (Pmax - shim))  # Spring Force at full shift

    # Secondary Ramp Angle & Rotation
    """ This will have to change for composite ramps"""
    ramp_angle = Ramps[e] / 180 * pi  # Ramp Angle in radians
    full_rotation = (28.575 / tan(ramp_angle)) / (pi * 90) * 2 * pi

    # Secondary Pretension
    pretension = SPTs[0, r] / 180 * pi

    # Secondary spring forces
    fs2x = SSs[t, 0]  # Spring Compression Force at no shift
    Fs2x = SSs[t, 1]  # Spring Compression Force at full shift
    f2t = SSs[t, 2] * pretension  # Spring Torsion Moments at pretension
    F2t = SSs[t, 2] * (pretension + full_rotation)  # Spring Torsion Moments at full shift

    y = 0 # Shift Percentage
    while y <= 1:
        rsint = FWs[q, (int(y * 4) + 3)]  # Flyweight geometric parameter

        cr = 3.83 - y * (3.83 - 0.76)  # Current ratio

        # Pulley diameters and wrap angles
        D1, D2 = pulley_diameters(ratio=cr)
        r1 = D1 / 2
        r2 = D2 / 2
        phi_1, phi_2 = wrap_angles(D1, D2)

        T2 = T1 * cr * eff * (0.1 if no_T2 else 1)

        # Interpolating Spring Forces
        Fs2t = f2t + y * (F2t - f2t)  # Secondary Torsion
        Fs2 = fs2x + y * (Fs2x - fs2x)  # Secondary Compression
        Fs1 = fs1x + y * (Fs1x - fs1x)  # Primary Compression

        if y == 0:
            # Idle Speeds
            rpm_idle = ((f0 / (MFW * rsint))**0.5) * 60 / (2 * pi)
            engine_rpms.append(rpm_idle)
            veh_speed.append(0)

            # Max Idle Speed
            rpm_idle_max = (60 / (2 * pi)) * (feng / (MFW * rsint))**0.5
            engine_rpms.append(rpm_idle_max)
            veh_speed.append(0)

            # Engagement Speed
            """ 
            Mulamba: Check and Fix!
            Why just add 900 rpm for engagement?! - It does match the data he got...

            TRY: Use dynamic friction coefficient instead of static friction coeff.
            TRY: From Aaen, not all torque from engine needs to be transmitted to get the vehicle moving
            ====> Calculate the takeoff torque or use Wheel Force Transducer data to estimate
            """

            w_eng = (((T_takeoff * tan(beta) / (r1 * cf_dyn)) + feng) / (MFW * rsint))**0.5
            rpm_eng = (60 / (2 * pi)) * w_eng
            engine_rpms.append(rpm_eng)
            veh_speed.append(0)

            # Clutching
            """Use the engagement rpm just solved for to guess clutching engine and evalute the engine torque at that point
            """
            T1_clu = engTorq(w_eng)

            w_clu = (((T1_clu * tan(beta) / (r1 * cf)) + feng) / ((MFW * rsint) - (m_prime * tan(beta) * phi_1 * r1**2)))**0.5
            rpm_clu = (60 / (2 * pi)) * w_clu
            vel_clu = rpm_clu / (cr * 11.3) / 60 * Wcirc * 3.6
            engine_rpms.append(rpm_clu)
            veh_speed.append(vel_clu)

        # Straight shift points:

        ## Iteration for convergence
        T_factor = 0.1 if no_T2 else 1 # For testbench data
        T1_guess = T1
        err = float('inf')
        tol = 1  # percent error bound
        max_iter = 50
        iter_count = 0

        ### Lists to store errors for this shift point
        T_err_iter = []
        F2_err_iter = []


        while err > tol and iter_count < max_iter:
            T2_guess = T1_guess * cr * eff * T_factor
            F2_guess = (0.5 * T2_guess + Fs2t) / (r_h * tan(ramp_angle)) + Fs2

            denom = MFW * rsint + m_prime * tan(beta) * (phi_1 * r1**2 - phi_2 * r2**2 / cr**2)
            w1 = ((F2_guess + Fs1) / denom)**0.5

            T1_new = engTorq(w1)
            T2_new = T1_new * cr * eff * T_factor
            F2_new = (0.5 * T2_new + Fs2t) / (r_h * tan(ramp_angle)) + Fs2

            T_err = percentError(T2_guess, T2_new)
            F2_err = percentError(F2_guess, F2_new)
            err = max(abs(T_err), abs(F2_err))

            # Store errors for this iteration
            T_err_iter.append(T_err)
            F2_err_iter.append(F2_err)

            T1_guess = T1_new
            iter_count += 1

        # print(f"shift: {y}, took {iter_count} iterations")

        # Append error histories for this shift point
        T_err_history.append(T_err_iter)
        F2_err_history.append(F2_err_iter)

        # Use converged values
        rpm = w1 * 60 / (2 * pi)
        engine_rpms.append(rpm)
        veh_speed.append(rpm / (cr * 11.3) / 60 * Wcirc * 3.6)

        # Overrun
        if y == 1:
            engine_rpms.append(ErpmMax)
            veh_speed.append(ErpmMax / (CVTH * 11.3) / 60 * Wcirc * 3.6)

        F1 = (MFW * rsint * (w1)**2 - Fs1)
        Fc1 = Fc(m_prime, w1, r1, phi_1)
        Rs1 = F1 / tan(beta)
        R1 = Fc1 + Rs1


        w2 = w1 / cr
        Fc2 = Fc(m_prime, w2, r2, phi_2)
        Rs2 = F2_new / tan(beta)
        R2 = Fc2 + Rs2

        # Use converged T1_new and T2_new
        T_err = percentError(T2_guess, T2_new)  # should be small now
        F2_err = percentError(F2_guess, F2_new)
        Terr_plt.append(T_err)
        F2_err_plt.append(F2_err)

        Tmax1 = maxT(cf, Rs1, Fc1, r1)
        Tmax2 = maxT(cf, Rs2, Fc2, r2)
        if Tmax1 <= T1 or Tmax2 <= T2:
            slip = 1

        F1_plt.append(F1)
        F2_plt.append(F2_new)
        Fc1_plt.append(Fc1)
        Fc2_plt.append(Fc2)
        Rs1_plt.append(Rs1)
        Rs2_plt.append(Rs2)
        R1_plt.append(R1)
        R2_plt.append(R2)
        T1_plt.append(T1)
        T2_plt.append(T2)
        Tmax1_plt.append(Tmax1)
        Tmax2_plt.append(Tmax2)
        ys.append(y)

        y += 0.25

    return {
        'veh_speed': veh_speed,
        'engine_rpms': engine_rpms,
        'slip': slip,
        'ys': ys,
        'F1_plt': F1_plt,
        'F2_plt': F2_plt,
        'Fc1_plt': Fc1_plt,
        'Fc2_plt': Fc2_plt,
        'Rs1_plt': Rs1_plt,
        'Rs2_plt': Rs2_plt,
        'R1_plt': R1_plt,
        'R2_plt': R2_plt,
        'T1_plt': T1_plt,
        'T2_plt': T2_plt,
        'Terr_plt': Terr_plt,
        'F2_err_plt': F2_err_plt,
        'Tmax1_plt': Tmax1_plt,
        'Tmax2_plt': Tmax2_plt,

        'Idle_rpm':rpm_idle_max,
        'Engage_rpm':rpm_eng,
        'Clutch_rpm':rpm_clu,
        'T_err_history': T_err_history,
        'F2_err_history': F2_err_history
    }


@app.function
def plot_simulation(result, q, w, e, r, t,shim, goal, Vsmin, Vsmax, ErpmMax, ErpmMin):
    """
    Generate plots for CVT simulation results.

    Parameters:
    result: Dictionary from cvt_simulation containing simulation data
    q, w, e, r, t: CVT parameters for labeling
    goal: Peak power RPM
    Vsmin, Vsmax: Min and max vehicle speeds
    ErpmMax, ErpmMin: Max and min engine RPMs

    Returns:
    dict with 'main_plot', 'rad_plot', 'err_plot'
    """
    veh_speed = result['veh_speed']
    engine_rpms = result['engine_rpms']
    slip = result['slip']
    ys = result['ys']
    Fc1_plt = result['Fc1_plt']
    Fc2_plt = result['Fc2_plt']
    Rs1_plt = result['Rs1_plt']
    Rs2_plt = result['Rs2_plt']
    R1_plt = result['R1_plt']
    R2_plt = result['R2_plt']
    Terr_plt = result['Terr_plt']
    F2_err_plt = result['F2_err_plt']

    # Main Plot (Engine RPM vs. Vehicle Speed)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[Vsmin, 0, Vsmax], y=[ErpmMax, 0, ErpmMax],
        mode='lines', line=dict(dash='dot', color='grey'),
        name='Low & High Ratios',
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=[goal, goal],
        mode='lines', line=dict(dash='dash', color='green'),
        name=f'Ideal shift ({goal}rpm)',
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=[ErpmMax, ErpmMax],
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=[ErpmMin, ErpmMin],
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
    ))

    lame = f"{[q, w, e, r, t]} - {shim:.0f}"
    if slip == 1:
        lame += " - SLIP"

    fig.add_trace(go.Scatter(
        x=veh_speed[0:2], y=engine_rpms[0:2],
        mode='lines+markers', marker=dict(symbol='star', size=6), line=dict(color='rgb(31, 119, 180)'),
        name='No Engagement',
        hoverinfo='x+y+name'
    ))
    fig.add_trace(go.Scatter(
        x=veh_speed[1:3], y=engine_rpms[1:3],
        mode='lines', line=dict(color='rgb(31, 119, 180)', dash='dot'),
        name='Total Slip Engagement (TBC)',
        hoverinfo='x+y+name'
    ))
    fig.add_trace(go.Scatter(
        x=veh_speed[2:5], y=engine_rpms[2:5],
        mode='lines+markers', marker=dict(symbol='circle', size=6), line=dict(color='rgb(255, 127, 14)', dash='dash'),
        name='Clutching & Low Ratio',
        hoverinfo='x+y+name'
    ))
    fig.add_trace(go.Scatter(
        x=veh_speed[4:-1], y=engine_rpms[4:-1],
        mode='lines+markers', marker=dict(symbol='circle', size=6), line=dict(color='blue'),
        name=lame+" (Straight Shift)",
        hoverinfo='x+y+name'
    ))
    fig.add_trace(go.Scatter(
        x=veh_speed[-2:], y=engine_rpms[-2:],
        mode='lines+markers', marker=dict(symbol='circle', size=6), line=dict(color='green'),
        name="Overrun",
        hoverinfo='x+y+name'
    ))


    fig.update_layout(
        template="plotly_white",
        title = lame,
        title_x=0.5,
        margin=dict(t=30, b=10),
        xaxis_title="Vehicle Speed in km/h",
        yaxis_title="Engine Speed in RPM",
        yaxis=dict(range=[1300, 4000]),
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=13)
        ),
        hovermode='closest',
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return {
        "main_plot": fig,
    }


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Other Functions
    Function used in the mathematical model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Radial Force Model
    Developedby Hannes (JL) Pretorius (2025) to be able to convert the side force model into radial forces to be able to include Centrifugal Forces from the belt. It is also more intuitive to understand how the shifting happens with this model.

    ### Cetrifugal Forces
    For a belt of wrap angle, $\phi$ and mass per meter of $m'$, the centripetal force, $F_c$ is given by
    $$F_c= m'\omega^2 r^2 \phi$$
    where $\omega$ is the angular velocity of the pulley and $r$ is the radius the belt wraps around.
    """)
    return


@app.function
def Fc(m,w,r,phi):
    return m * w**2 * r**2 * phi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Wrap Angles
    The first step is to solve for the wrap angles on the pulleys

    From Shigley Chapter 17 (Budynas and Nisbett, 2019) the wrap angles can be solved as follows, where C is the centre distance, and D and d is the large and small pulley diameters respectively:

    Small Pulley:
    $$ϕ_d=π-2 sin^{-1}\frac{D-d}{2C}$$
    Large Pulley:
    $$ϕ_d=π+2 sin^{-1}\frac{D-d}{2C}$$
    """)
    return


@app.function
def wrap_angles(D1,D2,C=0.2667):
    from numpy import pi, arcsin

    if D1<D2:
        d = D1
        D = D2
        phi_1 = pi - 2*arcsin((D-d)/(2*C))
        phi_2 = pi + 2*arcsin((D-d)/(2*C))

    elif D1>D2:
        d = D2
        D = D1
        phi_1 = pi - 2*arcsin((D-d)/(2*C))
        phi_2 = pi + 2*arcsin((D-d)/(2*C))

    else: phi_1 = phi_2 = pi

    return phi_1, phi_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diameters from Ratio (NB: Only for Algorithm)
    NB: Check what the actual belt length and centre distance is!

    Assuming no slip, the two diameters can be related through the CVT ratio, cr
    $$D_2=cr\cdot D_1$$
    The belt pitch length equation can then be rewritten as follows and solved as a quadratic equation for $D_1$.
    $$L=2C+\frac{\pi}{2}\cdot (D_1+D_2 )+(D_1-D_2 )^2/4C$$
    $$\Rightarrow L=2C+\frac{\pi}{2}\cdot (D_1+rD_1 )+(D_1-crD_1 )^2/4C$$
    $$\Rightarrow[(1-2cr+cr^2)/4C]\cdot D_1^2+[ \frac{\pi}{2}\cdot (1+cr)]\cdot D_1+[2C-L]=0$$
    """)
    return


@app.function
def pulley_diameters(ratio,C=0.266405,L=1.027):
    from numpy import pi
    import numpy.polynomial.polynomial as poly

    cr = ratio
    a = (1 - 2*cr + cr**2)/(4*C)
    b = (0.5*pi*(1+cr))
    c = (2*C-L)

    roots = poly.polyroots((c,b,a))
    # print(roots)

    Dp = roots[1]
    Ds = cr*Dp

    # Li = 2*C +0.5*pi*(Dp+Ds) + (Dp -Ds)**2 /(4*C)

    return Dp, Ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Belt Slipping / Max Transferable Torque
    ##Transmittable Torque

    The maximum torque that can be transferred by the sheaves before the belt slips can be determined from

    $$T_{max}=F_{t1}-F_{t2}⋅r=Δ F_{t,max}⋅r$$

    For this equation to be used, the belt tensions need to be determined independently of the transmitted torque. This can be done by relating the difference between the tensions, $\Delta F_t$ and the total frictional force around the wrap of the belt by

    $$ΔF_{t,max}=μ N$$

    where N is the total normal force the belt experiences from the pulley and can be defined by the radial force due to the side force, $R_s$ and the reduction of it by the centrifugal force, $F_c$ given resulting in

    $$\Delta F_{t,max} = \frac{\mu}{sin\beta} (R_s-F_c )=f' (R-F_c)$$

    That means that the max transmittable torque is

    $$T_{max}=f' (R_s-F_c )⋅r$$
    """)
    return


@app.function
def maxT (f,Rs,Fc,r):
    Tmax = f*(Rs-Fc)*r
    return Tmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kohler/Rehlko Engine Torque Curve

    Below is the power and torque curve for the rule-compliant Rehlko/Kohler engine with the Baja SAE restrictor plate installed. It has been converted into metric units from the original that was provided by Baja SAE (2025)

    It is then coded in a function to be used in the algorithm:

    $$ T_{engine} = -2\times 10^{-6} \cdot  rpm^2 + 0.0058\cdot rpm + 22.536 $$
    """)
    return


@app.function
def engTorq(w):
    import numpy as np
    rpm = w * 30/np.pi
    T_eng = -2*10**-6 * rpm**2 + 0.0058*rpm + 22.536
    return T_eng


@app.cell
def _(mo):
    mo.md(r"""
    ## Misc. Usefull Functions
    """)
    return


@app.function
def percentError(old, new):
    return 100*abs(old-new)/new


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Old Model for Comparison
    """)
    return


@app.cell
def _(ErpmMax, ErpmMin, GR, Vsmax, Vsmin, Wdia, e, mo, q, r, t, w):

    import plotly.io as pio
    import plotly.colors

    colors = plotly.colors.qualitative.Plotly

    old_result = old_cvt_model(q, w, e, r, t)
    old_x, old_y = old_result

    old_result_fix = old_cvt_model_fix(q, w, e, r, t)
    old_x_fix, old_y_fix = old_result_fix

    result_new = cvt_simulation(q, w, e, r, t)
    goal_x = result_new['veh_speed']
    goal_y = result_new['engine_rpms']


    # Create Plotly figure
    fig = go.Figure()

    v2s = (60*GR/(0.5*Wdia*3.6*2*np.pi))

    old_shift = dict(
        x=np.array(old_x)*v2s ,
        y=old_y,
        mode='lines+markers',
        line=dict(color=colors[7]),
        name=f"Old Model [{q,w,e,r,t}]",
    )

    old_shift_fix = dict(
        x=np.array(old_x_fix) *v2s,
        y=old_y_fix,
        mode='lines+markers',
        line=dict(dash='dash',color=colors[6]),
        name=f"Old Model Diameter Fix [{q,w,e,r,t}]",
    )

    goal_shift = dict(
            x=np.array(goal_x) *v2s,
            y=goal_y,
            mode='lines+markers',
            line=dict(color=colors[5],dash="dot"),
            name=f"New Model [{q,w,e,r,t}]",
        )


    # Plotting Lines to compare against
    ## Data for comparison lines
    Low = np.array([Vsmin,0, Vsmax])*v2s

    Govspeed = np.array([ErpmMax, ErpmMax])
    Idle = np.array([ErpmMin, ErpmMin])
    RPM = np.array([0, 3800])
    goal = 3500

    # Plotting Lines to compare against

    # Data for comparison lines
    PeakP = [goal, goal]
    Govspeed = [ErpmMax, ErpmMax]
    Idle = [ErpmMin, ErpmMin] 

    fig.add_trace(go.Scatter(
        x=Low, y=[ErpmMax, 0, ErpmMax],
        mode='lines', line=dict(dash='dot', color='grey'),
        name='Low & High Ratios',
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=PeakP,
        mode='lines', line=dict(dash='dash', color='green'),
        name=f'Ideal shift ({goal}rpm)',
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=Govspeed,
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
    ))
    fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=Idle,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
    ))
    # Add scatter trace
    fig.add_trace(go.Scatter(**old_shift))
    # fig.add_trace(go.Scatter(**old_shift_fix))
    fig.add_trace(go.Scatter(**goal_shift))


    # Update layout
    fig.update_layout(
        template = "plotly_white",
        dragmode='zoom',
        xaxis_title="Secondary Speed [RPM]",
        yaxis_title="Engine/Primary Speed [RPM]",
        yaxis=dict(range=[1000, 4500]),
        title=dict(text="Old vs Fix vs New", x=0.5, xanchor='center'),
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=14)
        ),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    pio.write_image(fig, f'Model Comparison [{q,w,e,r,t}].svg', scale=1, width=1080, height=540)
    mo.ui.plotly(fig)
    return


@app.function
def old_cvt_model(q=4, w=2, e=7, r=1, t=1):
    # Initialize part parameters
    FWs = FW()
    PSs = PS()
    SSs = SS()
    Ramps = Helix()
    SPTs = SPT()

    # Pulley Diameters
    dp = 0.063 # initial diameter of primary at 3.83 x2 = x1 - 0.0047
    Dp = 0.184 #0.173 # final diameter of primary  at 0.76

    ds = 0.24129 #0.2295 # initial diameter of secondary at 3.83 x2 = x1 - 0.0047
    Ds = 0.13984 #0.1315 # final diameter of secondary at 0.76

    # Ratio and Speed Definitions
    ## Gear Ratio and Torque Ratio Definitions
    GR = 11.3 #Gear Reduction Ratio
    CVTH = 0.76 # CVT High Ratio
    CVTL = 3.8 # CVT Low Ratio
    TRL = GR*CVTL # Torque Ratio Low
    TRH = GR*CVTH # Torque Ratio High

    eff = 1  # Efficiency of torque transfer
    floss = 1
    rr = 0.045  # Torque Feedback Ramp Radius
    P0 = 63.5  # static spring displacement
    Peng = 59  # Spring displacement when belt starts to engage
    Pmax = 30.226  # Spring displacement when clutch is fully shifted
    shim = 0  # Spring shim displacement
    cf = 0.55 # coeff friction between rubber and aliminium
    goal = 3600  # Peak Power RPM
    T1 = round(-2E-6 * goal**2 + 0.0058 * goal + 22.536, 1)  # Peak Torque trendline


    ## Engine RPM and Speed Definitions
    ErpmMax = 3800 # Max engine RPM
    ErpmMin = 1800  # Min engine RPM

    ## Vehicle Speed and Wheel Diameter Definitions
    Wdia = 23*0.0254 #wheel diameter in m
    Wcirc = 1.74 #Wdia * np.pi ### needs calibration
    Vsmax = 3800/TRH/60*Wcirc*3.6 # Max vehicle speed in km/h
    Vsmin = 3800/TRL/60*Wcirc*3.6 # Min vehicle speed in km/h

    '''
    # Data Points for Comparison Lines
    Low = [0,Vsmin]
    High = [0,Vsmax]
    Med = [0,((Vsmax+Vsmin)/2)]
    PeakP = [goal,goal]
    Govspeed = [3800,3800]
    RPM = [0,3800]

    # Close Previous Plots
    plt.close()

    # Plotting Engine RPM vs Vehicle Speed
    plt.figure(2,figsize=(10, 6))
    ## Plotting Lines to compare against
    plt.plot(Low,RPM,"--" ,label = "Low gear ratio")
    plt.plot(High,RPM,"--", label = "High gear ratio")
    plt.plot(High,PeakP,"--", label = "Ideal shift")
    plt.plot(High,Govspeed,"--", label = "Governer")
    plt.ylabel("RPM")
    plt.xlabel("Vehicle speed (km/h)")
    plt.title("Engine RPM along Vehicle Speed")
    plt.subplots_adjust(bottom=0.2)
    if inspect_optimised_results == True: plt.subplots_adjust(bottom=0.2), plt.suptitle(name + "\nPlot slip setups: "+str(plot_slip_setups), fontsize=10, y=0.1)
    plt.ylim(0,4000)
    #plt.yticks(np.arange(0,4050,50))
    '''

    # Initialize variables for plotting
    os = [] # Shift percentage
    f1 = [] # Possible Primary Side Force
    f2 = [] # Possible Secondary Side Force
    F_req1=[] # Required Primary Side Force
    F_req2=[] # Required Secondary Side Force
    F_req3=[] # Average Required  Side Force

    inspect = np.array([q,w,e,r,t])

    # Extracting the data form the CVT Parts database
    ## Flyweight Calculations
    mfw = FWs[int(inspect[0]),1] # Mass of flyweight

    ## Primary Spring Calculations
    f1eng = PSs[int(inspect[1]),2]*(PSs[int(inspect[1]),1]-(P0-shim)) # Spring Force at belt engagement
    fs1x = PSs[int(inspect[1]),2]*(PSs[int(inspect[1]),1]-(Peng-shim)) # Spring Force at zero shift
    Fs1x = PSs[int(inspect[1]),2]*(PSs[int(inspect[1]),1]-(Pmax-shim)) # Spring Force at full shift

    ## Ramp Calculations
    ra = Ramps[int(inspect[2])]/180*pi # Ramp Angle in radians
    rot = (28.575/tan(ra))/(pi*90)*2*pi # Ramp rotation in radians

    pre = SPTs[0,int(inspect[3])]/180*pi # Secondary Pretension in radians

    ## Secondary Spring Calculations
    fs2x = SSs[int(inspect[4]),0]
    Fs2x = SSs[int(inspect[4]),1]
    fs2t = SSs[int(inspect[4]),2]*pre
    Fs2t = SSs[int(inspect[4]),2]*(pre+rot)        

    o = 0 #Starting shift percentage

    rpms = []
    veh_speed = [] # Vehicle Speed in km/h (Was named "shifts" in the original code)

    while o<=1:

        rel1=1
        rel2=1

        # Flywieght Geometry Factor Evaluated at shift percentage
        rsint = FWs[int(inspect[0]),int(o*4+3)] #

        #Current Pulley Ratio   
        cr = 3.83 - o * (3.83-0.76)

        # Pulley  Diameters
        primD = dp + (Dp-dp)*o
        secD = ds + (Ds-ds)*o

        # Secondary Torque from Primary/Engine Torque
        T2=T1*cr*eff

        # Calculate the required side forces for the primary and secondary
        req1= ((T1/primD)/cf)*2 # 2*Torque/Diameter / friction coeff
        req2 = ((T2/secD)/cf)*2
        req3=(req1+req2)/2 # Average
        F_req1.append(req1)
        F_req2.append(req2)
        F_req3.append(req3)

        # Calculate side forces for the primary and secondary due to the springs
        F2t = fs2t + o * (Fs2t-fs2t) # Secondary Torsion
        F2x = fs2x + o * (Fs2x-fs2x) # Secondary Compression
        F1x = fs1x + o * (Fs1x-fs1x) # Primary Compression

        # Calculate the total side force on the secondary side
        F2 = ((0.5*T2+F2t)/(rr*tan(ra))*floss+F2x)*rel2

        # Imperically Determined Correction Factor (See Mills pg 15-16)
        if F2<req3:
            slip=1
            if o==0:
                rel1=0.94
            if o==0.25:
                rel1=0.9
            if o==0.5:
                rel1=0.9

        # Calculate the total side force on the primary side
        F1 = (mfw*3/1000*rsint*(goal/60*2*pi)**2-F1x)*rel1


        ## Record the forces for plotting
        f1.append(F1)
        f2.append(F2)
        os.append(o)



        ## Idle RPM   
        if o==0:
            rpm = ((f1eng/(mfw*3/1000*rsint))**0.5)*60/(2*pi)
            rpms.append(rpm)
            veh_speed.append(0)

        rpm=((((F2+F1x*rel1)/(mfw*3/1000*rsint*rel1))**0.5)*60/(2*pi))
        rpms.append(rpm)
        veh_speed.append(rpm/(cr*11.3)/60*Wcirc*3.6)


        # increase shift percentage by 25%
        o=o+0.25

    return veh_speed, rpms


@app.function
def old_cvt_model_fix(q=4, w=2, e=7, r=1, t=1):
    # Initialize part parameters
    FWs = FW()
    PSs = PS()
    SSs = SS()
    Ramps = Helix()
    SPTs = SPT()

    # Pulley Diameters
    dp = 0.063 # initial diameter of primary at 3.83 x2 = x1 - 0.0047
    Dp = 0.184 #0.173 # final diameter of primary  at 0.76

    ds = 0.24129 #0.2295 # initial diameter of secondary at 3.83 x2 = x1 - 0.0047
    Ds = 0.13984 #0.1315 # final diameter of secondary at 0.76

    # Ratio and Speed Definitions
    ## Gear Ratio and Torque Ratio Definitions
    GR = 11.3 #Gear Reduction Ratio
    CVTH = 0.76 # CVT High Ratio
    CVTL = 3.8 # CVT Low Ratio
    TRL = GR*CVTL # Torque Ratio Low
    TRH = GR*CVTH # Torque Ratio High

    eff = 1  # Efficiency of torque transfer
    floss = 1
    rr = 0.045  # Torque Feedback Ramp Radius
    P0 = 63.5  # static spring displacement
    Peng = 59  # Spring displacement when belt starts to engage
    Pmax = 30.226  # Spring displacement when clutch is fully shifted
    shim = 0  # Spring shim displacement
    cf = 0.55 # coeff friction between rubber and aliminium
    goal = 3600  # Peak Power RPM
    T1 = round(-2E-6 * goal**2 + 0.0058 * goal + 22.536, 1)  # Peak Torque trendline


    ## Engine RPM and Speed Definitions
    ErpmMax = 3800 # Max engine RPM
    ErpmMin = 1800  # Min engine RPM

    ## Vehicle Speed and Wheel Diameter Definitions
    Wdia = 23*0.0254 #wheel diameter in m
    Wcirc = 1.74 #Wdia * np.pi ### needs calibration
    Vsmax = 3800/TRH/60*Wcirc*3.6 # Max vehicle speed in km/h
    Vsmin = 3800/TRL/60*Wcirc*3.6 # Min vehicle speed in km/h

    '''
    # Data Points for Comparison Lines
    Low = [0,Vsmin]
    High = [0,Vsmax]
    Med = [0,((Vsmax+Vsmin)/2)]
    PeakP = [goal,goal]
    Govspeed = [3800,3800]
    RPM = [0,3800]

    # Close Previous Plots
    plt.close()

    # Plotting Engine RPM vs Vehicle Speed
    plt.figure(2,figsize=(10, 6))
    ## Plotting Lines to compare against
    plt.plot(Low,RPM,"--" ,label = "Low gear ratio")
    plt.plot(High,RPM,"--", label = "High gear ratio")
    plt.plot(High,PeakP,"--", label = "Ideal shift")
    plt.plot(High,Govspeed,"--", label = "Governer")
    plt.ylabel("RPM")
    plt.xlabel("Vehicle speed (km/h)")
    plt.title("Engine RPM along Vehicle Speed")
    plt.subplots_adjust(bottom=0.2)
    if inspect_optimised_results == True: plt.subplots_adjust(bottom=0.2), plt.suptitle(name + "\nPlot slip setups: "+str(plot_slip_setups), fontsize=10, y=0.1)
    plt.ylim(0,4000)
    #plt.yticks(np.arange(0,4050,50))
    '''

    # Initialize variables for plotting
    os = [] # Shift percentage
    f1 = [] # Possible Primary Side Force
    f2 = [] # Possible Secondary Side Force
    F_req1=[] # Required Primary Side Force
    F_req2=[] # Required Secondary Side Force
    F_req3=[] # Average Required  Side Force

    inspect = np.array([q,w,e,r,t])

    # Extracting the data form the CVT Parts database
    ## Flyweight Calculations
    mfw = FWs[int(inspect[0]),1] # Mass of flyweight

    ## Primary Spring Calculations
    f1eng = PSs[int(inspect[1]),2]*(PSs[int(inspect[1]),1]-(P0-shim)) # Spring Force at belt engagement
    fs1x = PSs[int(inspect[1]),2]*(PSs[int(inspect[1]),1]-(Peng-shim)) # Spring Force at zero shift
    Fs1x = PSs[int(inspect[1]),2]*(PSs[int(inspect[1]),1]-(Pmax-shim)) # Spring Force at full shift

    ## Ramp Calculations
    ra = Ramps[int(inspect[2])]/180*pi # Ramp Angle in radians
    rot = (28.575/tan(ra))/(pi*90)*2*pi # Ramp rotation in radians

    pre = SPTs[0,int(inspect[3])]/180*pi # Secondary Pretension in radians

    ## Secondary Spring Calculations
    fs2x = SSs[int(inspect[4]),0]
    Fs2x = SSs[int(inspect[4]),1]
    fs2t = SSs[int(inspect[4]),2]*pre
    Fs2t = SSs[int(inspect[4]),2]*(pre+rot)        

    o = 0 #Starting shift percentage

    rpms = []
    veh_speed = [] # Vehicle Speed in km/h (Was named "shifts" in the original code)

    while o<=1:

        rel1=1
        rel2=1

        # Flywieght Geometry Factor Evaluated at shift percentage
        rsint = FWs[int(inspect[0]),int(o*4+3)] #

        #Current Pulley Ratio   
        cr = 3.83 - o * (3.83-0.76)

        # Pulley  Diameters
         # Pulley diameters and wrap angles
        primD, secD = pulley_diameters(ratio=cr)

        # Secondary Torque from Primary/Engine Torque
        T2=T1*cr*eff

        # Calculate the required side forces for the primary and secondary
        req1= ((T1/primD)/cf)*2 # 2*Torque/Diameter / friction coeff
        req2 = ((T2/secD)/cf)*2
        req3=(req1+req2)/2 # Average
        F_req1.append(req1)
        F_req2.append(req2)
        F_req3.append(req3)

        # Calculate side forces for the primary and secondary due to the springs
        F2t = fs2t + o * (Fs2t-fs2t) # Secondary Torsion
        F2x = fs2x + o * (Fs2x-fs2x) # Secondary Compression
        F1x = fs1x + o * (Fs1x-fs1x) # Primary Compression

        # Calculate the total side force on the secondary side
        F2 = ((0.5*T2+F2t)/(rr*tan(ra))*floss+F2x)

        # Calculate the total side force on the primary side
        F1 = (mfw*3/1000*rsint*(goal/60*2*pi)**2-F1x)


        ## Record the forces for plotting
        f1.append(F1)
        f2.append(F2)
        os.append(o)



        ## Idle RPM   
        if o==0:
            rpm = ((f1eng/(mfw*3/1000*rsint))**0.5)*60/(2*pi)
            rpms.append(rpm)
            veh_speed.append(0)

        rpm=((((F2+F1x)/(mfw*3/1000*rsint))**0.5)*60/(2*pi))
        rpms.append(rpm)
        veh_speed.append(rpm/(cr*11.3)/60*Wcirc*3.6)


        # Calculate the total side force on the primary side
        o=o+0.25

    return veh_speed, rpms


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
