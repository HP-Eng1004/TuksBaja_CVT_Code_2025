import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo

    import numpy as np
    from matplotlib import pyplot as plt
    from numpy import pi as pi
    from numpy import tan as tan

    # Import from external modules
    from CVT_Parts_2025 import FW, PS, SS, Helix, SPT, FW_names, PS_names, SPT_names

    from CVT_Model_2025 import pulley_diameters, wrap_angles, Fc,  maxT, percentError, engTorq


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Investigate a setup (For Briggs & Straton Engine)
    #### Use the shorthand developed by Mills (2019)
    """
    )
    return


@app.cell(hide_code=True)
def _(T, es, qs, rs, ss, ts, ws):
    mo.vstack([
        mo.hstack([T,T.value,"%",f"({round(T.value*0.17,1)} [Nm]"], justify="start"),
        mo.hstack([qs,qs.value,";",ws,ws.value,";",ss,ss.value,"[mm]"], justify="start"),
        mo.hstack([es,es.value,";",rs,rs.value,";",ts,ts.value], justify="start")])

    return


@app.cell(hide_code=True)
def _(es, qs, rs, ss, ts, ws):
    # Get values from sliders below
    q = qs.value # Flyweights
    w = ws.value # Primary spring
    shim = ss.value

    e = es.value # Ramp angle
    r = rs.value # Secondary Pretension
    t = ts.value # Secondary Spring
    return e, q, r, shim, t, w


@app.cell
def _(T, e, q, r, shim, t, w):
    result = cvt_simulation_briggs(q, w, e, r, t, shim =shim, plot=True, no_T2 = False, goal = 3600,TorqPerc=T.value)
    print(result['veh_speed'])
    print(result['engine_rpms'])
    return


@app.function
def cvt_simulation_briggs(q=4, w=2, e=7, r=1, t=1, goal= 3400, shim=0, plot=False, no_T2 = False, TorqPerc = 100):
    """
    Simulate CVT based on parameters.

    Parameters:
    q: Flyweights index (0-8)
    w: Primary spring index (0-6)
    e: Ramp angle index (0-9)
    r: Secondary pretension index (0-4)
    t: Secondary spring index (0-2)
    plot: If True, generate and show plots (requires matplotlib and plotly)

    Returns:
    dict with 'veh_speed', 'engine_rpms', and other computed values.
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

    eff = 1  # Efficiency of torque transfer
    floss = 1
    r_h = 0.045  # Torque Feedback Ramp Radius
    P0 = 63.5  # static spring displacement
    Peng = 59  # Spring displacement when belt starts to engage
    Pmax = 30.226  # Spring displacement when clutch is fully shifted
    # shim = 0  # Spring shim displacement

    goal = goal  # Peak Power RPM
    T1 = 17*TorqPerc/100

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

    veh_speed_b = []
    engine_rpms_b = []

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

    y = 0  # Shift Percentage
    while y <= 1:  # Shift percentage
        rsint = FWs[q, (int(y * 4) + 3)]  # Flyweight geometric parameter
        cr = 3.83 - y * (3.83 - 0.76)  # Current ratio

        # Pulley diameters and wrap angles
        D1, D2 = pulley_diameters(ratio=cr)
        r1 = D1 / 2
        r2 = D2 / 2
        phi_1, phi_2 = wrap_angles(D1, D2)

        T2 = T1 * cr * eff

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
            cf_dyn = 0.25 * cf
            T_takeoff = 7.6 # Rough values from WFT
            rpm_eng = (60 / (2 * pi)) * (((T_takeoff*tan(beta) / (r1 * cf_dyn)) + feng) / (MFW * rsint))**0.5
            engine_rpms.append(rpm_eng)
            veh_speed.append(0)

            # Clutching
            """
            Dateer op, en sit in verslag
            """
            w_clu = (((T1*tan(beta) / (r1 * cf)) + feng) / ((MFW * rsint) - (m_prime *tan(beta)*phi_1*r1**2)) )**0.5
            rpm_clu = (60 / (2 * pi)) * w_clu
            vel_clu = rpm_clu / (cr * 11.3) / 60 * Wcirc * 3.6
            engine_rpms.append(rpm_clu)
            veh_speed.append(vel_clu)

        # Straight Shifting
        F2 = (0.5 * T2 + Fs2t) / (r_h * tan(ramp_angle)) + Fs2

        # Primary Speed
        w1 = ((F2 + Fs1) / (MFW * rsint + m_prime * tan(beta) * (phi_1 * r1**2 - phi_2 * r2**2 / cr**2)))**0.5
        rpm = w1 * 60 / (2 * pi)


        engine_rpms.append(rpm)
        veh_speed.append(rpm / (cr * 11.3) / 60 * Wcirc * 3.6)

        # Overrun
        if y==1:
            engine_rpms.append(ErpmMax)
            veh_speed.append(ErpmMax / (CVTH * 11.3) / 60 * Wcirc * 3.6)

        # Other forces
        F1 = (MFW * rsint * (w1)**2 - Fs1)
        Fc1 = Fc(m_prime, w1, r1, phi_1)
        Rs1 = F1 / tan(beta)
        R1 = Fc1 + Rs1

        T1_new = engTorq(w1)
        T2_new = T1_new * cr * eff
        F2_new = (0.5 * T2_new + Fs2t) / (r_h * tan(ramp_angle)) + Fs2

        w2 = w1 / cr
        Fc2 = Fc(m_prime, w2, r2, phi_2)
        Rs2 = F2_new / tan(beta)
        R2 = Fc2 + Rs2

        T_err = percentError(T2, T2_new)
        F2_err = percentError(F2, F2_new)
        Terr_plt.append(T_err)
        F2_err_plt.append(F2_err)

        Tmax1 = maxT(cf, Rs1, Fc1, r1)
        Tmax2 = maxT(cf, Rs2, Fc2, r2)
        if Tmax1 <= T1 or Tmax2 <= T2:
            slip = 1

        # Append to lists
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

    # Optional plotting
    if plot:
        import plotly.graph_objs as go

        # Data for comparison lines
        PeakP = [goal, goal]
        Govspeed = [ErpmMax, ErpmMax]
        Idle = [ErpmMin, ErpmMin] 

        fig = go.Figure()
        # Plotting Lines to compare against
        fig.add_trace(go.Scatter(
            x=[Vsmin,0,Vsmax], y=[ErpmMax, 0, ErpmMax],
            mode='lines', line=dict(dash='dot', color='grey'),
            name='Low & High Ratios',
        ))
        fig.add_trace(go.Scatter(
            x=[0, Vsmax], y=PeakP,
            mode='lines', line=dict(dash='dash', color='green'),
            name=f'Ideal shift ({goal}rpm)',
        ))
        fig.add_trace(go.Scatter(
            x=[0, Vsmax], y=Govspeed,
            mode='lines', line=dict(dash='dash', color='red'),
            name='Governor',
        ))
        fig.add_trace(go.Scatter(
            x=[0, Vsmax], y=Idle,
            mode='lines', line=dict(dash='dash', color='grey'),
            name='Idle',
        ))

        # Label for the last trace
        lame = str([q, w, e, r, t]) + '-'+ str(shim)
        if slip == 1:
            lame = str([q, w, e, r, t]) + " - SLIP"

        # Plotting vehicle speed vs. engine RPMs
        fig.add_trace(go.Scatter(
            x=veh_speed[0:2], y=engine_rpms[0:2],
            mode='lines+markers', marker=dict(symbol='star', size=6), line=dict(color='rgb(31, 119, 180)'),  # C0 equivalent
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
            mode='lines+markers', marker=dict(symbol='circle', size=6), line=dict(color='rgb(255, 127, 14)', dash='dash'),  # C1 equivalent
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

        # Update layout
        fig.update_layout(
            template = "plotly_white",
            title=dict(text="Briggs Model (Const. Torque)", x=0.5, xanchor='center'),
            xaxis_title="Vehicle Speed in km/h",
            yaxis_title="Engine Speed in RPM",
            yaxis=dict(range=[1200, 4000]),
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=-0.4,
                xanchor="center",
                x=0.5,
                orientation="h",
                traceorder="normal",
                itemsizing="constant",
                font=dict(size=10)
            ),
            # margin=dict(b=150),  # Adjust bottom margin for legend
            hovermode='closest',
            width=800,
            height=600
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        # Radial Force plot
        fig_rad, ax_rad = plt.subplots(figsize=(8, 6))
        ax_rad.set_title("Radial Force along Shift %")
        ax_rad.set_xlabel("Shift Percentage")
        ax_rad.set_ylabel("Radial Force [N]")
        ax_rad.plot(ys, Fc1_plt, '--', color="C0", label="Primary Centrifugal Force")
        ax_rad.plot(ys, Fc2_plt, '--', color="C1", label="Secondary Centrifugal Force")
        ax_rad.plot(ys, Rs1_plt, '-.', color="C0", label="Primary Side to Radial Force")
        ax_rad.plot(ys, Rs2_plt, '-.', color="C1", label="Secondary Side to Radial Force")
        ax_rad.plot(ys, R1_plt, '-', color="C0", label="Primary Total Radial Force")
        ax_rad.plot(ys, R2_plt, '-', color="C1", label="Secondary Total Radial Force")
        ax_rad.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
        plt.subplots_adjust(bottom=0.4)
        ax_rad.grid()

        # Error plot
        fig_err, ax_err = plt.subplots(figsize=(8, 6))
        ax_err.set_title("Percentage error between the initial and updated Torque and Primary Side Force")
        ax_err.set_xlabel("Shift Percentage")
        ax_err.set_ylabel("% Difference Error")
        ax_err.plot(ys, F2_err_plt, label="Secondary Side Force Error")
        ax_err.plot(ys, Terr_plt, label="Torque Error")
        ax_err.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=8)
        plt.subplots_adjust(bottom=0.4)
        ax_err.grid()
        ax_err.set_ylim(0, 50)

        mo.output.append(
            mo.vstack([
                mo.hstack([
                    fig,
                    mo.md(
                       f"""      
                       **CVT Setup:** {q,w,e,r,t}-{shim}\n
                       **Flyweight [{q}]:** {FWn[q]}\n
                       **Spring 1 [{w}]:** {PSn[w]}\n
                       **Shim:** {shim}mm\n
                       **Ramp Angle [{e}]:** {e*2 +32}\n
                       **Pretension [{r}]:** {SPTn[r]}\n
                       **Spring 2 [{t}]:** {SSn[t]}\n
                       \n

                       **Useful Output Parameters:** \n
                        - Max Idle: {rpm_idle_max:.0f} rpm \n
                        - Engagement: {rpm_eng:.0f} rpm\n
                        - Low Gear: {rpm_clu:.0f} rpm\n
                        - Ave. Shift: {np.average(engine_rpms[4:]):.0f} rpm\n
                        - Max. Speed: {veh_speed[-1]:.0f}km/h
                        """),


                   ],justify="start")]))

    if plot == True:
        return {
            'veh_speed': veh_speed,
            'engine_rpms': engine_rpms,
            'slip': slip,
            'ys': ys,
            "rad_plot": fig_rad,
            "err_plot": fig_err,
        }
    else:
        return {
            'veh_speed': veh_speed,
            'engine_rpms': engine_rpms,
            'slip': slip,
            'ys': ys,        
        # Add other lists if needed, e.g., 'F1_plt': F1_plt, etc.
    }


@app.cell
def _(ErpmMax, Govspeed, Idle, T, Vsmax, Vsmin, go):
        #list = np.array([[7,3,8,0,0,0],[7,3,9,0,0,0],[7,3,8,2,0,0],[7,3,8,0,1,0],[7,3,9,0,2,0],]) # List for secondary component influence

    list = np.array([[7,3,8,0,0,0],[5, 2, 7, 1, 1,10],[7, 1, 6, 0, 0, 5]]) # List for model validation
    goal = 3600

    _fig = go.Figure()
    # Plotting Lines to compare against
    _fig.add_trace(go.Scatter(
        x=[Vsmin,0,Vsmax], y=[ErpmMax, 0, ErpmMax],
        mode='lines', line=dict(dash='dot', color='grey'),
        name='Low & High Ratios',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=[goal,goal],
        mode='lines', line=dict(dash='dash', color='green'),
        name=f'Ideal shift ({goal}rpm)',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=Govspeed,
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=Idle,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
    ))

    for setup in list:

        _q=setup[0]
        _w=setup[1]
        _e=setup[2]
        _r=setup[3]
        _t=setup[4]
        _shim=setup[5]
        _result = cvt_simulation_briggs(_q, _w, _e, _r, _t, shim =_shim, plot=False, no_T2 = False, goal = 3600,TorqPerc=T.value) 


        # Label for the last trace
        _lame = str([_q, _w, _e, _r, _t]) + '-'+ str(_shim)

        _fig.add_trace(go.Scatter(
                x=_result["veh_speed"], y=_result["engine_rpms"],
                mode='lines+markers',
                name=_lame,
                hoverinfo='x+y+name'
        ))

    _fig.update_layout(
                template = "plotly_white",
                title=dict(text="Briggs Model (Const. Torque)", x=0.5, xanchor='center'),
                xaxis_title="Vehicle Speed in km/h",
                yaxis_title="Engine Speed in RPM",
                yaxis=dict(range=[1500, 4000]),
                showlegend=True,
                legend=dict(
                    yanchor="bottom",
                    y=-0.4,
                    xanchor="center",
                    x=0.5,
                    orientation="h",
                    traceorder="normal",
                    itemsizing="constant",
                    font=dict(size=10)
                ),
                # margin=dict(b=150),  # Adjust bottom margin for legend
                hovermode='closest',
                width=800,
                height=600
            )
    _fig.update_xaxes(showgrid=True)
    _fig.update_yaxes(showgrid=True)

    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Old Model Comparisons""")
    return


@app.function
def old_cvt_model_briggs(q=4, w=2, e=7, r=1, t=1):
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
    T1 = 17


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
    T1 = 17  # Peak Torque trendline


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
def _(e, q, r, t, w):
    import plotly.graph_objs as go
    import plotly.io as pio
    import plotly.colors

    colors = plotly.colors.qualitative.Plotly

    old_result = old_cvt_model_briggs(q, w, e, r, t)
    old_x, old_y = old_result

    old_result_fix = old_cvt_model_fix(q, w, e, r, t)
    old_x_fix, old_y_fix = old_result_fix

    result_new = cvt_simulation_briggs(q, w, e, r, t, plot=False)
    goal_x = result_new['veh_speed']
    goal_y = result_new['engine_rpms']

    GR = 11.3  # Gear Reduction Ratio
    CVTH = 0.76  # CVT High Ratio
    CVTL = 3.8  # CVT Low Ratio
    TRL = GR * CVTL  # Torque Ratio Low
    TRH = GR * CVTH  # Torque Ratio High
    ErpmMax = 3700  # Max engine RPM
    ErpmMin = 1800  # Min engine RPM
    Wdia = 23 * 0.0254  # wheel diameter in m
    Wcirc = 1.74  # Wdia * np.pi ### needs calibration
    Vsmax = 3800 / TRH / 60 * Wcirc * 3.6  # Max vehicle speed in km/h
    Vsmin = 3800 / TRL / 60 * Wcirc * 3.6  # Min vehicle speed in km/h


    # Create Plotly figure
    _fig = go.Figure()

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
            line=dict(color=colors[5]),
            name=f"New Model [{q,w,e,r,t}]",
        )


    # Plotting Lines to compare against
    ## Data for comparison lines
    Low = np.array([0, Vsmin])*v2s
    High = np.array([0, Vsmax])*v2s
    Govspeed = np.array([ErpmMax, ErpmMax])
    Idle = np.array([ErpmMin, ErpmMin])
    RPM = np.array([0, 3800])


    _fig.add_trace(go.Scatter(
        x=[0, Vsmin*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Low gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[Vsmin*v2s, Vsmax*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='High gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=Govspeed,
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=Idle,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
    ))
    # Add scatter trace
    _fig.add_trace(go.Scatter(**old_shift))
    _fig.add_trace(go.Scatter(**old_shift_fix))
    _fig.add_trace(go.Scatter(**goal_shift))

    # Update layout
    _fig.update_layout(
        template = "plotly_white",
        dragmode='zoom',
        xaxis_title="Secondary Speed [RPM]",
        yaxis_title="Engine/Primary Speed [RPM]",
        yaxis=dict(range=[1000, 4500]),
        title=dict(text="Diameter Fix Comparison (Briggs)", x=0.5, xanchor='center'),
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
            font=dict(size=10)
        ),
    )
    _fig.update_xaxes(showgrid=True)
    _fig.update_yaxes(showgrid=True)

    mo.ui.plotly(_fig)
    return ErpmMax, Govspeed, Idle, Vsmax, Vsmin, go


@app.cell
def _():
    # Define the sliders for above
    qs = mo.ui.slider(
        start=0, 
        stop=8, 
        step=1, 
        value=7, 
        label="Flyweights"
    )

    ws = mo.ui.slider(
        start=0, 
        stop=6, 
        step=1, 
        value=3, 
        label="Primary Spring"
    )

    ss = mo.ui.slider(
        start=0, 
        stop=40, 
        step=1, 
        value=0, 
        label="Primary Spring Shim"
    )

    es = mo.ui.slider(
        start=0, 
        stop=9, 
        step=1, 
        value=7, 
        label="Ramp Angle"
    )

    rs = mo.ui.slider(
        start=0, 
        stop=4, 
        step=1, 
        value=0, 
        label="Secondary Pretension"
    )

    ts = mo.ui.slider(
        start=0, 
        stop=2, 
        step=1, 
        value=0, 
        label="Secondary Spring"
    )

    T = mo.ui.slider(
        start=50, 
        stop=150, 
        step=5, 
        value=100, 
        label="Torque Percentage (of 17Nm)"
    )
    return T, es, qs, rs, ss, ts, ws


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
