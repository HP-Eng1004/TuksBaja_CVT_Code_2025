import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objs as go
    from CVT_Model_2025 import cvt_simulation
    from CVT_Plotting_2025 import plot_torque_transfer, plot_force_balance, plot_slip_risk, plot_rpm_surface, plot_engagement_rpms, plot_exhaustive_search
    return (
        cvt_simulation,
        mo,
        np,
        plot_engagement_rpms,
        plot_exhaustive_search,
        plot_force_balance,
        plot_rpm_surface,
        plot_slip_risk,
        plot_torque_transfer,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Modified Mills' (2019) Algorithm
    ### With parts from Mulambu (2021)
    Initial Shifting Stages
    #### JL (Hannes) Pretorius - (2025)
    - Fixed the relation for the pulley diameters
    - Changed equation for $\omega_1$ (Engine Speed) to include centrifugal forces
    - Add an updated way to predict if the belt will slip
    """
    )
    return


@app.cell
def _(mo):
    goalRPM_slider = mo.ui.slider(start=3000, stop=3800, step=50, value=3600)
    tolerance_slider = mo.ui.slider(start=0, stop=400, step=10, value=50)
    idle_slider = mo.ui.slider(start=1200, stop=2500, step=50, value=1800)
    return goalRPM_slider, idle_slider, tolerance_slider


@app.cell
def _(goalRPM_slider, idle_slider, mo, tolerance_slider):
    mo.hstack([
        mo.md(f"**Shift Goal**: {goalRPM_slider.value} RPM {goalRPM_slider} "),
        mo.md(f"**Idle**: {idle_slider.value} RPM {idle_slider} "),
        mo.md(f"**Tolerance**: {tolerance_slider.value} RPM {tolerance_slider} ")
    ], justify="start")
    return


@app.cell
def _(
    cvt_simulation,
    goalRPM_slider,
    idle_slider,
    mo,
    np,
    plot_engagement_rpms,
    plot_exhaustive_search,
    plot_force_balance,
    plot_rpm_surface,
    plot_slip_risk,
    plot_torque_transfer,
    tolerance_slider,
):
    goal = goalRPM_slider.value
    tolerance = tolerance_slider.value
    engagement = idle_slider.value
    T1 = round(-2E-6 * goal**2 + 0.0058 * goal + 22.536, 1)

    # General CVT Parameters
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

    # Store results for all valid setups
    results = []
    setup = []
    sets = 0

    # Run cvt_simulation once per setup
    for q in range(9):
        for w in range(7):
            for e in range(10):
                for r in range(5):
                    for t in range(3):
                        result = cvt_simulation(q, w, e, r, t, goal=goal, shim=0, no_T2=False)
                        engine_rpms = np.array(result['engine_rpms'])
                        veh_speed = result['veh_speed']
                        slip = result['slip']
                        rpm_idle_max = engine_rpms[1]

                        if rpm_idle_max >= engagement and np.max(engine_rpms) <= ErpmMax:
                            straight_rpms = engine_rpms[4:9]
                            if np.all(np.abs(straight_rpms - goal) <= tolerance):
                                results.append({
                                    'params': [q, w, e, r, t],
                                    'veh_speed': veh_speed,
                                    'engine_rpms': engine_rpms,
                                    'slip': slip,
                                    'ys': result['ys'],
                                    'F1_plt': result['F1_plt'],
                                    'F2_plt': result['F2_plt'],
                                    'Fc1_plt': result['Fc1_plt'],
                                    'Fc2_plt': result['Fc2_plt'],
                                    'Rs1_plt': result['Rs1_plt'],
                                    'Rs2_plt': result['Rs2_plt'],
                                    'R1_plt': result['R1_plt'],
                                    'R2_plt': result['R2_plt'],
                                    'T1_plt': result['T1_plt'],
                                    'T2_plt': result['T2_plt'],
                                    'Terr_plt': result['Terr_plt'],
                                    'F2_err_plt': result['F2_err_plt'],
                                    'Tmax1_plt': result['Tmax1_plt'],
                                    'Tmax2_plt': result['Tmax2_plt']
                                })
                                sets += 1
                                setup.extend([q, w, e, r, t, slip])

    # Generate plots using stored results
    fig_exhaustive = plot_exhaustive_search(results, goal, Vsmin, Vsmax, ErpmMax, ErpmMin)
    fig_torque = None
    shim = 0
    if results:
        fig_torque = plot_torque_transfer(results[0],shim, *results[0]['params'])
    q_range, w_range = range(9), range(7)
    fig_force = plot_force_balance(results, q_range, w_range)
    fig_slip = plot_slip_risk(results, goal, tolerance)
    fig_rpm_surface = plot_rpm_surface(results, q_range, w_range, shift_idx=4)
    fig_engagement = plot_engagement_rpms(results)

    # Process results for text output
    results_array = np.zeros([sets, 6])
    i = 0
    nums = 0
    while i < sets:
        for ii in range(6):
            results_array[i, ii] = int(setup[nums])
            nums += 1
        i += 1

    r_slip = [r for r in results_array if r[5] == 1]
    r_no_slip = [r for r in results_array if r[5] == 0]

    print(f"Number of NONSLIP setups within {tolerance} RPM from {goal} RPM and only moves to engage after {engagement} RPM: {sets}")
    print([["mfw,ps,ra,pre,ss"]])
    print(f"Number of setups that do not slip: {len(r_no_slip)}")
    print("No-slip setups:\n", np.array2string(np.array(r_no_slip), separator=','))
    print(f"Number of setups that slip: {len(r_slip)}")
    print("Slip setups:\n", np.array2string(np.array(r_slip), separator=','))

    text = mo.vstack([
        mo.md(f"Number of NONSLIP setups within {tolerance} RPM from {goal} RPM and only moves to engage after {engagement} RPM: {sets}"),
        mo.md(f"Number of setups that do not slip: {len(r_no_slip)}"),
        mo.md(f"No-slip setups: \n {np.array2string(np.array(r_no_slip), separator=',')}"),
        mo.md(f"Number of setups that slip: {len(r_slip)}"),
        mo.md(f"Slip setups: \n {np.array2string(np.array(r_slip), separator=',')}"),
    ])

    # Display plots and text
    plots = [mo.ui.plotly(fig_exhaustive)]
    if fig_torque:
        plots.append(mo.ui.plotly(fig_torque))
    plots.extend([
        mo.ui.plotly(fig_force),
        mo.ui.plotly(fig_slip),
        mo.ui.plotly(fig_rpm_surface),
        mo.ui.plotly(fig_engagement)
    ])
    mo.vstack(plots + [text])
    return fig_engagement, fig_force, fig_rpm_surface, fig_slip


@app.cell
def _(fig_engagement, fig_force, fig_rpm_surface, fig_slip, mo):
    extra_plots=[
        mo.ui.plotly(fig_force),
        mo.ui.plotly(fig_slip),
        mo.ui.plotly(fig_rpm_surface),
        mo.ui.plotly(fig_engagement)
    ]

    mo.vstack(extra_plots)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
