import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Car X CVT Tests (Hannes 2025)""")
    return


@app.cell
def _():
    # Specify the folder where the readV3 is stored on your computer
    import sys 
    import os # Add the directory containing your module to the Python path 
    module_path = os.path.abspath('/Users/hanne/OneDrive - University of Pretoria/Documents/MyPyMod/VDG_DAQ') 
    if module_path not in sys.path: 
        sys.path.append(module_path)
    return


@app.cell
def _():
    # Import General modules
    import numpy as np
    from scipy import stats
    from matplotlib import pyplot as plt

    import readV3 # This file is provided by VDG to read the .bin files
    return np, readV3


@app.cell
def _(mo):
    # File picker for selecting .bin files
    initial_path="C:\\Users\\hanne\\OneDrive - University of Pretoria\\Desktop\\DAQ Files"

    file_picker = mo.ui.file_browser(
        label="Select DAQ Data File",
        filetypes=[".bin",".txt"],
        initial_path=initial_path,
        multiple = False
    )
    file_picker  # Display the file picker
    return file_picker, initial_path


@app.cell
def _(file_picker, mo):
    type = str(file_picker.path())[-4:]
    mo.md("**File Type: **"+type)
    return (type,)


@app.cell
def _(file_picker, initial_path, readV3, read_txt, type):
    # Read the selected file
    Data = None
    if file_picker.value:
        file_path = file_picker.path()

        path = str(file_picker.path())

        code = path[len(initial_path)+1:len(initial_path)+9]

        if type == ".bin":
            Data = readV3.readV3(file_path)
            # This is the read file for the normal old blue Baja DAQ
        elif type == ".txt":
            Data = read_txt(file_path)
            # See this read function defined at the end of the document for the txt file of the new small black DAQ
    else:
        code = "63800"
    return Data, code


@app.cell
def _():
    """
    Define Channel Info
    # This is where channel info, and calibration is defined
    Channel Info array storing:
    - Name
    - Calibration Equation (Linear Coefficients ax + b => [a,b])
    - Axis Labels
    """

    # Turk box calibration eqn

    ## Wheel
    _N = 6 # Number of reflectors
    _r = 11.2 # ratio between reflectors and where the speed is needed
    _n = 30*100 # Turk box rpm                                               # CHECK THE TURK BOXES

    m_wheel = _r* _n/_N /10 # gradient of the calibration line
    print("m_wheel = ",m_wheel)

    ## Primary
    _N = 3 # Number of reflectors 
    _r = 1 # ratio between reflectors and where the speedin needed
    _n = 150*100 # Turk box frequency in Hz

    m_prime = _r *_n/_N /10 # gradient of the calibration line
    print("m_prime = ",m_prime)


    '''["Accel X", (19.475, -46.626), ["X-Label", "Acceleration [m/s^2]"]], #4
        ["Accel Y", (19.5248, -46.3303), ["X-Label5", "Y-Label5"]], #5
        ["Accel Z", (1,0), ["X-Label6", "Y-Label6"]], #6'''

    Channel_Info = [ # Car X Tests 25 September onwards
        # ["Name", (a,b), ["X-Label", "Y-Label"]]
        ["Front Wheels", (m_prime,0), ["X-Label", "RPM"]], #0
        ["Laser Distance", (41.221, 32.88), ["X-Label", "Distance [mm]"]], #1
        ["Primary", (m_prime,75), ["X-Label", "RPM"]], #2
        ["Secondary", (m_wheel,89.5), ["X-Label3", "RPM"]], #3
        ["Accel X", (1, 0), ["X-Label", "Distance [mm]"]], #4
        ["Front Press.", (1, 0), ["X-Label5", "Y-Label5"]], #5
        ["Rear Press.", (1,0), ["X-Label6", "Y-Label6"]], #6
        ["Brake Force", (1,0), ["X-Label", "Y-Label"]], #7
        ["5th Wheel", (1 ,0),  ["X-Label","RPM"]], #8
        ["NaN", (1,0),  ["X-Label","RPM"]], #9
        ["NaN", (1,0),  ["X-Label","RPM?"]], #10
        ["NaN", (1,0), ["X-Label", "Y-Label"]], #11
        ["NaN", (1,0), ["X-Label", "Y-Label"]], #12
        ["NaN", (1,0), ["X-Label", "Y-Label"]], #13
        ["NaN", (1,0), ["X-Label", "Y-Label"]], #14
        ["NaN", (1,0), ["X-Label", "Y-Label"]], #15
    ]


    #end
    return (Channel_Info,)


@app.cell(hide_code=True)
def _(Channel_Info, Data, mo):
    channel_options = {}
    if Data is None:
        mo.output.append(mo.md(
                f"""
            /// details | **NB: Select data file above**
                type: warn
            Select the .bin file above. \n 
            You can change the default folder in the code
            ///
            """))
    else: 
        mo.md(f"""**Number of Channels: **{len(Data)}""")
        # Channel selector for plotting
        channel_options = {f"Ch.{i} ({Channel_Info[i][0]})" : i for i in range(len(Data))}
        # print(channel_options)
    return (channel_options,)


@app.cell
def _(code, mo):
    setup_input = mo.ui.text(
        value=code,
        label="Setup Number: ",
    )

    setup_input
    return (setup_input,)


@app.cell
def _(mo, shim):
    if shim: value=shim
    else: value = 0

    shim_input = mo.ui.number(
        value,
        step=5,
        label="Shim [mm]: ",)
    shim_input
    return


@app.cell
def _(setup_input):
    # Extract numerical values and run simulation
    setup_values = setup_input.value  # This calls the on_change function, returning the dictionary
    q = int(setup_values[0])
    w = int(setup_values[1])
    e = int(setup_values[2])
    r = int(setup_values[3])
    t = int(setup_values[4])
    shim = int(setup_values[6])*10 + int(setup_values[7])

    print(q,w,e,r,t,shim)
    return e, q, r, setup_values, shim, t, w


@app.cell
def _(T_takeoff, cf_dyn, e, q, r, shim, t, w):
    from CVT_Briggs_Algorithm_2025 import cvt_simulation_briggs, old_cvt_model_briggs


    result = cvt_simulation_briggs(q, w, e, r, t, shim = shim, cf_dyn = cf_dyn, T_takeoff = T_takeoff)
    old_result = old_cvt_model_briggs(q, w, e, r, t)

    goal_x = result['veh_speed']
    goal_y = result['engine_rpms']

    old_x, old_y = old_result
    return cvt_simulation_briggs, goal_x, goal_y, old_x, old_y


@app.cell(hide_code=True)
def _(channel_options, mo):
    theme = mo.ui.dropdown(
        options=["light", "dark"],
        value="light",
        label="Select Theme"
    )

    # Channel selector for plotting
    channel_selector_multi = mo.ui.multiselect(
        options=channel_options,
        label="Select Channels to Plot Together"
    )
    channel_selector_multi  # Display the selector

    # Button to toggle calibration
    calibrate_button = mo.ui.switch(label="Apply custom calibration equation")

    # Set sampling frequency
    Sample_freq = mo.ui.number(start=100, stop=2000, step = 100, value = 1000, label="Sample Rate [Hz]")

    # Slider for cutoff frequency (Hz)
    cutoff_slider = mo.ui.slider(
        start=5,
        stop=500,
        step=5,
        value=25,
        label="Cutoff Frequency [Hz]",
        show_value=True
    )

    # Button to toggle filter
    filter_button = mo.ui.switch(label="Low-Pass Filter")

    show_unfiltered_but = mo.ui.switch(label="Show Unfiltered Data", value=False)

    mo.vstack([
        mo.hstack([theme],justify="start"),
        mo.hstack([Sample_freq,filter_button,cutoff_slider, show_unfiltered_but],justify="start"),

    ])
    return (
        Sample_freq,
        cutoff_slider,
        filter_button,
        show_unfiltered_but,
        theme,
    )


@app.cell
def _(mo):
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
    return T_start, mu_d


@app.cell
def _(T_start, mu_d):
    cf_dyn = mu_d.value #Dynamic Coeff. of Friction
    T_takeoff = T_start.value # Takeoff Torque
    return T_takeoff, cf_dyn


@app.cell
def _(Data, Sample_freq, mo):
    from math import floor

    if Data is not None:
        # Double-sided slider for time range (seconds)
        sample_rate = Sample_freq.value
        Time = Data[0][2]
        Data_Duration = len(Time)
        round_end = floor(Data_Duration/(sample_rate))

        time_slider = mo.ui.range_slider(
            start=0,
            stop= round_end,
            step=5,
            value=[0, round_end],
            label="**Time Range [s]** (Use to focus on a section, or when the data size is too large)",
            show_value=True,
            full_width = True
        )

        mo.output.append(time_slider)
    return Data_Duration, sample_rate, time_slider


@app.cell(hide_code=True)
def _(Data, Data_Duration, mo, pio, sample_rate, theme, time_slider):
    if Data is not None:
        # Trim signals based on time range
        start_idx = int(time_slider.value[0] * sample_rate)
        end_idx = int(time_slider.value[1] * sample_rate)
        section = [start_idx,end_idx]
        print([time_slider.value[0],time_slider.value[1]],section,Data_Duration)
        mo.output.append(mo.md(f"**Start:** {time_slider.value[0]}s, **End:** {time_slider.value[1]}s"))

    # Set Plotly template based on theme
    if theme.value == "dark":
        pio.templates.default = "plotly_dark"
    else:
        pio.templates.default = "plotly_white"
    return (section,)


@app.cell
def _(
    Channel_Info,
    Data,
    cutoff_slider,
    filter_button,
    mo,
    plot_multi,
    section,
    show_unfiltered_but,
):
    # Plot selected channels
    if Data is not None:
        _fig = plot_multi(
            data=Data,
            section = section,
            Ch_info=Channel_Info,
            Ch_plot=[2,3],
            calibrated_data=True,
            Title='Raw RPM Data',
            custom_labels= False, #{'x': 'Time [s]', 'y': 'Acceleration [m/s^2]'},
            apply_low_pass=filter_button.value,
            cutoff_freq=cutoff_slider.value,
            show_unfiltered = show_unfiltered_but.value
        )

        mo.output.append(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plotting Speed vs Speed""")
    return


@app.cell(hide_code=True)
def _(T_start, mo, mu_d):
    mo.hstack([T_start,T_start.value,"[Nm] ; ", mu_d,mu_d.value], justify="start")
    return


@app.cell
def _(
    Channel_Info,
    Data,
    cutoff_slider,
    filter_button,
    goal_x,
    goal_y,
    old_x,
    old_y,
    plot_vs,
    setup_values,
):
    plot_vs(
        data=Data,
        Ch_info=Channel_Info,
        Ch_x=3,
        Ch_y=2,
        veh_speed = goal_x,
        engine_rpms = goal_y,
        old_veh = old_x,
        old_rpm= old_y,
        calibrated_data=True,
        subTitle=f'Speed - Speed Diagram [{setup_values}]',
        custom_labels={'x': 'Secondary [RPM]', 'y': 'Engine [RPM'},
        apply_low_pass=filter_button.value,
        cutoff_freq=cutoff_slider.value,)
    return


@app.cell
def _(Channel_Info, Data, cutoff_slider, filter_button, plot_ratio):
    plot_ratio(
        data=Data,
        Ch_info=Channel_Info,
        Ch_2=3,
        Ch_1=2,
        calibrated_data=True,
        subTitle='Pulley Speed Ratio',
        custom_labels={'x': 'Secondary [RPM]', 'y': 'Engine [RPM'},
        apply_low_pass=filter_button.value,
        cutoff_freq=cutoff_slider.value,
        saveFig=False,
    )
    return


@app.cell
def _(Channel_Info, Data, plot_data, section):
    time = Data[1].time
    diameter = Data[1].Data[section[0]:section[1]]
    diameter = apply_calibration(diameter, Channel_Info[1][1])

    DS = 203
    offset = 120
    diameter = DS - (diameter-offset)

    plot_data(x=time,
              y=diameter,
              section = section,
              Title="Secondary Diameter",
              name = "2nd Diam.",
              custom_labels=["Time [s]","Diameter [mm]"], 
              apply_low_pass=True, 
              cutoff_freq=10, 
              show_unfiltered= True, 
              color = 4)
    return diameter, time


@app.cell
def _(
    Channel_Info,
    Data,
    diameter,
    low_pass_filter,
    np,
    plot_data,
    pulley_diameters,
    section,
):
    # Write new data
    x = Data[2].time[section[0]:section[1]]
    _y1 = Data[2].Data[section[0]:section[1]]
    _y2 = Data[3].Data[section[0]:section[1]]

    # Transform Data using Calibration Values

    _y1 = apply_calibration(_y1, Channel_Info[2][1])
    _y2 = apply_calibration(_y2, Channel_Info[3][1])

    # Apply Low-Pass Filter
    _cutoff_freq = 60
    x = low_pass_filter(x, _cutoff_freq)
    _y1 = low_pass_filter(_y1, _cutoff_freq)
    _y2 = low_pass_filter(_y2, _cutoff_freq)

    cr_diam = []

    for i in range(len(x)):
        cr_diam.append(pulley_diameters(_y1[i]/(_y2[i]))[1])

    cr_diam = np.array(cr_diam)

    diff = ((diameter-cr_diam*1000)/diameter) *100

    plot_data(x=x,
              y=diff,
              section = section,
              Title="Slip",
              name = "2nd Diam.",
              custom_labels=["Time [s]","% Slip"], 
              apply_low_pass=True, 
              cutoff_freq=10, 
              show_unfiltered= True, 
              color = 5)
    return (cr_diam,)


@app.cell
def _(cr_diam, cutoff_freq, diameter, go, low_pass_filter, plotly, time):
    _fig = go.Figure()

    # Add scatter trace
    diamete_filtered = low_pass_filter(diameter, cutoff_freq)

    _fig.add_trace(
        go.Scatter(
            x=time,
            y=diamete_filtered,
            mode='lines',
            line=dict(color=plotly.colors.qualitative.Plotly[4]),
            name=f"Diameter from Laser Sensor",
            hovertemplate=
                f'X: %{{x:.2f}}<br>' +
                f'Y: %{{y:.2f}}<br>' +
                '<extra></extra>',
        ))

    _fig.add_trace(
        go.Scatter(
            x=time,
            y=cr_diam*1000,
            mode='lines',
            line=dict(color=plotly.colors.qualitative.Plotly[3]),
            name=f"Diameter from Pulley Speed Ratio",
            hovertemplate=
                f'X: %{{x:.2f}}<br>' +
                f'Y: %{{y:.2f}}<br>' +
                '<extra></extra>',
        ))

    _fig.update_layout(
            template = "plotly_white",
            dragmode='zoom',
            xaxis_title="Time [s]",
            yaxis_title="Secondary Diameter [mm]",
            title=dict(text="Secondary Diameter", x=0.5, xanchor='center'),
            showlegend=True,
            hovermode='closest',
            legend=dict(yanchor="top", y=-0.2, xanchor="center", x=0.5)  # Legend below plot
        )
    _fig.update_xaxes(showgrid=True)
    _fig.update_yaxes(showgrid=True)
    return


@app.cell
def _(mo):
    mo.md(r"""# Compare Setups""")
    return


@app.cell(hide_code=True)
def _(Channel_Info, cvt_simulation_briggs, go, low_pass_filter, np, read_txt):
    # Setup 1
    file1 = "C:\\Users\\hanne\\OneDrive - University of Pretoria\\Desktop\\DAQ Files\\73800-00\\04 Okt 2025\\T3 op &af.txt"
    data1 = read_txt(file1)
    section1 = np.array([0,80])*1000
    result1 = cvt_simulation_briggs(7, 3, 8, 0, 0, 0, cf_dyn = 0.3, T_takeoff = 17.4)
    goal_x1 = result1['veh_speed']
    goal_y1 = result1['engine_rpms']

    # Setup 2
    file2 = "C:\\Users\\hanne\\OneDrive - University of Pretoria\\Desktop\\DAQ Files\\62501-10\\62501-10 (4 Okt 2025).txt"
    data2 = read_txt(file2)
    section2 = np.array([65,95])*1000
    result2 = cvt_simulation_briggs(6, 2, 5, 0, 1, shim = 10, cf_dyn = 0.3, T_takeoff = 17.4)
    goal_x2 = result2['veh_speed']
    goal_y2 = result2['engine_rpms']

    cutoff_freq = 25

    Ch_x = 3
    Ch_y = 2
    Ch_info = Channel_Info

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
    v2s = (60*GR/(0.5*Wdia*3.6*2*np.pi))


    # Create Plotly figure
    _fig = go.Figure()

    # Extract data1
    x1 = data1[Ch_x].Data[section1[0]:section1[1]]
    y1 = data1[Ch_y].Data[section1[0]:section1[1]]

    x1 = apply_calibration(x1, Ch_info[Ch_x][1])
    y1 = apply_calibration(y1, Ch_info[Ch_y][1])

    x1 = low_pass_filter(x1, cutoff_freq)
    y1 = low_pass_filter(y1, cutoff_freq)

    # Prepare scatter trace
    scatter1 = dict(
        x=x1,
        y=y1,
        mode='lines',
        name=f"Test Data 73800-00",

    )
    _fig.add_trace(go.Scatter(**scatter1))

    goal_shift1 = dict(
        x=np.array(goal_x1) *v2s,
        y=goal_y1,
        mode='lines+markers',
        line=dict(dash='dash', color='blue'),
        name=f"Model [73800-00]",
    )
    _fig.add_trace(go.Scatter(**goal_shift1))


    # Extract data2
    x2 = data2[Ch_x].Data[section2[0]:section2[1]]
    y2 = data2[Ch_y].Data[section2[0]:section2[1]]

    x2 = apply_calibration(x2, Ch_info[Ch_x][1])
    y2 = apply_calibration(y2, Ch_info[Ch_y][1])

    x2 = low_pass_filter(x2, cutoff_freq)
    y2 = low_pass_filter(y2, cutoff_freq)

    scatter2 = dict(
        x=x2,
        y=y2,
        mode='lines',
        name=f"Test Data 62501-10",

    )
    _fig.add_trace(go.Scatter(**scatter2))

    v2s = (60*GR/(0.5*Wdia*3.6*2*np.pi))

    # Model Plots


    goal_shift2 = dict(
        x=np.array(goal_x2) *v2s,
        y=goal_y2,
        mode='lines+markers',
        line=dict(dash='dash', color='green'),
        name=f"Model [62501-10]",
    )
    _fig.add_trace(go.Scatter(**goal_shift2))

    # Plotting Lines to compare against
    Low = np.array([0, Vsmin])*v2s
    High = np.array([0, Vsmax])*v2s
    Govspeed = np.array([ErpmMax, ErpmMax])
    Idle = np.array([ErpmMin, ErpmMin])
    RPM = np.array([0, 3800])


    # Add scatter traces

    # fig.add_trace(go.Scatter(**old_shift))




    _fig.add_trace(go.Scatter(
        x=[0, Vsmin*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Low gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='High gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=Govspeed,
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=Idle,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
        xaxis = 'x2'
    ))


    factor = 1/v2s
    # Update layout
    _fig.update_layout(
        xaxis=dict(
            title='RPM',
            range=[0, 4800],  # Set to include your data range
            tick0=0,
            dtick=1000,  # Major ticks at 1000 increments
            gridcolor='lightgrey',
            showgrid=True
        ),
        xaxis2=dict(
            title='Vehicle Speed (km/h)',  # Adjust units as needed
            overlaying='x',  # Overlays the primary x-axis
            side='top',  # Position at the top
            range=[0, 4800 * factor],  # Scale range based on factor
            tick0=0,
            dtick= 1000 * factor,  # Match increments scaled by factor
            showgrid=False,  # Avoid clutter from secondary grid
            tickformat=".0f"
        ),
        dragmode='zoom',
        xaxis_title="Secondary Speed [RPM]",
        yaxis_title="Engine/Primary Speed [RPM]",
        yaxis=dict(range=[1000, 4200]),
        title=dict(text="Experiment Comparison", x=0.5,y=0.95, xanchor='center'),
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            font = dict(size=16)
        ),
    )
    _fig.update_xaxes(showgrid=True)
    _fig.update_yaxes(showgrid=True)
    return (
        Ch_info,
        Ch_x,
        Ch_y,
        Govspeed,
        Idle,
        RPM,
        Vsmax,
        Vsmin,
        cutoff_freq,
        factor,
        file2,
        result2,
        v2s,
    )


@app.cell(hide_code=True)
def _(
    Ch_info,
    Ch_x,
    Ch_y,
    Govspeed,
    Idle,
    RPM,
    Vsmax,
    Vsmin,
    cutoff_freq,
    cvt_simulation_briggs,
    factor,
    go,
    low_pass_filter,
    np,
    read_txt,
    v2s,
):
    # Slow Accel 73800-00
    file3 = "C:\\Users\\hanne\\OneDrive - University of Pretoria\\Desktop\\DAQ Files\\73800-00\\04 Okt 2025\\T1.txt"
    data3 = read_txt(file3)
    section3 = np.array([0,41])*1000
    result3_1 = cvt_simulation_briggs(7, 3, 8, 0, 0, 0, cf_dyn = 0.3, T_takeoff = 17.4)
    goal_x3_1 = result3_1['veh_speed']
    goal_y3_1 = result3_1['engine_rpms']
    result3_2 = cvt_simulation_briggs(7, 3, 8, 0, 0, 0, cf_dyn = 0.65, T_takeoff = 17.4)
    goal_x3_2 = result3_2['veh_speed']
    goal_y3_2 = result3_2['engine_rpms']


    # Create Plotly figure
    _fig = go.Figure()

    # Extract data1
    x3 = data3[Ch_x].Data[section3[0]:section3[1]]
    y3 = data3[Ch_y].Data[section3[0]:section3[1]]

    x3 = apply_calibration(x3, Ch_info[Ch_x][1])
    y3 = apply_calibration(y3, Ch_info[Ch_y][1])

    x3 = low_pass_filter(x3, cutoff_freq)
    y3 = low_pass_filter(y3, cutoff_freq)

    # Prepare scatter trace
    scatter3 = dict(
        x=x3,
        y=y3,
        mode='lines',
        name=f"Test Data 73800-00",

    )
    _fig.add_trace(go.Scatter(**scatter3))

    goal_shift3_1 = dict(
        x=np.array(goal_x3_1) *v2s,
        y=goal_y3_1,
        mode='lines+markers',
        line=dict(dash='dash', color='blue'),
        name=f"Model [73800-00] ; cf_dyn = 0.3",
    )
    _fig.add_trace(go.Scatter(**goal_shift3_1))

    goal_shift3_2 = dict(
        x=np.array(goal_x3_2) *v2s,
        y=goal_y3_2,
        mode='lines+markers',
        line=dict(dash='dashdot', color='#1F77B4'),
        name=r"Model [73800-00] ; cf_dyn = 0.65",
    )
    _fig.add_trace(go.Scatter(**goal_shift3_2))


    _fig.add_trace(go.Scatter(
        x=[0, Vsmin*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Low gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='High gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=Govspeed,
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=Idle,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
        xaxis = 'x2'
    ))



    # Update layout
    _fig.update_layout(
        xaxis=dict(
            title='RPM',
            range=[0, 4800],  # Set to include your data range
            tick0=0,
            dtick=1000,  # Major ticks at 1000 increments
            gridcolor='lightgrey',
            showgrid=True
        ),
        xaxis2=dict(
            title='Vehicle Speed (km/h)',  # Adjust units as needed
            overlaying='x',  # Overlays the primary x-axis
            side='top',  # Position at the top
            range=[0, 4800 * factor],  # Scale range based on factor
            tick0=0,
            dtick= 1000 * factor,  # Match increments scaled by factor
            showgrid=False,  # Avoid clutter from secondary grid
            tickformat=".0f"
        ),
        dragmode='zoom',
        xaxis_title="Secondary Speed [RPM]",
        yaxis_title="Engine/Primary Speed [RPM]",
        yaxis=dict(range=[1000, 4200]),
        title=dict(text="Experiment Comparison", x=0.5,y=0.95, xanchor='center'),
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            font = dict(size=16)
        ),
    )
    _fig.update_xaxes(showgrid=True)
    _fig.update_yaxes(showgrid=True)
    return (goal_shift3_2,)


@app.cell
def _(
    Ch_info,
    Ch_x,
    Ch_y,
    Govspeed,
    Idle,
    RPM,
    Vsmax,
    Vsmin,
    cutoff_freq,
    cvt_simulation_briggs,
    factor,
    file2,
    go,
    goal_shift3_2,
    low_pass_filter,
    np,
    read_txt,
    result2,
    v2s,
):
    # Create Plotly figure
    _fig = go.Figure()

    # Setup 4
    file4 = "C:\\Users\\hanne\\OneDrive - University of Pretoria\\Desktop\\DAQ Files\\62501-10\\62501-10 (4 Okt 2025).txt"
    data4 = read_txt(file2)
    section4 = np.array([0,35])*1000
    result4 = cvt_simulation_briggs(6, 2, 5, 0, 1, shim = 10, cf_dyn = 0.3, T_takeoff = 17.4)
    goal_x4 = result2['veh_speed']
    goal_y4 = result2['engine_rpms']

    _fig.add_trace(go.Scatter(**goal_shift3_2))

    # Extract data2
    x4 = data4[Ch_x].Data[section4[0]:section4[1]]
    y4 = data4[Ch_y].Data[section4[0]:section4[1]]

    x4 = apply_calibration(x4, Ch_info[Ch_x][1])
    y4 = apply_calibration(y4, Ch_info[Ch_y][1])

    x4 = low_pass_filter(x4, cutoff_freq)
    y4 = low_pass_filter(y4, cutoff_freq)

    scatter4 = dict(
        x=x4,
        y=y4,
        mode='lines',
        name=f"Test Data 62501-10",

    )

    _fig.add_trace(go.Scatter(**scatter4))

    # Model Plots


    goal_shift4 = dict(
        x=np.array(goal_x4) *v2s,
        y=goal_y4,
        mode='lines+markers',
        line=dict(dash='dash', color='green'),
        name=f"Model [62501-10]",
    )
    _fig.add_trace(go.Scatter(**goal_shift4))


    # Add scatter traces

    _fig.add_trace(go.Scatter(
        x=[0, Vsmin*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Low gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=RPM,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='High gear ratio',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax*v2s], y=Govspeed,
        mode='lines', line=dict(dash='dash', color='red'),
        name='Governor',
    ))
    _fig.add_trace(go.Scatter(
        x=[0, Vsmax], y=Idle,
        mode='lines', line=dict(dash='dash', color='grey'),
        name='Idle',
        xaxis = 'x2'
    ))



    # Update layout
    _fig.update_layout(
        xaxis=dict(
            title='RPM',
            range=[0, 4800],  # Set to include your data range
            tick0=0,
            dtick=1000,  # Major ticks at 1000 increments
            gridcolor='lightgrey',
            showgrid=True
        ),
        xaxis2=dict(
            title='Vehicle Speed (km/h)',  # Adjust units as needed
            overlaying='x',  # Overlays the primary x-axis
            side='top',  # Position at the top
            range=[0, 4800 * factor],  # Scale range based on factor
            tick0=0,
            dtick= 1000 * factor,  # Match increments scaled by factor
            showgrid=False,  # Avoid clutter from secondary grid
            tickformat=".0f"
        ),
        dragmode='zoom',
        xaxis_title="Secondary Speed [RPM]",
        yaxis_title="Engine/Primary Speed [RPM]",
        yaxis=dict(range=[1000, 4200]),
        title=dict(text="Experiment Comparison", x=0.5,y=0.95, xanchor='center'),
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            orientation="h",
            traceorder="normal",
            font = dict(size=16)
        ),
    )
    _fig.update_xaxes(showgrid=True)
    _fig.update_yaxes(showgrid=True)
    return


@app.cell
def _(e, go, low_pass_filter, mo, np, q, r, section, shim, t, w):
    def plot_vs(data, Ch_info, Ch_x, Ch_y, veh_speed, engine_rpms, old_veh, old_rpm, calibrated_data=False, subTitle="Custom Subtitle", 
                    custom_labels=None, apply_low_pass=False, cutoff_freq=60.0):

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
            fig = go.Figure()

            # Extract data
            x = data[Ch_x].Data[section[0]:section[1]]
            y = data[Ch_y].Data[section[0]:section[1]]

            # Extract time data
            time_data = data[Ch_x].time[section[0]:section[1]]

            # Transform Data using Calibration Values
            if calibrated_data:
                x = apply_calibration(x, Ch_info[Ch_x][1])
                y = apply_calibration(y, Ch_info[Ch_y][1])

            # Apply Low-Pass Filter
            if apply_low_pass:
                x = low_pass_filter(x, cutoff_freq)
                y = low_pass_filter(y, cutoff_freq)

            # Set labels
            XL = custom_labels.get('x', Ch_info[Ch_x][2][0]) if custom_labels else Ch_info[Ch_x][2][0]
            YL = custom_labels.get('y', Ch_info[Ch_y][2][1]) if custom_labels else Ch_info[Ch_y][2][1]

            # Prepare scatter trace
            scatter_params = dict(
                x=x,
                y=y,
                mode='lines',
                name=f"Test Data",
                hovertemplate=
                    f"{Ch_info[Ch_x][0]} vs {Ch_info[Ch_y][0]}<br>" +
                    f"X: %{{x:.2f}}<br>" +
                    f"Y: %{{y:.2f}}<br>" +
                    f"Time: %{{customdata:.2f}}<br>" +
                    "<extra></extra>",
                customdata=time_data  # Add time data to hover
            )

            v2s = (60*GR/(0.5*Wdia*3.6*2*np.pi))

            goal_shift = dict(
                x=np.array(veh_speed) *v2s,
                y=engine_rpms,
                mode='lines+markers',
                name=f"New Model [{q,w,e,r,t}-{shim}]",
            )

            old_shift = dict(
                x=np.array(old_veh)*v2s ,
                y=old_rpm,
                mode='lines+markers',
                name=f"Old Model [{q,w,e,r,t}-{shim}]",
            )


            # Plotting Lines to compare against
            Low = np.array([0, Vsmin])*v2s
            High = np.array([0, Vsmax])*v2s
            Govspeed = np.array([ErpmMax, ErpmMax])
            Idle = np.array([ErpmMin, ErpmMin])
            RPM = np.array([0, 3800])


            # Add scatter traces
            fig.add_trace(go.Scatter(**scatter_params))
            fig.add_trace(go.Scatter(**old_shift))
            fig.add_trace(go.Scatter(**goal_shift))



            fig.add_trace(go.Scatter(
                x=[0, Vsmin*v2s], y=RPM,
                mode='lines', line=dict(dash='dash', color='grey'),
                name='Low gear ratio',
            ))
            fig.add_trace(go.Scatter(
                x=[0, Vsmax*v2s], y=RPM,
                mode='lines', line=dict(dash='dash', color='grey'),
                name='High gear ratio',
            ))
            fig.add_trace(go.Scatter(
                x=[0, Vsmax*v2s], y=Govspeed,
                mode='lines', line=dict(dash='dash', color='red'),
                name='Governor',
            ))
            fig.add_trace(go.Scatter(
                x=[0, Vsmax], y=Idle,
                mode='lines', line=dict(dash='dash', color='grey'),
                name='Idle',
                xaxis = 'x2'
            ))


            factor = 1/v2s
            # Update layout
            fig.update_layout(
                xaxis=dict(
                    title='RPM',
                    range=[0, 4800],  # Set to include your data range
                    tick0=0,
                    dtick=1000,  # Major ticks at 1000 increments
                    gridcolor='lightgrey',
                    showgrid=True
                ),
                xaxis2=dict(
                    title='Vehicle Speed (km/h)',  # Adjust units as needed
                    overlaying='x',  # Overlays the primary x-axis
                    side='top',  # Position at the top
                    range=[0, 4800 * factor],  # Scale range based on factor
                    tick0=0,
                    dtick= 1000 * factor,  # Match increments scaled by factor
                    showgrid=False,  # Avoid clutter from secondary grid
                    tickformat=".0f"
                ),
                dragmode='zoom',
                xaxis_title="Secondary Speed [RPM]",
                yaxis_title="Engine/Primary Speed [RPM]",
                yaxis=dict(range=[1000, 4200]),
                title=dict(text=subTitle, x=0.5,y=0.95, xanchor='center'),
                showlegend=True,
                hovermode='closest',
                legend=dict(
                    yanchor="bottom",
                    y=-0.4,
                    xanchor="center",
                    x=0.5,
                    orientation="h",
                    traceorder="normal",
                    font = dict(size=14)
                ),
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)


            return mo.ui.plotly(fig)
    return (plot_vs,)


@app.cell
def _(go, low_pass_filter, mo, pio, plotly, theme):
    def plot_data(x,y,section = [1,-1],
                  Title = "Custom Title",
                  name = "Data Name",
                  custom_labels=["Time [s]","Torque [Nm]"], 
                  apply_low_pass=True, 
                  cutoff_freq=60.0, 
                  show_unfiltered= False, 
                  color = 0):

        # Create Plotly figure
        fig = go.Figure()

        # Write new data
        x = x[section[0]:section[1]]
        y = y[section[0]:section[1]]

        if apply_low_pass:
            y_unfiltered = y.copy()
            y = low_pass_filter(y, 10)

            if show_unfiltered:
                fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_unfiltered,
                    mode='lines',
                    name=f"{name} (Unfiltered)",
                    line=dict(color=plotly.colors.qualitative.Plotly[color], dash='dash', width=1),  # Faded with dashed line
                    opacity=0.5,  # Make unfiltered data faded
                )
            )

        # Add scatter trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name=f"{name}",
                line=dict(color=plotly.colors.qualitative.Plotly[color]),
                hovertemplate=
                    f'X: %{{x:.2f}}<br>' +
                    f'Y: %{{y:.2f}}<br>' +
                    '<extra></extra>',
            )
        )

        # Set Plotly template based on theme
        if theme.value == "dark":
            pio.templates.default = "plotly_dark"
        else:
            pio.templates.default = "plotly_white"

        # Update layout
        fig.update_layout(
            dragmode='zoom',
            xaxis_title=custom_labels[0],
            yaxis_title=custom_labels[1],
            title=dict(text=Title, x=0.5, xanchor='center'),
            showlegend=True,
            hovermode='closest',
            legend=dict(yanchor="top", y=-0.2, xanchor="center", x=0.5)  # Legend below plot
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)


        return mo.ui.plotly(fig)
    return (plot_data,)


@app.cell
def _(
    calculate_selection_average,
    go,
    low_pass_filter,
    mo,
    pio,
    plotly,
    theme,
):
    # Plotting Functions
    def plot_multi(data, section, Ch_info, Ch_plot, calibrated_data=False, Title="Custom Title", custom_labels=None, apply_low_pass=False, cutoff_freq=60.0, show_unfiltered= False, saveFig=False, color_offset = 0):

        # Create Plotly figure
        title = f"Channels: {Ch_plot}"
        fig = go.Figure()
        # Use Plotly's qualitative color scale
        colors = plotly.colors.qualitative.Plotly

        # Add each channel as a trace
        for idx, ch in enumerate(Ch_plot):
            # Write new data
            x = data[ch].time[section[0]:section[1]]
            y = data[ch].Data[section[0]:section[1]]


            # Transform Data using Calibration Values
            if calibrated_data:
                print(f"CH {ch} - Calibration Equation:")
                y = apply_calibration(y, Ch_info[ch][1])

            # Store unfiltered data for plotting
            y_unfiltered = y.copy()


            # Apply Low-Pass Filter
            if apply_low_pass:
                y = low_pass_filter(y, cutoff_freq)

            # Write legend
            name = f"Ch.{ch} - {Ch_info[ch][0]}"
            XL = custom_labels.get('x', Ch_info[ch][2][0]) if custom_labels else "Time [s]"
            YL = custom_labels.get('y', Ch_info[ch][2][1]) if custom_labels else Ch_info[ch][2][1]

            # Add filtered trace
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=f"{name} (Filtered)" if apply_low_pass else name,
                    line=dict(color=colors[idx % len(colors) + color_offset]),  # Cycle through Plotly colors
                    hovertemplate=
                        f"{name} (Filtered)<br>" +
                        f'X: %{{x:.2f}}<br>' +
                        f'Y: %{{y:.2f}}<br>' +
                        '<extra></extra>',  # Removes extra trace info
                )
            )

            # Add unfiltered trace if requested
            if show_unfiltered and apply_low_pass:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_unfiltered,
                        mode='lines',
                        name=f"{name} (Unfiltered)",
                        line=dict(color=colors[idx % len(colors) + color_offset], dash='dash', width=1),  # Faded with dashed line
                        opacity=0.5,  # Make unfiltered data faded
                        hovertemplate=
                            f"{name} (Unfiltered)<br>" +
                            f'X: %{{x:.2f}}<br>' +
                            f'Y: %{{y:.2f}}<br>' +
                            '<extra></extra>',
                    )
                )

            # Set Plotly template based on theme
            if theme.value == "dark":
                pio.templates.default = "plotly_dark"
            else:
                pio.templates.default = "plotly_white"

            # Update layout
            fig.update_layout(
                dragmode='zoom',
                # template="plotly_white",
                xaxis_title=XL,
                yaxis_title=YL,
                title=dict(text=Title, x=0.5, xanchor='center', font=dict(size=16)),
                showlegend=True,
                legend=dict(
                    x=0.5,
                    y=-0.2,
                    xanchor="center",
                    yanchor="top",
                    orientation="h",
                    traceorder="normal",
                    font = dict(size=14)
                ),
                hovermode='closest',
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

            # Add callback for selection event
            fig.data[0].on_selection(calculate_selection_average)

        # Save Figure
        if saveFig:
            fig.write_image(title + '.png')

        return mo.ui.plotly(fig)
    return (plot_multi,)


@app.function
def apply_calibration(data, calibration_values = (1,0)):
    """
    Apply calibration to the data.

    Parameters:
    data : list or numpy array
        The data to be calibrated.

    calibration_values : tuple
        A tuple containing the calibration coefficients (a, b) where y = a*x + b.

    Returns:
    calibrated_data : list or numpy array
        The calibrated data.
    """
    a, b = calibration_values
    calibrated_data = a * data + b
    print(f"y = {a:.2f}x + {b:.2f}")
    return calibrated_data


@app.cell
def _(np):
    def low_pass_filter(data, cutoff_freq, sampling_rate=1000, order=5, min_segment_length=10):
        """
        Apply a low-pass filter to continuous non-NaN segments of the data, skipping NaN values.

        Parameters:
            data: list or numpy array
                The data to be filtered.
            cutoff_freq: float
                The cutoff frequency for the low-pass filter.
            sampling_rate: int, optional
                The sampling rate of the data. Default is 1000 Hz.
            order: int, optional
                The order of the filter. Default is 5.
            min_segment_length: int, optional
                Minimum length of a non-NaN segment to filter. Shorter segments are left as-is.

        Returns:
            filtered_data: numpy array
                The filtered data with NaNs preserved in their original positions.
        """
        from scipy.signal import butter, filtfilt

        data = np.asarray(data, dtype=float)

        if len(data) == 0:
            return data

        # Initialize output array with original data (preserving NaNs)
        filtered_data = data.copy()

        # Validate filter parameters
        nyquist_freq = 0.5 * sampling_rate
        if cutoff_freq <= 0 or cutoff_freq >= nyquist_freq:
            raise ValueError(f"Cutoff frequency must be between 0 and {nyquist_freq} Hz")

        # Design Butterworth filter
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # Find continuous non-NaN segments
        mask = ~np.isnan(data)
        segment_starts = []
        segment_ends = []
        i = 0
        while i < len(data):
            if mask[i]:
                # Start of a new segment
                start = i
                while i < len(data) and mask[i]:
                    i += 1
                end = i
                if end - start >= min_segment_length:  # Only process sufficiently long segments
                    segment_starts.append(start)
                    segment_ends.append(end)
            else:
                i += 1

        # Apply filter to each valid segment
        for start, end in zip(segment_starts, segment_ends):
            segment = data[start:end]
            try:
                filtered_segment = filtfilt(b, a, segment)
                filtered_data[start:end] = filtered_segment
            except ValueError as e:
                # If filtering fails (e.g., too short segment), leave as-is
                print(f"Warning: Filtering failed for segment {start}:{end} ({e}). Skipping.")

        return filtered_data
    return (low_pass_filter,)


@app.cell
def _():
    from collections import namedtuple
    import pandas as pd

    Channel = namedtuple('Channel', ['ChannelName', 'ChannelNumber', 'Data', 'time'])

    def read_txt(file_path):
        # Load the data
        df = pd.read_csv(file_path, sep="\t")

        # Get time vector
        time = df["Time [s]"].to_numpy()

        # Create list to store channels
        channels = []

        # Create Channel namedtuple for each channel
        for i in range(8):
           # Try both possible column name formats
            channel_name = f"Channel{i}"
            if channel_name not in df.columns:
                channel_name = f"Channel {i}"  # Try with space
            if channel_name not in df.columns:
                raise KeyError(f"Column for channel {i} not found in DataFrame")
            channel_data = df[channel_name].to_numpy().astype(float)
            channel = Channel(
                ChannelName=channel_name,
                ChannelNumber=i,
                Data=channel_data,
                time=time
            )
            channels.append(channel)

        return channels
    return (read_txt,)


@app.cell
def _(e, go, low_pass_filter, mo, plotly, q, r, section, t, w):
    from CVT_Model_2025 import pulley_diameters

    def plot_ratio(data, Ch_info, Ch_1, Ch_2, calibrated_data=True, subTitle="Custom Subtitle", 
                custom_labels=None, apply_low_pass=False, cutoff_freq=60.0, saveFig=False, 
                interactive=True):

        # Create Plotly figure
        title = f"CVT Ratio"
        fig = go.Figure()

        # Write new data
        x = data[Ch_1].time[section[0]:section[1]]
        y1 = data[Ch_1].Data[section[0]:section[1]]
        y2 = data[Ch_2].Data[section[0]:section[1]]

        # Transform Data using Calibration Values
        if calibrated_data:
            y1 = apply_calibration(y1, Ch_info[Ch_1][1])
            y2 = apply_calibration(y2, Ch_info[Ch_2][1])

        # Apply Low-Pass Filter
        if apply_low_pass:
            y1 = low_pass_filter(y1, cutoff_freq)
            y2 = low_pass_filter(y2, cutoff_freq)

        # Set labels
        XL = "Time [s]"
        YL = "CVT Pulley Ratio"

        # Add scatter trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y1/y2,
                mode='lines',
                name=f"CVT Pulley Ratio",
                line=dict(color=plotly.colors.qualitative.Plotly[0]),
                hovertemplate=
                    f'X: %{{x:.2f}}<br>' +
                    f'Y: %{{y:.2f}}<br>' +
                    '<extra></extra>',
            )
        )

        # Update layout
        fig.update_layout(
            dragmode='zoom',
            xaxis_title=XL,
            yaxis_title=YL,
            title=dict(text=subTitle, x=0.5, xanchor='center'),
            showlegend=True,
            hovermode='closest',
            legend=dict(yanchor="top", y=-0.2, xanchor="center", x=0.5)  # Legend below plot
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True,range=[0.5, 4])

        # Save Figure
        if saveFig:
            fig.write_image(f"{q,w,e,r,t}_ss" + '.png', engine='kaleido')

        return mo.ui.plotly(fig)
    return plot_ratio, pulley_diameters


@app.cell
def _(Data, mo):
    def calculate_selection_average(ch,section):
        y = Data[ch].Data[section[0]:section[1]]
        avg=y.mean()
        mo.output.append(mo.md(f"Average of selected points: {avg:.4f}"))
    return (calculate_selection_average,)


@app.cell
def _():
    import plotly.graph_objs as go
    import plotly.io as pio
    import plotly.colors
    return go, pio, plotly


if __name__ == "__main__":
    app.run()
