import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    /// attention | NB: Note on functions below

    All the arrays below are defined as functions to enable it to be imported into othe notebooks or scripts, while still keeping the explaining notes.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Flyweight Geometric Coefficient Correlations (Mills, 2019)
    The following is a recreation of the plot of the flyweight geometric coefficient (Mills, 2019, fig. 9). The x-axis is changed to present the data in a more understandable way that reflects the way it was measured better. Mills (2019) measured each flyweight’s geometric coefficient, Rsin(θ) at idle (plotted just before zero shift), and then at 0%, 25%, 50%, 75%, 100% shift percentages.
    """
    )
    return


@app.function
def FW():
    return np.array([      # Series, WEIGHT, RSINT(IDLE), RSINT(0), RSINT(0.25), RSINT(0.5), RSINT(0.75), RSINT(1) # 1=10, 2=D, 3=S
    [1,45,0.028333333,0.0353,0.033346667,0.0307,0.029666667,0.0285],  #0
    [1,51,0.030666667,0.0388,0.036433333,0.033266667,0.031166667,0.0328],  #1
    [3,55,0.0515,0.038766667,0.03704,0.034926667,0.033675,0.036066667],  #2
    [1,60,0.031333333,0.041,0.0392,0.035866667,0.034041667,0.0355],  #3
    [1,62,0.03245	,0.0424,0.0403,0.03821,0.03655,0.0341],  #4
    [1,66,0.0305,0.0405,0.039526667,0.037426667,0.036089167,0.033563333],  #5
    [1,70,0.039833333,0.039966667,0.03932,0.037933333,0.036908333,0.0365],  #6
    [1,72,0.034533333,0.039466667,0.03932,0.038353333,0.037441667,0.037966667],  #7
    [2,80,0.023566667,0.0305,0.03444,0.034353333,0.033841667,0.034366667],  #8
    ])


@app.function
def FW_names():
    return np.array(["1045","1051","S55","1060","1062","1066","1070","1072","D80"])


@app.cell
def _():
    FWs = FW()
    FWnms = FW_names()

    x = np.array([-0.03,0,0.25,0.5,0.75,1])*100

    fig = plt.figure(figsize=(8,5))
    plt.title(r"Flyweight Geometic Coefficient, $Rsin(\theta)$ Across Shift Percentage")

    for i in range(len(FWs)):
        plt.plot(x,FWs[i,2:],"-o",label = FWnms[i])

    plt.xlabel("Shift Percentage")
    plt.ylabel(r"$Rsin(\theta)$")

    plt.ylim(0.02,0.055)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=5)
    # plt.subplots_adjust(bottom=0.25)
    plt.grid()

    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Primary Springs""")
    return


@app.cell
def _():

    '''
    This data is from tests done by JL Pretorius (2025), it was compared to the data obtained by Mills (2019) and most corresponded but Steel [1] had a 21% difference, and Steel [4] was missing, and the brown spring was added although it seemed way to stiff to suit any setups

    ''' 

    """Old data, Mills (2019)
    np.array([        # Number, Static length, Stiffness  [N/mm]
        [1,60,16.8],  #BLUE/WHITE        0 
        [2,75,16.2],  #STEEL        1 
        [3,101.5,4.28],  #BLACK    2 
        [4,106,8.24], #PURPLE       3             
        [5,114,6.73],  #STEEL          4     
        [6,138,1.25], #STEEL       5 
        [7,144,1.26],  #YELLOW/RED  6 
        ])
    """
    return


@app.function
def PS():
    return np.array([        # Number, Static length, Stiffness  [N/mm]
    [0,60,17],  #BLUE/WHITE        0 
    [1,75,12.7],  #STEEL        1 
    [2,101.5,4.3],  #BLACK    2 
    [3,106,8.2], #PURPLE       3             
    [5,114,6.7],  #Steel (Missing)         4     
    [5,138,1.4], #STEEL/GREY       5 
    [6,144,1.2],  #YELLOW/RED  6 
    [7,82,18.9],  #BROWN  6 
    ])


@app.function
def PS_names(): 
    return np.array(["BLUE/WHITE (0)","Steel 1 (1)", "Black (2)","Purple (3)", "Steel MISSING (4)", "Steel/Silver (5)", "YELLOW/RED (6)", "Brown [7]"])


@app.cell
def _():
    PSs = PS()
    PS_nms = PS_names()

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.set_title(r"Primary Spring Stiffnesses and Static Lengths")
    ax2.set_xlabel(r"Spring Names")
    ax2.set_ylabel(r"Stiffness [N/m]", color='#1f77b4')  # Color for stiffness axis

    # Plot stiffness as bars
    colors = ['#1f77b4', 'Grey', '#3d3b3b', 'Purple','Grey' , 'grey', 'yellow','brown']
    for ii in range(len(PSs)):
        ax2.bar(PS_nms[ii], PSs[ii,2], color=colors[ii % len(colors)], label=PS_nms[ii] if ii == 0 else "")

    # Create secondary y-axis for static lengths
    ax3 = ax2.twinx()
    ax3.set_ylabel(r"Static Length [mm]", color='red')  # Color for length axis

    # Plot static lengths as a line plot
    ax3.plot(PS_nms, PSs[:,1], color='red', marker='o', linestyle='-', linewidth=2, markersize=8, label="Static Length")

    # Rotate x-axis labels
    ax2.set_xticks(range(len(PS_nms)), PS_nms, rotation=30)

    # Add legends for both axes
    ax3.legend(loc='center right')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Secondary Springs
    For now the function contains all three data sets that can be found in the reports from previious students at TuksBaja. They are all different with no way to replicate the results.  For now use the parameters from Mills' Appendix

    /// attention | NB: Need to verify results
    Need to do own experiments to confirm which results are the most accurate
    ///

    /// admonition  | Data to use: 
    For now use the Parameters from Mills' Appendix
    ///
    """
    )
    return


@app.function
def SS():  

    # initial force, final force, torsional stiffness [Nmm/rad] 
    # Addapted from Nathan Mills' Appendix (2019)
    Mills_Appx = np.array([[117, 239, 4.55], #0 #Red
                           [140, 270, 5.64], #1 #Black
                           [159, 313, 7.3]   #2 #Silver
                          ])

    # Maganezi's (2019) data according to Mulambu (2021)
    Maganezi = np.array([[108, 214.57, 1.52],  #0 #Red
                         [140, 251.16, 1.81],  #1 #Black
                         [159, 290.16, 2.27]   #2 #Silver
                        ])

    # Nathan Mills' first attempt at converting from data in the Polaris cataloques (2019)
    con = 4.4482216 # convert pound force to newtons
    conmom = con * 0.0254 # convert pound force inches to newton meters
    rng = (150-67)/180*np.pi
    Mills_Conv = np.array([     # f2x, F2x, k2t
                            [21*con, 45*con,   47*conmom/rng],      #RED
                            [24*con, 49*con,   55*conmom/rng],      #BLACK
                            [25*con, 54.5*con, 66*conmom/rng],    #SILVER
                            ])
    return Mills_Appx


@app.cell(hide_code=True)
def _():
    mo.md(r"""## New way to define it using initial (installed) force, and axial stiffness""")
    return


@app.function
def SS_new ():  

    # initial force, axial stiffness [N/mm], torsional stiffness [Nmm/rad] 
    # Addapted from Nathan Mills' Appendix (2019)
    Mills_Appx_Transformed = np.array([[117, 4.27, 4.55], #0 #Red
                           [140, 4.55, 5.64], #1 #Black
                           [159, 5.39, 7.3]   #2 #Silver
                          ])

    return Mills_Appx_Transformed


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Secondary Spring Pre-tension""")
    return


@app.function
def SPT():
    return np.array([
        [22,44,66,88,110] #
        ])


@app.function
def SPT_names():
    return np.array(["22 = (H1-S3)","44 = (H1-S2)","66 = (H2-S3)","88 = (H1-S1)","110 = (H2-S2)",])


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Torque Feedback Ramps
    /// admonition  | Improvements to make: 
    Currently this is only constant values, to investigate composite ramps this will need to be changed
    ///
    """
    )
    return


@app.function
def Helix():
    return np.array([
        32,34,36,38,40,42,44,46,48,50
        ])


if __name__ == "__main__":
    app.run()
