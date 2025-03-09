### WELL DATA ###


def read_well_data():
    """
    X,
    Y,
    OBJECTID,
    BOREHOLE_ID,
    WELL_NAME,
    COMMUNITY,
    PURPOSE,
    WELL_DEPTH_FTBGS,
    DEPTH_TO_BEDROCK_FTBGS,
    ESTIMATED_YIELD_GPM,
    YIELD_METHOD,
    STATIC_WATER_LEVEL_FTBTOC,
    DRILL_YEAR,
    DRILL_MONTH,
    DRILL_DAY,
    CASING_OUTSIDE_DIAM_IN,
    TOP_OF_SCREEN_FTBGS,
    BOTTOM_OF_SCREEN_FTBGS,
    TOP_OF_CASING_ELEVATION_MASL,
    GROUND_LEVEL_ELEVATION_MASL,
    WELL_HEAD_STICKUP_M,
    WELL_LOG,
    LINK,
    QUALITY,
    LOCATION_SOURCE,
    LATITUDE_DD,
    LONGITUDE_DD
    """

    # X: longitude
    # Y: latitude
    # WELL_DEPTH_FTBGS:
    # DEPTH_TO_BEDROCK_FTBGS:
    # GROUND_LEVEL_ELEVATION_MASL:

    df = pd.read_csv("./data/yukon_datasets/Water_wells.csv")

    lons = df["X"]
    lats = df["Y"]
    well_depth = df["WELL_DEPTH_FTBGS"]
    depth_to_bedrock = df["DEPTH_TO_BEDROCK_FTBGS"]
    ground_level_elevation = df["GROUND_LEVEL_ELEVATION_MASL"]

    depth_to_bedrock = (
        depth_to_bedrock.str.replace(">", "").str.replace("<", "").values.astype(float)
    )

    inds = (
        (lons > -136)
        & (lons < -134)
        & (lats > 60)
        & (lats < 61)
        & (depth_to_bedrock < 2800)
    )
    plt.scatter(lons[inds], lats[inds], c=depth_to_bedrock[inds])
    plt.colorbar()
    plt.xlim([-135.3, -134.9])
    plt.ylim([60.65, 60.81])

    plt.show()
