import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import ogr
import fiona
import zipfile

from matplotlib.patches import Polygon

import numpy as np


def read_yukon_datasets(dir_name, data_name, suffix=""):
    dir_path = "./data/yukon_datasets/" + dir_name + "/" + data_name + "/"

    gdb_file = gpd.read_file(
        dir_path + data_name + suffix + ".gdb", driver="OpenFileGDB"
    )

    shp_file = gpd.read_file(
        dir_path + data_name + suffix + ".shp", driver="ESRI Shapefile"
    )

    kml_file = gpd.read_file(dir_path + data_name + suffix + ".kmz", driver="libkml")

    return gdb_file, shp_file, kml_file


def read_drillhole_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets(
        "DrillHoles", "DrillholeLocations", "_250k"
    )
    """
    'LATITUDE_DD', 'LONGITUDE_DD', 'DEPOSIT_MATERIAL', 'DEPOSIT_TYPE_1', 'DEPOSIT_TYPE_2',
    'DEPOSIT_TYPE_3', 'DEPOSIT_TYPE_4', 'DEPOSIT_TYPE_LABEL'
    """

    """
    gbd_file
    
    ['DRILLHOLE_ID', 'PROPERTY', 'DDH_NUMBER', 'ZONE_DESC', 'YEAR_DRILLED',
       'ELEVATION_M', 'AZIMUTH', 'DIP', 'TOTAL_LENGTH_M',
       'CORE_CONDITION_DESC', 'TOTAL_BOXES', 'COMPLETE_HOLE', 'CORE_SIZE_DESC',
       'PUBLICLY_AVAILABLE', 'MINFILE_NUMBER', 'MINFILE_NAME',
       'LINK_TO_MINFILE', 'COMMODITIES', 'REFERENCES', 'CORE_OWNER',
       'END_OF_HOLE_M', 'SUBMITTED_BY', 'PUBLIC_DATE', 'LATITUDE_DD',
       'LONGITUDE_DD', 'DEPOSIT_MATERIAL', 'DEPOSIT_TYPE_1', 'DEPOSIT_TYPE_2',
       'DEPOSIT_TYPE_3', 'DEPOSIT_TYPE_4', 'DEPOSIT_TYPE_LABEL',
       'LINK_TO_DATA', 'geometry']
    
    
    shp_file

    ['ID', 'PROPERTY', 'DDH_NUMBER', 'ZONE_DESC', 'YEAR_DRILL', 'ELEV_M',
       'AZIMUTH', 'DIP', 'TOT_LEN_M', 'CORE_CON', 'TOT_BOXES', 'COMPLETE',
       'CORE_SIZE', 'PUBLIC', 'MINFILENUM', 'MINFILE', 'MINF_LINK',
       'COMMODITIE', 'REFERENCES', 'CORE_OWNER', 'END_HOLE_M', 'SUBMIT_BY',
       'PUBLIC_DAT', 'LATI_DD', 'LONG_DD', 'DEPOSIT_MA', 'DEP_TYPE_1',
       'DEP_TYPE_2', 'DEP_TYPE_3', 'DEP_TYPE_4', 'D_TYP_LABL', 'LINK_DATA',
       'geometry']
    """


def read_geotechnical_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets(
        "DrillHoles", "GeotechnicalBoreholes"
    )

    """
    'LATITUDE_DD', 'LONGITUDE_DD', 'HOLE_DEPTH'

    gbd

    ['SITE_ID', 'PROJECT_NAME', 'PROJECT_NUMBER', 'LOCATION_DESC',
       'HOLE_DEPTH', 'CONSULTANT', 'CONTRACTOR', 'CLIENT', 'START_DATE',
       'END_DATE', 'LINK_TO_DATA', 'LATITUDE_DD', 'LONGITUDE_DD', 'geometry']
    
    
    shp

    ['SITE_ID', 'NAME', 'PROJECTNUM', 'LOCATION', 'HOLE_DEPTH', 'CONSULTANT',
       'CONTRACTOR', 'CLIENT', 'START_DATE', 'END_DATE', 'DATA_LINK',
       'LATITUDE', 'LONGITUDE', 'geometry']
    """


def read_geothermal_boreholes_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets(
        "DrillHoles", "Geothermal_Boreholes"
    )

    """
    gbd

    ['NAME', 'LOCATION', 'PURPOSE', 'STATUS', 'SCREEN_INT', 'STATIC_WL',
       'CAPACITY', 'DISCHARGE', 'TEMP_WATER', 'DRILL_DEPTH', 'BROCK_DEPTH',
       'TEMP_MIN', 'TEMP_MAX', 'INT_TOP', 'INT_BOTTOM', 'INT_THERM',
       'BOTTOM_UNK', 'THERM_GRAD', 'TG_CONFIDENCE', 'TG_METHOD',
       'TG_CORRECTION', 'TG_DATE', 'TG_LOG_NUM', 'THERM_COND', 'HEAT_FLOW',
       'HF_CORRECTION', 'GT_SILICA', 'GT_ALKALI', 'PH', 'CONDUCTIVITY',
       'HARDNESS', 'CA_MG_L', 'MG_MG_L', 'NA_MG_L', 'K_MG_L', 'HCO3_MG_L',
       'SO4_MG_L', 'CO3_MG_L', 'SIO2_MG_L', 'CL_MG_L', 'F_MG_L', 'E_ISO_18O',
       'E_ISO_2H', 'RAD_ISO_3H', 'GAS_NOBLE', 'GAS_DISS', 'GAS_EMITTED',
       'REFERENCE', 'REF_LINK', 'COMMENTS', 'ELEVATION', 'LOC_SOURCE',
       'LATITUDE_DD', 'LONGITUDE_DD', 'geometry']
    
       
    shp

    ['NAME', 'LOCATION', 'PURPOSE', 'STATUS', 'SCREEN_INT', 'STATIC_WL',
       'CAPACITY', 'DISCHARGE', 'TEMP_WATER', 'DRILLDEPTH', 'BROCKDEPTH',
       'TEMP_MIN', 'TEMP_MAX', 'INT_TOP', 'INT_BOTTOM', 'INT_THERM',
       'BOTTOM_UNK', 'THERM_GRAD', 'TG_CNFDNCE', 'TG_METHOD', 'TG_CORRCTN',
       'TG_DATE', 'TG_LOG_NUM', 'THERM_COND', 'HEAT_FLOW', 'HF_CORRCTN',
       'GT_SILICA', 'GT_ALKALI', 'PH', 'CONDCTVTY', 'HARDNESS', 'CA_MG_L',
       'MG_MG_L', 'NA_MG_L', 'K_MG_L', 'HCO3_MG_L', 'SO4_MG_L', 'CO3_MG_L',
       'SIO2_MG_L', 'CL_MG_L', 'F_MG_L', 'E_ISO_18O', 'E_ISO_2H', 'RAD_ISO_3H',
       'GAS_NOBLE', 'GAS_DISS', 'GAS_EMITTD', 'REFERENCE', 'REF_LINK',
       'COMMENTS', 'ELEVATION', 'L_SOURCE', 'LAT_DD', 'LONG_DD', 'geometry']
    """


def read_water_wells_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets("DrillHoles", "WaterWells")

    """
    gbd

    ['BOREHOLE_ID', 'WELL_NAME', 'COMMUNITY', 'PURPOSE', 'WELL_DEPTH_FTBGS',
       'DEPTH_TO_BEDROCK_FTBGS', 'ESTIMATED_YIELD_GPM', 'YIELD_METHOD',
       'STATIC_WATER_LEVEL_FTBTOC', 'DRILL_YEAR', 'DRILL_MONTH', 'DRILL_DAY',
       'CASING_OUTSIDE_DIAM_IN', 'TOP_OF_SCREEN_FTBGS',
       'BOTTOM_OF_SCREEN_FTBGS', 'TOP_OF_CASING_ELEVATION_MASL',
       'GROUND_LEVEL_ELEVATION_MASL', 'WELL_HEAD_STICKUP_M', 'WELL_LOG',
       'LINK', 'QUALITY', 'LOCATION_SOURCE', 'LATITUDE_DD', 'LONGITUDE_DD',
       'geometry']
    """

    """
    shp

    ['BOREHOLEID', 'WELL_NAME', 'COMMUNITY', 'PURPOSE', 'WELL_DEPTH',
       'DEPBEDROCK', 'EST_YIELD', 'YIELD_METH', 'STATWATER', 'DRILL_YEAR',
       'DRILLMONTH', 'DRILL_DAY', 'CASINGDIAM', 'TOP_SCREEN', 'BOT_SCREEN',
       'CAS_ELEV', 'GROUNDELEV', 'HEAD_STICK', 'WELL_LOG', 'LINK', 'QUALITY',
       'LAT_DD', 'LONG_DD', 'geometry']
    """

    depths = gdb_file["DEPTH_TO_BEDROCK_FTBGS"]

    gt_depths_inds, gt_depths, lt_depths_inds, lt_depths, exact_depths_inds = (
        [],
        [],
        [],
        [],
        [],
    )
    for ind, d in enumerate(depths):
        if d is None:
            continue
        if ">" in d:
            gt_depths_inds.append(ind)
            gt_depths.append(float(d.replace(">", "")))
        elif "<" in d:
            lt_depths_inds.append(ind)
            lt_depths.append(float(d.replace("<", "")))
        else:
            exact_depths_inds.append(ind)

    cm = plt.cm.get_cmap("RdYlBu")

    plt.subplot(1, 2, 1)
    sc = plt.scatter(
        gdb_file["LONGITUDE_DD"].iloc[gt_depths_inds],
        gdb_file["LATITUDE_DD"].iloc[gt_depths_inds],
        c=gt_depths,
        vmin=0,
        vmax=500,
        s=35,
        cmap=cm,
    )
    plt.colorbar(sc)

    plt.xlim([-135.3, -134.9])
    plt.ylim([60.65, 60.81])
    plt.title("bedrock depth greater than")

    """
    plt.subplot(1, 3, 2)
    sc = plt.scatter(
        gdb_file["LONGITUDE_DD"].iloc[lt_depths_inds],
        gdb_file["LATITUDE_DD"].iloc[lt_depths_inds],
        c=lt_depths,
        vmin=np.min(lt_depths),
        vmax=np.max(lt_depths),
        s=35,
        cmap=cm,
    )
    plt.colorbar(sc)
    """

    plt.subplot(1, 2, 2)

    exact_depths = (
        gdb_file["DEPTH_TO_BEDROCK_FTBGS"].iloc[exact_depths_inds].astype(float)
    )
    sc = plt.scatter(
        gdb_file["LONGITUDE_DD"].iloc[exact_depths_inds],
        gdb_file["LATITUDE_DD"].iloc[exact_depths_inds],
        c=exact_depths,
        vmin=0,
        vmax=500,
        s=35,
        cmap=cm,
    )
    plt.colorbar(sc)
    plt.title("bedrock depth")

    plt.xlim([-135.3, -134.9])
    plt.ylim([60.65, 60.81])

    plt.tight_layout()
    plt.show()


def read_bedrock_geology_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets("DrillHoles", "BedrockGeology")

    """
    gbd

    ['UNIT_1M', 'UNIT_250K', 'UNIT_ORIG', 'ASSEMBLAGE', 'SUPERGROUP',
       'GP_SUITE', 'FORMATION', 'MEMBER', 'NAME', 'TERRANE', 'TERR_LABEL',
       'TECT_ELEM', 'ERA_MAX', 'PERIOD_MAX', 'EPOCH_MAX', 'STAGE_MAX',
       'AGE_MAX_MA', 'ERA_MIN', 'PERIOD_MIN', 'EPOCH_MIN', 'STAGE_MIN',
       'AGE_MIN_MA', 'ROCK_CLASS', 'ROCK_SUBCL', 'SHORT_DESC', 'ROCK_MAJOR',
       'ROCK_MINOR', 'ROCK_NOTES', 'LABEL_250K', 'LABEL_1M', 'COMMENTS', 'RED',
       'GREEN', 'BLUE', 'Shape_Length', 'Shape_Area', 'geometry']
    """

    """
    shp

    ['UNIT_1M', 'UNIT_250K', 'UNIT_ORIG', 'ASSEMBLAGE', 'SUPERGROUP',
       'GP_SUITE', 'FORMATION', 'MEMBER', 'NAME', 'TERRANE', 'TERR_LABEL',
       'TECT_ELEM', 'ERA_MAX', 'PERIOD_MAX', 'EPOCH_MAX', 'STAGE_MAX',
       'AGE_MAX_MA', 'ERA_MIN', 'PERIOD_MIN', 'EPOCH_MIN', 'STAGE_MIN',
       'AGE_MIN_MA', 'ROCK_CLASS', 'ROCK_SUBCL', 'SHORT_DESC', 'ROCK_MAJOR',
       'ROCK_MINOR', 'ROCK_NOTES', 'LABEL_250K', 'LABEL_1M', 'COMMENTS', 'RED',
       'GREEN', 'BLUE', 'geometry']
    """

    # UNIT_250K
    # TERRANE
    # ROCK_CLASS
    # ROCK_MAJOR
    # ASSEMBLAGE

    map_category = "ROCK_MAJOR"
    # my_geoseries = my_geoseries.set_crs(epsg=4326)
    shp_file = shp_file.to_crs("EPSG:4326")
    # "EPSG:3578"
    # print(shp_file.crs)

    fig, ax = plt.subplots(figsize=(10, 6))

    shp_file.plot(
        ax=ax,
        column=map_category,
        legend=True,
        categorical=True,
        legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.5, 1)},
    )

    # ax.set_extent([-135.3, -134.9, 60.65, 60.81])

    ax.set_xlim(-135.3, -134.9)
    ax.set_ylim(60.65, 60.81)

    plt.title(map_category)
    plt.show()

    #

    # print(gdb_file)
    # print(shp_file.columns)
    # print(kml_file.columns)


def read_bedrock_map_index_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets("Geology", "BedrockGeology")

    """
    gbd

    ['MAP_NAME', 'NTS_MAP', 'REFERENCE', 'PUBLICATION_YEAR', 'MAP_YEAR',
       'PUBLICATION_SCALE', 'MAP_SCALE', 'AGENCY', 'DATA_SOURCE', 'CONFIDENCE',
       'AREA_KM', 'Shape_Length', 'Shape_Area', 'geometry']
    
    shp

    ['MAP_NAME', 'NTS_MAP', 'REFERENCE', 'PUB_YEAR', 'MAP_YEAR', 'PUB_SCALE',
       'MAP_SCALE', 'AGENCY', 'DATASOURCE', 'CONFIDENCE', 'AREA_KM',
       'geometry']
    """


def read_yukon_bedrock_geology_files():
    dir_path = "./data/yukon_datasets/Geology/BedrockGeology/"

    """
    ['Source', 'NTS_MAPS', 'PUB_SCALE', 'MAP_NAME', 'Plot', 'Source_type',
       'PUB_YEAR', 'AGENCY', 'MAP_SCALE', 'CONFIDENCE', 'MAP_YEAR', 'Area',
       'Shape_Length', 'Shape_Area', 'geometry']
    """
    gdb_file = gpd.read_file(
        dir_path
        + "Yukon_Bedrock_Geology_Complete.fgdb/Yukon_Bedrock_Geology_June_2023.gdb",
        driver="OpenFileGDB",
    )

    """

    ['UNIT_1M', 'UNIT_250K', 'UNIT_ORIG', 'ASSEMBLAGE', 'SUPERGROUP',
       'GP_SUITE', 'FORMATION', 'MEMBER', 'NAME', 'TERRANE', 'TERR_LABEL',
       'TECT_ELEM', 'ERA_MAX', 'PERIOD_MAX', 'EPOCH_MAX', 'STAGE_MAX',
       'AGE_MAX_MA', 'ERA_MIN', 'PERIOD_MIN', 'EPOCH_MIN', 'STAGE_MIN',
       'AGE_MIN_MA', 'ROCK_CLASS', 'ROCK_SUBCL', 'SHORT_DESC', 'ROCK_MAJOR',
       'ROCK_MINOR', 'ROCK_NOTES', 'LABEL_250K', 'LABEL_1M', 'COMMENTS', 'RED',
       'GREEN', 'BLUE', 'geometry']
    """
    shp_file = gpd.read_file(
        dir_path + "Yukon_Bedrock_Geology_Complete.shp/Bedrock_Geology.shp",
        driver="ESRI Shapefile",
    )

    print(gdb_file.columns)
    print(shp_file.columns)


def read_faults_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets("Geology", "Faults")

    """
    'FEATURE', 'TYPE', 'SUBTYPE', 'NAME'


    gbd

    ['FEATURE', 'TYPE', 'SUBTYPE', 'CONFIDENCE', 'NAME', 'REFERENCE',
       'SCALE', 'SYMBOL_DIR', 'COMMENTS', 'Shape_Length', 'geometry']
    
    
    shp

    ['FEATURE', 'TYPE', 'SUBTYPE', 'CONFIDENCE', 'NAME', 'REFERENCE',
       'SCALE', 'SYMBOL_DIR', 'COMMENTS', 'geometry']
    """


def read_sedimentary_basin_files():
    gdb_file, shp_file, kml_file = read_yukon_datasets(
        "Geology", "SedimentaryBasins", "_250k"
    )

    """
    'BASIN_NAME', 'LOCATION'

    gbd

    ['SEDIMENT_BASIN_NAME', 'SEDIMENT_BASIN_ID', 'BASIN_LOCATION',
       'DESCRIPTION', 'REGION_NAME', 'AREA_KILOMETRES', 'AREA_ACRES',
       'AREA_HECTARES', 'Shape_Length', 'Shape_Area', 'geometry']
       
    shp

    ['BASIN_ID', 'BASIN_NAME', 'LOCATION', 'DESC_', 'REGION', 'AREA_KM',
       'AREA_ACRES', 'AREA_HA', 'geometry']
    """


def read_sedimentary_extents_files():
    dir_path = "./data/yukon_datasets/Geology/SedimentaryBasins/"

    """
    ['REGION_NAME', 'AREA_KILOMETRES', 'AREA_ACRES', 'AREA_HECTARES',
       'Shape_Length', 'Shape_Area', 'geometry']
    """
    gdb_file = gpd.read_file(
        dir_path + "Sedimentary_Extents_1M.gdb", driver="OpenFileGDB"
    )

    """
    ['REG_NAME', 'AREA_KM', 'AREA_ACRES', 'AREA_HA', 'geometry']
    """
    shp_file = gpd.read_file(
        dir_path + "Sedimentary_Extents_1M.shp", driver="ESRI Shapefile"
    )

    kml_file = gpd.read_file(dir_path + "Sedimentary_Extents_1M.kmz", driver="libkml")

    print(gdb_file.columns)
    print(shp_file.columns)
    print(kml_file.columns)


def read_surficial_geology_files():
    dir_path = "./data/yukon_datasets/Geology/SurficialGeology/"

    """
    ['REGION_NAME', 'AREA_KILOMETRES', 'AREA_ACRES', 'AREA_HECTARES',
       'Shape_Length', 'Shape_Area', 'geometry']
    """
    gdb_file = gpd.read_file(
        dir_path + "Surficial_Geology_Complete.fgbd/YukonSurficialGeology_2023_CSW.gdb",
        driver="OpenFileGDB",
    )

    """
    ['REG_NAME', 'AREA_KM', 'AREA_ACRES', 'AREA_HA', 'geometry']
    """
    # shp_file = gpd.read_file(
    #    dir_path + "Surficial_Geology_Complete.shp", driver="ESRI Shapefile"
    # )

    print(gdb_file.columns)
    # print(shp_file.columns)


def read_terranes_files():
    dir_path = "./data/yukon_datasets/Geology/Terranes/"

    """
    ['TERRANE', 'T_GROUP', 'AFFINITY', 'T_NAME', 'SUBTERRANE', 'T_GP_SIMPLE',
       'TECT_SETTING', 'AGE_RANGE', 'DESCRIPTION', 'Shape_Length',
       'Shape_Area', 'geometry']
    """
    gdb_file = gpd.read_file(dir_path + "Terranes.gdb", driver="OpenFileGDB")

    """
    ['TERRANE', 'T_GROUP', 'AFFINITY', 'T_NAME', 'SUBTERRANE', 'TGP_SIMPLE',
       'TECT_SET', 'AGE_RANGE', 'DESCRIPN', 'geometry']
    """
    shp_file = gpd.read_file(dir_path + "Terranes.shp", driver="ESRI Shapefile")

    kml_file = gpd.read_file(dir_path + "Terranes.kmz", driver="libkml")

    print(gdb_file.columns)
    print(shp_file.columns)
    print(kml_file.columns)


def read_curie_depth_files():
    # read .tif
    # read xlsx
    pass


def read_geothermal_dataset_files():
    # read .tif file

    dir_path = "./data/yukon_datasets/Geothermal/GeothermalDataset/"

    """
    ['CPD__km_', 'Interval', 'Shape_Length', 'geometry']
    """
    gdb_file = gpd.read_file(
        dir_path + "YGS_GeothermalCompilation_20231106.gdb", driver="OpenFileGDB"
    )

    """
    ['NAME', 'LATITUDE_D', 'LONGITUDE_', 'ELEVATION', 'LOC_SOURCE',
       'LOCATION', 'PURPOSE', 'STATUS', 'SCREEN_INT', 'STATIC_WL', 'CAPACITY',
       'DISCHARGE', 'TEMP_WATER', 'DRILL_DEPT', 'BROCK_DEPT', 'TEMP_MIN',
       'TEMP_MAX', 'INT_TOP', 'INT_BOTTOM', 'INT_THERM', 'BOTTOM_UNK',
       'THERM_GRAD', 'TG_CONFIDE', 'TG_CORRECT', 'TG_METHOD', 'TG_DATE',
       'TG_LOG_NUM', 'THERM_COND', 'HEAT_FLOW', 'HF_CORRECT', 'GT_SILICA',
       'GT_ALKALI', 'PH', 'CONDUCTIVI', 'HARDNESS', 'CA_MG_L', 'MG_MG_L',
       'NA_MG_L', 'K_MG_L', 'HCO3_MG_L', 'SO4_MG_L', 'CO3_MG_L', 'SIO2_MG_L',
       'CL_MG_L', 'F_MG_L', 'E_ISO_18O', 'E_ISO_2H', 'RAD_ISO_3H', 'GAS_NOBLE',
       'GAS_DISS', 'GAS_EMITTE', 'REFERENCE', 'REF_LINK', 'COMMENTS', 'PUBLIC',
       'created_da', 'last_edite', 'last_edi_1', 'geometry']
    """
    shp_file = gpd.read_file(dir_path + "Shapefiles", driver="ESRI Shapefile")

    print(gdb_file.columns)
    print(shp_file.columns)


def read_permafrost_point_files():
    dir_path = "./data/yukon_datasets/Geothermal/Permafrost/"

    """
    ['TITLE', 'AUTHOR', 'YEAR', 'REPORT_TYPE', 'DESCRIPTION', 'REPORT_LINK',
       'geometry']
    """
    gdb_file = gpd.read_file(
        dir_path + "Permafrost_Reports_Point.gdb", driver="OpenFileGDB"
    )

    """
    ['TITLE', 'AUTHOR', 'YEAR', 'REPORTTYPE', 'DESC_', 'REPORTLINK',
       'geometry']
    """
    shp_file = gpd.read_file(
        dir_path + "Permafrost_Reports_Point.shp", driver="ESRI Shapefile"
    )

    kml_file = gpd.read_file(dir_path + "Permafrost_Reports_Point.kmz", driver="libkml")

    print(gdb_file.columns)
    print(shp_file.columns)
    print(kml_file.columns)


def read_permafrost_polygon_files():
    dir_path = "./data/yukon_datasets/Geothermal/Permafrost/"

    """
    ['TITLE', 'AUTHOR', 'YEAR', 'REPORT_TYPE', 'DESCRIPTION', 'REPORT_LINK',
       'Shape_Length', 'Shape_Area', 'geometry']
    """
    gdb_file = gpd.read_file(
        dir_path + "Permafrost_Reports_Polygon.gdb", driver="OpenFileGDB"
    )

    """
    ['TITLE', 'AUTHOR', 'YEAR', 'REPORTTYPE', 'DESC_', 'REPORTLINK',
       'geometry']
    """
    shp_file = gpd.read_file(
        dir_path + "Permafrost_Reports_Polygon.shp", driver="ESRI Shapefile"
    )

    kml_file = gpd.read_file(
        dir_path + "Permafrost_Reports_Polygon.kmz", driver="libkml"
    )

    print(gdb_file.columns)
    print(shp_file.columns)
    print(kml_file.columns)


def read_radiogenic_heat_files():
    dir_path = "./data/yukon_datasets/Geothermal/RadiogenicHeat/"

    """
    ['TITLE', 'AUTHOR', 'YEAR', 'REPORT_TYPE', 'DESCRIPTION', 'REPORT_LINK',
       'Shape_Length', 'Shape_Area', 'geometry']
    """
    gdb_file = gpd.read_file(
        dir_path + "Geothermal_Radiogenic_Heat_Production.gdb", driver="OpenFileGDB"
    )

    """
    ['TITLE', 'AUTHOR', 'YEAR', 'REPORTTYPE', 'DESC_', 'REPORTLINK',
       'geometry']
    """
    shp_file = gpd.read_file(
        dir_path + "Geothermal_Radiogenic_Heat_Production.shp", driver="ESRI Shapefile"
    )

    kml_file = gpd.read_file(
        dir_path + "Geothermal_Radiogenic_Heat_Production.kmz", driver="libkml"
    )

    print(gdb_file.columns)
    print(shp_file.columns)
    print(kml_file.columns)


def read_water_wells_csv_files():
    dir_path = "./data/yukon_datasets/DrillHoles/WaterWells/"
    df = pd.read_csv(dir_path + "Waterwells.csv")
