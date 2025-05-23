
import matplotlib.pyplot as plt
import numpy as np
# from scalebar import scale_bar
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
import cartopy.io.img_tiles as cimgt
from math import floor
from matplotlib import patheffects

def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    return floor( ( lon + 180 ) / 6) + 1

def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy, str(length) + ' ' + units, transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, zorder=3)




# Google image tiling
request1 = cimgt.GoogleTiles(style='satellite')
request2 = cimgt.GoogleTiles(url='https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}.jpg')

# Map projection
proj = ccrs.AlbersEqualArea(central_longitude=-135.076167, \
                            central_latitude=60.729549, \
                            false_easting=0.0, \
                            false_northing=0.0, \
                            standard_parallels=(50, 70.0), \
                            globe=None)

# Create figure and axis (you might want to edit this to focus on station coverage)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection=proj)
ax.set_extent([-135.3, -134.9, 60.60, 60.81]) 

# Add background
ax.add_image(request2, 13)
ax.add_image(request1, 13, alpha=0.5)

# Draw gridlines
gl1 = ax.gridlines(draw_labels=True, xlocs=np.arange(-136.0,-134.0,0.1), \
                                     ylocs=np.arange(60.0,61.0,0.1), linestyle = ":", color='w', zorder=2)

# Turn off labels on certin sides of figure
gl1.top_labels = False
gl1.right_labels = False

# Update label fontsize
gl1.xlabel_style = {'size': 10}
gl1.ylabel_style = {'size': 10}


# Example of plotting a station location
sta_lon = -135.076167
sta_lat = 60.729549
ax.plot(sta_lon, sta_lat, 'k', marker= "^", markersize=10, transform=ccrs.PlateCarree(), zorder = 9)

# Add scalebar
scale_bar(ax, proj, 2)

#Save figure
plt.savefig('test_map.png', dpi=300, bbox_inches='tight')
