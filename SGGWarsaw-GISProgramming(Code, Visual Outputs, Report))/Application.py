# Import the required libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as mat_pyplt
from matplotlib.colors import ListedColormap
import folium
import contextily as ctx
import numpy as np
import rasterio
from rasterio.plot import show_hist

# Loading the Cambodia Provinces data collected from DIVAGIS
cambodia_provinces = gpd.read_file("./Data/Inputs/KHM_adm/KHM_adm1.shp")
cambodia_provinces.plot(color='None', edgecolor='Grey').set_title('Cambodia Provinces') # Plotting the provinces

# Loading our Area of Interest
aoi_shape = gpd.read_file('./Data/Inputs/Aoi/PhnomPenh.shp')
aoi_shape.plot(color="None", edgecolor='Grey').set_title("Area of Interest: Phnom Penh, Cambodia") #Plotting the AOI Boundary


# Showing the location of AOI in Cambodia

# AOI and Cambodia Provinces are in different crs. So Making them same CRS by reprojecting
cambodia_provinces_reprojected = cambodia_provinces.to_crs(aoi_shape.crs)

# Creating a figure with two subplots side by side
fig, (ax1, ax2) = mat_pyplt.subplots(1, 2, figsize=(12, 6))

# Ploting the AOI shape in the first subplot
aoi_shape.plot(ax=ax1, facecolor='mediumpurple', edgecolor='Black')
ax1.set_title("Area of Interest: Phnom Penh")

# Ploting the provinces and AOI shape in the second subplot
cambodia_provinces_reprojected.plot(ax=ax2, facecolor='lightblue', edgecolor='Black')
aoi_shape.plot(ax=ax2, facecolor='mediumpurple', edgecolor='Yellow')
ax2.set_title("Phnom Penh Location in Cambodia")
mat_pyplt.suptitle("Area of Interest Boundary and Location")    # Adding a title for the whole plot
mat_pyplt.tight_layout()                                        # Adjusting the spacing between subplots
mat_pyplt.show()                                                # Displaying the plot

# Saving the plot as an image file
fig.savefig("./Data/Outputs/Area_of_Interest_Boundary_and_Location.png", #Saving Location
            dpi=300,  # dots per inch
            bbox_inches='tight',  # Adjust the bounding box to include the entire plot
            format='jpg',  # Specify the file format e.g., 'png', 'jpg', 'svg'
           )


# Reading the geopackage file with Khum(Level 3 Administrative Unit) information
cambodia_khums = gpd.read_file("./Data/Inputs/KHM_adm/KHM_adm3.gpkg", layer = 'KHM_adm3')
unique_khum_names = cambodia_khums.NAME_3.unique()              # finding unique Khums in the country
cambodia_total_khums = len(cambodia_khums.NAME_3.unique())      # Number of total khums in country

print("Total Number of Khums in Cambodia:",cambodia_total_khums)

print("Finding Khums only in Area of Interest")

# Ensure that the CRS matches, if not throw an exception and continue. This will help from code breaking in the middle
if aoi_shape.crs != cambodia_khums.crs:
    print('CRS Differs Between Layers. Trying to reproject...')
    
    try:
        cambodia_khums_reproj = cambodia_khums.to_crs(crs=aoi_shape.crs, epsg=None, inplace=False)
        print('Both layers converted to:',aoi_shape.crs)
    except:
        print('Reprojection Failed! Please Debug The Code')

# FInding the khums in AOI
aoi_khums = cambodia_khums_reproj.overlay(aoi_shape, how = 'intersection')

#Saving the khum in AOI info as Geopackage
aoi_khums.to_file('./Data/Outputs/Khums_in_Area_of_Interest.gpkg', driver='GPKG', layer='Khums_in_Area_of_Interest')  
aoi_khums_count = len(aoi_khums.NAME_3.unique())
print("Total Khums "+str(aoi_khums_count)+" Found in Area of Interest")


# Plotting the khums for whole country and khums in aoi
# Creating a figure with two subplots side by side
fig, (ax1, ax2) = mat_pyplt.subplots(1, 2, figsize=(10, 5))

# Ploting the cambodia_khums in the first subplot
cambodia_khums.plot(ax=ax1,facecolor= 'mediumpurple', edgecolor='yellow', linewidth=0.25)
ax1.set_title('Khums in Cambodia')

aoi_khums.plot(ax=ax2, facecolor='mediumpurple', edgecolor='yellow') # Plot the aoi_khums in the second subplot
ax2.set_title('Khums in Area of Interest')
mat_pyplt.tight_layout()                                             # Adjust the spacing between subplots, so that it doesnot break while saving
mat_pyplt.suptitle("Khums of Cambodia", fontweight='bold')           # Adding a title for the whole plot
mat_pyplt.subplots_adjust(top=0.85)                                 # otherwise breaks while saving

# Display the plot
mat_pyplt.show()
fig.savefig("./Data/Outputs/Khums_in_Area_of_Interest.png", #Saving Location
            dpi=300,  # dots per inch
            bbox_inches='tight',  # Adjust the bounding box to include the entire plot
            format='png',  # Specify the file format e.g., 'png', 'jpg', 'svg'
)


# Find the Khums only on Phnom Phn Main Municipality (Doing this just because I have to implement difference)
khums_in_phnom_munic =  aoi_khums[aoi_khums['NAME_2']=='Phnom Penh']
khums_outside_phnm_munic = gpd.overlay(aoi_khums, khums_in_phnom_munic, how = 'difference')

#Saving information on Khums only in Phonm Penh Main Municpality
khums_in_phnom_munic.to_file('./Data/Outputs/khums_in_phnom_munic.geojson', driver='GeoJSON') # Saving as GEOJSON

#Saving information on Khums only outside Phonm Penh Main Municpality
khums_outside_phnm_munic.to_file('./Data/Outputs/khums_outside_phnom_munic.shp')


# Create subplots with one row and two columns
fig, (ax1, ax2) = mat_pyplt.subplots(1, 2, figsize=(10, 5))

# Plot Khums inside Phnom Penh Main Municipality
ax1.set_title('Khums inside Phnom Penh Main Municipality')
aoi_shape.plot(ax=ax1, facecolor='White', edgecolor='Black')
khums_in_phnom_munic.plot(ax=ax1, facecolor='mediumpurple', edgecolor='Black')

# Plot Khums outside Phnom Penh Main Municipality
ax2.set_title('Khums outside Phnom Penh Main Municipality')
khums_outside_phnm_munic.plot(ax=ax2, facecolor='mediumpurple', edgecolor='Black')

# Adjust spacing between subplots and main title
mat_pyplt.subplots_adjust(top=0.85)

# Add a title for the whole plot
mat_pyplt.suptitle("Khums Inside and Outside Phnom Penh Main Municipality",fontweight='bold')

# Display the plots
mat_pyplt.tight_layout()
mat_pyplt.show()
fig.savefig("./Data/Outputs/Khums_Inside_and_Outside_Phnom_Penh_Main_Municipality.png", #Saving Location
            dpi=300,  # dots per inch
            bbox_inches='tight',  # Adjust the bounding box to include the entire plot
            format='png',  # Specify the file format e.g., 'png', 'jpg', 'svg'
)


# Now that we have all the khums inside the Phnom Penh main Municipality, we can get the border of municipality by 
# clipping total border of phnom penh state by khum data frame
phnom_penh_municipality_border = gpd.clip(aoi_shape, khums_in_phnom_munic, keep_geom_type=False)
khums_in_phnom_munic.to_file('./Data/Outputs/phnom_penh_municipality_border.shp')






# Importing the landuse data collected from field survey. Source: HNEE 
phnom_penh_munc_landuse = gpd.read_file("./Data/Inputs/inside_phnom_penh_land_use.geojson", driver='GeoJSON')
# first let's check the column names
print('The dataframe has the following columns:', phnom_penh_munc_landuse.columns.values)
print('The dataframe total rows:', phnom_penh_munc_landuse.shape[0])

# C_L2 columns hold the land use info. Let's check the unique values and their counts in C_L2
print('\nCounts of rows for each land use type:\n', phnom_penh_munc_landuse['C_L2'].value_counts())

# Combine the polygons of the same land use type
dissolved_phnom_penh_munc_landuse = phnom_penh_munc_landuse.dissolve(by="C_L2")
dissolved_phnom_penh_munc_landuse.reset_index(inplace=True)

# Print the counts of rows for each land use type after dissolving
print('\nCounts of rows for each land use type after dissolving:\n', dissolved_phnom_penh_munc_landuse['C_L2'].value_counts())

# We can not understand which code stands for what class, lets add the names for each class number
class_names = pd.read_csv('./Data/Inputs/CL2_NL2_Names.csv')
print(class_names)

# Adding the N_L2 values from class_names file to the dissolved class files 
dissolved_phnom_penh_munc_landuse = dissolved_phnom_penh_munc_landuse.merge(class_names, on = 'C_L2')
print('\n\n\n',dissolved_phnom_penh_munc_landuse.head(2))


# Right now the landuse data does not include which in which khum is the polygon located
# We want to add khum info to the polygons by doing spatial join
# CRS of both dataframes should be same
cambodia_khums_reprojected = cambodia_khums.to_crs(dissolved_phnom_penh_munc_landuse.crs)
dissolved_phnom_penh_munc_landuse_with_khum = dissolved_phnom_penh_munc_landuse.sjoin(cambodia_khums_reprojected, how="left")

# however, we dont need all columns
columns_to_drop = ["index_right", "ID_0", "ISO", "NAME_0", "ID_1", "NAME_1", "ID_2", "ID_3","TYPE_3", "ENGTYPE_3", "NL_NAME_3", "VARNAME_3" ]

# removing the unnecessary columns 
dissolved_phnom_penh_munc_landuse_with_khum = dissolved_phnom_penh_munc_landuse_with_khum.drop(columns=columns_to_drop)

# Define the desired colors for each land use class
colors = ['#27AE60', '#145A32', '#F1C40F', '#5499C7', '#F6DDCC','#58D68D' , '#E74C3C']

# Create a custom color map with the specified colors
color_map = ListedColormap(colors)

# Plot phnom_penh_munc_landuse with customized colors for each unique value in the N_L2 column
fig, ax = mat_pyplt.subplots(figsize=(10, 14))
dissolved_phnom_penh_munc_landuse_with_khum.plot(ax=ax, column='N_L2', cmap=color_map, edgecolor='black', linewidth=0.25, alpha=0.85, legend=True)

# Add a basemap from OpenStreetMap
ctx.add_basemap(ax, crs=dissolved_phnom_penh_munc_landuse_with_khum.crs.to_string())
# Set the title
ax.set_title('Land Use in Phnom Penh Municipality')
# Move the legend box outside of the main frame
legend = ax.get_legend()
legend.set_bbox_to_anchor((0.67, -0.02))
legend.set_title('Land Use Types')
# Show the plot
mat_pyplt.show()

# Save the figure with adjusted padding
fig.savefig("./Data/Outputs/Land_Use_in_Phnom_Penh_Municipality.png", bbox_inches='tight')


# Creating the the Folium Interactive map
m = folium.Map([11.55, 104.9], zoom_start=11)
# Define the desired colors for each class in C_L2
colors = {
    11000: 'DarkRed',
    12000: 'Red',
    13000: 'Yellow',
    14000: 'Green',
    51000: 'Blue',
    32000: 'darkpurple',
    31000: 'darkgreen'
}

# Add a GeoJSON layer for aoi_shape with a name
folium.GeoJson(aoi_shape, name="AOI Shape", tooltip='Phnom Penh Boundary').add_to(m)
# Add a GeoJSON layer for dissolved_phnom_penh_munc_landuse with customized styling and a name
folium.GeoJson(dissolved_phnom_penh_munc_landuse_with_khum, 
               style_function=lambda x: {
                   "fillColor": colors.get(x['properties']['C_L2'], 'gray'),
                   "color": "black",
                   "weight": 0.5,
                   "fillOpacity": 0.7
               },
               name="Land Use",
               tooltip=folium.GeoJsonTooltip(fields=['N_L2'], labels=False, sticky=True)).add_to(m)

# Adding the layer control to browse between the layers
folium.LayerControl().add_to(m)

# Display the map
m
# Saving the map as HTML
m.save("./Data/Outputs/InteractiveMap.html")



# Importing the raster bands for AOI
bands_path = './Data/Inputs/LC09_L2SP_126052_20230622_20230624_02_T1' # Defining the bands folder

b1 = rasterio.open(bands_path+'/B1.tif') # ultra blue, coastal aerosol
b2 = rasterio.open(bands_path+'/B2.tif') # blue
b3 = rasterio.open(bands_path+'/B3.tif') # green
b4 = rasterio.open(bands_path+'/B4.tif') # red
b5 = rasterio.open(bands_path+'/B5.tif') # near infrared
b6 = rasterio.open(bands_path+'/B6.tif') # shortwave infrared 1
b7 = rasterio.open(bands_path+'/B7.tif') # shortwave infrared 2
b10 = rasterio.open(bands_path+'/B10.tif')

all_bands = [b1,b2,b3,b4,b5,b6,b7,b10]


bands_data = [] # holds the bands main data array
stats = [] # holds the band statistics

# Checking the meta information
for band in all_bands:    
    band_data = band.read(1)
    bands_data.append(band_data)
    
    # calculate the stats
    band_stats = {'min': band_data.min(),
                  'mean': band_data.mean(),
                  'median': np.median(band_data),
                  'max': band_data.max()
                 }
    stats.append(band_stats)
    print("\nName:", band.name)
    # Check type of the variable 'raster'
    print("\tType:",type(band))
    # Projection
    print('\tProjection:',band.crs)
    # Dimensions
    print("\tWidth:", band.width, "Height:",band.height)
    # Number of bands
    print("\tCount of bands:", band.count)
    # Bounds of the file
    print("\tBounds of raster:", band.bounds)
    # Driver (data format)
    print("\tRaster File Type:",band.driver)
    # No data values for all channels
    print("\tNo Data Values: ", band.nodatavals)
    print("\tBand Statistic:",band_stats)
    # Affine transform (how raster is scaled, rotated, skewed, and/or translated) # will use this later for exporting
    print("\tAffine:",band.transform)

# Plotting the bands
fig, (ax1, ax2, ax3) = mat_pyplt.subplots(ncols=3, nrows=1, figsize=(10, 6), sharey=True)

# Ploting red, green, and blue bands
ax1.imshow(bands_data[3], cmap='Reds')
ax2.imshow(bands_data[2], cmap='Greens')
ax3.imshow(bands_data[1], cmap='Blues')

# Adding titles
ax1.set_title("Band4")
ax2.set_title("Band3")
ax3.set_title("Band2")
mat_pyplt.show()
# Save the figure with adjusted padding
fig.savefig("./Data/Outputs/Individual Bands (B4, B3, B2).png", bbox_inches='tight')


# Showing the histograms for each individual bands
num_bands = len(all_bands)
num_rows = (num_bands + 3) // 4
fig, axs = mat_pyplt.subplots(num_rows, 4, figsize=(25, 5 * num_rows))

for idx, band in enumerate(all_bands):
    rowidx = idx // 4
    colidx = idx % 4

    ax = axs[rowidx, colidx]

    # Plot the histogram of the band
    show_hist(band, ax=ax, bins=50, lw=0.0, stacked=False, alpha=0.8)

    # Set the band title
    ax.set_title(f"Band {idx + 1}")

# Add title
mat_pyplt.suptitle('Histogram of the Bands DN values',  fontsize=16)
# Adjust subplot spacing
mat_pyplt.tight_layout()
mat_pyplt.show()
# Saving the figure with adjusted padding
fig.savefig("./Data/Outputs/Bands_Histograms.png", bbox_inches='tight')


# The values are spread in a big range. we will normalize the bands
def normalize(array):
    """Normalizes numpy arrays into scale 0.0-1.0"""
    """Credit: Dr. Tracz class lecture codes"""
    
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

bands_data_normalized = []

for index, band in enumerate(bands_data):
    # Normalize the bands
    bandN = normalize(band)
    bands_data_normalized.append(bandN) # saving the normalized band to the normalized bands list


# Creating false color composite of nir, r, g bands
nrg = np.dstack((bands_data_normalized[4], bands_data_normalized[3], bands_data_normalized[2]))

# Creating false color composite of nir, r, g bands
rgb = np.dstack((bands_data_normalized[3], bands_data_normalized[2], bands_data_normalized[1]))

fig, axs = mat_pyplt.subplots(1, 2, figsize=(12, 6))

# Plot the first image (nrg) in the first subplot
axs[0].imshow(nrg)
axs[0].set_title('NIR-R-G Composite')

# Plot the second image (rgb) in the second subplot
axs[1].imshow(rgb)
axs[1].set_title('R-G-B Composite')

# Adding title
mat_pyplt.suptitle('False and True Color Composits',  fontsize=15)

# Adjusting spacing between subplots
mat_pyplt.tight_layout()

# Display the plot
mat_pyplt.show()

# Saving the figure with adjusted padding
fig.savefig("./Data/Outputs/False_and_True_Color_Composits.png", bbox_inches='tight')

meta = b4.meta
meta.update(count=3)
meta.update(driver='GTiff')
meta.update(dtype='float32')

# Defining the destination file for nrg
nrg_data = (bands_data_normalized[4], bands_data_normalized[3], bands_data_normalized[2])
nrg_composite_path = './Data/Outputs/nrg_composite.tif'
dst_nrg = rasterio.open(nrg_composite_path, 'w', **meta)

# Writing all bands to destination file
for index, band in enumerate(nrg_data):
    dst_nrg.write(band, index+1)

dst_nrg.close() # we have to close it as we are in writing mode, and kernel has lock on the file

# Defining the destination file for rgb
rgb_data = (bands_data_normalized[3], bands_data_normalized[2], bands_data_normalized[1])
rgb_composite_path = './Data/Outputs/rgb_composite.tif'
dst_rgb = rasterio.open(rgb_composite_path, 'w', **meta)

# Writing all bands to destination file
for index, band in enumerate(rgb_data):
    dst_rgb.write(band, index+1)
dst_rgb.close()

# Now calculate lst for the clipped area of Phnom Penh Municipality
# Change the reaction of numpy in order to not complain about dividing with zero values using np
np.seterr(divide='ignore', invalid='ignore')

# For LST we need Emmisivity, 
# for Emmisivity , we need Proportional Vegetation, 
# for Proportional Vegetation we need NDVI, 
# for NDVI We nee NIR and R bands

#Picking the required bands
nir_data = bands_data_normalized[4]
red_data = bands_data_normalized[3]
thermal_data = bands_data_normalized[7]


# Compute the NDVI for images
ndvi = (nir_data - red_data) /( nir_data + red_data)
min_ndvi = np.min(ndvi)
max_ndvi = np.max(ndvi)

print("Minimum NDVI:", min_ndvi)
print("Maximum NDVI:", max_ndvi)


#proportional vegetation
pvi = ((ndvi - min_ndvi) / (max_ndvi - min_ndvi)) ** 2

min_pvi = np.min(pvi)
max_pvi = np.max(pvi)

print("Minimum PVI:", min_pvi)
print("Maximum PVI:", max_pvi)

#Emissivity
a = 0.004
b = 0.986

emissivity = pvi * a + b
min_emissivity = np.min(pvi)
max_emissivity = np.max(pvi)

print("Minimum Emissivity:", min_emissivity)
print("Maximum Emissivity:", max_emissivity)


lst = (thermal_data / (1 + (10.8 * (thermal_data / 14388)) * np.log(emissivity))) - 273.15

min_lst = np.min(pvi)
max_lst = np.max(pvi)

print("Minimum LST:", min_lst)
print("Maximum LST:", max_lst)


# Create a figure with subplots
fig, axs = mat_pyplt.subplots(1, 4, figsize=(16, 4))

# Plot NDVI
im0 = axs[0].imshow(ndvi, cmap='RdYlGn')
axs[0].set_title('NDVI')
fig.colorbar(im0, ax=axs[0], label='NDVI', orientation='horizontal')

# Plot PVI
im1 = axs[1].imshow(pvi, cmap='Spectral')
axs[1].set_title('PVI')
fig.colorbar(im1, ax=axs[1], label='PVI', orientation='horizontal')

# Plot Emissivity
im2 = axs[2].imshow(emissivity, cmap='jet')
axs[2].set_title('Emissivity')
fig.colorbar(im2, ax=axs[2], label='Emissivity', orientation='horizontal')

# Plot LST
im3 = axs[3].imshow(lst, cmap='turbo')
axs[3].set_title('LST')
fig.colorbar(im3, ax=axs[3], label='LST (°C)', orientation='horizontal')

# Add supertitle
fig.suptitle('Vegetation Indices and LST', fontsize=16, fontweight='bold')

# Adjust spacing between subplots
mat_pyplt.tight_layout()

# Save the figure
fig.savefig('./Data/Outputs/Indices.png')

# Show the plot
mat_pyplt.show()


# Define the output file path
lst_output_path = './Data/Outputs/lst.tif'

# Define the CRS and transformation parameters
crs = b4.crs
transform = b4.transform


# Write the ndvi_change array to a TIFF file with CRS, transform, and metadata
with rasterio.open(lst_output_path, 'w', driver='GTiff', height=lst.shape[0], width=lst.shape[1], count=1, dtype=lst.dtype, crs=crs, transform=transform, nodata=None) as dst:
    dst.write(lst, 1)
