#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 08:45:36 2024

@author: williamtaylor
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
plt.style.use('default')
import json
import csv
from descartes import PolygonPatch
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.dates as mdates
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
# general flow:
# isolate flood values (by timestep)
# calculate mean and std values of the dataset
# calculate z-score
# calculate percentile
# translate to return

# longer record from Brett Whitin
with open('data/ordc1f_2019.qme', 'r') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    
    # Skip the header if it exists
    next(reader)  # Skip the header row if it exists
    
    rows_data = []
    # Iterate over each row in the CSV file
    for row in reader:
        
        columns = row[0].split() # split by whitespace
        
        rows_data.append(columns)
        

df = pd.DataFrame(rows_data[6:], columns=['Site','MMYY','Day','Inflow (cfs)']) # skip the first 6 rows which contain header info

# Split 'MMYY', 'Day', and 'Inflow (cfs)' columns by dash if necessary
for col in ['MMYY', 'Day', 'Inflow (cfs)']:
    # Find columns containing dashes
    mask = df[col].str.contains('-')
    if mask.any():
        # Split columns by dash
        df.loc[mask, col] = df.loc[mask, col].str.split('-').str[-1]


# deal with some of the inflow values that got moved to the 'Day' column
moved_inflow_rows = df['Day'].str.contains('\d+.\d+')
df.loc[moved_inflow_rows, 'Inflow (cfs)'] = df.loc[moved_inflow_rows, 'Day']

problem = df[moved_inflow_rows]
#print(problem)
# replacing these days manually
df.loc[13551,'Day'] = '07'
df.loc[13844,'Day'] = '27'
df.loc[13845,'Day'] = '28'
df.loc[13924,'Day'] = '15'
df.loc[14254,'Day'] = '11'
df.loc[14258,'Day'] = '15'
df.loc[14578,'Day'] = '30'
df.loc[14585,'Day'] = '06'
df.loc[14587,'Day'] = '08'
df.loc[14595,'Day'] = '16'
df.loc[14612,'Day'] = '03'
df.loc[14656,'Day'] = '16'
df.loc[14677,'Day'] = '07'
df.loc[21160,'Day'] = '07'
df.loc[21163,'Day'] = '10'
df.loc[21455,'Day'] = '29'

# convert Inflow to float values
df['Inflow (cfs)'] = df['Inflow (cfs)'].astype(float)
df['Day'] = df['Day'].astype(str)

# ensure the MMYY column has leading zeros for the months
df['MMYY'] = df['MMYY'].apply(lambda x: x.zfill(4))
df['Day'] = df['Day'].str.zfill(2)

# fix dates
df['Date'] = df['MMYY'] + df['Day']
df['Date'] = pd.to_datetime(df['Date'], format='%m%y%d')

start_date = pd.to_datetime('1960-10-01')
df['Date'] = start_date + pd.to_timedelta(df.index, unit='D')

# make the date the index
df.set_index('Date', inplace=True)

df.drop(columns=['Site','MMYY','Day'])
df['Year'] = df.index.year
annual_max = df.groupby('Year')['Inflow (cfs)'].max()



#%% TOCS plot
import datetime
import matplotlib.dates as mdates

colors = sns.color_palette('colorblind')

def get_tocs(x, d):
  tp = [0, x[1], x[2], x[3], 366]
  sp = [1, x[4], x[4], 1, 1]
  return np.interp(d, tp, sp)

days = np.arange(0,366,1)
params = json.load(open('data/params.json'))
tocs = get_tocs(params['ORO'], days)

# convert days to datetime object
start_date = datetime.datetime(2000,10,1)
dates = [start_date + datetime.timedelta(days=int(d)) for d in days]

# tocs line
plt.figure(figsize=(6,4))
plt.plot(dates, tocs*3524, linestyle='--', c='grey')

# lines
plt.axhline(3524, c='grey',linestyle='--')
plt.axhline(3850, c='black',linestyle='solid')

# text
plt.annotate('Top of Dam', xy=(dates[20],3900), fontweight='bold', fontsize=12)
plt.annotate('Surcharge Pool', xy=(dates[20],3600), fontweight='bold', fontsize=12)
plt.annotate('Flood Pool', xy=(dates[20],2880), fontweight='bold', fontsize=12)
plt.annotate('Conservation Pool', xy=(dates[20],2500), fontweight='bold', fontsize=12)

# fill
bottom = np.zeros(366)
plt.fill_between(dates, bottom, tocs*3524, color=colors[0], alpha=0.5)
plt.fill_between(dates, tocs*3524, 3524, color=colors[1], alpha=0.5)
plt.fill_between(dates, 3524, 3850,color=colors[2], alpha=0.5)

# flood visualization
flood = 616.85 # peak one day volume for 1997 flood
flood_start = np.min(tocs) * 3524
flood_end = flood_start + flood
flood_day = datetime.datetime(2001,2,15)
plt.annotate("",
             xy = (flood_day, flood_end), xytext=(flood_day, flood_start),
             arrowprops=dict(facecolor='black', edgecolor='black', linewidth=2, arrowstyle='<->'))
plt.annotate('Peak single day volume\n1997 flood', xy=(dates[143], 3250), fontsize=10)

# format axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.ylabel('Storage (TAF)')
plt.ylim(2000,4000)
plt.xlim(dates[0], dates[-1])
plt.tight_layout()
plt.show()


#%%
from matplotlib.patches import FancyArrow, Rectangle

fig = plt.figure()
fig.set_size_inches([8, 6])

# map plot
gs1 = GridSpec(1, 1, left = 0.01, right = 0.50)
ax1 = fig.add_subplot(gs1[0])

# Create a Basemap instance
m = Basemap(llcrnrlat=38.3, urcrnrlat=40.7, llcrnrlon=-122, urcrnrlon=-120, resolution='i', ax=ax1)
m.arcgisimage(service='World_Shaded_Relief')

m.drawrivers(color='dodgerblue', linewidth=1.0, zorder=1)
m.drawcountries(color='k', linewidth=1.25)
m.drawstates()

# import the shapefiles
gdf = gpd.read_file('shapefiles/HUC8_US.shp')
selected_hucs = ['18020121', '18020122', '18020123']
filtered_gdf = gdf[gdf['HUC8'].isin(selected_hucs)]

#filtered_gdf.plot(figsize=(10, 10), edgecolor='black', cmap='Blues')

# Plot the shapefile data
for geom in filtered_gdf['geometry']:
    if geom.geom_type == 'Polygon':
        coords = [m(x,y) for x,y in geom.exterior.coords]
        patch = PolygonPatch(geom, edgecolor='green', facecolor='green', alpha=1, zorder=2)
        ax1.add_patch(patch)
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom:
            coords = [m(x,y) for x,y in poly.exterior.coords]
            patch = PolygonPatch(poly, edgecolor='green', facecolor='green', alpha=1, zorder=2)
            ax1.add_patch(patch)

# Oroville Marker
nodes = json.load(open('shapefiles/nodes.json'))
x,y = m(nodes['ORO']['coords'][0],nodes['ORO']['coords'][1])
m.scatter(x,y, facecolor='black', edgecolor='black', s=170, marker='^', zorder=3)
ax1.text(x-0.15, y-0.1, 'Oroville Reservoir', fontsize=9, ha='center', va='center', color='k', zorder=3)

# Sacramento Marker
x,y = m(-121.4944, 38.5816)
m.scatter(x,y, facecolor='red', edgecolor='black', s=170, marker='*')
ax1.text(x-0.1, y+0.1, 'Sacramento', fontsize=9, ha='center', va='center', color='k')

# plot inset of greater geographical area
axins = zoomed_inset_axes(ax1, 0.03, loc='upper left')
axins.set_xlim(-124,-120)
axins.set_ylim(36,42)

# scale bar
scalebar_length_km = 50
lon_ref = -120.5
lat_ref = 38.4
km_per_degree = 111
scalebar_length_deg = scalebar_length_km / km_per_degree
x1, y1 = m(lon_ref, lat_ref)
x2, _ = m(lon_ref + scalebar_length_deg, lat_ref)

height = (m.urcrnrlat - m.llcrnrlat) * 0.005


scalebar = Rectangle((x1, y1), x2-x1, height, edgecolor='black', facecolor='black', linewidth=2)
ax1.add_patch(scalebar)
ax1.text((x1 + x2) / 2, y1 + height * 2, f'{scalebar_length_km} km', fontsize=8, ha='center', color='k')


# north arrow
# ax1.annotate(
#     'N', xy=(0.92, 0.85), xytext=(0.92,0.93),
#     arrowprops=dict(facecolor='black', edgecolor='black', width=3, headwidth=10),
#     ha='center', va='center', fontsize=12, fontweight='bold', color='k', xycoords='axes fraction'
# )

# Define arrow position (adjust if needed)
lon_arrow = -120.28  # Longitude of arrow base
lat_arrow = 38.6    # Latitude of arrow base

# Convert lat/lon to map coordinates
x, y = m(lon_arrow, lat_arrow)

# Define arrow properties (length, direction)
arrow_length = (m.urcrnrlat - m.llcrnrlat) * 0.05  # 10% of map height
dx, dy = 0, arrow_length  # dx=0 (no horizontal movement), dy=upward movement

# Add north arrow
north_arrow = FancyArrow(x, y, dx, dy, width=arrow_length * 0.3, head_width=arrow_length * 0.5, 
                         head_length=arrow_length * 0.5, color='black')
ax1.add_patch(north_arrow)

ax1.text(x, y + arrow_length * 1.5, 'N', fontsize=12, ha='center', va='bottom', fontweight='bold')


# add basemap to inset map
m2 = Basemap(llcrnrlat=29, urcrnrlat=43.5, llcrnrlon=-125, urcrnrlon=-114, resolution='i', ax=axins)
m2.arcgisimage(service='World_Shaded_Relief')
m2.drawstates(color='k',linewidth=0.5)
m2.drawcountries(color='k', linewidth=0.5)
for geom in filtered_gdf['geometry']:
    if geom.geom_type == 'Polygon':
        patch = PolygonPatch(geom, facecolor='g', edgecolor='g', alpha=1, zorder=2)
        axins.add_patch(patch)
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom:
            x, y = poly.exterior.xy
            m2.plot(x, y, marker=None, color='m')

# remove  tick marks
plt.xticks(visible=False)
plt.yticks(visible=False)

ax1.set_title('(a) Location overview', loc='left', fontweight='bold', fontsize=12)
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# current encroachment rules
# Step 2: Create a figure and gridspec layout
gs2 = GridSpec(2, 1, left=0.58, right=1, hspace=0.35)
ax2 = fig.add_subplot(gs2[1])

ax2.plot(dates, tocs*3524, linestyle='--', c='grey')

# lines
ax2.axhline(3524, c='grey',linestyle='--')
ax2.axhline(3850, c='black',linestyle='solid')

# text
ax2.annotate('Top of Dam', xy=(dates[5],3900), fontsize=10)
ax2.annotate('Surcharge Pool', xy=(dates[5],3600), fontsize=10)
ax2.annotate('Flood Pool', xy=(dates[5],2880), fontsize=10)
ax2.annotate('Conservation Pool', xy=(dates[5],2500), fontsize=10)

# fill
bottom = np.zeros(366)
ax2.fill_between(dates, bottom, tocs*3524, color=colors[0], alpha=0.5)
ax2.fill_between(dates, tocs*3524, 3524, color=colors[1], alpha=0.5)
ax2.fill_between(dates, 3524, 3850,color=colors[2], alpha=0.5)

# flood visualization
flood = 616.85 # peak one day volume for 1997 flood
flood_start = np.min(tocs) * 3524
flood_end = flood_start + flood
flood_day = datetime.datetime(2001,2,1)
ax2.annotate("",
             xy = (flood_day, flood_end), xytext=(flood_day, flood_start),
             arrowprops=dict(facecolor='black', edgecolor='black', linewidth=2, arrowstyle='<->'))
ax2.annotate('Peak single day\ninflow volume\n(1997 flood)', xy=(dates[130], 3100), fontsize=8)

# format axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.set_ylabel('Storage (TAF)')
ax2.set_ylim(2000,4050)
ax2.set_xlim(dates[0], dates[-1])
ax2.set_title('(c) Existing seasonal rule curve', fontweight='bold', loc='left', fontsize=12)


ax3 = fig.add_subplot(gs2[0])
ax3.plot(annual_max/1000, c='blue')
ax3.set_title('(b) Annual maximum daily inflow', loc='left', fontweight='bold', fontsize=12)
ax3.set_xlabel('')
ax3.set_ylabel('Inflow (kcfs)')
ax3.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('/Users/williamtaylor/Documents/Github/Robust-FIRO/figures/figure_1.pdf', format='pdf', bbox_inches='tight', transparent=False)
plt.show()


