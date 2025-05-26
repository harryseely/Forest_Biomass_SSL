@echo off

set TR=22.56
set SRC_NODATA=-3.4e+38
set DST_NODATA=-3.4e+38
set INPUT_FILE="E:/RMF/LiDAR Summary Metrics/CHM.tif"
set OUTPUT_FILE="E:/RMF/LiDAR Summary Metrics/CHM_22.56m_resampled.tif"
set TARGET_SRS="EPSG:6660"

gdalwarp -tr %TR% %TR% -overwrite -of GTiff -srcnodata %SRC_NODATA% -dstnodata %DST_NODATA% %INPUT_FILE% %OUTPUT_FILE%