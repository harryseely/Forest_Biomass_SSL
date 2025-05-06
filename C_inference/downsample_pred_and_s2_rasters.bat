@echo off

set TR=200

:: Define input files
set INPUT_FILE1="E:/rq3_rmf_inference/tile_predictions_clean.tif"
set INPUT_FILE2="E:/RMF/rmf_s2_rgb_cloud_free_comp_epsg6600.tif"

:: Define output files
set OUTPUT_FILE1="E:/rq3_rmf_inference/tile_predictions_clean_200m.tif"
set OUTPUT_FILE2="E:/rq3_rmf_inference/rmf_s2_rgb_cloud_free_comp_epsg6600_200m.tif"

:: Process first file
echo Processing file: %INPUT_FILE1%
echo Output file: %OUTPUT_FILE1%
echo Target resolution: %TR% m

gdalwarp -tr %TR% %TR% -overwrite -of GTiff "%INPUT_FILE1%" "%OUTPUT_FILE1%"

echo Resampling completed for %INPUT_FILE1%

:: Process second file
echo Processing file: %INPUT_FILE2%
echo Output file: %OUTPUT_FILE2%
echo Target resolution: %TR% m

gdalwarp -tr %TR% %TR% -overwrite -of GTiff "%INPUT_FILE2%" "%OUTPUT_FILE2%"

echo Resampling completed for %INPUT_FILE2%

echo All files processed.