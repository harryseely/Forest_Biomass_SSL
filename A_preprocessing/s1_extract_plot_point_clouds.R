# Get packages
library(lidR)
library(here)
library(sf)
library(dplyr)

source("preprocessing/preprocessing_utils.R")

#Read the las catalog
rmf_lidar_dir <- "E:/RMF/RMF_SPL100/LAS_Classified_Point_Clouds_Normalized"
ctg <- readLAScatalog(rmf_lidar_dir)

#Read the plot coordinates and ensure CRS is correct
plot_coords <- st_read(here("data/romeo_malette/plot_locations/MasterPlotLocationUTM_December6.shp")) %>%
    st_transform(lidR::crs(ctg))

#Buffer plots by radius
plots_buffered <- st_buffer(plot_coords, dist = 11.28)

#Read the study area boundary
aoi <- st_read(here("data/romeo_malette/romeo_malette_boundary.gpkg")) %>%
    st_transform(st_crs(plot_coords))

#View plot coordinates
plot(st_geometry(aoi), col = "grey")
plot(st_geometry(plot_coords), add = TRUE, col = "red", cex = 0.3)

#Set output dir for clipped point clouds
output_dir <- here("data/plot_point_clouds")

metrics_df_ls <- list()

start_time <- Sys.time()

#Extract the point clouds for each plot
for(i in 1:nrow(plots_buffered)){
    
    #Subset plot and extract geometry
    plot_id <- plots_buffered$PlotID[i]
    plot_geom <- plots_buffered[plots_buffered$PlotID == plot_id,]
    
    #Clip las
    las_clipped <- lidR::clip_roi(ctg, plot_geom)
    
    #Get metrics
    metrics_i <- lidR::cloud_metrics(las_clipped, func = custom_metrics(x=X, y=Y, z=Z, z_percentiles = TRUE))
    
    #Convert metrics to df with plot id
    metrics_i$plot_id <- plot_id
    metrics_df_i <- as.data.frame(metrics_i)
    
    #Store metrics df in list
    metrics_df_ls[[i]] <- metrics_df_i
    
    #Write clipped las
    out_fname <- paste0("plot_", plot_id, ".las")
    writeLAS(las_clipped, file.path(output_dir, out_fname))

    print(paste0("Extracted point cloud for plot ", plot_id))

    
}

#Combine metrics dfs
metrics_df <- do.call(rbind.data.frame, metrics_df_ls)

#Export
write.csv(metrics_df, file = here("data/labeled_plot_metrics.csv"), row.names = F)

# Report runtime and summarize session
end_time <- Sys.time()

elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
elapsed_hours <- floor(elapsed_time / 3600)
elapsed_minutes <- floor((elapsed_time %% 3600) / 60)
elapsed_seconds <- elapsed_time %% 60

cat("Elapsed time:", elapsed_hours, "hours,", elapsed_minutes, "minutes,", elapsed_seconds, "seconds.")