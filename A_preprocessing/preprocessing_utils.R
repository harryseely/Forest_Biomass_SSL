library(dplyr)
library(lidRmetrics)

check_df_for_nas <- function(df, col_names=c(), col_patterns=c(), quiet=F, return_col_names=F){

  if(length(col_names)>0){
    df <- df %>% select(all_of(col_names))

  } else if(length(col_patterns)>0){
    df <- df %>% select(contains(col_patterns))

  } else{
    if(!quiet){
      print("Checking all columns for NAs.")
    }
  }

  #Check by column for NAs
  col_check <- colSums(is.na(df))

  na_presence <- sum(col_check) > 0

  if(na_presence){
    na_df <- col_check[col_check > 0]

    print(paste("There are", nrow(na_df), "NaNs present in", names(na_df), "column"))

    if(return_col_names){

      return(na_df)

    }else{stop("Stopping code due to NAs.")}

  }else{

    if(!quiet){
      print("No NaNs in specified cols.")

    }
  }

}

get_biomass <- function(a, b, c, dbh, height, use_height=FALSE){ #DBH in cm, height in meters

  if(use_height){

      Biomass_kg = a * dbh^b * height^c

  } else{

      Biomass_kg <- a * (dbh^b)
  }

  return(Biomass_kg)
}

get_tree_comp_biomass <- function(tree_df, use_height=FALSE){

  if(!use_height){
    tree_df['height'] <- NA
    tree_df['c_Bark'] <- NA
    tree_df['c_Wood'] <- NA
    tree_df['c_Foliage'] <- NA
    tree_df['c_Branches'] <- NA
  }
  
  tree_df['bark_biomass'] <- get_biomass(
                                         a=tree_df['a_Bark'], 
                                         b=tree_df['b_Bark'], 
                                         c=tree_df['c_Bark'],
                                         dbh=tree_df['dbh'],
                                         height=tree_df['height'],
                                         use_height=use_height)

  tree_df['wood_biomass'] <- get_biomass(
                                         a=tree_df['a_Wood'], 
                                         b=tree_df['b_Wood'], 
                                         c=tree_df['c_Wood'],
                                         dbh=tree_df['dbh'],
                                         height=tree_df['height'],
                                         use_height=use_height)

  tree_df['foliage_biomass'] <- get_biomass(
                                            a=tree_df['a_Foliage'], 
                                            b=tree_df['b_Foliage'], 
                                            c=tree_df['c_Foliage'],
                                            dbh=tree_df['dbh'],
                                            height=tree_df['height'],
                                            use_height=use_height)

  tree_df['branches_biomass'] <- get_biomass(
                                             a=tree_df['a_Branches'], 
                                             b=tree_df['b_Branches'], 
                                             c=tree_df['c_Branches'],
                                             dbh=tree_df['dbh'],
                                             height=tree_df['height'],
                                             use_height=use_height)
  
  tree_df$tree_AGB <- tree_df$wood_biomass +
    tree_df$bark_biomass +
    tree_df$branches_biomass + 
    tree_df$foliage_biomass
  
  return(tree_df)
  
}

get_z <- function(x, mn, sd){
  Z <- (x - mn)/sd
  return(Z)}

convert_to_z_score <- function(vector){

  mn <- mean(vector)
  sd <- sd(vector)

  z_scores <- unlist(lapply(vector, FUN = get_z, mn=mn, sd=sd))

  return(z_scores)

}

custom_metrics <- function(x, y, z, z_percentiles=FALSE){
  
  m_basic  <- lidRmetrics::metrics_basic(z=z)
  
  m_vox <- metrics_voxels(x=x, y=y, z=z, vox_size = 1)
  
  m_lad <- metrics_lad(z=z)
  
  m_dens <- metrics_canopydensity(z=z, interval_count=10)

  m_percabv <- metrics_percabove(z=z, threshold = c(2, 5))

  m_disper <- metrics_dispersion(z=z, dz = 1)

  m_interv <- metrics_interval(z=z, zintervals = c(0, 0.15, 2, 5, 10, 20, 30))

  m <- c(m_basic, m_vox, m_lad, m_dens, m_percabv, m_disper, m_interv)

  #Add percentiles if requested
  if(z_percentiles){
    m_percentiles <- lidRmetrics::metrics_percentiles(z=z)
    m <- c(m, m_percentiles)
  }

  return(m)
}

#Define function to remove points from the val/test dataset that are closest to the train cluster centroid
rm_pts_close_to_train <- function(clust_pts, split, target_n_pts, t_centroid){

    #Reduce to target split
    clust_pts_subset <- clust_pts %>%
                        filter(cluster == split)    

    #In the subset cluster, get the current number of points
    n_current <- nrow(clust_pts_subset)

    #In the val dataset get the distance of each point to the train dataset center
    clust_pts_subset['dist_to_train_cent'] <- st_distance(x = clust_pts_subset, 
                                                          y = t_centroid)

    #Re-order subset by distance to train centroid and take the target_n_pts closest points
    clust_pts_subset <- clust_pts_subset %>% 
                    arrange(desc(dist_to_train_cent)) %>%
                    slice_head(n = target_n_pts)
    
    return(clust_pts_subset)
}

#Define function to perform spatial clustering for train, val, and test sets
spatial_split <- function(plots, n_train, n_val, n_test, kmeans_seed){

    #Only need the PlotID column
    plots <- plots %>% select(PlotID)

    # Extract coordinates from the sf points data frame
    coords <- st_coordinates(plots)

    # Apply k-means clustering to the coordinates
    set.seed(kmeans_seed)
    kmeans_result <- kmeans(coords, centers = 3)

    # Assign the cluster labels back to the sf data frame
    plots$cluster <- as.factor(kmeans_result$cluster)

    # Associated each cluster with the train, val, and test sets
    plots <- plots %>%
      mutate(cluster = case_when(cluster == 1 ~ 'train',
                                cluster == 2 ~ 'val',
                                cluster == 3 ~ 'test'))
    
    #Get the train cluster centroid
    train_centroid <- plots %>% 
            filter(cluster == "train") %>%
            st_coordinates() %>%
            as.data.frame() %>%
            summarise(x = mean(X), y = mean(Y)) %>%
            st_as_sf(coords = c("x", "y"), crs = st_crs(plots))
    
    #Reduce the number of points in the val and test datasets
    val_clust_pts <- rm_pts_close_to_train(clust_pts = plots, 
                                            split = "val", 
                                            target_n_pts = n_val, 
                                            t_centroid = train_centroid)

    test_clust_pts <- rm_pts_close_to_train(clust_pts = plots, 
                                            split = "test", 
                                            target_n_pts = n_test, 
                                            t_centroid = train_centroid)

    #Re-Allocated points to clusters after dropping points
    plots <- plots %>%
                    mutate(cluster = if_else(PlotID %in% val_clust_pts$PlotID, "val",
                                    if_else(PlotID %in% test_clust_pts$PlotID, "test", "train")))
    
    #Return order of clusters matching the input order
    clusters_vec <- plots$cluster 

    return(clusters_vec)

}