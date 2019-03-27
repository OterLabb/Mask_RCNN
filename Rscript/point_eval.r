#
# Script to see some statistics for how many evaluation points are predicted right
#

library(arcgisbinding)
arc.check_product()

points <- arc.open("C:/Users/Oddbjorn/Documents/MASTER/Log_semester/point_and_detected.gdb/points")
detected <- arc.open("C:/Users/Oddbjorn/Documents/MASTER/Log_semester/point_and_detected.gdb/detected")

require(sp)
library(sf)

points.dataframe <- arc.select(object = points, fields = c("OBJECTID", "Shape"))
polygon.dataframe <- arc.select(object = detected, fields = c("OBJECTID", "Confidence"))

points.sp.df <- arc.data2sp(points.dataframe)
polygon.sp.df <- arc.data2sp(polygon.dataframe)

# Check coordinate system
crs1 <- proj4string(points.sp.df)
crs2 <- proj4string(polygon.sp.df)

if (crs1 != crs2) {
  print("CRS is not the same, transforming to the same crs")
  crs.new <- CRS(crs1)
  polygon.sp.df <- spTransform(polygon.sp.df, crs.new)
  points.sp.df <- spTransform(points.sp.df, crs.new)
} else {
  print("CRS is the same for both inputs")
}

# Convert to sf
points = st_as_sf(points.sp.df)
polygon = st_as_sf(polygon.sp.df)

# Intersecting points with polygon
results <- st_intersection(points, polygon)

# Calculate percent hit ratio
percent <- length(results$OBJECTID) / length(points.sp.df) * 100
cat(percent, "percent of the logs were predicted right")
cat(mean(results$Confidence), "mean confidence of the correctly predicted logs")
mean_intersected <- mean(results$Confidence)

# Histogram
hist(results$Confidence,
     breaks = 20,
     col = rgb(160,160,160, maxColorValue=255),
     xlab = 'Confidence',
     ylab = 'Frequency',
     main = 'Confidence of predicted resulst'
     )
# Add mean vertical line
abline(v = mean(results$Confidence), col="red", lwd=3, lty=2)

# Subset all other points, that were not intersected
require(Hmisc)
z <- subset(polygon, OBJECTID %nin% results$OBJECTID.1)
mean <- mean(z$Confidence)
std <- sd(z$Confidence)

# Histogram
hist(z$Confidence,
     breaks = 20,
     col = rgb(160,160,160, maxColorValue=255),
     xlab = 'Confidence',
     ylab = 'Frequency',
     main = 'Predicted logs with confidence values'
)
# Add mean vertical line
abline(v = mean(z$Confidence), col="red", lwd=3, lty=2)

# Clear all
rm(list=ls())
