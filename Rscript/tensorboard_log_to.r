#
# Script to convert tensorboard log from Mask R-CNN to a multiple plot in R
#

require(reshape2)

# Set path for tensorboard .csv files
filepath = 'F:/ArcGIS/Semester_Log/OUT_RGB_INT_DSM/tensorboard/'

# Find all .csv files in the file path
filenames <- list.files(path=filepath, pattern = '*.csv')

# Create a dataframe off all the .csv files
df <- lapply(filenames,function(i){
  i <- paste(filepath,i,sep="")
  read.csv(i, header=TRUE)
})

# Clean up names
filenames <- gsub("run_.-tag-","",filenames)
names(df) <- gsub(".csv","",filenames)

# Melt dataframe into long format
one_df <- melt(df, id.vars = c("Step", "Value"))

# Function to create uppper case for first charachter
firstup <- function(x) {
  substr(x, 1, 1) <- toupper(substr(x, 1, 1))
  x
}

# Set plot settings
par(mfrow = c(3,4))

# For loop to create plots for all 12 different variables
for (i in unique(one_df$L1)){

  # Subset dataframe based on unique value in column 'L1'
  dat <- one_df[one_df$L1 == i,]

  # Further clean up of names
  filenames <- gsub("_"," ",i)
  Capitalized <- firstup(filenames)

  # Plot each .csv file based on step and value
  plot(dat$Step,dat$Value,
       main = Capitalized,
       col=rgb(0.1, 0.5, 0.3, 0.3),
       pch = 19,
       xlab = 'Step',
       ylab = 'Value')

  # Add a line through all points
  lines(dat$Step,dat$Value)

  # Create X and Y variables for cleaner view in script
  x <- dat$Step
  y <- dat$Value

  # Add a exponential line to each plot
  f <- function(x,a,b) {a * exp(b * x)}
  fit <- nls(y ~ f(x,a,b), start = c(a=1, b=0))
  co <- coef(fit)
  curve(f(x, a=co[1], b=co[2]), add = TRUE, col="red", lwd=1)
}

# Print message
print("All done")

# Clean environment
rm(list=ls())
