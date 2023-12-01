## Global_sustainable_energy
### Summary
The provided R code conducts a thorough analysis of global sustainable energy data spanning from 2000 to 2020, sourced from Kaggle. The code encompasses essential tasks such as data cleaning, visualization, correlation analysis, and developing a linear regression model for predicting CO2 emissions in Nepal.

### Important
Most output could be generated using simple functions, But I have used loops to understand better
key concepts of reproducible programming.

### the format of files folders is:
```
├───figs
├───raw_data
└───rendered_document_files
    ├───figure-html
    └───libs
        ├───bootstrap
        ├───clipboard
        ├───htmlwidgets-1.6.2
        ├───jquery-1.12.4
        ├───leaflet-1.3.1
        │   └───images
        ├───leaflet-binding-2.1.2
        ├───leafletfix-1.0.0
        ├───proj4-2.6.2
        ├───Proj4Leaflet-1.0.1
        ├───quarto-html
        └───rstudio_leaflet-1.3.1
            └───images
```
## Packages used 
 -library(tidyverse)
- library(here)
- library(janitor)
- library(rnaturalearth)
- library(sf)
- library(kableExtra)

### Key Steps:

1. **Data Cleaning:**
   - The code reads energy data from a CSV file, checks for duplicate rows, and ensures correct data types for each column.
   - Column names are shortened to enhance model interpretation.

2. **Visualization:**
   - The code maps global energy data onto a world map using the `rnaturalearth` and `sf` libraries.
   - Correlation analysis is performed to understand associations between variables.

3. **Country-Specific Analysis (Nepal):**
   - Nepal's energy data is isolated, and trends for various variables are visualized over the years.
   - Columns with all NA values and redundant data are removed.

4. **Regression Modeling:**
   - A linear regression model is built to predict CO2 emissions in Nepal.
   - Step-wise regression is performed to select significant predictors.
   - Feature selection results in six significant variables influencing CO2 emissions.
   - The dataset is split into training and testing sets, and a linear regression model is fitted on the training data.

5. **Model Evaluation:**
   - The accuracy of the model is assessed using Mean Absolute Error (MAE), normalized for better interpretation.
   - The code also includes a forecast using a naive method and evaluates its accuracy.

6. **Summary and Documentation:**
   - The code provides a comprehensive summary of the entire analysis, making it suitable for inclusion in a `readme.md` file.
   - Key findings, model accuracy, and forecasting results are highlighted.

This code serves as a robust framework for exploring, analyzing, and modeling sustainable energy data, specifically tailored for the case of Nepal. 

