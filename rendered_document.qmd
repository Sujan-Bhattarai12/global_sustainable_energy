---
title: "Global Data on Sustainable Energy (2000-2020)"
author: "Sujan Bhattarai"
format: html
editor: visual
---

The required data is download from the Kaggle

(link: https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy)

# **Potential Use cases**

1.  **Energy Consumption Prediction:** Predict future energy usage, aid planning, and track SDG 7 progress.

2.  **Carbon Emission Forecasting:** Forecast CO2 emissions, support climate strategies.

3.  **Energy Access Classification:** Categorize regions for infrastructure development, understand sustainable energy's role.

4.  **Sustainable Development Goal Tracking:** Monitor progress towards Goal 7, evaluate policy

Load required libraries

```{r  message=FALSE, warning=FALSE}
library(tidyverse)
library(here)
library(janitor)
library(rnaturalearth)
library(sf)
library(kableExtra)
```

Read the data, view first few columns, and sum the NA values for each column

```{r  message=FALSE, warning=FALSE}
energy_data <- read_csv(here("raw_data","energy_data.csv"))
head(energy_data)
data.frame(total_NA_count = colSums(is.na(energy_data)))
```

Check duplicate rows if present

```{r}
sum(duplicated(energy_data)) 
```

Check the class of each columns

```{r}
##build function to check columns class iteratively
for (i in seq_along(energy_data)){
  print(class(energy_data[[i]]))
}
```

Data-type is in correct format except for country. Change the column type. (This code should change data type and it is vital to run this before moving to next step)

```{r}
for (i in seq_along(energy_data)){
  if(class(energy_data[[i]]) == "character"){
    energy_data[[i]] = as.factor(energy_data[[i]])
  }else
    energy_data[[i]] = as.numeric(energy_data[[i]])
}
```

Shorten column names from energy_data. Short names are better in models and corr-plot.

```{r}
energy_data1 <- energy_data %>%
  rename( popElectric = "Access to electricity (% of population)",
          cleanfuels = "Access to clean fuels for cooking",
          renewable_elec_cap =  "Renewable-electricity-generating-capacity-per-capita",
          finance_flow = "Financial flows to developing countries (US $)",
          renewable_inTotal= "Renewable energy share in the total final energy consumption (%)",
          fossil_eng= "Electricity from fossil fuels (TWh)",
          n_elece = "Electricity from nuclear (TWh)",
          electric = "Low-carbon electricity (% electricity)",
          electric_re = "Electricity from renewables (TWh)",
          energy_cap = "Primary energy consumption per capita (kWh/person)",
          energy_in = "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
          co2_emis = "Value_co2_emissions_kt_by_country",
          renewables = "Renewables (% equivalent primary energy)",
          gdp = "gdp_growth",
          gdp_cap = "gdp_per_capita",
          pop_density = "Density\\n(P/Km2)",
          land_area = "Land Area(Km2)",
          Lat = "Latitude",
          Lon = "Longitude")
```

Map the data in worldmap to see the change

```{r}
world_map <- ne_countries(returnclass = "sf", scale = "medium")
map <- world_map %>% 
       ggplot() +
       geom_sf()
```

Join the raw energy data with the global data based on ID so that data can be integrated in world map

```{r  message=FALSE, warning=FALSE}
joined <- right_join(world_map,energy_data1,
                     by=c("sovereignt"="Entity")) %>% 
          filter(Year==2020)

for (i in which(colnames(joined)=="Year"): which(colnames(joined)=="pop_density")){
  maps <- world_map %>% 
    st_transform(crs="+proj=robin") %>% 
    ggplot()+
    geom_sf(color="darkgrey")+
    geom_sf(data=joined, aes(fill=joined[[i+1]]))+
    xlab(colnames(joined[i+1][1]))+
    theme(legend.title= element_text("none"))
  print(maps)
}

```

Correlation to see association between variables, it is important to understand the presence of multi-collinearity in the dataset

```{r}
data_for_correlation <- as.matrix(energy_data1[10:length(energy_data1)])
corelation_matrix <- reshape2::melt(
  round(cor(data_for_correlation, use = 'pairwise.complete.obs'), 2)
  )
ggplot(data= corelation_matrix, aes(Var1, Var2)) + 
  geom_tile(aes(fill = value)) +
  scale_fill_gradient(low="blue", high ="orange")
```

Energy data contains energy trend since 2000 to 2020 for 178 countries. This project focuses in Nepal, model and forecast the future energy changes.

```{r}
nepal_energy_data <- energy_data %>% 
                     filter(Entity=="Nepal")
```

Observe the general trend for all variables

```{r}
for (i in which(colnames(nepal_energy_data)=="Year"):which(colnames    (nepal_energy_data)== "Density\\n(P/Km2)")){
  graph =ggplot(nepal_energy_data, aes(Year, nepal_energy_data[[i]]))+
    geom_line()+
    xlab(colnames(nepal_energy_data[i][1]))+ ylab("Measure")+
    theme_bw()
  print(graph)
}

```

Column with all NA and pop density has same data for two decades . Both of them can be removed.

1.  **Energy Consumption Prediction:** Predict future energy usage, aid planning, and track SDG 7 progress

    Create the regression model

    Co2 emission as a function of energy use and other variables, select required variables only using domain knowledge

    Nepal_energy_data\$Value_co2_emissions_kt_by_country \~ nepal_energy_data\$\`Access to electricity (% of population)\`: nepal_energy_data\$\`Energy intensity level of primary energy (MJ/\$2017 PPP GDP)\`

Step 1: rearrange the columns

```{r}
nepal_energy_data <- nepal_energy_data %>% 
                     select(Year, `Value_co2_emissions_kt_by_country`, 
                            `Access to electricity (% of population)`: `Energy intensity level of primary energy (MJ/$2017 PPP GDP)`)
```

step 2: Step wise regression to see if all variables are predictors for CO2 emission

```{r}
intercept_only <- lm(`Value_co2_emissions_kt_by_country` ~ 1, 
                      data=nepal_energy_data)

#define model with all predictors
all <- lm(`Value_co2_emissions_kt_by_country` ~ ., 
           data=nepal_energy_data)

#perform forward stepwise regression
forward <- step(intercept_only, direction="both" , 
                scope=formula(all), trace=0)

kableExtra:: kable(forward$anova)
kableExtra:: kable(forward$coefficients)

```

Step3: Feature selection

result from step-wise regression shows that only six variables are significant in defining co2 emission. this could be wrong but I believe in statistics though its seldom correct. So, keep only those columns

```{r}
nepal_feature_selected_data  <- nepal_energy_data %>% 
        select(Year,
Co2_emission = `Value_co2_emissions_kt_by_country`, 
Energy_perCap = `Primary energy consumption per capita (kWh/person)`,
renewble_fraction = `Renewable energy share in the total final energy consumption (%)`, 
electricity_perCap = `Renewable-electricity-generating-capacity-per-capita`,
low_carbon_electricity = `Low-carbon electricity (% electricity)`,
energy_intensity = `Energy intensity level of primary energy (MJ/$2017 PPP GDP)`
)

```

Split the data into test and train model

```{r}
set.seed(123)
smp_size <- floor(0.75 * nrow(nepal_feature_selected_data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(nepal_feature_selected_data)), size = smp_size)

train <- nepal_feature_selected_data[train_ind, ]
test <-  nepal_feature_selected_data[-train_ind, ] 
```

Fit the linear regression model on train data

```{r}
lm_model <- lm(data = train,
                   Co2_emission ~ (`Energy_perCap` +
                                     renewble_fraction + 
                                     electricity_perCap +
                                     low_carbon_electricity + 
                                     energy_intensity))
summary(lm_model)
t(broom::glance(lm_model)) ##glance creates in long columns, transpose changes it to one column format
```

Test model accuracy using test dataset

```{r}
train["model_predict"] <- predict(lm_model, train)

ggplot(train, aes(Year, Co2_emission))+
  geom_line()+
  geom_line(aes(Year, model_predict), color="green")
```

Linear model has high accuracy for the data set. Now use the same model to forecast the co2 consumption for next 10 years.

Use mathematic metrics MAE to check the accuracy of the model

```{r}
MAE <- (1/nrow(train))*sum((train$model_predict- train$Co2_emission), na.rm = TRUE)
print(MAE)
```

The MAE calculation gave output a number. It's hard to interpret. Normalize it between 0 to 1, so that low values of MAE is the best fit

```{r}
range_actual <- max(train$Co2_emission, na.rm = TRUE) - min(train$Co2_emission, 
                                                            na.rm=TRUE)
# Normalize MAE
normalized_MAE <- (MAE/range_actual)
normalized_MAE

##shorcut to measure more accuracy metrics
forecast:: accuracy(train$Co2_emission, train$model_predict)
```

Forecast using Naive method, or simply using linear regression model
