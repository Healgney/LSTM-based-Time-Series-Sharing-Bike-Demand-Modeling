# The Study of LSTM-based Time-Series Sharing Bike Demand Modeling​
A_PostG_Student_Course_Project

1.  **Introduction**

With the popularity of the low-carbon transportation concept and the influence of new technologies such as mobile payment, the sharing economy is gradually challenging the traditional transportation industry. With the introduction of low-carbon life and green transportation, the return and revival of non-motorized vehicles is an unstoppable trend. The urban public bicycle system is a product of cities promoting low-carbon environmental protection, which meets the needs of green travel and urban development, reduces carbon emissions, exercises the body, and raises residents' awareness of green environmental protection. The emergence and development of shared bicycles respond to the green and low-carbon environmental protection needs, so shared bicycles can develop steadily, and the shared bicycle transportation system will become increasingly mature. Bike-sharing improves connections to other transportation modes and is considered a very effective way to increase urban mobility (O'Brien et al., 2014). In the past few years, bike-sharing has proliferated worldwide and has become an essential component of urban multimodal transportation networks. The popularity and development of bike-sharing can solve the "last mile" problem in the transportation process, reduce traffic congestion, and address residents' demand for short-distance transportation (Lin et al., 2013). 

Although the development of bike-sharing has significantly impacted today's traffic situation, the analysis of bike-sharing travel patterns in current studies is still inadequate. The rapid development of shared bicycles and their explosive growth in a concise period have made it more convenient for citizens to use bicycles. However, problems such as disorderly parking and indiscriminate wearing and trespassing have emerged one after another, and how to solve these problems is a new challenge in urban traffic management planning issues (Zhou, 2015). A complete non-motorized travel network needs to be established to manage urban traffic. However, the current site-based bicycle sharing (SBBS) has limitations regarding supply and demand constraints. Many previous studies have pointed out that existing planned bicycle lanes and parking areas are insufficient to meet the rapidly growing demand for bicycles, which may lead to low availability of shared bicycles in some locations during peak hours while other stations are filled with parking spaces, resulting in problems such as inability to return bicycles (Chen et al., 2016). Such innovative modes of travel bring new pressures and challenges to urban governance, mainly in the following aspects： 

\(1\) Over-deployment and waste of public resources 

In the stage of rough growth of shared bicycles, the differentiation of shared bicycles among enterprises is not apparent, and the entry threshold is relatively low. In order to seize the market share and improve the industry's competitiveness, companies often tend to form a capital competition barrier by placing bicycles on a large scale. This has resulted in an oversupply of bicycle resources, crowding out urban road resources. As part of the public transportation service, the launch and dispatch of shared bicycles should be regulated so that they can realize the original purpose of the sharing economy, and the focus of management should shift from expanding the launch to matching supply and demand. 

 (2) Uneven distribution and difficulty in balancing supply and demand 

Although there is a surplus of shared bicycles, the phenomenon that one bicycle is hard to find in some areas still exists. This is due to the uneven spatial and temporal distribution of bicycle-sharing resources and the imbalance between supply and demand. For example, the overall flow of bicycles in the morning is from living areas to work areas, and many bicycles are docked around subway stations and office buildings. In contrast, demand in other areas is difficult to meet, resulting in a loss of potential demand. Therefore, if the level of service satisfaction is to be improved and the advantages of shared bicycles are to be entirely played, operation managers must first deal with the balance between supply and demand and effectively rebalance the bicycles. 

Many recent papers that have used machine learning to study problems related to urban bicycle transportation systems show that machine learning is applicable to data fusion problems and has an excellent performance in short-term prediction problems.

2.  **Literature Review**

Scholars around the world have investigated bicycle sharing, and the current research on bicycle sharing primarily focuses on the development history and evaluation of bicycle sharing, the travel characteristics mining of bicycle sharing, the traffic demand prediction and site optimization of bicycle sharing, and the optimal scheduling of bicycle sharing. It is essential to accurately predict the demand for borrowing and returning bicycles at each station or area of the bicycle sharing system in order to control the number of bicycles deployed, increase the utilization rate of shared bicycles, and optimize the scheduling problems associated with shared bicycles.

Based on our research questions, we review the research conducted by scholars in the last decade on the problem of demand forecasting for shared bicycles. The current forecasting models for bicycle sharing demand forecasting models can be summarized into three main categories: time-series forecasting models based on statistical methods, traditional machine learning models, and deep learning models.

**2.1 Traditional Time Series Forecasting Models**

Traditional time series forecasting models are comprised primarily of the History Average (HA) method and the Autoregressive Integrated Moving Average (ARIMA) model and its variants(Contreras et al., 2003; Williams & Hoel, 2003). Among all the time series methods, the ARIMA is the most widely used forecasting model based on historical data, which(Belloni et al., 2014) use the values of the input variables and the error term to forecast. For instance, Kaltenbrunner et al. used the ARMA model for the first time to predict the number of public bicycles in the Barcelona region over a short period of time, and then analyzed the geographic movement patterns of public bicycles based on the prediction outcomes (Kaltenbrunner et al., 2010). Yoon et al. suggested a spatial-temporal prediction system based on the ARIMA model in order to address the non-stationarity problem inherent in the ARMA model (Yoon et al., 2012). Compared to ARMA, their findings indicate a small improvement in favor of ARIMA. Lin and Zhang examined the features of Mobike's Shanghai inhabitants' trip, compared to previous forecasting approaches, ARIMA avoids some complex influencing aspects, performs a more objective and realistic sustainable projection, and improves the prediction's accuracy (Mehdizadeh Dastjerdi & Morency, 2022).

However, traditional time-series forecasting models rely solely on historical data of bicycle sharing to predict demand, and rarely take into account other factors that may influence the future demand for bicycle sharing. Moreover, traditional time-series forecasting models heavily rely on the linearity and smoothness assumptions of the data, which is contrary to the highly nonlinear and dynamic nature of the demand changes for bicycle sharing, resulting in the poor performance of these models in practical applications.

**2.2 Traditional Machine Learning Models**

Due to the inaccuracy of traditional time-series prediction models, numerous researchers have begun to attempt to predict the demand for shared bikes using machine learning models, with the methods primarily consisting of Support Vector Machines (SVMs), Artificial Neural Networks (ANNs), and some integrated learning methods based on decision trees. For example, Bacciu et al. used support vector machine and random forest to predict the short-time return of docked bike-sharing sites, which provide a useful addition in economic research (Belloni et al., 2014). Yang et al. used the sliding window method to extract features from historical bike-sharing borrowing and returning quantities, which were then used to train a Random Forest Model using the Hangzhou docked bicycle sharing dataset (Yang et al., 2016). Marco A et al. proposed a new model selection method using SVM to combine different criteria, given a set of candidate models, rather than considering any individual criteria, and train the SVM at each prediction source to select the best model (Villegas et al., 2018).

However, machine learning model can model shared bicycle demand and extract more complex correlations. Moreover, in addition to utilizing historical demand In addition to constructing features based on historical demand, machine learning models can also obtain similar sites to the target site by clustering. Additionally, external factors can be utilized effectively, such as weather, holidays, etc. However, these methods are frequently incapable of handling the original sample data and rely heavily on feature engineering, which is highly subjective; and feature engineering is typically complex because it requires careful engineering design and substantial domain expertise.

**2.3 Deep Learning Models**

Instead of traditional machine learning algorithms, deep learning models for temporal prediction excel at representation learning, which means they do not need to manually extract features from raw data, and they tend to perform well when sufficient data is available for training. Deep learning models for temporal prediction are comprised primarily of Recurrent Neural Networks (RNN) and their variants, which can implement more complex mapping relationships for deep learning. Numerous researchers are currently employing diverse temporal prediction deep learning models to investigate the bike-sharing demand prediction problem. For instance, Yan Pan et al. created a bicycle traffic network in Chicago using CitiBike data and trained an LSTM model to predict bicycle rentals and returns utilizing the gating mechanism of long and short-term memory and the capacity of recurrent neural networks to interpret sequential input (Pan et al., 2019). Collini et al. compared emerging machine learning methods and deep learning methods for the short-term prediction of the number of available bikes and free bike slots in bike share stations (Collini et al., 2021). This study concluded that deep learning, particularly Bi-directional Long Short-Term Memory (Bi-LSTM), provided a robust method to predict available bicycles quickly and accurately, even with limited historical data. Mehdizadeh et al. utilized a four-group LSTM-based model to predict bike-sharing pickup demand under COVID-19, and adopted a univariate ARIMA model to compare results as a benchmark (Mehdizadeh Dastjerdi & Morency, 2022). Results indicate that deep learning models outperform ARIMA models significantly.

Unfortunately, these deep learning models based on time-series prediction cannot show deeper connections between stations or regions and do not use relevant information on spatial distribution, resulting in a lack of prediction accuracy for peak hours, which is crucial for predicting these high-demand moments accurately. In addition, many researchers have also applied deep learning techniques such as Convolutional Neural Networks (CNN), Graph Convolutional Network (GCN), and Temporal Convolutional Networks (TCN) to various facets of traffic flow prediction. As neural network research progresses, more and more scholars are integrating deep learning techniques into traffic prediction research. The results of the example analysis demonstrate that such combined models significantly reduce the workload of manual feature extraction and can frequently produce more efficient and accurate prediction results than single models and conventional time-series prediction methods with sufficient training samples.

Among similar deep learning algorithms, there is another integrated model called T-GCN (Temporal Graph Convolutional Network), which is a neural network-based traffic prediction method. Unlike CNN (Convolutional Neural Network), T-GCN combines Graph Convolutional Network (GCN) and Gated Recursive Unit (GRU) to capture spatial dependencies by learning complex topology with GCN, and temporal dependencies by learning dynamic changes of traffic data with GRU. Therefore, T-GCN has the capability of temporal feature learning and can capture both spatial and temporal dependencies simultaneously. Zhao et al. proposed T-GCN model which combines the GCN with the GRU. This model can extract the spatio-temporal correlation from traffic data, and its predictions on real-world traffic datasets exceed state-of-the-art baselines (Zhao et al., 2020).

3.  **Objectives **

- Dealing with weather and temporal features. Explore, analyze and quantify how weather and temporal features effect the demand of shared bikes.

- Use the LSTM algorithm to predict and analyze the demand for shared bicycles on a whole system scale at each time period.

4.  **Study Area and Datasets**

&nbsp;

1.  Study Area

New York is located at 74°00′ W, 40°43′N, at the mouth of the Hudson River on the northeastern coast of the United States, bordering the Atlantic Ocean. It is bounded by New Jersey to the west and Rhode Island to the north. It has a total area of 1,214 square kilometers, of which 425 square kilometers is water and 789 square kilometers is land.

New York is located in the Hudson River estuary plain, the city's topography is low, the highest elevation is in Staten Island's Mount Todd, about 124.9 meters, is the highest in the eastern coast of the United States south of Maine.

New York has a temperate continental humid climate with cold winters and cool summers, with average temperatures below 0°C in January and 21°C in July. Average annual precipitation ranges from 820-1100 mm. The warm season lasts about 3.5 months, from early June to mid-September, with an average daily high temperature of 29°C and a low temperature of 22°C. The cold season lasts about 3 months, from early December to early March, with an average daily high temperature of 4°C and a low temperature of -2°C. The rainy season is from late March to late August. The rainy season is usually accompanied by a hurricane season, with relatively few hurricanes and tropical storms, but they do occur occasionally.

The Hudson River in New York flows through the Hudson Valley and enters New York Bay, dividing New York City and New Jersey. The East River flows through Long Island Sound, separating the Bronx and Manhattan from Long Island. The Harlem River lies between the East River and the Hudson River, separating the boroughs of Manhattan and the Bronx. The Bronx River, which flows through the Bronx and Westchester County, is the only river in New York City that is all fresh water.

![Study Areajpg](https://user-images.githubusercontent.com/8214596/209771068-1a0debd0-4e9f-4618-9417-f3ea41eddd36.jpg)

2.  Datasets

We obtained the shared bicycle data from July 1, 2021, 0:00 to July 31, 2022, 23:00 from citibike official website. The original data recorded 32785175 trips, sampled at hourly frequency, and the length of the sampled data set is 9504. 

For weather-related data, we have obtained hourly data on the visual crossing website for July 2021 to July 2022, including temperature, weather conditions, precipitation, etc.

To deal with temporal features, we searched for the local holidays including the Forth of July, Veteran Day, Christmas & New Year Holiday, etc. The ‘is holiday/weekend’ attribute is coded with 1 with corresponding time, while other times are coded with 0. The ‘is weekday’ attribute is also coded with 1 with corresponding time, while other times are also coded with 0.

From the related bicycle sharing research articles, we know that the weather factor has a great influence on the need hand climate factor of bicycle sharing, so we take the weather factor as a main research direction. We assign a value to each different weather condition, and the worse the weather condition is, the higher his value is. For example, the value of weather condition for a sunny day is 0, and the value of weather condition for a snowy day is 5. However, there are lots of complex weather condition in real life. For instance, as shown in table 2, rain often comes with overcast, that's a combination of 5. In addition to the weather factor, we also included weekdays, weekend and holidays as influencing factors due to the different commuting and transportation needs on weekdays and weekends. 

**Table 1.** Corresponding Cumulative Weight of Different Conditions 

| Conditions in detail                          | Value |
| --------------------------------------------- | ----- |
| Rain                                          | 3     |
| Snow, Rain, Overcast                          | 10    |
| Freezing Drizzle/Freezing Rain, Ice, Overcast | 12    |
| Snow, Overcast                                | 7     |
| Ice, Overcast                                 | 8     |
| Snow, Rain, partially cloudy                  | 9     |
| Overcast                                      | 2     |
| Clear                                         | 0     |
| Rain, Ice, Overcast                           | 11    |
| Rain, Overcast                                | 5     |
| Rain, partially cloudy                        | 4     |
| Freezing Drizzle/Freezing Rain, Overcast      | 6     |
| Snow, partially cloudy                        | 6     |
| Partially cloudy                              | 1     |
| Snow, Rain                                    | 8     |

![monthly demand](https://user-images.githubusercontent.com/8214596/209771132-108c7078-0374-49b8-9021-5eeee487fb97.png)

![annual daily demand](https://user-images.githubusercontent.com/8214596/209771183-49e1869c-8eb6-4e4a-9bd6-665abcbac951.png)



**Figure 1.** Amount of Bike Shares

Looking at the monthly usage trends for a full year of bike-sharing data, the demand for bikes starts to drop in October, reaches the lowest point at February, then increases to reach the peak at August. When we look at the daily bicycle usage statistics, there're some significant drop in bicycle sharing usage on certain days, which we consider them as public holidays. 

![bike count](https://user-images.githubusercontent.com/8214596/209771200-3ced9ef0-69f2-4fb6-8087-63c80d9a700d.png)

**Figure 2.** Box Plot on Bike Share Count

The box plot of overall bike-sharing demand shows that the data spans a wide range, while half of the data is distributed between 800 and 5200, and the median demand is about 2800. 

![12monthly demand](https://user-images.githubusercontent.com/8214596/209771211-d3821a80-b5b1-4515-b939-90a949486b7f.png)

**Figure 3.** Box Plot on Count Across Months

The box plot of bike-sharing demand for each month of the year shows that the period of the year with the highest demand for bike-sharing is from June to September, considering that the temperature and climate in summer are more suitable for people to Choose cycling as a mode of transportation. More specifically, the demand for bikes in summer is about twice as high as in winter. 

![weather cond](https://user-images.githubusercontent.com/8214596/209771225-ccf53533-f2de-4b3b-a62c-01895199e528.png)

**Figure 4.** Box Plot on Count Across Weather Conditions

We next examined the weather conditions index and saw that more users chose bike-sharing trips when the weather conditions value was between 0 and 4. However, when the weather conditions value rises to 5, the demand for bike-sharing rapidly decreases and there are more outliers, indicating that the choice of bike-sharing users for weather conditions value of 5 varies more. When the weather is more severe, very few users continue to choose bicycle travel. 

![week:day](https://user-images.githubusercontent.com/8214596/209771237-1b72261c-5675-4969-9ad3-2d509ca253d6.png)

**Figure 5.** Box Plot on Count Across Working Days

The statistics for weekend and weekday bicycle demand show that there is no significant difference between the two figures, with values ranging roughly from 1000 to 5500. 

![hourly demand](https://user-images.githubusercontent.com/8214596/209771247-9f34dc9e-932c-45f4-8564-f1b72c6669a0.png)

**Figure 6.** Box Plot on Count Across Hour of the Day

When the demand for shared bikes is counted on an hourly basis, it can be seen that 8 a.m. and 6 p.m. are the two peak times for shared bike use. These two times are also the peak commuting times for most citizens. Bike usage at 6 p.m. is significantly higher than 8 a.m., indicating that more users prefer riding off work than riding to work. 

![temperature:demand](https://user-images.githubusercontent.com/8214596/209771254-5317f036-ddf6-42f9-9003-7ea59d661624.png)

** Figure 7.** Box Plot on Count Across Temperature

The graph of the relationship between temperature and bike-sharing usage shows that few people choose to travel by bike when the temperature is below zero. When the temperature is between 0 and 10 degrees, the demand for bikes rises, however, the usage is mostly 400 to 800 per hour, which is still a relatively small value. When the temperature is above 10 degrees, the demand rises as the temperature rises, and at a temperature of around 28 to 30 degrees, the maximum hourly bicycle demand occurs, at around 12,400. 

5.  **Methodology**

**4.1 Introduction of LSTM  **

LSTM network is another variant model based on Recurrent Neural Network (RNN), in which each independent RNN unit in the transmission chain passes the information extracted by itself to the next independent unit, and when the length of the transmission chain increases to a certain level, information is lost and "forgetting" occurs. ". To solve this dependency problem, Hochreiter et al. first proposed LSTM (Hochreiter & Schmidhuber, 1997) in 1997. The independent RNN unit and RNN structure are shown in Figs. 7 and 8. 

![ind_rnn_unit](https://user-images.githubusercontent.com/8214596/209771272-fb0189fd-1a54-4516-8eff-dfc72d3820a2.png)

**Figure 8.** Independent RNN Unit 

![rnn](https://user-images.githubusercontent.com/8214596/209771294-a1696a0c-794a-4edc-abeb-73954235183b.png)

**Figure 9.** RNN Structure 

![lstm_cell](https://user-images.githubusercontent.com/8214596/209771306-566d087b-0c4e-48ee-aa18-b17e82d7a0ff.png)

**Figure 10.** LSTM Cell Structure 

LSTM employs three gating mechanisms to solve the long-term dependency problem in traditional RNNs: Forget Gate *f*, Input Gate *i,* and Output Gate *o*. The LSTM cell structure is shown in figure 10. In all of the following equations, 𝜎 is *Sigmoid Function*, $Sigmoid = \frac{1}{1 + e^{- x}}$, will transform the input content into a vector in the range (0, 1). *W_(f), W_(i), W_(o),* and *W_(c)*indicate the weight coefficient matrixes of LSTM cell states in the update process; *b_(f), b_(i), b_(o)*, and *b_(c)*represent the bias coefficient matrixes in the cell state update process; ${\widetilde{C}}_{t}$ Indicates candidate cell unit information. 

In the LSTM, the forgetting gate is used to control which information needs to be "forgotten" for the current cell state, and the cell state is updated through the forgetting gate, as shown in equation (1). 

$f_{t} = \sigma(W_{f}\left\lbrack h_{t - 1},x_{t} \right\rbrack + b_{f})$ （1）

The role of input gate *i*, also known as memory gate, is that it can control the degree of retention of the current network input $x_{t}$ in cell unit state $C_{t}$, and its update mechanism is shown in equations (2), (3) and (4). 

$i_{t} = \sigma(W_{i}\left\lbrack h_{t - 1},x_{t} \right\rbrack + b_{i})$ （2）

${\widetilde{C}}_{t} = tanh(W_{c}\left\lbrack h_{t - 1},x_{t} \right\rbrack + b_{c})$ （3）

${C_{t} = f}_{t}\  \times \ {\widetilde{C}}_{t - 1}{+ i}_{t}\  \times \ {\widetilde{C}}_{t}$ （4）

The role of the output gate is to control how much information is output for the current cell state, and its update mechanism is shown in equations (5) and (6). 

$o_{t} = \sigma(W_{o}\left\lbrack h_{t - 1},x_{t} \right\rbrack + b_{o})$ （5）

$h_{t} = o_{t} \times tanh(C_{t})$ （6）

**4.2 Bi-LSTM **

![bi_lstm_structure](https://user-images.githubusercontent.com/8214596/209771340-7c9839d5-fa32-4d32-a840-c703949b486e.png)

**Figure 11.** Bi-LSTM Structure 

Bidirectional LSTM (Bi-LSTM) is a recurrent neural network used primarily for natural language processing. As an improvement of RNN, unlike standard LSTM, the input flows in both directions and can utilize information from both sides. bi-LSTM can consider both forward and backward temporal state information, and backward inference is used to complement temporal inference to avoid the limitation of temporal causality in the learning process. 

The Structure of Bi-LSTM is shown in the figure 11. In summary, Bi-LSTM adds one more LSTM layer, which reverses the direction of information flow. Briefly, it means that the input sequence flows backward in the additional LSTM layer. Then we combine the outputs from both LSTM layers in several ways, such as average, sum, multiplication, or concatenation. 

A Bi-LSTM algorithm is used in this study. 

**4.3 Modelling Tasks **

In this study, the model code is written in Python language, and the Tensorflow-Keras-based development framework is used to build the LSTM model to implement the computation. The model is trained using Apple Silicon M1 Pro GPU. 80% of the randomly selected dataset is used as the training set for feature learning of the model, and 20% is used as the test set to evaluate the model prediction capability. 

According to the model training objective, the following training task is defined: to predict 1 future record using 24 historical records of bike-sharing and their attribute features. A sample containing 24 bike-sharing history records of length and their attribute features is used as the model input, followed by 1 bike-sharing record of length as the true value. A sliding window with a window size of 25 is used to generate all data samples with a step size of 1. 

The experiment optimizes the model parameters with the samples from the training set, optimizes the hyperparameters with the results of the runs on the evaluation set, and validates the performance of the model on the test set. The performance evaluation metrics include the root mean square error (RMSE) and the mean absolute error (MAE), which are calculated as follows. 

$RMSE = \sqrt{\frac{1}{N}\sum_{i = 1}^{N}{(Y_{i} - {\widetilde{Y}}_{i})}^{2}}$ （7）

$MAE = \frac{1}{N}\sum_{i = 1}^{N}{|Y_{i} - {\widetilde{Y}}_{i}|}$ （8）

The smaller the deviation between the predicted and true values, the smaller the corresponding RMSE value, and the better the prediction fit of the model. 

In the training process, the Mean Square Error (MSE) loss function is used to measure the test error in the test set after each epoch iteration, the adaptive estimation Adam is chosen as the optimizer, and the model parameters are saved at this time. 

The model is trained according to the model structure and learning strategy that have been set. In the training process, the number of samples input to the model, i.e., the size of the training batch of samples (Batch size), has an impact on the training time and accuracy of the model. too large or too small a batch-size selection will affect the model prediction effect and lead to an increase in the prediction result error. When training the model in this study, different combinations of training hyperparameters are set to perform the training task. The training batch sizes of 32 and 64 are used for each sample, the number of training rounds epoch is 30 and 50, the corresponding test data of the model is specified for each round, and each training record is output. After training according to each combination of training hyperparameters, the error results are recorded and compared. 

**Table 2.** Comparison of different training settings 

| Epoch  | Batchsize | RMSE    | MAE     |
| ------ | --------- | ------- | ------- |
| 50     | 64        | 649     | 444     |
| **30** | **32**    | **632** | **391** |
| 50     | 32        | 644     | 420     |
| 30     | 64        | 655     | 426     |

![loss plot](https://user-images.githubusercontent.com/8214596/209771354-13e1b569-0aef-4337-893f-1d0ccff92f77.png)

**Figure 12.** Loss Function

Figure 12 shows the changing loss function of the model in the training and test sets during the training process, and it can be seen that the curve is decreasing and finally converges to zero. 

6.  **Results Visualization **

Choose 400 prediction results arbitrarily and visualize in the graph below.  

![pred_result](https://user-images.githubusercontent.com/8214596/209771364-5b6f602d-9a68-4516-a873-695b0dfc0da8.png)

**Figure 13.** Prediction Results

The Bi-LSTM model can predict the data well, and the predicted hourly usage curve matches the trend of the actual vehicle usage curve, and the model fits well and meets the empirical error requirement in the prediction process

- Quantifying the Factorial Impact Effect  

> In exploring and quantify the influential features for weather and temporal, we are setting four sets of experiments for comparison: all sets are modeled with the best model settings(Batchsize:32, Epoch:30). 

1.  Complete data 

&nbsp;

2.  Complete data without weather features 

&nbsp;

3.  Complete data without temporal features 

&nbsp;

4.  Complete data without both weather features and temporal features 

**Table 3.** Feature Results 

| Features                                     | Error Evaluation |                    |         |                    |
| -------------------------------------------- | ---------------- | ------------------ | ------- | ------------------ |
|                                              | RMSE             | Accuracy Decreased | MAE     | Accuracy Decreased |
| **Complete data**                            | **632**          | **-**              | **391** | **-**              |
| Complete data wo/ temporal features          | 663              | 4.91%              | 442     | 13.04%             |
| Complete data wo/ weather features           | 727              | 14.33%             | 494     | 23.30%             |
| Complete data wo/ weather, temporal features | 707              | 10.32%             | 469     | 15.79%             |

(Unit: times / 13 months)  

In order to verify the effect of adding weather and temporal features on the prediction accuracy of the model, different combinations of features were used to construct the prediction model to test the test set, and then the performance of the model was evaluated using RMSE and MAE metrics for different combinations, respectively. The evaluation results are shown in the table. 

The prediction result of the complete dataset trained with the best model settings is 632 in RMSE, 391 in MAE. Smaller RMSE/MAE indicates more accurate prediction results. From the table, it can be seen that by reducing one feature in the model training, the root means square error and mean absolute error of the model prediction results increase. It can be seen that the prediction accuracy obtained by adding weather features and temporal features is higher than that of the data without weather features and temporal features(trip counts time series alone). 

However, it can be seen that temporal features are less influential than weather features in affecting the demands of shared bikes. Adding temporal features alone could possibly introducing inaccurate priori knowledge to the deep learning model that resulted in lower accuracy of the prediction results. Therefore, more accurate prediction are the results of adding the combination of temporal and weather features.  

7.  **Conclusion **

In this paper, we first obtained the open-source bicycle sharing data of citibike, dated from July 1, 2021 to July 31, 2022, with a total of 32785175 bicycle sharing trip records. And the trip records are sampled and accumulated in hourly frequency, and the resampled data are concatenated with corresponding weather features and time features. The following conclusions based on the data features are concluded:

- Monthly demand for shared bicycles is highest in August and lowest in February.

- Daily demand drops significantly on individual public holidays.

- Hourly demand figures for bicycle sharing span a wide range, with a median of about 2,800, and seasonally, demand is about twice as high in the summer as it is in the winter.

- In terms of weather, bike-sharing demand is higher with the weather condition of little or no precipitation.

- Throughout the 24-hour period of the day, bike-sharing demand peaked at 8 a.m. and 6 p.m. during peak commuting hours.

- The relationship between temperature and bike-sharing demand is that demand rises with temperature, with the highest hourly demand occurring when the temperature is between 28 and 30. 

- Separating holiday/weekend and weekday statistics within a year reveals no significant difference in demand between the two.

Then, we used the Bi-LSTM model with selected different model hyperparameter settings to predict the hourly demand before and after removing features. The results show that the accuracy of the prediction results decreases to different degrees after the removal of features, indicating that the demand for shared bicycles is inextricably linked with weather features and temporal features. Moreover, it can be seen in the results that weather features are the major features influencing the demand of share bikes. The influence for temporal features like Holiday, Weekend have no significant impact on the demand of shared bikes.  The model is able to predict the trend of bike-sharing demand well. LSTM can effectively predict the demand of bike shares. 

**  
**



**References**

Belloni, A., Chernozhukov, V., & Hansen, C. (2014). High-Dimensional Methods and Inference on Structural and Treatment Effects. *Journal of Economic Perspectives*, *28*(2), 29–50. https://doi.org/10.1257/jep.28.2.29

Chen, L., Zhang, D., Wang, L., Yang, D., Ma, X., Li, S., Wu, Z., Pan, G., Nguyen, T.-M.-T., & Jakubowicz, J. (2016). Dynamic cluster-based over-demand prediction in bike sharing systems. *Proceedings of the 2016 ACM International Joint Conference on Pervasive and Ubiquitous Computing*, 841–852. https://doi.org/10.1145/2971648.2971652

Collini, E., Nesi, P., & Pantaleo, G. (2021). Deep Learning for Short-Term Prediction of Available Bikes on Bike-Sharing Stations. *IEEE Access*, *9*, 124337–124347. https://doi.org/10.1109/ACCESS.2021.3110794

Contreras, J., Espinola, R., Nogales, F. J., & Conejo, A. J. (2003). ARIMA models to predict next-day electricity prices. *IEEE Transactions on Power Systems*, *18*(3), 1014–1020. https://doi.org/10.1109/TPWRS.2002.804943

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kaltenbrunner, A., Meza, R., Grivolla, J., Codina, J., & Banchs, R. (2010). Urban cycles and mobility patterns: Exploring and predicting trends in a bicycle-based public transport system. *Pervasive and Mobile Computing*, *6*(4), 455–466. https://doi.org/10.1016/j.pmcj.2010.07.002

Lin, J.-R., Yang, T.-H., & Chang, Y.-C. (2013). A hub location inventory model for bicycle sharing system design: Formulation and solution. *Computers & Industrial Engineering*, *65*(1), 77–86. https://doi.org/10.1016/j.cie.2011.12.006

Mehdizadeh Dastjerdi, A., & Morency, C. (2022). Bike-Sharing Demand Prediction at Community Level under COVID-19 Using Deep Learning. *Sensors*, *22*(3), Article 3. https://doi.org/10.3390/s22031060

O'Brien, O., Cheshire, J., & Batty, M. (2014). Mining bicycle sharing data for generating insights into sustainable transport systems. *Journal of Transport Geography*, *34*, 262–273. https://doi.org/10.1016/j.jtrangeo.2013.06.007

Pan, Y., Zheng, R. C., Zhang, J., & Yao, X. (2019). Predicting bike sharing demand using recurrent neural networks. *Procedia Computer Science*, *147*, 562–566. https://doi.org/10.1016/j.procs.2019.01.217

Villegas, M. A., Pedregal, D. J., & Trapero, J. R. (2018). A support vector machine for model selection in demand forecasting applications. *Computers & Industrial Engineering*, *121*, 1–7. https://doi.org/10.1016/j.cie.2018.04.042

Williams, B. M., & Hoel, L. A. (2003). Modeling and Forecasting Vehicular Traffic Flow as a Seasonal ARIMA Process: Theoretical Basis and Empirical Results. *Journal of Transportation Engineering*, *129*(6), 664–672. https://doi.org/10.1061/(ASCE)0733-947X(2003)129:6(664)

Yang, Z., Hu, J., Shu, Y., Cheng, P., Chen, J., & Moscibroda, T. (2016). Mobility Modeling and Prediction in Bike-Sharing Systems. *Proceedings of the 14th Annual International Conference on Mobile Systems, Applications, and Services*, 165–178. https://doi.org/10.1145/2906388.2906408

Yoon, J. W., Pinelli, F., & Calabrese, F. (2012). Cityride: A Predictive Bike Sharing Journey Advisor. *2012 IEEE 13th International Conference on Mobile Data Management*, 306–311. https://doi.org/10.1109/MDM.2012.16

Zhao, L., Song, Y., Zhang, C., Liu, Y., Wang, P., Lin, T., Deng, M., & Li, H. (2020). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction. *IEEE Transactions on Intelligent Transportation Systems*, *21*(9), 3848–3858. https://doi.org/10.1109/TITS.2019.2935152

Zhou, X. (2015). Understanding Spatiotemporal Patterns of Biking Behavior by Analyzing Massive Bike Sharing Data in Chicago. *PLOS ONE*, *10*(10), e0137922. https://doi.org/10.1371/journal.pone.0137922
