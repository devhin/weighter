# packages loading
library(shiny)
library(xlsx)
library(neuralnet)
library(arm)

# ui definition
ui <- fluidPage(
  # sidebar
  sidebarLayout(
    sidebarPanel(
      # Short introduction and instructions
      HTML("<h3>Weighter</h3>
           <br>
           Use it to predict your part weight based on the following parameters.<br>
           "),br(),
      
      # user inputs
      fluidRow(
        column(12, textInput("TT", "TT (0 ou 1)", 0)),
        column(12, textInput("TA", "TA (0 ou 1)", 0)),
        column(12, textInput("Fluo", "Fluo (0 ou 1)", 0)),
        column(12, textInput("Pieds", "Pieds (nb)", 0)),
        column(12, textInput("Poles", "Poles (nb)", 0)),
        column(12, textInput("Hauteur", "Hauteur (cm)", 0))
      ),
      
      # Compute button
      actionButton("compute", "Compute weight", class = "actn_button"),
      br(), br(),br(), br(), br(),br(), br(), br(), br(),br(), br(), br(), br(),
      width = 4),
    
    # main panel
    mainPanel(
      # set buttons style of to blue with white text
      tags$head(tags$style(".actn_button{width: 100%;margin-bottom:22px;} .actn_button{background-color: #0069a7; color: white;} 
                           .dl_button{width: 100%;margin-bottom:22px;} .dl_button{background-color: #0069a7; color: white;}")),
      tags$head(tags$style("#predicted_weight_lm{color: blue;font-size: 20px;font-style: bold;}")),
      tags$head(tags$style("#predicted_weight_nn{color: blue;font-size: 20px;font-style: bold;}")),
      tags$head(tags$style("#mse_lm{color: blue;font-size: 20px;font-style: bold;}")),
      tags$head(tags$style("#mse_nn{color: blue;font-size: 20px;font-style: bold;}")),
      br(),
      
      # display results of computation
      fluidRow(
        column(6, align="center", textOutput("predicted_weight_lm")),
        column(6, align="center", textOutput("predicted_weight_nn"))
      ),
      
      # display models structure (coeffs and graph)
      fluidRow(
        column(6, align="center", plotOutput("structure_lm")),
        column(6, align="center", plotOutput("structure_nn"))
      ),
      
      # display models accuracy on test set
      fluidRow(
        column(6, align="center", plotOutput("plot_lm")),
        column(6, align="center", plotOutput("plot_nn"))
      ), 
      
      # display models mean square error
      fluidRow(
        column(6, align="center", textOutput("mse_lm")),
        column(6, align="center", textOutput("mse_nn"))
      )
      
    )
  )
)

#server definition
server <- function(input, output, session) {
  
  # load data to model from .xlsx to dataframe
  data = read.xlsx2("data.xlsx", 1, header=TRUE, colClasses="numeric")
  
  # format data loaded to numeric
  data$TT = as.numeric(as.character(data$TT))
  data$TA = as.numeric(as.character(data$TA))
  data$Fluo = as.numeric(as.character(data$Fluo))
  data$Pieds = as.numeric(as.character(data$Pieds))
  data$Poles = as.numeric(as.character(data$Poles))
  data$Hauteur = as.numeric(as.character(data$Hauteur))
  data$Poids = as.numeric(as.character(data$Poids))

  # check no value is missing (ie set to NA)
  apply(data,2,function(x) sum(is.na(x)))
  
  # define training and test sets fom a random set of indexes: 75% data in training set
  index <- sample(1:nrow(data),round(0.75*nrow(data)))
  train <- data[index,]
  test <- data[-index,]
  
  # fit the linear model based on training set
  lm.fit <- glm(Poids~., data=train)
  summary(lm.fit)
  
  # apply the linear model to test set and compute mean square error
  pr.lm <- predict(lm.fit,test)
  MSE.lm <- sum((pr.lm - test$Poids)^2)/nrow(test)
  
  # prepare data to be modelled by neural net : min centered (max-min) scaling
  maxs <- apply(data, 2, max) 
  mins <- apply(data, 2, min)
  scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
  train_ <- scaled[index,]
  test_ <- scaled[-index,]
  
  # train neural net on training set (scaled). hidden=c(5,3,3) defines the nn structure
  n <- names(train_)
  f <- as.formula(paste("Poids ~", paste(n[!n %in% "Poids"], collapse = " + ")))
  nn <- neuralnet(f,data=train_,hidden=c(5,3,3),linear.output=T)
  
  # apply neural net to test set (scaled)
  pr.nn <- compute(nn,test_[,1:6])
  
  # unscale predicted test set and original scaled test set
  pr.nn_ <- pr.nn$net.result*(max(data$Poids)-min(data$Poids))+min(data$Poids)
  test.r <- (test_$Poids)*(max(data$Poids)-min(data$Poids))+min(data$Poids)
  
  # compute mean square error and display it for both lm and nn
  MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)

  # plot lm structure
  output$structure_lm <- renderPlot({
    coefplot(lm.fit)
  })
  
  #plot nn structure
  output$structure_nn <- renderPlot({
    plot(nn, rep="best")
  })
  
  # plot Real vs predicted lm test data (unscaled)
  output$plot_lm <- renderPlot({
    plot(test$Poids,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
    abline(0,1,lwd=2)
    legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)
  })
  
  # plot Real vs predicted NN test data (unscaled)
  output$plot_nn <- renderPlot({
    plot(test$Poids,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
    abline(0,1,lwd=2)
    legend('bottomright',legend='NN',pch=18,col='red', bty='n')
  })
  
  # declare and initialize reactive variables
  param <- reactiveValues(param = data[1,])
  param_scaled <- reactiveValues(param_scaled = data[1,])
  predicted_weight <- reactiveValues(weight_lm = 0, weight_nn = 0)
  
  # apply the hereabove defined linear model and neural net to user inputs when user clicks on compute button
  observeEvent(input$compute,{
    param$param$TT = as.numeric(as.character(input$TT))
    param$param$TA = as.numeric(as.character(input$TA))
    param$param$Fluo = as.numeric(as.character(input$Fluo))
    param$param$Pieds = as.numeric(as.character(input$Pieds))
    param$param$Poles = as.numeric(as.character(input$Poles))
    param$param$Hauteur = as.numeric(as.character(input$Hauteur))
    param_scaled$param_scaled <- param$param
    
    # compute weight predicted by linear model
    predicted_weight$weight_lm <- predict(lm.fit, param$param)
    
    # compute weight predicted by neural net (scale user inputs first)
    param_scaled$param_scaled <- as.data.frame(scale(param_scaled$param_scaled, center = mins, scale = maxs - mins))
    predicted_weight$weight_nn <- compute(nn,param_scaled$param_scaled[,1:6])
    predicted_weight$weight_nn <- predicted_weight$weight_nn$net.result*(max(data$Poids)-min(data$Poids))+min(data$Poids)
  })
  
  # display the predicted weight by both models
  output$predicted_weight_lm <- renderText({paste0("Weight predicted from user data by lm : ", predicted_weight$weight_lm, " kg")})
  output$predicted_weight_nn <- renderText({paste0("Weight predicted from user data by nn : ", predicted_weight$weight_nn, " kg")})
  
  # display lm and nn mean square error
  output$mse_lm <- renderText({paste0("Mean Square Error of lm : ", MSE.lm)})
  output$mse_nn <- renderText({paste0("Mean Square Error of nn : ", MSE.nn)})

}

shinyApp(ui = ui, server = server)